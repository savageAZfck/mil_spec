//! apple_decode_bench — drive an Apple-path (.mlmodelc shard dir) ANE
//! decode and time tokens/sec. This is the "their toolchain" lane of
//! the head-to-head vs qwen_decode (self-emitted HWX).
//!
//!   apple_decode_bench <shard_dir> <steps> [seed_token]

use mil_infer::ane_pipeline::Pipeline;
use mil_infer::e5rt::DeviceMask;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(&args[1]);
    let steps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let mut tok: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(9707);

    let (layers, heads) = mil_infer::ane_pipeline::shard_paths(&dir)?;
    println!("{} layer shards, {} head shards", layers.len(), heads.len());
    let pipe = Pipeline::open_dir(&dir, DeviceMask::Ane)?;
    println!("pipeline open; shared inputs: {:?}", {
        let names = pipe.state_names();
        let mut v: Vec<String> = Vec::new();
        for n in ["attn_mask", "kv_write_mask", "rope_cos", "rope_sin"] {
            if let Some(e) = pipe.shared_elems(n) {
                v.push(format!("{n}={e}"));
            }
        }
        let _ = names;
        v
    });

    let emb = std::fs::File::open(dir.join("embedding.f16"))?;
    let (slot, h) = pipe.embedding_slot()?;
    let h = h as usize;
    let seq = pipe.shared_elems("attn_mask").unwrap_or(2048);
    let rope_half = pipe.shared_elems("rope_cos").unwrap_or(64);
    pipe.reset_state()?;

    let mut tokens = vec![tok];
    let mut times = Vec::new();
    for pos in 0..steps {
        // embed current token
        let mut row = vec![0u8; h * 2];
        emb.read_exact_at(&mut row, (tok * h * 2) as u64)?;
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), slot as *mut u8, row.len()) };

        // masks + rope for this position
        let mut attn = vec![f16(-10000.0); seq];
        for i in 0..=pos.min(seq - 1) {
            attn[i] = 0;
        }
        pipe.set_shared("attn_mask", &attn).ok();
        let mut kvw = vec![0u16; seq.min(pipe.shared_elems("kv_write_mask").unwrap_or(seq))];
        if pos < kvw.len() {
            kvw[pos] = f16(1.0);
        }
        pipe.set_shared("kv_write_mask", &kvw).ok();
        let (mut rc, mut rs) = (vec![0u16; rope_half], vec![0u16; rope_half]);
        for i in 0..rope_half {
            let freq = 1e6f64.powf(-2.0 * i as f64 / (rope_half * 2) as f64);
            let ang = pos as f64 * freq;
            rc[i] = f16(ang.cos() as f32);
            rs[i] = f16(ang.sin() as f32);
        }
        pipe.set_shared("rope_cos", &rc).ok();
        pipe.set_shared("rope_sin", &rs).ok();

        let t = std::time::Instant::now();
        pipe.step()?;
        let us = t.elapsed().as_secs_f64() * 1e6;
        times.push(us);

        let logits = pipe.logits();
        tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| {
                half::f16::from_bits(*a.1)
                    .to_f32()
                    .total_cmp(&half::f16::from_bits(*b.1).to_f32())
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        tokens.push(tok);
        println!("step {pos}: {us:.0}us -> {tok}");
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "apple path: min {:.0}us mean {:.0}us p50 {:.0}us => {:.2} tok/s",
        times[0],
        mean,
        times[times.len() / 2],
        1e6 / mean
    );
    println!("tokens: {tokens:?}");
    Ok(())
}
