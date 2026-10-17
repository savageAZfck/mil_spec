//! e5_conform — bisect the direct path vs the MLModel path.
//!
//!   cargo run -p mil_infer --example e5_conform -- <shard_dir> <n_layers> <token>
//!
//! Runs ONE real decode step (embedding row, pos-0 rope/masks) through
//! `n_layers` shards with NO heads, then prints the hidden vector head —
//! comparable element-for-element against the Swift single-layer probe.

use mil_infer::ane_pipeline::{shard_paths, Pipeline};
use mil_infer::e5rt::DeviceMask;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(&args[1]);
    let n: usize = args[2].parse()?;
    let token: u64 = args[3].parse()?;

    let with_heads = std::env::var("HEADS").is_ok();
    let (mut layers, heads) = shard_paths(&dir)?;
    layers.truncate(n);
    let pipe = Pipeline::open(
        &layers,
        if with_heads { &heads } else { &[] },
        DeviceMask::Ane,
    )?;

    // embedding row
    let emb = std::fs::File::open(dir.join("embedding.f16"))?;
    let (slot, h) = pipe.embedding_slot()?;
    let mut row = vec![0u8; h as usize * 2];
    emb.read_exact_at(&mut row, token * h * 2)?;
    unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), slot as *mut u8, row.len()) };

    // pos-0 inputs: rope cos=1 sin=0; attn opens col0; write-mask[0]=1
    let seq = 2048usize;
    let mut attn = vec![
        half::f16::from_f32(-10000.0).to_bits();
        pipe.shared_elems("attn_mask").unwrap_or(seq)
    ];
    let rows = attn.len() / seq;
    for r in 0..rows {
        attn[r * seq] = 0;
    }
    pipe.set_shared("attn_mask", &attn)?;
    let mut kv = vec![0u16; seq];
    kv[0] = half::f16::from_f32(1.0).to_bits();
    pipe.set_shared("kv_write_mask", &kv)?;
    pipe.set_shared("rope_cos", &[half::f16::from_f32(1.0).to_bits(); 64])?;
    pipe.set_shared("rope_sin", &[0u16; 64])?;
    pipe.reset_state()?;

    pipe.step()?;

    let hidden = pipe.last_hidden(layers.len()).unwrap();
    print!("hidden[0..16]:");
    for v in &hidden[..16] {
        print!(" {:.6}", half::f16::from_bits(*v).to_f32());
    }
    println!();
    let nz = hidden.iter().filter(|v| **v != 0).count();
    println!("nonzero: {nz}/{}", hidden.len());
    if with_heads {
        let logits = pipe.logits();
        let (arg, &top) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                half::f16::from_bits(**a)
                    .to_f32()
                    .total_cmp(&half::f16::from_bits(**b).to_f32())
            })
            .unwrap();
        println!("argmax: {arg} ({:.5})", half::f16::from_bits(top).to_f32());
        print!("logits[0..8]:");
        for v in &logits[..8] {
            print!(" {:.5}", half::f16::from_bits(*v).to_f32());
        }
        println!();
    }
    if let Some(k) = pipe.state_f16(0, "k_cache_0") {
        let k_nz = k.iter().filter(|v| **v != 0).count();
        print!("k_cache_0 head:");
        for v in &k[..8] {
            print!(" {:.5}", half::f16::from_bits(*v).to_f32());
        }
        println!("  nonzero {k_nz}/{}", k.len());
    }
    Ok(())
}
