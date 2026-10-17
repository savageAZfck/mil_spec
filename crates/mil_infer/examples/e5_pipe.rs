//! e5_pipe — chained multi-shard decode on the ANE.
//!
//!   cargo run -p mil_infer --example e5_pipe -- <shard_dir> [n_layers] [steps]
//!
//! Loads `layer_*`/`head_*` .mlmodelc shards, binds them into one
//! execution stream (hidden ping-pong, shared masks/rope, resident KV),
//! and steps the whole model with a single submission per step.

use mil_infer::ane_pipeline::{shard_paths, Pipeline};
use mil_infer::e5rt::DeviceMask;
use std::path::Path;
use std::time::Instant;

fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7fffff;
    if e - 112 <= 0 {
        return s as u16;
    }
    (s | (((e - 112) as u32) << 10) | (m >> 13)) as u16
}

fn f32_of(v: u16) -> f32 {
    let s = (v as u32 & 0x8000) << 16;
    let e = ((v as u32 >> 10) & 0x1f) as i32;
    let m = (v as u32 & 0x3ff) << 13;
    if e == 0 {
        return f32::from_bits(s | (m >> 13));
    }
    f32::from_bits(s | (((e + 112) as u32) << 23) | m)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    let n_layers: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    let (mut layers, mut heads) = shard_paths(&dir)?;
    let total = layers.len();
    layers.truncate(n_layers);
    if layers.len() < total {
        heads.clear(); // partial chain: no heads
    }
    println!(
        "[pipe] {} layer shards, {} head shards (dir: {})",
        layers.len(),
        heads.len(),
        dir.display()
    );

    let t = Instant::now();
    let pipe = Pipeline::open(&layers, &heads, DeviceMask::Ane)?;
    println!("[pipe] stream encoded in {:.1?}", t.elapsed());
    println!("[pipe] states: {:?}", pipe.state_names().first());

    // seed embedding slot with a constant token vector
    let (slot, n) = pipe.embedding_slot()?;
    let vec: Vec<u16> = (0..n).map(|_| f16(0.01)).collect();
    unsafe { std::ptr::copy_nonoverlapping(vec.as_ptr(), slot, n as usize) };

    // seed shared live inputs (rope/masks) — nonzero so the graph computes
    for name in ["rope_cos", "rope_sin", "kv_write_mask", "attn_mask"] {
        let v = match name {
            "rope_cos" => f16(1.0),
            "kv_write_mask" => f16(1.0),
            _ => f16(0.0),
        };
        let _ = pipe.set_shared(name, &vec![v; 1 << 14]);
    }
    pipe.reset_state()?;

    for s in 0..steps {
        let t = Instant::now();
        pipe.step()?;
        println!("step {}: {:.3}ms", s, t.elapsed().as_secs_f64() * 1e3);
    }

    let logits = pipe.logits();
    if logits.is_empty() {
        println!("[pipe] no heads bound (partial chain) — body ran clean");
    } else {
        let nz = logits.iter().filter(|v| **v != 0).count();
        let (arg, &top) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| f32_of(**a).total_cmp(&f32_of(**b)))
            .unwrap();
        println!(
            "[pipe] logits: {} elems, {} nonzero, argmax {} ({:.4})",
            logits.len(),
            nz,
            arg,
            f32_of(top)
        );
    }
    Ok(())
}
