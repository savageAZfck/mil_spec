//! `chain_exec` — exec an ANEC conv-chain Mach-O through the espresso
//! `_ANEClient` path, then (optionally) with mutated descriptor ids.
//!
//!   chain_exec <macho> <cin> <cout> <mutate>
//!
//! Builds a minimal espresso package around the artifact, loads it,
//! fills `x` with a known pattern, evals, and prints `y*`.
//! `mutate`: 0 = as-is, 1 = rewrite every `00{id}00{idx}` desc word's
//! id to 0x60, 2 = add +1 to each mid id (54-56 range only).

use mil_infer::espresso::EspressoProgram;
use std::path::Path;

fn f16(v: f32) -> u16 {
    let f = v.to_bits();
    let s = ((f >> 16) & 0x8000) as u16;
    let e = ((f >> 23) & 0xff) as i32 - 127 + 15;
    let m = (f >> 13) & 0x3ff;
    if e <= 0 {
        return s;
    }
    if e >= 31 {
        return s | 0x7c00;
    }
    s | ((e as u16) << 10) | m as u16
}
fn f32_of(h: u16) -> f32 {
    let s = (h & 0x8000) != 0;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    let v = if e == 0 {
        m / 1024.0 * 2f32.powi(-14)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if s {
        -v
    } else {
        v
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let macho = &a[1];
    let cin: usize = a[2].parse().unwrap();
    let cout: usize = a[3].parse().unwrap();
    let mutate: u32 = a[4].parse().unwrap();

    let mut bytes = std::fs::read(macho).unwrap();
    if mutate > 0 {
        let mut n = 0;
        for i in (0..bytes.len() - 4).step_by(4) {
            let w = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
            let id = (w >> 16) & 0xff;
            let idx = w & 0xffff;
            if w >> 24 == 0 && ((w >> 8) & 0xff) == 0 && (0x54..=0x64).contains(&id) && idx <= 0x20
            {
                let nid = if mutate == 1 { 0x60 } else { id + 1 };
                let nw = (nid << 16) | (w & 0xffff);
                bytes[i..i + 4].copy_from_slice(&nw.to_le_bytes());
                n += 1;
            }
            if mutate == 3 && i >= bytes.len() / 2 && i < bytes.len() / 2 + 64 && w != 0 {
                bytes[i..i + 4].copy_from_slice(&0u32.to_le_bytes());
                n += 1;
            }
            if mutate == 6 && i >= 4096 {
                bytes[i..i + 4].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
                n += 1;
            }
            if mutate == 5 && i < 64 {
                bytes[i..i + 4].copy_from_slice(&0u32.to_le_bytes());
                n += 1;
            }
            if mutate == 4 && (w >> 16) == 0xfff8 {
                bytes[i..i + 4].copy_from_slice(&0u32.to_le_bytes());
                n += 1;
            }
        }
        println!("mutated {n} desc words (mode {mutate})");
    }

    let pkg = Path::new(&a[5]);
    std::fs::create_dir_all(pkg).unwrap();
    std::fs::write(pkg.join("model.H16.espresso.hwx"), &bytes).unwrap();
    std::fs::write(
        pkg.join("model.espresso.net"),
        "{\n \"storage\": \"model.espresso.weights\",\n \"layers\": [\n  {\"name\":\"c0\",\"type\":\"elementwise\",\"bottom\":\"x\",\"top\":\"y4\",\"operation\":0,\"alpha\":1,\"beta\":0,\"fused_relu\":0,\"weights\":{},\"debug_info\":\"\"}\n ],\n \"properties\": {},\n \"analyses\": {},\n \"format_version\": 200,\n \"metadata_in_weights\": []\n}\n",
    )
    .unwrap();
    std::fs::write(
        pkg.join("model.espresso.shape"),
        format!("{{\"layer_shapes\":{{\n \"x\":{{\"n\":1,\"k\":{cin},\"h\":1,\"w\":1,\"_rank\":4}},\n \"y4\":{{\"n\":1,\"k\":{cout},\"h\":1,\"w\":1,\"_rank\":4}}\n}}}}\n"),
    )
    .unwrap();
    std::fs::write(pkg.join("model.espresso.weights"), [0u8; 64]).unwrap();
    std::fs::write(
        pkg.join("model.H16.espresso.precompilation_info"),
        "{\"ANE_COMPILER\":{\"kernel_validation_status\":{\"visit_order\":[],\"to_fallback\":[],\"any_fallback\":0,\"unsupported\":[]}}}\n",
    )
    .unwrap();

    let mut m = EspressoProgram::load(pkg).expect("load");
    println!(
        "loaded: inputs={:?} outputs={:?}",
        m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>()
    );
    // fill input x: lane k = (k%16)+1 as fp16
    let d = m.input_descs[0].clone();
    let b = m.input_mut(0);
    for k in 0..d.k {
        let off = d.offset(0, 0, k, 0, 0);
        b[off..off + 2].copy_from_slice(&f16(((k % 16) + 1) as f32).to_le_bytes());
    }
    m.eval().expect("eval");
    let d = m.output_descs[0].clone();
    let out = m.output(0);
    let vals: Vec<f32> = (0..d.k.min(8))
        .map(|k| {
            let off = d.offset(0, 0, k, 0, 0);
            f32_of(u16::from_le_bytes([out[off], out[off + 1]]))
        })
        .collect();
    println!("y4[0..8] = {vals:?}");
}
