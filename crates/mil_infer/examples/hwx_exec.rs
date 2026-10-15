//! `hwx_exec` — load a bare .macho/.hwx as a precompiled ANE model via
//! `_ANEModel modelAtURL:` + `ANEFModelPreCompiled` option, then eval.
//!
//!   hwx_exec <macho> [mutate-mode]
//!
//! Proves whether our emitted HWX bytes are what `aned` executes:
//! mutating the artifact must change behaviour if it is consumed.

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
    let macho = a[1].clone();
    let mutate: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let extra: Vec<(String, u64)> = a[3..]
        .iter()
        .filter_map(|s| {
            s.split_once('=')
                .map(|(k, v)| (k.to_string(), v.parse().unwrap_or(1)))
        })
        .collect();
    let extra_ref: Vec<(&str, u64)> = extra.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let is_dir = Path::new(&macho).is_dir();
    let mut bytes = if is_dir {
        Vec::new()
    } else {
        std::fs::read(&macho).unwrap()
    };
    if mutate > 0 && !is_dir {
        let tmp = Path::new("/tmp/hwx_exec_mut.macho");
        let mut n = 0;
        for i in (0..bytes.len() - 4).step_by(4) {
            let w = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
            let id = (w >> 16) & 0xff;
            if w >> 24 == 0 && ((w >> 8) & 0xff) == 0 && (0x54..=0x64).contains(&id) {
                let nw = (0x60u32 << 16) | (w & 0xffff);
                bytes[i..i + 4].copy_from_slice(&nw.to_le_bytes());
                n += 1;
            }
            if mutate == 2 && i >= 4096 && w != 0 {
                bytes[i..i + 4].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
                n += 1;
            }
        }
        println!("mutated {n} words (mode {mutate})");
        std::fs::write(tmp, &bytes).unwrap();
        std::env::set_var("HWX", tmp);
    }
    let path = std::env::var("HWX").unwrap_or(macho.clone());

    let mut m =
        EspressoProgram::load_precompiled(Path::new(&path), &extra_ref).expect("load_precompiled");
    println!(
        "loaded: ph={} inputs={:?} outputs={:?} procs={:?}",
        m.program_handle,
        m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        m.procedures
    );
    if m.input_descs.is_empty() {
        println!("no input descs — attrs may be absent for bare hwx");
        return;
    }
    let d = m.input_descs[0].clone();
    let b = m.input_mut(0);
    let fill_mode: u32 = std::env::var("FILL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for k in 0..d.k {
        let off = d.offset(0, 0, k, 0, 0);
        let v = match fill_mode {
            1 => 0.0f32,
            2 => 1.0,
            3 => (k as f32) * 0.25,
            _ => ((k % 16) + 1) as f32,
        };
        b[off..off + 2].copy_from_slice(&f16(v).to_le_bytes());
    }
    m.eval().expect("eval");
    let d = m.output_descs[0].clone();
    let out = m.output(0);
    let vals: Vec<f32> = (0..d.k.min(8))
        .map(|k| {
            f32_of(u16::from_le_bytes([
                out[d.offset(0, 0, k, 0, 0)],
                out[d.offset(0, 0, k, 0, 0) + 1],
            ]))
        })
        .collect();
    println!("out[0..8] = {vals:?}");
}
