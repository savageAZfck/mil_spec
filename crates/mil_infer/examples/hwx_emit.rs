//! `hwx_emit` — emit an ANE Mach-O without ANECompiler.framework and
//! verify it two ways: byte-equality against the reference corpus
//! (same inputs → same bytes) and live execution on the ANE through
//! the espresso `_ANEClient` path.
//!
//! Usage:
//!   hwx_emit emit <op> <n> <out.macho>          emit image
//!   hwx_emit verify <refdir>                   byte-compare vs corpus
//!   hwx_emit exec <pkg-dir>                  run via EspressoProgram

use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{
    emit_conv1x1, emit_elementwise2, emit_matmul, emit_unary_lut, ConvShape, EwOp, LutOp, MmShape,
};
use std::path::{Path, PathBuf};

fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7fffff;
    if e == 0 {
        return s;
    }
    if e >= 143 {
        return s | 0x7c00;
    }
    if e < 103 {
        return s;
    }
    s | (((e - 112) as u16) << 10) | ((m >> 13) as u16)
}

fn f32_of(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(s);
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | (((e + 112) as u32) << 23) | (m << 13))
}

enum Op {
    Ew(EwOp),
    Lut(LutOp),
}

fn op_of(s: &str) -> Op {
    match s {
        "add" => Op::Ew(EwOp::Add),
        "mul" => Op::Ew(EwOp::Mul),
        "sub" => Op::Ew(EwOp::Sub),
        "sigmoid" => Op::Lut(LutOp::Sigmoid),
        "tanh" => Op::Lut(LutOp::Tanh),
        other => {
            eprintln!("unknown op {other}");
            std::process::exit(2);
        }
    }
}

fn emit(op: Op, n: u32, mil: &str, out: &str) -> Vec<u8> {
    match op {
        Op::Ew(o) => emit_elementwise2(o, n, mil, out),
        Op::Lut(o) => emit_unary_lut(o, n, mil, out),
    }
}

fn conv_shape(s: &str) -> ConvShape {
    match s {
        "c32w1" => ConvShape::C32W1,
        "c64w1" => ConvShape::C64W1,
        "c64w5" => ConvShape::C64W5,
        "c64w8" => ConvShape::C64W8,
        "c128w1" => ConvShape::C128W1,
        other => {
            eprintln!("unknown conv shape {other} (c32w1|c64w1|c64w5|c64w8|c128w1)");
            std::process::exit(2);
        }
    }
}

fn mm_shape(s: &str) -> MmShape {
    match s {
        "32" => MmShape::N32,
        "64" => MmShape::N64,
        "128" => MmShape::N128,
        "256" => MmShape::N256,
        other => {
            eprintln!("unknown matmul size {other} (32|64|128|256)");
            std::process::exit(2);
        }
    }
}

/// Raw `u16` weights file, either bare fp16 data or a deadbeef blob
/// (data starts at byte 128).
fn load_weights(p: &str) -> Vec<u16> {
    let b = std::fs::read(p).unwrap();
    let off = if b.len() > 132 && &b[64..68] == b"\xef\xbe\xad\xde" {
        128
    } else {
        0
    };
    b[off..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn diff_report(a: &[u8], b: &[u8], label: &str) -> usize {
    let mut first = usize::MAX;
    let mut count = 0usize;
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            count += 1;
            if first == usize::MAX {
                first = i;
            }
        }
    }
    count += a.len().abs_diff(b.len());
    if count == 0 {
        println!("  {label}: BYTE-IDENTICAL");
    } else {
        let lo = first.saturating_sub(4);
        println!(
            "  {label}: {count} diffs, first @{first:#x}: ours={:02x?} ref={:02x?}",
            &a[lo..(first + 16).min(a.len())],
            &b[lo..(first + 16).min(b.len())]
        );
    }
    count
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("emit") => {
            let op = op_of(&a[2]);
            let n: u32 = a[3].parse().unwrap();
            let out = PathBuf::from(&a[4]);
            let mil = format!("{}.mil", out.display());
            let img = emit(op, n, &mil, &out.display().to_string());
            std::fs::write(&out, &img).unwrap();
            println!("emitted {} ({} bytes)", out.display(), img.len());
        }
        Some("emit-conv") => {
            // emit-conv <shape> <weights.bin> <out.macho>
            let shape = conv_shape(&a[2]);
            let w = load_weights(&a[3]);
            let out = PathBuf::from(&a[4]);
            let mil = format!("{}.mil", out.display());
            let img = emit_conv1x1(shape, &w, &mil, &out.display().to_string()).unwrap();
            std::fs::write(&out, &img).unwrap();
            println!("emitted {} ({} bytes)", out.display(), img.len());
        }
        Some("emit-mm") => {
            // emit-mm <n> <weights.bin> <out.macho>
            let shape = mm_shape(&a[2]);
            let w = load_weights(&a[3]);
            let out = PathBuf::from(&a[4]);
            let mil = format!("{}.mil", out.display());
            let img = emit_matmul(shape, &w, &mil, &out.display().to_string()).unwrap();
            std::fs::write(&out, &img).unwrap();
            println!("emitted {} ({} bytes)", out.display(), img.len());
        }
        Some("verify") => {
            // verify <dir>: for each net.macho reference at <dir>/*/out,
            // re-emit with the recorded inputs and compare.
            let dir = Path::new(&a[2]);
            std::env::set_var("HWX_SIG_ANEC", "1");
            let mut total = 0usize;
            let cases: Vec<(String, u32, &str)> = vec![
                ("n32".into(), 32, "add"),
                ("n128".into(), 128, "add"),
                ("n256".into(), 256, "add"),
                ("m48".into(), 48, "add"),
                ("m96".into(), 96, "add"),
                ("q20".into(), 20, "add"),
                ("q36".into(), 36, "add"),
                ("q56".into(), 56, "add"),
                ("same_add".into(), 64, "add"),
                ("same_mul".into(), 64, "mul"),
                ("same_sub".into(), 64, "sub"),
                ("sig32".into(), 32, "sigmoid"),
                ("sig128".into(), 128, "sigmoid"),
                ("same_sigmoid".into(), 64, "sigmoid"),
                ("same_tanh".into(), 64, "tanh"),
            ];
            for (name, n, op) in cases {
                let reff = dir.join(&name).join("out").join("net.macho");
                if !reff.exists() {
                    println!("  {name}: no reference, skipped");
                    continue;
                }
                let reff_bytes = std::fs::read(&reff).unwrap();
                let mil = format!("{}/{name}/net.mil", dir.display());
                let out = format!("{}/{name}/out/net.macho", dir.display());
                let ours = emit(op_of(op), n, &mil, &out);
                total += diff_report(&ours, &reff_bytes, &format!("{name} ({op}@{n})"));
            }
            // weight-bearing families: perm-weight probes are the refs.
            for (name, shape) in [
                ("conv32p", ConvShape::C32W1),
                ("conv64p", ConvShape::C64W1),
                ("conv64s5p", ConvShape::C64W5),
                ("conv64s8p", ConvShape::C64W8),
                ("conv128p", ConvShape::C128W1),
            ] {
                let reff = dir.join(name).join("out").join("net.macho");
                if !reff.exists() {
                    println!("  {name}: no reference, skipped");
                    continue;
                }
                let c = shape.channels();
                let w: Vec<u16> = (0..(c * c) as u16).collect();
                let mil = format!("{}/{name}/net.mil", dir.display());
                let out = format!("{}/{name}/out/net.macho", dir.display());
                let ours = emit_conv1x1(shape, &w, &mil, &out).unwrap();
                total += diff_report(&ours, &std::fs::read(&reff).unwrap(), name);
            }
            for (name, shape) in [("mm32p", MmShape::N32), ("mm64p", MmShape::N64)] {
                let reff = dir.join(name).join("out").join("net.macho");
                if !reff.exists() {
                    println!("  {name}: no reference, skipped");
                    continue;
                }
                let n = shape.n();
                let w: Vec<u16> = (0..(n * n) as u16).collect();
                let mil = format!("{}/{name}/net.mil", dir.display());
                let out = format!("{}/{name}/out/net.macho", dir.display());
                let ours = emit_matmul(shape, &w, &mil, &out).unwrap();
                total += diff_report(&ours, &std::fs::read(&reff).unwrap(), name);
            }
            println!("total diffs: {total}");
        }
        Some("exec") => {
            // exec <pkg-dir> — same as gate0 ownexec but for our emitted hwx.
            let mut m = EspressoProgram::load(Path::new(&a[2])).expect("load");
            println!(
                "loaded: ph={} inputs={:?} outputs={:?}",
                m.program_handle,
                m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
                m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>()
            );
            for (i, v) in [(0usize, f16(1.0)), (1, f16(2.5))] {
                let d = m.input_descs[i].clone();
                let b = m.input_mut(i);
                for k in 0..d.k {
                    let off = d.offset(0, 0, k, 0, 0);
                    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            let t = std::time::Instant::now();
            m.eval().expect("eval");
            let us = t.elapsed().as_secs_f64() * 1e6;
            let d = m.output_descs[0].clone();
            let out = m.output(0);
            let want = f16(3.5);
            let good = (0..d.k)
                .filter(|k| {
                    let off = d.offset(0, 0, *k, 0, 0);
                    u16::from_le_bytes([out[off], out[off + 1]]) == want
                })
                .count();
            let lanes: Vec<f32> = (0..4)
                .map(|k| {
                    let off = d.offset(0, 0, k, 0, 0);
                    f32_of(u16::from_le_bytes([out[off], out[off + 1]]))
                })
                .collect();
            println!(
                "eval {us:.0} us, z[0..4] = {lanes:?}, lanes==3.5: {good}/{}",
                d.k
            );
        }
        Some("exec1") => {
            // exec1 <pkg-dir> <input-f32> <expected-f32> — unary programs.
            let mut m = EspressoProgram::load(Path::new(&a[2])).expect("load");
            let x: f32 = a[3].parse().unwrap();
            let want_f: f32 = a[4].parse().unwrap();
            let d = m.input_descs[0].clone();
            let b = m.input_mut(0);
            for k in 0..d.k {
                let off = d.offset(0, 0, k, 0, 0);
                b[off..off + 2].copy_from_slice(&f16(x).to_le_bytes());
            }
            let t = std::time::Instant::now();
            m.eval().expect("eval");
            let us = t.elapsed().as_secs_f64() * 1e6;
            let d = m.output_descs[0].clone();
            let out = m.output(0);
            let got = (0..d.k)
                .map(|k| {
                    let off = d.offset(0, 0, k, 0, 0);
                    f32_of(u16::from_le_bytes([out[off], out[off + 1]]))
                })
                .collect::<Vec<_>>();
            let good = got.iter().filter(|v| (*v - want_f).abs() < 0.01).count();
            println!(
                "eval {us:.0} us, z[0..4] = {:?}, lanes~={want_f}: {good}/{}",
                &got[..4.min(got.len())],
                d.k
            );
        }
        other => eprintln!("usage: hwx_emit emit|verify|exec|exec1 ... ({other:?})"),
    }
}
