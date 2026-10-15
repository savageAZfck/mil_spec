//! `e5_mut` — descriptor-id mutation experiment via the e5rt path.
//!
//!   e5_mut <mil> <in_name> <in_k> <out_name> <out_k> <mutate>
//!
//! 1. compile the mil into the e5 bundle cache (/tmp/e5_mut_cache)
//! 2. exec baseline and print the output head
//! 3. mutate every `00{id}00{idx}` desc word found in any file under the
//!    cache dir (mutate=1: id->0x60, 2: id+1)
//! 4. load the mutated bundle directly and exec again, print output

use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use std::path::{Path, PathBuf};

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

fn walk(d: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(d).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mil = PathBuf::from(&a[1]);
    let in_name = a[2].clone();
    let in_k: u64 = a[3].parse().unwrap();
    let out_name = a[4].clone();
    let out_k: u64 = a[5].parse().unwrap();
    let mutate: u32 = a[6].parse().unwrap();

    let cache = PathBuf::from("/tmp/e5_mut_cache");
    let cc = CompileCfg {
        cache_dir: cache.clone(),
        force_recompile: true,
    };
    let live: Vec<(&str, u64)> = vec![(in_name.as_str(), in_k)];
    let outs: Vec<(&str, u64)> = vec![(out_name.as_str(), out_k)];
    let inouts: Vec<(&str, u64)> = vec![];

    let prog =
        Program::compile_cfg(&mil, DeviceMask::Ane, &live, &outs, &inouts, &cc).expect("compile");
    let inp: Vec<u16> = (0..in_k)
        .map(|i| half::f16::from_f32(((i % 16) + 1) as f32).to_bits())
        .collect();
    prog.set_input_f16(&in_name, &inp).unwrap();
    prog.execute().expect("exec");
    let base = prog.output_f16(&out_name).unwrap();
    let head: Vec<f32> = base.iter().take(8).map(|b| f32_of(*b)).collect();
    println!("baseline: {head:?} ({} elems)", base.len());
    drop(prog);

    // find the bundle + mutate desc words in every file
    let mut files = Vec::new();
    walk(&cache, &mut files);
    let mut bundle_dir: Option<PathBuf> = None;
    for f in &files {
        println!(
            "  cache file: {} ({} B)",
            f.display(),
            f.metadata().unwrap().len()
        );
        if f.extension().map(|e| e == "bundle").unwrap_or(false) {
            bundle_dir = Some(f.clone());
        }
        if f.is_dir() {
            continue;
        }
        let mut d = std::fs::read(f).unwrap();
        let mut n = 0;
        for i in (0..d.len().saturating_sub(4)).step_by(4) {
            let w = u32::from_le_bytes(d[i..i + 4].try_into().unwrap());
            let id = (w >> 16) & 0xff;
            let idx = w & 0xffff;
            if w >> 24 == 0 && ((w >> 8) & 0xff) == 0 && (0x54..=0x64).contains(&id) && idx <= 0x20
            {
                let nid = if mutate == 1 { 0x60 } else { id + 1 };
                d[i..i + 4].copy_from_slice(&((nid << 16) | (w & 0xffff)).to_le_bytes());
                n += 1;
            }
        }
        if n > 0 {
            std::fs::write(f, &d).unwrap();
            println!("    -> mutated {n} desc words in {}", f.display());
        }
    }
    let bundle = bundle_dir.expect("no bundle dir found in cache");
    let (prog2, _map) = Program::from_e5_bundle(&bundle).expect("bundle load");
    prog2.set_input_f16(&in_name, &inp).unwrap();
    prog2.execute().expect("exec2");
    let out2 = prog2.output_f16(&out_name).unwrap();
    let head2: Vec<f32> = out2.iter().take(8).map(|b| f32_of(*b)).collect();
    println!("mutated : {head2:?} ({} elems)", out2.len());
    println!("match: {}", base == out2);
}
