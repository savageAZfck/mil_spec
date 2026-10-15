//! kv_probe — KV-cache append on ANE via aliased persistent surfaces.
//!
//!   prog MUL: tmp = kv * mask        (x=kv, y=mask, both surfaces)
//!   prog ADD: cache = cache + tmp    (x=cache, y=tmp, z aliased to x)
//!
//! `tmp` lives on one IOSurface adopted by the ADD program as its y
//! input; `cache` is the ADD program's x input and z output sharing
//! one surface, so state persists across evals. Each step writes a
//! one-hot mask selecting slot `t` of the cache and verifies the
//! appended values are present and previous slots unchanged.
//!
//! Append-only semantics: correct for decode (each slot written once,
//! starting zeroed). Overwrite of a live slot additionally needs a
//! `cache = cache * keep` pass first.

use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_elementwise2, EwOp};
use std::path::Path;

const N: usize = 1024; // flat cache: 8 slots x 128 lanes (big-n layout, n>256)
const SLOT: usize = 128;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

// element axis: channel axis for the n<=256 dim-1 layout, batch axis
// for the n>256 dim-0 layout
fn eoff(d: &mil_infer::espresso::TensorDesc, pos: usize) -> usize {
    if d.n > 1 {
        d.offset(pos as u64, 0, 0, 0, 0)
    } else {
        d.offset(0, 0, pos as u64, 0, 0)
    }
}
fn put(buf: &mut [u8], d: &mil_infer::espresso::TensorDesc, pos: usize, v: f32) {
    let off = eoff(d, pos);
    if off + 2 <= buf.len() {
        buf[off..off + 2].copy_from_slice(&f16(v).to_le_bytes());
    }
}
fn get(buf: &[u8], d: &mil_infer::espresso::TensorDesc, pos: usize) -> f32 {
    let off = eoff(d, pos);
    if off + 2 <= buf.len() {
        f32_of(u16::from_le_bytes([buf[off], buf[off + 1]]))
    } else {
        f32::NAN
    }
}

fn main() {
    // unique paths per run — aned's patch cache keys on the embedded
    // output path and can replay a stale patched image otherwise
    let pid = std::process::id();
    let mul_path = format!("/tmp/kv_mul_{pid}.macho");
    let add_path = format!("/tmp/kv_add_{pid}.macho");
    std::fs::write(
        &mul_path,
        emit_elementwise2(EwOp::Mul, N as u32, "kv_mul.mil", &mul_path),
    )
    .unwrap();
    std::fs::write(
        &add_path,
        emit_elementwise2(EwOp::Add, N as u32, "kv_add.mil", &add_path),
    )
    .unwrap();

    let mut pm = EspressoProgram::load_precompiled(Path::new(&mul_path), &[]).expect("load mul");
    let mut pa = EspressoProgram::load_precompiled(Path::new(&add_path), &[]).expect("load add");
    println!(
        "mul in={:?} out={:?} | add in={:?} out={:?}",
        pm.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        pm.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        pa.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        pa.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
    );

    let d_kv = pm.input_descs[0].clone();
    let d_mask = pm.input_descs[1].clone();
    let d_cache = pa.input_descs[0].clone();

    // tmp (mul output 0) becomes add's y input 1.
    let tmp_surf = pm.output_surf(0);
    pa.adopt_input(1, tmp_surf).expect("adopt tmp");
    // cache: add output 0 aliases add input 0 -> in-place accumulate.
    pa.alias_out_to_in(0, 0);

    // zero the cache
    for b in pa.input_mut(0).iter_mut() {
        *b = 0;
    }

    let mut bad = 0usize;
    for t in 0..4usize {
        // kv token t: value (t*10 + lane) at its slot lanes, 0 elsewhere
        {
            let b = pm.input_mut(0);
            for b in b.iter_mut() {
                *b = 0;
            }
            for l in 0..SLOT {
                put(b, &d_kv, t * SLOT + l, (t * 10 + l % 10) as f32 + 1.0);
            }
        }
        // one-hot mask for slot t
        {
            let b = pm.input_mut(1);
            for b in b.iter_mut() {
                *b = 0;
            }
            for l in 0..SLOT {
                put(b, &d_mask, t * SLOT + l, 1.0);
            }
        }
        pm.eval().expect("mul eval");
        pa.eval().expect("add eval");

        // verify: slot t holds kv_t, all previous slots hold their kv
        let cache = pa.input_bytes(0);
        for s in 0..=t {
            for l in [0usize, 1, 64, 127] {
                let want = (s * 10 + l % 10) as f32 + 1.0;
                let got = get(cache, &d_cache, s * SLOT + l);
                if got != want {
                    bad += 1;
                    if bad < 8 {
                        println!("MISMATCH step {t} slot {s} lane {l}: got {got} want {want}");
                    }
                }
            }
        }
        // untouched slots stay zero
        for s in t + 1..8 {
            let got = get(cache, &d_cache, s * SLOT);
            if got != 0.0 {
                bad += 1;
                if bad < 8 {
                    println!("LEAK step {t} slot {s}: got {got} want 0");
                }
            }
        }
    }
    if bad == 0 {
        println!("KV-append PASS: slots written, all verified, no leakage");
    } else {
        println!("FAIL: {bad} mismatches");
        std::process::exit(1);
    }
}
