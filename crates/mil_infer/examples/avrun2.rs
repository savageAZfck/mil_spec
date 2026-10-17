//! avrun2 — probe [1,T,1,1] x [1,T,1,W] matmul semantics (attn@V shape).
use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    for d in m.input_descs.iter().chain(m.output_descs.iter()) {
        println!(
            "{}: n={} k={} h={} w={} bs={} ps={} rs={} es={}",
            d.name, d.n, d.k, d.h, d.w, d.batch_stride, d.plane_stride, d.row_stride, d.elem
        );
    }
    let (_da, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    {
        let b = m.input_mut(0);
        for i in 0..(b.len() / 2) {
            let v = f16(2.0);
            b[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    {
        let b = m.input_mut(1);
        for t in 0..64u64 {
            for w in 0..64u64 {
                let o = dv.offset(0, 0, t, 0, w);
                b[o..o + 2].copy_from_slice(&f16((t * 64 + w) as f32 / 64.0).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    for c in [1u64, 2, 3, 4, 5, 6, 7] {
        let row: Vec<f32> = (0..6u64)
            .map(|w| {
                let o = dz.offset(0, 0, c, 0, w);
                f32_of(u16::from_le_bytes([z[o], z[o + 1]]))
            })
            .collect();
        println!(
            "z[{}][0..6]={:?}  -> a-src={:?}",
            c,
            row,
            row.iter()
                .enumerate()
                .map(|(w, v)| v / ((c * 64 + w as u64) as f32 / 64.0))
                .collect::<Vec<_>>()
        );
    }
}
