//! which a-slot drives channel c? write a[c*S]=c for stride S, z[c][*]/V[c][*] reveals it.
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
    let stride: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    let (da, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    println!(
        "a: k={} w={} bs={} ps={} es={}",
        da.k, da.w, da.batch_stride, da.plane_stride, da.elem
    );
    {
        let b = m.input_mut(0);
        for c in 0..64usize {
            let o = c * stride;
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16((c + 1) as f32).to_le_bytes());
            }
        }
    }
    {
        let b = m.input_mut(1);
        for t in 0..64u64 {
            for w in 0..64u64 {
                let o = dv.offset(0, 0, t, 0, w);
                b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    for c in 0..8u64 {
        let o = dz.offset(0, 0, c, 0, 0);
        println!(
            "z[c={}]={} (expect {})",
            c,
            f32_of(u16::from_le_bytes([z[o], z[o + 1]])),
            c + 1
        );
    }
}
