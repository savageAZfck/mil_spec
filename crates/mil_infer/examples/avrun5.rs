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
    let byte: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    let (dv, dz) = (m.input_descs[1].clone(), m.output_descs[0].clone());
    {
        let b = m.input_mut(0);
        b[byte..byte + 2].copy_from_slice(&f16(7.0).to_le_bytes());
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
    let mut nz: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..64u64 {
        for w in 0..64u64 {
            let o = dz.offset(0, 0, c, 0, w);
            let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
            if v != 0.0 && v.is_finite() {
                nz.push((c, w, v));
            }
        }
    }
    println!(
        "byte={} nz={} first8={:?}",
        byte,
        nz.len(),
        &nz[..nz.len().min(8)]
    );
}
