//! parametric matvec check: z[c]=dot(a[0..K], V[c][0..K]) for a=[1,1,1,K], V=[1,R,1,K]
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
    let r: u64 = std::env::args().nth(2).unwrap().parse().unwrap();
    let k: u64 = std::env::args().nth(3).unwrap().parse().unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    let (da, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    // a[w] = w/512 ; V[c][w] = (c+w)/512 — small values, fp16-safe
    {
        let b = m.input_mut(0);
        for w in 0..k {
            let o = da.offset(0, 0, 0, 0, w);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(w as f32 / 512.0).to_le_bytes());
            }
        }
    }
    {
        let b = m.input_mut(1);
        for c in 0..r {
            for w in 0..k {
                let o = dv.offset(0, 0, c, 0, w);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16((c + w) as f32 / 512.0).to_le_bytes());
                }
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut mm = 0;
    for c in 0..r {
        let o = dz.offset(0, 0, c, 0, 0);
        if o + 2 > z.len() {
            mm += 1;
            continue;
        }
        let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
        let exp: f32 = (0..k)
            .map(|w| (w as f32 / 512.0) * ((c + w) as f32 / 512.0))
            .sum();
        if (v - exp).abs() > exp * 0.02 + 0.05 {
            mm += 1;
            if mm < 4 {
                println!("c={} z={} exp={:.3}", c, v, exp);
            }
        }
    }
    println!("z[c]=dot(a,V[c]) mismatches: {}/{}", mm, r);
}
