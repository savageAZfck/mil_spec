//! probe avw_ft: a=[1,1,1,64] x V=[1,64,1,64] ty=true — broadcast row-dot?
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
    let (da, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    println!(
        "a:k{} h{} w{} ps{} es{} | v:k{} h{} w{} ps{} es{} | z:k{} h{} w{} ps{} es{}",
        da.k,
        da.h,
        da.w,
        da.plane_stride,
        da.elem,
        dv.k,
        dv.h,
        dv.w,
        dv.plane_stride,
        dv.elem,
        dz.k,
        dz.h,
        dz.w,
        dz.plane_stride,
        dz.elem
    );
    // a[w] = w/64 position-coded (dense w-axis); V[c][w] = c*0.25 + w/256
    {
        let b = m.input_mut(0);
        for w in 0..64u64 {
            let o = da.offset(0, 0, 0, 0, w);
            b[o..o + 2].copy_from_slice(&f16(w as f32 / 64.0).to_le_bytes());
        }
    }
    {
        let b = m.input_mut(1);
        for c in 0..64u64 {
            for w in 0..64u64 {
                let o = dv.offset(0, 0, c, 0, w);
                b[o..o + 2].copy_from_slice(&f16(c as f32 * 0.25 + w as f32 / 256.0).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    // expected row-dot: z[c] = sum_w a[w]*V[c][w]
    let mut mm = 0;
    for h in 0..64u64 {
        let o = dz.offset(0, 0, 0, h, 0);
        let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
        let exp: f32 = (0..64)
            .map(|w| (w as f32 / 64.0) * (h as f32 * 0.25 + w as f32 / 256.0))
            .sum();
        if (v - exp).abs() > 0.5 {
            mm += 1;
            if mm < 5 {
                println!("h={} z={} exp={:.2}", h, v, exp);
            }
        }
    }
    println!("z[0][h]=dot(a,V[h]) mismatches: {}/64", mm);
}
