use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let (t, d) = (64u64, 32u64);
    let mut m =
        EspressoProgram::load_precompiled(Path::new("/tmp/avw3_out/avw3.macho"), &[]).unwrap();
    let (ds, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    for (tag, x) in [("s", &ds), ("v", &dv), ("z", &dz)] {
        println!(
            "{tag} n{} k{} h{} w{} ps{} rs{} bs{}",
            x.name, x.k, x.h, x.w, x.plane_stride, x.row_stride, x.batch_stride
        );
    }
    {
        let b = m.input_mut(0);
        for i in 0..t {
            let o = ds.offset(0, 0, 0, i, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(i as f32 / 64.0).to_le_bytes());
            }
        }
    }
    {
        let b = m.input_mut(1);
        for i in 0..t {
            for j in 0..d {
                let o = dv.offset(0, 0, 0, i, j);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16(((i * 3 + j) % 9) as f32 / 8.0).to_le_bytes());
                }
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut mm = 0;
    for j in 0..d {
        let o = dz.offset(0, 0, 0, 0, j);
        if o + 2 > z.len() {
            mm += 1;
            continue;
        }
        let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
        let exp: f32 = (0..t)
            .map(|i| (i as f32 / 64.0) * (((i * 3 + j) % 9) as f32 / 8.0))
            .sum();
        if (v - exp).abs() > exp * 0.02 + 0.05 {
            mm += 1;
            if mm < 4 {
                println!("j={} z={} exp={:.3}", j, v, exp);
            }
        }
    }
    println!("attnV h-axis mismatches: {}/{}", mm, d);
}
