use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let (t, k) = (64u64, 64u64);
    let mut m =
        EspressoProgram::load_precompiled(Path::new("/tmp/outer_o/outer.macho"), &[]).unwrap();
    let (de, dr, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    for (tag, x) in [("e", &de), ("r", &dr), ("z", &dz)] {
        println!(
            "{tag} n{} k{} h{} w{} ps{} rs{} bs{}",
            x.name, x.k, x.h, x.w, x.plane_stride, x.row_stride, x.batch_stride
        );
    }
    {
        let b = m.input_mut(0);
        for i in 0..t {
            let o = de.offset(0, 0, i, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(if i == 5 { 1.0 } else { 0.0 }).to_le_bytes());
            }
        }
    }
    {
        let b = m.input_mut(1);
        for j in 0..k {
            let o = dr.offset(0, 0, 0, 0, j);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16((j + 1) as f32).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut mm = 0;
    for i in 0..t {
        for j in 0..k {
            let o = dz.offset(0, 0, i, 0, j);
            if o + 2 > z.len() {
                mm += 1;
                continue;
            }
            let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
            let exp = if i == 5 { (j + 1) as f32 } else { 0.0 };
            if (v - exp).abs() > 0.01 {
                mm += 1;
                if mm < 4 {
                    println!("i={} j={} z={} exp={}", i, j, v, exp);
                }
            }
        }
    }
    println!("outer(e5,r) mismatches: {}/{}", mm, t * k);
}
