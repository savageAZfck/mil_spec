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
    let (d0, d1, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    println!("in0={} in1={} out={}", d0.name, d1.name, dz.name);
    {
        let b = m.input_mut(0);
        for o in (0..b.len()).step_by(2) {
            b[o..o + 2].copy_from_slice(&f16(2.0).to_le_bytes());
        }
    }
    {
        let b = m.input_mut(1);
        for o in (0..b.len()).step_by(2) {
            b[o..o + 2].copy_from_slice(&f16(3.0).to_le_bytes());
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    for (c, h) in [(0u64, 0u64), (0, 1), (1, 0), (2, 3)] {
        let row: Vec<f32> = (0..4u64)
            .map(|w| {
                let o = dz.offset(0, 0, c, h, w);
                f32_of(u16::from_le_bytes([z[o], z[o + 1]]))
            })
            .collect();
        println!("z[c={}][h={}][0..4]={:?}", c, h, row);
    }
}
