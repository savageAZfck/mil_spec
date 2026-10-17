//! avrun — semantics probe for matmul(tx=true,ty=false) over [1,64,1,64].
//! x=I -> z should equal a function of y exposing the contraction.
use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn put(b: &mut [u8], d: &mil_infer::espresso::TensorDesc, c: u64, w: u64, v: f32) {
    let o = d.offset(0, 0, c, 0, w);
    b[o..o + 2].copy_from_slice(&f16(v).to_le_bytes());
}

fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    for d in m.input_descs.iter().chain(m.output_descs.iter()) {
        println!(
            "{}: n={} k={} h={} w={} bs={} ps={} rs={}",
            d.name, d.n, d.k, d.h, d.w, d.batch_stride, d.plane_stride, d.row_stride
        );
    }
    let (dx, dy, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    // x[r][c] = 1 iff r==c+1 (subdiagonal shift) — shows index movement
    {
        let b = m.input_mut(0);
        for r in 0..64 {
            for c in 0..64 {
                put(b, &dx, r, c, if r == c + 1 { 1.0 } else { 0.0 });
            }
        }
    }
    {
        let b = m.input_mut(1);
        for r in 0..64 {
            for c in 0..64 {
                put(b, &dy, r, c, (r * 64 + c) as f32);
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    // scan planes: z[c][h][w] — find nonzero regions
    for cc in 0..4u64 {
        for hh in 0..4u64 {
            let row: Vec<f32> = (0..6u64)
                .map(|w| {
                    let o = dz.offset(0, 0, cc, hh, w);
                    f32_of(u16::from_le_bytes([z[o], z[o + 1]]))
                })
                .collect();
            println!("z[c={}][h={}][0..6]={:?}", cc, hh, row);
        }
    }
}
