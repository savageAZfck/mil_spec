use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_mm_part, MmPart};
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
    let p = "/tmp/mmact.macho";
    std::fs::write(p, emit_mm_part(MmPart::Act64, "/tmp/m.mil", p)).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(p), &[]).unwrap();
    let (dx, dy, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    // x: every row = q where q[k]=k/64 (broadcast query)
    {
        let b = m.input_mut(0);
        for r in 0..64 {
            for c in 0..64 {
                put(b, &dx, r, c, c as f32 / 64.0);
            }
        }
    }
    // y[r][c] = r*64+c
    {
        let b = m.input_mut(1);
        for r in 0..64 {
            for c in 0..64 {
                put(b, &dy, r, c, (r * 64 + c) as f32 / 64.0);
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut vals = [0f32; 64];
    for i in 0..64u64 {
        let o = dz.offset(0, 0, i, 0, 0);
        vals[i as usize] = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
    }
    println!("z[0..8]={:?}", &vals[..8]);
    println!("z[56..64]={:?}", &vals[56..]);
    // hypothesis: z[i] = q . y[i,:] = sum_k (k/64)*(i*64+k)
    let mut bad = 0;
    for i in 0..64 {
        let want: f32 = (0..64)
            .map(|k| (k as f32 / 64.0) * ((i * 64 + k) as f32 / 64.0))
            .sum();
        let got = vals[i];
        if (got - want).abs() > want.abs() * 0.01 + 0.5 {
            bad += 1;
            if bad < 6 {
                println!("z[{i}]={got} want={want}");
            }
        }
    }
    println!("broadcast-dot mismatches={bad}");
}
