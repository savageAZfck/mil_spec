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
    let mode: u32 = std::env::args().nth(1).unwrap().parse().unwrap();
    let mut m =
        EspressoProgram::load_precompiled(Path::new("/tmp/avw2_out/avw2.macho"), &[]).unwrap();
    let (ds, dv, dz) = (
        m.input_descs[0].clone(),
        m.input_descs[1].clone(),
        m.output_descs[0].clone(),
    );
    {
        let b = m.input_mut(0);
        match mode {
            0 => {
                for i in 0..t {
                    let o = ds.offset(0, 0, i, 0, 0);
                    if o + 2 <= b.len() {
                        b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
                    }
                }
            }
            1 => {
                for i in 0..t {
                    let o = (i * 2) as usize;
                    if o + 2 <= b.len() {
                        b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
                    }
                }
            }
            _ => {
                for i in 0..b.len() / 2 {
                    if i % 2 == 0 {
                        b[2 * i..2 * i + 2].copy_from_slice(&f16(1.0).to_le_bytes());
                    }
                }
                b.iter_mut().for_each(|_| {});
                for c in 0..b.len() {
                    b[c] = 0;
                }
                for i in 0..t {
                    let o = ds.offset(0, 0, i, 0, 0);
                    if o + 2 <= b.len() {
                        b[o..o + 2].copy_from_slice(&f16(i as f32).to_le_bytes());
                    }
                }
            }
        }
    }
    {
        let b = m.input_mut(1);
        for i in 0..t {
            for j in 0..d {
                let o = dv.offset(0, 0, i, 0, j);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
                }
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut hits = 0;
    for c in 0..64 {
        for j in 0..d {
            let o = dz.offset(0, 0, c, 0, j);
            if o + 2 > z.len() {
                continue;
            }
            let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
            if v != 0.0 {
                hits += 1;
                if hits < 8 {
                    print!("c{}j{}:{} ", c, j, v);
                }
            }
        }
    }
    println!("hits={}", hits);
}
