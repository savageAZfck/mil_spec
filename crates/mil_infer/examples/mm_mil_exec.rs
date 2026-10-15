//! mm_mil_exec — load the emit_from_mil matmul-M4 artifact, fill the
//! channel-major input x[k,m] = k*16+m (m in 0..W), eval, verify the
//! identity matmul per position.

use mil_infer::espresso::EspressoProgram;
use std::path::Path;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn main() {
    let macho = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/mmroute/out.macho".into());
    let wd: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let mut m =
        EspressoProgram::load_precompiled(Path::new(&macho), &[]).expect("load_precompiled");
    println!(
        "loaded: ph={} inputs={:?} outputs={:?}",
        m.program_handle,
        m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
    );
    let d = m.input_descs[0].clone();
    {
        let b = m.input_mut(0);
        for k in 0..d.k {
            for p in 0..wd {
                let off = d.offset(0, 0, k, 0, p);
                let v = (k * 16 + p) as f32;
                b[off..off + 2].copy_from_slice(&f16(v).to_le_bytes());
            }
        }
    }
    m.eval().expect("eval");
    let d = m.output_descs[0].clone();
    let out = m.output(0);
    let (mut bad, mut tot) = (0u32, 0u32);
    for n in 0..d.k {
        for p in 0..wd {
            let off = d.offset(0, 0, n, 0, p);
            let v = f32_of(u16::from_le_bytes([out[off], out[off + 1]]));
            let want = (n * 16 + p) as f32;
            tot += 1;
            if v != want {
                if bad < 8 {
                    println!("z[{n},{p}] = {v} want {want}");
                }
                bad += 1;
            }
        }
    }
    println!("{bad}/{tot} mismatches");
}
