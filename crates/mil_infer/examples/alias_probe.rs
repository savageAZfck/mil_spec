//! alias_probe — prove IOSurface-level aliasing gives in-place state.
//! Loads the stateless `add(x,x)->y` artifact, aliases output y onto
//! input x's surface, evals twice: x -> 2x -> 4x if aliasing holds
//! (doubling proves the second eval read the first eval's write).

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
        .unwrap_or("/tmp/stv/o_out/o.macho".into());
    let mut m = EspressoProgram::load_precompiled(Path::new(&macho), &[]).expect("load");
    println!(
        "inputs={:?} outputs={:?}",
        m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
        m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>()
    );
    let d = m.input_descs[0].clone();
    // x = all 1.0 (element dim is `w` for dense scalars)
    {
        let b = m.input_mut(0);
        for p in 0..d.w.max(1).min(16) {
            let off = d.offset(0, 0, 0, 0, p);
            if off + 2 <= b.len() {
                b[off..off + 2].copy_from_slice(&f16(1.0).to_le_bytes());
            }
        }
    }
    m.alias_out_to_in(0, 0);
    for step in 0..3 {
        m.eval().expect("eval");
        let data = m.input_bytes(0);
        let off = d.offset(0, 0, 0, 0, 0);
        let v = f32_of(u16::from_le_bytes([data[off], data[off + 1]]));
        println!("step {step}: x[0] = {v}");
    }
}
