use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_repack64;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let p = format!("/tmp/rpk_{}.macho", std::process::id());
    std::fs::write(&p, emit_repack64("/tmp/rpk.mil", &p)).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    let dx = m.input_descs[0].clone();
    let dz = m.output_descs[0].clone();
    {
        let b = m.input_mut(0);
        for c in 0..64u64 {
            let o = dx.offset(0, 0, c, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(c as f32).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let mut mm = 0;
    for w in 0..64u64 {
        let o = dz.offset(0, 0, 0, 0, w);
        if o + 2 > z.len() {
            mm += 1;
            continue;
        }
        if f32_of(u16::from_le_bytes([z[o], z[o + 1]])) != w as f32 {
            mm += 1;
        }
    }
    println!("repack mismatches: {}/64", mm);
}
