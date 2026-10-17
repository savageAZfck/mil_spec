use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_matmul_ns;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn main() {
    let pid = std::process::id();
    let d = 64usize;
    let w: Vec<u16> = (0..d * d)
        .map(|i| f16(((i / d * 7 + i % d * 3) % 17) as f32 / 64.0))
        .collect();
    let p = format!("/tmp/hdbg_{pid}.macho");
    std::fs::write(&p, emit_matmul_ns(d, d, &w, "/tmp/hd.mil", &p).unwrap()).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    let dx = m.input_descs[0].clone();
    let dz = m.output_descs[0].clone();
    println!(
        "x k{} w{} ps{} | z k{} h{} w{} ps{} surf?",
        dx.k, dx.w, dx.plane_stride, dz.k, dz.h, dz.w, dz.plane_stride
    );
    {
        let b = m.input_mut(0);
        for i in 0..d {
            let o = dx.offset(0, 0, 0, 0, i as u64);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(i as f32 / 64.0).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    println!("eval ok");
}
