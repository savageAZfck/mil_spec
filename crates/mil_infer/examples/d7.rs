use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_elementwise2, EwOp};
use std::path::Path;
fn main() {
    let pid = std::process::id();
    for n in [256u32, 512, 1024] {
        let p = format!("/tmp/d7_{n}_{pid}.macho");
        std::fs::write(&p, emit_elementwise2(EwOp::Mul, n, "/tmp/d7.mil", &p)).unwrap();
        let m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
        for d in m.input_descs.iter().take(1) {
            println!(
                "mul{n} in {}: n{} k{} h{} w{} ps{} rs{} bs{}",
                d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride, d.batch_stride
            );
        }
        for d in m.output_descs.iter().take(1) {
            println!(
                "mul{n} out {}: n{} k{} h{} w{} ps{} rs{} bs{}",
                d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride, d.batch_stride
            );
        }
    }
}
