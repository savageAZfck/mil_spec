use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_unary_lut, LutOp};
use std::path::Path;
fn main() {
    let pid = std::process::id();
    for n in [128u32, 256, 512] {
        let p = format!("/tmp/d8_{n}_{pid}.macho");
        std::fs::write(&p, emit_unary_lut(LutOp::Sigmoid, n, "/tmp/d8.mil", &p)).unwrap();
        match EspressoProgram::load_precompiled(Path::new(&p), &[]) {
            Ok(m) => {
                for d in m.input_descs.iter().take(1) {
                    println!(
                        "sig{n} in: n{} k{} h{} w{} ps{}",
                        d.n, d.k, d.h, d.w, d.plane_stride
                    );
                }
                for d in m.output_descs.iter().take(1) {
                    println!(
                        "sig{n} out: n{} k{} h{} w{} ps{}",
                        d.n, d.k, d.h, d.w, d.plane_stride
                    );
                }
            }
            Err(e) => println!("sig{n}: load failed {e:?}"),
        }
    }
    // also describe the banked sig256 skeleton
    let m = EspressoProgram::load_precompiled(Path::new("/tmp/bank256/sig256.macho"), &[]).unwrap();
    for d in m.input_descs.iter().take(1) {
        println!(
            "sig256-skel in: n{} k{} h{} w{} ps{}",
            d.n, d.k, d.h, d.w, d.plane_stride
        );
    }
}
