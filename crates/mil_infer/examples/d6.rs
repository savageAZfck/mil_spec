use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for p in std::env::args().skip(1) {
        let m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
        for d in &m.input_descs {
            println!(
                "{} in {} k{} h{} w{} ps{}",
                p, d.name, d.k, d.h, d.w, d.plane_stride
            );
        }
        for d in &m.output_descs {
            println!(
                "{} out {} k{} h{} w{} ps{}",
                p, d.name, d.k, d.h, d.w, d.plane_stride
            );
        }
    }
}
