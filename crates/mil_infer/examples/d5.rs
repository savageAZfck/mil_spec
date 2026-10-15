use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    let m =
        EspressoProgram::load_precompiled(Path::new("/tmp/maskadd_o/maskadd.macho"), &[]).unwrap();
    for d in &m.input_descs {
        println!(
            "in {} k{} h{} w{} ps{}",
            d.name, d.k, d.h, d.w, d.plane_stride
        );
    }
    for d in &m.output_descs {
        println!(
            "out {} k{} h{} w{} ps{}",
            d.name, d.k, d.h, d.w, d.plane_stride
        );
    }
}
