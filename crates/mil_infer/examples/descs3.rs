use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for tag in ["t", "r"] {
        let p = format!("/tmp/repack_{}_out/repack_{}.macho", tag, tag);
        let m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
        for d in &m.input_descs {
            println!(
                "{tag} in {} k{} h{} w{} ps{} rs{}",
                d.name, d.k, d.h, d.w, d.plane_stride, d.row_stride
            );
        }
        for d in &m.output_descs {
            println!(
                "{tag} out {} k{} h{} w{} ps{} rs{}",
                d.name, d.k, d.h, d.w, d.plane_stride, d.row_stride
            );
        }
    }
}
