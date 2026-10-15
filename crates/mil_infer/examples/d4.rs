use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    let m = EspressoProgram::load_precompiled(Path::new("/tmp/add4d_o/add4d.macho"), &[]).unwrap();
    for d in &m.input_descs {
        println!(
            "in {} n{} k{} h{} w{} ps{} rs{} bs{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride, d.batch_stride
        );
    }
    for d in &m.output_descs {
        println!(
            "out {} n{} k{} h{} w{} ps{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride
        );
    }
}
