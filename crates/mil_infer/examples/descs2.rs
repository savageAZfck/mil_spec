use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    let m = EspressoProgram::load_precompiled(Path::new("/tmp/smc_out/smc.macho"), &[]).unwrap();
    for d in &m.input_descs {
        println!(
            "in {} n{} k{} h{} w{} ps{} rs{} e{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride, d.elem
        );
    }
    for d in &m.output_descs {
        println!(
            "out {} n{} k{} h{} w{} ps{} rs{} e{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride, d.elem
        );
    }
}
