use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    let m = EspressoProgram::load_precompiled(Path::new("/tmp/kv2_mul.macho"), &[]).unwrap();
    for d in m.input_descs.iter().chain(m.output_descs.iter()) {
        println!(
            "{}: n={} k={} h={} w={} bs={} ds={} ps={} rs={} elem={}",
            d.name,
            d.n,
            d.k,
            d.h,
            d.w,
            d.batch_stride,
            d.depth_stride,
            d.plane_stride,
            d.row_stride,
            d.elem
        );
    }
}
