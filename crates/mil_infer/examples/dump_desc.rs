use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for p in std::env::args().skip(1) {
        let m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
        for d in m.input_descs.iter().chain(m.output_descs.iter()) {
            println!(
                "{p} {}: n={} k={} h={} w={} bs={} ps={} elem={} bytes={}",
                d.name,
                d.n,
                d.k,
                d.h,
                d.w,
                d.batch_stride,
                d.plane_stride,
                d.elem,
                d.bytes()
            );
        }
    }
}
