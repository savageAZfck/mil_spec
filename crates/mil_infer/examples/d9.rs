use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for p in std::env::args().skip(1) {
        let m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
        for d in m.input_descs.iter().take(1) {
            println!(
                "{p} in: n{} k{} h{} w{} ps{} bs{}",
                d.n, d.k, d.h, d.w, d.plane_stride, d.batch_stride
            );
        }
        for d in m.output_descs.iter().take(1) {
            println!(
                "{p} out: n{} k{} h{} w{} ps{} bs{}",
                d.n, d.k, d.h, d.w, d.plane_stride, d.batch_stride
            );
        }
    }
}
