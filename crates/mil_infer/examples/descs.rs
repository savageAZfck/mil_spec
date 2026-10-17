use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_avw, emit_softmax};
use std::path::Path;
fn show(tag: &str, d: &mil_infer::espresso::TensorDesc) {
    println!(
        "{tag} [{}] n{} k{} h{} w{} ps{} rs{} bs{} ds{} e{}",
        d.name,
        d.n,
        d.k,
        d.h,
        d.w,
        d.plane_stride,
        d.row_stride,
        d.batch_stride,
        d.depth_stride,
        d.elem
    );
}
fn main() {
    let pid = std::process::id();
    let p1 = format!("/tmp/d_avw_{pid}.macho");
    std::fs::write(&p1, emit_avw(64, 64, "/tmp/d1.mil", &p1)).unwrap();
    let m1 = EspressoProgram::load_precompiled(Path::new(&p1), &[]).unwrap();
    for d in &m1.input_descs {
        show("avw in", d);
    }
    for d in &m1.output_descs {
        show("avw out", d);
    }
    let p2 = format!("/tmp/d_sm_{pid}.macho");
    std::fs::write(&p2, emit_softmax(64, "/tmp/d2.mil", &p2).unwrap()).unwrap();
    let m2 = EspressoProgram::load_precompiled(Path::new(&p2), &[]).unwrap();
    for d in &m2.input_descs {
        show("sm in", d);
    }
    for d in &m2.output_descs {
        show("sm out", d);
    }
}
