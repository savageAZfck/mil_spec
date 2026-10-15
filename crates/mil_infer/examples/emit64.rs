use mil_infer::hwx::{emit_elementwise2, EwOp};
fn main() {
    for (op, name) in [(EwOp::Mul, "m1024"), (EwOp::Add, "a1024")] {
        let p = format!("/tmp/{name}.macho");
        std::fs::write(&p, emit_elementwise2(op, 1024, "/tmp/a.mil", &p)).unwrap();
    }
}
