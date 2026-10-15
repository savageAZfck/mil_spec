use mil_infer::hwx::{emit_elementwise2, EwOp};
fn main() {
    let a = emit_elementwise2(EwOp::Mul, 256, "kv_mul.mil", "/tmp/kv_mul.macho");
    std::fs::write("/tmp/kv_a.macho", &a).unwrap();
    let b = emit_elementwise2(EwOp::Mul, 256, "kv_mul.mil", "/tmp/kv_mul.macho");
    std::fs::write("/tmp/kv_b.macho", &b).unwrap();
    println!("identical={}", a == b);
}
