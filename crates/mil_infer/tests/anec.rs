//! Direct ANECompiler.framework validation tests.
//!
//! Gated on the private framework being loadable — skipped anywhere
//! that isn't macOS with the dyld cache present.

use std::path::PathBuf;

const GOOD: &str = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 4]> x, tensor<fp16, [1, 4]> y) {
        tensor<fp16, [1, 4]> z = add(x = x, y = y)[name = string("z")];
    } -> (z);
}
"#;

const BAD_INT: &str = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 4]> x, tensor<int32, [1]> pos) {
        tensor<int32, [1]> one = const()[val = tensor<int32, [1]>(1)];
        tensor<int32, [1]> p2 = add(x = pos, y = one)[name = string("p2")];
    } -> (x);
}
"#;

fn fixture(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("anec_test_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("model.mil");
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn anec_validate_accepts_fp_compute() {
    let Ok(ane) = mil_infer::Anec::open() else {
        eprintln!("ANECompiler.framework unavailable — skipping");
        return;
    };
    let mil = fixture("good", GOOD);
    let v = ane.validate(&mil, "h16g").expect("validate call");
    assert_eq!(v.rc, 0, "rc={} errors={:?}", v.rc, v.errors);
    assert_eq!(v.status, 0, "status={:#x} errors={:?}", v.status, v.errors);
    assert!(v.errors.is_empty(), "errors={:?}", v.errors);
}

#[test]
fn anec_validate_rejects_int_compute() {
    let Ok(ane) = mil_infer::Anec::open() else {
        eprintln!("ANECompiler.framework unavailable — skipping");
        return;
    };
    let mil = fixture("bad_int", BAD_INT);
    let v = ane.validate(&mil, "h16g").expect("validate call");
    assert!(
        v.rc != 0 || v.status != 0 || !v.errors.is_empty(),
        "int32 arithmetic should fail the eligibility gate"
    );
}
