//! Emit a tiny `.mlpackage` — `y = x * x` — and print the path.
//!
//! Run: `cargo run --example mlpackage_add`
//! Compile (macOS): `xcrun coremlc compile /tmp/mil_spec_add.mlpackage .`

use mil_spec::*;

fn main() -> std::io::Result<()> {
    let mut b = Block::new();
    let y = b.mul("x", "x", &[1, 4, 1, 1], "y");
    b.outputs = vec![y];

    let inputs = [Feature {
        name: "x".into(),
        shape: vec![1, 4, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 4, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fn_inputs = [NVT {
        name: "x".into(),
        ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])),
    }];

    let spec = encode_model(&inputs, &outputs, &[], &b, &fn_inputs, &ModelMeta::new(8, "CoreML5"));

    let dir = std::path::Path::new("/tmp/mil_spec_add.mlpackage");
    write_mlpackage(dir, &spec, None)?;
    println!("wrote {}", dir.display());
    println!("compile: xcrun coremlc compile {} .", dir.display());
    Ok(())
}
