//! `lint_spec` (the `milc lint <pkg>` path) must give the same verdicts
//! as `lint_block` on the in-memory graph the package was written from:
//! dtype and rank are decoded from the MIL `TensorType`
//! (`dataType = 1`, `rank = 2`, `Dimension`s at 3), not read as an
//! `ArrayFeatureType`.

use mil_lint::{lint_block, Unit};
use mil_spec::*;

fn build() -> (Block, Vec<u8>) {
    let mut b = Block::new();
    let mut wb = WeightBin::new();
    // rank-2 fp16 blob weight (was misread as a "str const" -> CPU)
    let wdata: Vec<u8> = (0..32)
        .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
        .collect();
    let off = wb.put("w2", DType::Fp16, &[8, 4], &wdata);
    let w2 = b.konst_blob(
        "w2",
        "@model_path/weights/weight.bin",
        off,
        DType::Fp16,
        &[8, 4],
    );
    let _ = w2;
    // string const, int consts, fp16 scalar
    b.konst_str("pad_type_s", "valid");
    b.konst_i32("axes", &[1]);
    let half_c = b.konst_f16("half", 0.5);
    // fp16 elementwise (ANE)
    let t = b.mul("x", &half_c, &[1, 4, 2, 2], "t16");
    // fp32 island: cast up, fp32 elementwise, cast back
    let up = b.cast(&t, "fp32", &[1, 4, 2, 2], "up32", true);
    let sq = b.o1(
        "mul",
        vec![("x".into(), bind(&up).1), ("y".into(), bind(&up).1)],
        "sq32",
        ValueType::Tensor(TensorType {
            dtype: DType::Fp32,
            shape: vec![1, 4, 2, 2],
        }),
    );
    let down = b.cast(&sq, "fp16", &[1, 4, 2, 2], "down16", false);
    // rank-5 tensor (GPU by the rank guard)
    let r5 = b.reshape(&down, &[1, 4, 2, 1, 2], "r5");
    let r5b = b.add(&r5, &r5, &[1, 4, 2, 1, 2], "r5b");
    let back = b.reshape(&r5b, &[1, 4, 2, 2], "back4");
    b.outputs = vec![back.clone()];
    let shape = [1, 4, 2, 2];
    let feat = |n: &str| Feature {
        name: n.into(),
        shape: shape.to_vec(),
        dtype: DType::Fp16,
        is_state: false,
    };
    let fin = vec![NVT {
        name: "x".into(),
        ty: ValueType::Tensor(TensorType::f16(&shape)),
    }];
    let spec = encode_model(
        &[feat("x")],
        &[feat(&back)],
        &[],
        &b,
        &fin,
        &ModelMeta::new(10, "CoreML9"),
    );
    (b, spec)
}

#[test]
fn decoded_package_verdicts_equal_in_memory_verdicts() {
    let (b, spec) = build();
    let mem = lint_block(&b);
    let pkg = mil_verify::lint_spec(&spec).expect("decode");
    assert_eq!(mem.verdicts.len(), pkg.verdicts.len());
    for (m, p) in mem.verdicts.iter().zip(&pkg.verdicts) {
        assert_eq!(m.op, p.op);
        assert_eq!(m.name, p.name, "{}", m.op);
        assert_eq!(
            m.unit, p.unit,
            "{} {}: {} vs {}",
            m.op, m.name, m.rule, p.rule
        );
        assert_eq!(m.rule, p.rule, "{} {}", m.op, m.name);
        assert_eq!(m.shape, p.shape, "{} {} shape", m.op, m.name);
    }
}

#[test]
fn decoded_package_flags_fp32_rank5_and_keeps_weights_off_cpu() {
    let (_, spec) = build();
    let r = mil_verify::lint_spec(&spec).unwrap();
    let find = |name: &str| r.verdicts.iter().find(|v| v.name == name).unwrap();
    assert_eq!(find("sq32").unit, Unit::Gpu);
    assert_eq!(find("sq32").rule, "elementwise fp32");
    assert_eq!(find("r5b").rule, "rank>4");
    assert_eq!(find("r5").unit, Unit::Gpu);
    // the rank-2 fp16 weight const is not a string const
    let w = find("w2");
    assert_eq!(w.shape.as_deref(), Some(&[8i64, 4][..]));
    assert_ne!(w.rule, "str const");
    assert_eq!(find("t16").unit, Unit::Ane);
    // a real string const stays a string const
    assert_eq!(find("pad_type_s").rule, "str const");
}
