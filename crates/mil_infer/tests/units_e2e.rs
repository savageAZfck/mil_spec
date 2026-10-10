//! Independent check of the `ComputeUnits` -> `MLComputeUnits` mapping:
//! a large fp16 elementwise (tanh) chain is far slower on the CPU than on the GPU, so
//! `CpuAndGpu` must beat `CpuOnly` by a wide margin. (With the old
//! pass-through discriminants, `CpuOnly` selected CPU+GPU and `CpuAndGpu`
//! selected All, so this ordering inverted or collapsed.)
//!
//! Timing-based, so `#[ignore]`d. Run:
//! `cargo test -p mil_infer --test units_e2e --release -- --ignored --nocapture`

use mil_infer::{ComputeUnits, Input, Model};
use mil_spec::*;

fn median_ms(m: &Model, ins: &[Input<'_>], runs: usize) -> f64 {
    for _ in 0..2 {
        m.predict(ins).expect("warmup");
    }
    let mut v: Vec<f64> = (0..runs)
        .map(|_| m.predict(ins).expect("predict").latency.as_secs_f64() * 1e3)
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "timing; needs coremlc + GPU"]
fn gpu_units_are_faster_than_cpu_on_big_tanh_chain() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    // 16M-element tanh chain: embarrassingly parallel transcendental
    // math, slow on CPU cores, fast on the GPU.
    let shape = [1, 16, 1024, 1024];
    let mut b = Block::new();
    let mut cur = "x".to_string();
    for i in 0..8 {
        cur = b.o1(
            "tanh",
            vec![("x".into(), bind(&cur).1)],
            &format!("t{i}"),
            ValueType::Tensor(TensorType::f16(&shape)),
        );
    }
    let last = cur.clone();
    b.outputs = vec![last.clone()];
    let feat = |name: &str| Feature {
        name: name.into(),
        shape: shape.to_vec(),
        dtype: DType::Fp16,
        is_state: false,
    };
    let fin: Vec<NVT> = ["x"]
        .iter()
        .map(|n| NVT {
            name: n.to_string(),
            ty: ValueType::Tensor(TensorType {
                dtype: DType::Fp16,
                shape: shape.to_vec(),
            }),
        })
        .collect();
    let spec = encode_model(
        &[feat("x")],
        &[feat(&last)],
        &[],
        &b,
        &fin,
        &ModelMeta::new(10, "CoreML9"),
    );
    let dir = std::env::temp_dir().join(format!("mil_units_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let pkg = dir.join("m.mlpackage");
    write_mlpackage(&pkg, &spec, None).unwrap();
    let cm = mil_compile::compile(&pkg, &dir.join("c")).unwrap();

    let data: Vec<u8> = (0..16 * 1024 * 1024_i64)
        .flat_map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.01).to_le_bytes())
        .collect();
    let ins = [Input {
        name: "x",
        shape: &shape,
        data: &data,
        dtype: DType::Fp16,
    }];
    let mut t = std::collections::BTreeMap::new();
    for (name, cu) in [
        ("CpuOnly", ComputeUnits::CpuOnly),
        ("CpuAndGpu", ComputeUnits::CpuAndGpu),
        ("All", ComputeUnits::All),
        ("CpuAndNeuralEngine", ComputeUnits::CpuAndNeuralEngine),
    ] {
        match Model::load(&cm.path, cu) {
            Ok(m) => {
                let ms = median_ms(&m, &ins, 5);
                println!("UNITS {name}: median {ms:.2} ms");
                t.insert(name, ms);
            }
            Err(e) => println!("UNITS {name}: load failed: {e}"),
        }
    }
    if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
        let _ = std::fs::remove_dir_all(&dir);
    }
    let (cpu, gpu) = (t["CpuOnly"], t["CpuAndGpu"]);
    assert!(
        gpu * 2.0 < cpu,
        "CpuAndGpu ({gpu:.2} ms) is not clearly faster than CpuOnly ({cpu:.2} ms)"
    );
}
