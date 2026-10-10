//! End-to-end pipeline test: safetensors → mil_convert → package →
//! verify → lint → coremlc compile. This is the proof that the whole
//! workspace produces artifacts Apple's own compiler accepts.

use mil_convert::{convert, Options, Quant};
use std::io::Write;
use std::path::Path;

fn write_safetensors(path: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
    let mut header = String::from("{");
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape, vals) in tensors {
        let off = data.len() as u64;
        for v in vals {
            data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{shape_str}],\"data_offsets\":[{off},{}]}}",
            data.len()
        ));
    }
    header.push_str(&entries.join(","));
    header.push('}');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
}

fn tiny_qwen3(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": false
        }"#,
    )
    .unwrap();
    let mut tensors: Vec<(String, Vec<i64>, Vec<f32>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<i64>, fill: f32| {
        let n: i64 = shape.iter().product();
        tensors.push((
            name.to_string(),
            shape,
            (0..n).map(|i| fill + (i as f32 % 7.0) * 0.01).collect(),
        ));
    };
    for l in 0..2 {
        push(
            &format!("model.layers.{l}.input_layernorm.weight"),
            vec![16],
            1.0,
        );
        push(
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            vec![16],
            1.0,
        );
        push(
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            vec![16, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            vec![8, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            vec![8, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            vec![16, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.self_attn.q_norm.weight"),
            vec![8],
            1.0,
        );
        push(
            &format!("model.layers.{l}.self_attn.k_norm.weight"),
            vec![8],
            1.0,
        );
        push(
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            vec![32, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            vec![32, 16],
            0.02,
        );
        push(
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            vec![16, 32],
            0.02,
        );
    }
    push("model.norm.weight", vec![16], 1.0);
    push("lm_head.weight", vec![64, 16], 0.02);
    let refs: Vec<(&str, Vec<i64>, Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
        .collect();
    write_safetensors(&dir.join("model.safetensors"), &refs);
}

#[test]
fn full_pipeline_compiles() {
    let root = std::env::temp_dir().join(format!("pipe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("model");
    tiny_qwen3(&model);
    let pkg = root.join("tiny.mlpackage");

    let r = convert(
        &model,
        &pkg,
        &Options {
            seq: 1,
            max_kv: 16,
            quant: Quant::Int8,
            lm_head: true,
            embed: false,
            spec_version: 10,
            opset: "CoreML9".into(),
            lora: None,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(r.op_count > 0);

    // 1. verify the package structure
    let vr = mil_verify::verify_package(&pkg);
    assert!(vr.is_ok(), "package verify failed: {}", vr);

    // 2. summarize the spec
    let spec = std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    let s = mil_verify::summarize(&spec).unwrap();
    assert_eq!(s.states.len(), 1, "expected one packed KV state");
    assert_eq!(s.states[0].name, "kv");
    assert!(s.inputs.iter().any(|f| f.name == "mask"));
    assert!(s.inputs.iter().any(|f| f.name == "pos"));
    assert!(s.op_counts.iter().any(|(t, _)| t == "conv"));
    assert!(s.op_counts.iter().any(|(t, _)| t == "slice_update"));
    assert!(s.op_counts.iter().any(|(t, _)| t == "write_state"));

    // 3. compile with coremlc — the acceptance test. Skipped when CLT is absent.
    if mil_compile::coremlc_path().is_some() {
        let out = root.join("compiled");
        let m = mil_compile::compile(&pkg, &out);
        match m {
            Ok(cm) => {
                assert!(cm.path.exists());
                println!("compiled via {}: {}", cm.backend, cm.path.display());
            }
            Err(e) => panic!("coremlc rejected the generated graph:\n{e}"),
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// The whole point of the toolchain: load the compiled stateful decoder,
/// run two sequential predicts on one MLState, and prove the KV cache
/// actually threads — logits at pos=1 must differ from pos=0, and a
/// fresh state must reproduce pos=0's logits exactly.
#[test]
fn stateful_predict_threads_kv() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("st_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("model");
    tiny_qwen3(&model);
    let pkg = root.join("tiny.mlpackage");
    convert(
        &model,
        &pkg,
        &Options {
            seq: 1,
            max_kv: 16,
            quant: Quant::Int8,
            lm_head: true,
            embed: false,
            spec_version: 10,
            opset: "CoreML9".into(),
            lora: None,
            ..Options::default()
        },
    )
    .unwrap();
    let compiled = mil_compile::compile(&pkg, &root.join("compiled")).unwrap();

    use mil_infer::{ComputeUnits, Input, Model};
    let m = Model::load(&compiled.path, ComputeUnits::All).unwrap();
    let state = m.new_state().expect("model should expose MLState");

    let f16s = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    };
    // Two different hidden inputs — real decode steps get different x.
    // (Same x at both positions would write identical K/V and correctly
    // produce identical logits — a false negative.)
    let x0: Vec<f32> = (0..16).map(|i| 0.05 * (i as f32 + 1.0)).collect();
    let x1: Vec<f32> = (0..16).map(|i| -0.03 * (i as f32 + 2.0)).collect();
    let cos = f16s(&[1.0; 8]); // rope identity
    let sin = f16s(&[0.0; 8]);
    let run = |pos: i32, xb: &[u8], st: &mil_infer::State| -> Vec<f32> {
        // causal mask: 0 up to and including pos, -1e4 after
        let mut mk = vec![-1e4f32; 16];
        for m_ in mk.iter_mut().take(pos as usize + 1) {
            *m_ = 0.0;
        }
        let p = m
            .predict_with_state(
                Some(st),
                &[
                    Input {
                        name: "x",
                        shape: &[1, 16, 1, 1],
                        data: xb,
                        dtype: mil_spec::DType::Fp16,
                    },
                    Input {
                        name: "cos",
                        shape: &[1, 1, 1, 8],
                        data: &cos,
                        dtype: mil_spec::DType::Fp16,
                    },
                    Input {
                        name: "sin",
                        shape: &[1, 1, 1, 8],
                        data: &sin,
                        dtype: mil_spec::DType::Fp16,
                    },
                    Input {
                        name: "mask",
                        shape: &[1, 1, 1, 16],
                        data: &f16s(&mk),
                        dtype: mil_spec::DType::Fp16,
                    },
                    Input {
                        name: "pos",
                        shape: &[1],
                        data: &pos.to_le_bytes(),
                        dtype: mil_spec::DType::Int32,
                    },
                ],
            )
            .unwrap();
        assert_eq!(p.outputs.len(), 1);
        assert_eq!(p.outputs[0].shape, vec![1, 64, 1, 1]);
        let v = p.outputs[0].values();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite logits");
        v
    };

    let xb0 = f16s(&x0);
    let xb1 = f16s(&x1);
    let l0 = run(0, &xb0, &state);
    let l1 = run(1, &xb1, &state);
    assert_ne!(
        l0, l1,
        "pos=1 logits identical to pos=0 — KV state is not threading"
    );

    // fresh state must reproduce pos=0 exactly
    let s2 = m.new_state().unwrap();
    let l0b = run(0, &xb0, &s2);
    assert_eq!(l0, l0b, "same pos on fresh state must be deterministic");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn determinism_same_source_same_spec() {
    let root = std::env::temp_dir().join(format!("det_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("model");
    tiny_qwen3(&model);
    let opts = Options {
        seq: 1,
        max_kv: 16,
        quant: Quant::Int8,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..Options::default()
    };
    let p1 = root.join("a.mlpackage");
    let p2 = root.join("b.mlpackage");
    convert(&model, &p1, &opts).unwrap();
    convert(&model, &p2, &opts).unwrap();
    let s1 = std::fs::read(p1.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    let s2 = std::fs::read(p2.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    assert_eq!(s1, s2, "conversion must be byte-for-byte deterministic");
    assert!(mil_verify::diff_specs(&s1, &s2).is_empty());
    let _ = std::fs::remove_dir_all(&root);
}
