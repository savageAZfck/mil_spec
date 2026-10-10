//! EnumeratedShapes end-to-end: convert a tiny model with
//! `--seq-lens`, compile via coremlc, predict at a non-default
//! sequence length, and compare logits with a fixed-length package.
//!
//! Run: cargo test -p mil_convert --test enum_shapes_e2e --release -- --ignored --nocapture

#![cfg(target_os = "macos")]

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

fn f16s(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

/// The shared input set at sequence length `s`: hidden states, rope
/// tables (deterministic non-trivial values), causal mask, pos=0 and —
/// for the flexible model — the runtime `seq` scalar.
fn inputs_at(s: i64, max_kv: i64, with_seq: bool) -> Vec<mil_infer::Input<'static>> {
    let d = 16i64;
    let sh = |v: &[i64]| -> &'static [i64] { Box::leak(v.to_vec().into_boxed_slice()) };
    let hd = 8i64;
    // deterministic-but-varied activations
    let x: Vec<f32> = (0..d * s)
        .map(|i| 0.05 * ((i * 7 % 13) as f32) - 0.3)
        .collect();
    let mut cs = Vec::new();
    let mut sn = Vec::new();
    for p in 0..s {
        for c in 0..hd {
            let f = (p * hd + c) as f32 * 0.1;
            cs.push(f.cos());
            sn.push(f.sin());
        }
    }
    let mut mk = vec![-1e4f32; (s * max_kv) as usize];
    for p in 0..s {
        for k in 0..=p {
            mk[(p * max_kv + k) as usize] = 0.0;
        }
    }
    let mut v = vec![
        mil_infer::Input {
            name: "x",
            shape: sh(&[1, d, s, 1]),
            data: Box::leak(f16s(&x).into_boxed_slice()),
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "cos",
            shape: sh(&[1, 1, s, hd]),
            data: Box::leak(f16s(&cs).into_boxed_slice()),
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "sin",
            shape: sh(&[1, 1, s, hd]),
            data: Box::leak(f16s(&sn).into_boxed_slice()),
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "mask",
            shape: sh(&[1, 1, s, max_kv]),
            data: Box::leak(f16s(&mk).into_boxed_slice()),
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "pos",
            shape: sh(&[1]),
            data: Box::leak(Box::new(0i32.to_le_bytes())),
            dtype: mil_spec::DType::Int32,
        },
    ];
    if with_seq {
        v.push(mil_infer::Input {
            name: "seq",
            shape: sh(&[1]),
            data: Box::leak(Box::new((s as i32).to_le_bytes())),
            dtype: mil_spec::DType::Int32,
        });
    }
    v
}

fn predict(m: &mil_infer::Model, inputs: &[mil_infer::Input]) -> Vec<f32> {
    let st = m.new_state().expect("state");
    let p = m.predict_with_state(Some(&st), inputs);
    let p = match p {
        Ok(p) => p,
        Err(e) => panic!("predict failed on input shapes {:?}: {e}", inputs[0].shape),
    };
    assert_eq!(p.outputs.len(), 1);
    p.outputs[0].values()
}

/// Convert the same checkpoint twice — fixed `--seq 4` and
/// `--seq-lens 1,4,8` — then prove the flexible model's logits at seq 4
/// match the fixed package, and that seq 1 (the default entry) also
/// runs.
#[test]
#[ignore = "e2e: needs coremlc"]
fn enumerated_shapes_match_fixed_package() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("enum_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("model");
    tiny_qwen3(&model);

    let fixed_pkg = root.join("fixed4.mlpackage");
    convert(
        &model,
        &fixed_pkg,
        &Options {
            seq: 4,
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

    let fixed8_pkg = root.join("fixed8.mlpackage");
    convert(
        &model,
        &fixed8_pkg,
        &Options {
            seq: 8,
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

    let flex_pkg = root.join("flex.mlpackage");
    convert(
        &model,
        &flex_pkg,
        &Options {
            seq_lens: vec![1, 4, 8],
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

    let fixed_c = mil_compile::compile(&fixed_pkg, &root.join("cf")).unwrap();
    let fixed8_c = mil_compile::compile(&fixed8_pkg, &root.join("c8")).unwrap();
    let flex_c = mil_compile::compile(&flex_pkg, &root.join("cx")).unwrap();
    let mf = mil_infer::Model::load(&fixed_c.path, mil_infer::ComputeUnits::All).unwrap();
    let m8 = mil_infer::Model::load(&fixed8_c.path, mil_infer::ComputeUnits::All).unwrap();
    let mx = mil_infer::Model::load(&flex_c.path, mil_infer::ComputeUnits::All).unwrap();

    eprintln!("predict flex s=1 (default)…");
    let g1 = predict(&mx, &inputs_at(1, 16, true));
    assert_eq!(g1.len(), 64);
    assert!(g1.iter().all(|v| v.is_finite()));
    // non-default entry: seq=4 (default is the first entry, 1)
    eprintln!("predict fixed s=4…");
    let want = predict(&mf, &inputs_at(4, 16, false));
    eprintln!("predict flex s=4…");
    let got = predict(&mx, &inputs_at(4, 16, true));
    assert_eq!(got.len(), want.len());
    let mut max_err = 0f32;
    for (g, w) in got.iter().zip(&want) {
        max_err = max_err.max((g - w).abs());
    }
    eprintln!("seq=4: flex vs fixed max|Δ| = {max_err}");
    assert!(
        max_err < 0.02,
        "flex s=4 logits diverge from fixed package (max|Δ|={max_err})"
    );

    // the default entry (seq=1) must also run
    let got1 = predict(&mx, &inputs_at(1, 16, true));
    assert_eq!(got1.len(), 64);
    assert!(got1.iter().all(|v| v.is_finite()));

    // and the third entry (seq=8)
    let want8 = predict(&m8, &inputs_at(8, 16, false));
    let got8 = predict(&mx, &inputs_at(8, 16, true));
    let mut max8 = 0f32;
    for (g, w) in got8.iter().zip(&want8) {
        max8 = max8.max((g - w).abs());
    }
    eprintln!("seq=8: flex vs fixed max|Δ| = {max8}");
    assert!(max8 < 0.02);

    if std::env::var("MIL_KEEP_ARTIFACTS").is_err() {
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// Bisect the real flex block: truncate at each op prefix, compile,
/// predict at s=1. The first failing prefix fingers the op CoreML
/// can't execute under the symbolic/enum shape declarations.
#[test]
#[ignore = "e2e: needs coremlc"]
fn enum_prefix_bisect() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    use mil_convert::builder::{self, WeightEmitter};
    use mil_convert::safetensors;
    use mil_convert::{plan, ModelConfig};
    let root = std::env::temp_dir().join(format!("enum_bisect_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("model");
    tiny_qwen3(&model);
    let cfg = ModelConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let opts = Options {
        seq_lens: vec![1, 4, 8],
        max_kv: 16,
        quant: Quant::Int8,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..Options::default()
    };
    let build = || -> mil_convert::builder::Built {
        let src = safetensors::open_dir(&model).unwrap();
        let plan = plan::Plan::uniform(opts.quant);
        let tmp_w = root.join("w.bin");
        let mut writer = mil_spec::BlobWriter::create(&tmp_w).unwrap();
        let mut em = WeightEmitter {
            w: &mut writer,
            quant: opts.quant,
            file: "@model_path/weights/weight.bin".into(),
            src: &src,
            plan: &plan,
        };
        let built = builder::build(&cfg, &opts, &mut em).unwrap();
        writer.finish().unwrap();
        built
    };
    let n_ops = build().block.ops.len();
    eprintln!("flex block: {n_ops} ops");
    if std::env::var("MIL_DUMP_OPS").is_ok() {
        for (i, op) in build().block.ops.iter().enumerate() {
            eprintln!("  op {}: {} -> {}", i + 1, op.ty, op.outputs[0].name);
        }
    }

    // Compile+predict a prefix at s=1 and s=4. Ok(false) = skipped
    // (output can't be a model output); Err = this prefix fails.
    let try_prefix = |upto: usize| -> Result<bool, String> {
        let mut built = build();
        built.block.ops.truncate(upto);
        // Every prefix reads `kv` but the terminal write_state lives at
        // the end of the full graph — re-attach it (writing the
        // untouched read value back) or CoreML rejects the state at
        // load time. Only when the prefix doesn't already carry one.
        if !built.block.ops.iter().any(|o| o.ty == "write_state") {
            built.block.ops.push(mil_spec::Op {
                ty: "write_state".into(),
                inputs: vec![
                    ("input".into(), mil_spec::bind("kv").1),
                    ("data".into(), mil_spec::bind("kv_0").1),
                ],
                outputs: vec![],
                attrs: vec![("name".into(), mil_spec::Value::Str("kv_write_state".into()))],
            });
        }
        let last = &built.block.ops[upto - 1];
        if matches!(
            last.ty.as_str(),
            "const" | "constexpr_blockwise_shift_scale" | "read_state" | "write_state"
        ) {
            return Ok(false);
        }
        let (oname, oshape, odtype) = match &last.outputs[0].ty {
            mil_spec::ValueType::Tensor(t) => {
                (last.outputs[0].name.clone(), t.shape.clone(), t.dtype)
            }
            _ => return Ok(false),
        };
        if odtype != mil_spec::DType::Fp16 {
            return Ok(false);
        }
        built.block.outputs = vec![oname.clone()];
        let outputs = vec![mil_spec::Feature {
            name: oname.clone(),
            shape: oshape.clone(),
            dtype: odtype,
            is_state: false,
        }];
        let mut flex = built.enum_shapes.take().unwrap_or_default();
        if let Some(marks) = built.syms.get(&oname) {
            let s0 = opts.seq_lens[0];
            flex.insert(
                oname.clone(),
                mil_spec::EnumeratedShapes {
                    shapes: opts
                        .seq_lens
                        .iter()
                        .map(|&v| {
                            oshape
                                .iter()
                                .enumerate()
                                .map(|(i, &d)| {
                                    if marks.get(i).is_some_and(|m| m.is_some()) {
                                        d * v / s0
                                    } else {
                                        d
                                    }
                                })
                                .collect()
                        })
                        .collect(),
                },
            );
        }
        let spec = mil_spec::encode_model_flex(
            &built.inputs,
            &outputs,
            &built.states,
            &built.block,
            &built.fn_inputs,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
            &flex,
            &built.syms,
        );
        let dir = root.join(format!("p{upto}"));
        let pkg = dir.join("m.mlpackage");
        std::fs::create_dir_all(&dir).unwrap();
        mil_spec::write_mlpackage_stream(&pkg, &spec, Some(&root.join("w.bin")))
            .map_err(|e| format!("pkg: {e}"))?;
        let comp = mil_compile::compile(&pkg, &dir.join("c")).map_err(|e| format!("cc: {e}"))?;
        let m = mil_infer::Model::load(&comp.path, mil_infer::ComputeUnits::All)
            .map_err(|e| format!("load: {e}"))?;
        let st = m.new_state().map_err(|e| format!("state: {e}"))?;
        m.predict_with_state(Some(&st), &inputs_at(1, 16, true))
            .map_err(|e| format!("s=1 predict: {e}"))?;
        let _ = std::fs::remove_dir_all(&dir);
        Ok(true)
    };

    // Binary search for the first failing prefix — a broken producer
    // keeps every downstream prefix broken, so failures are monotone
    // in practice. ~log2(n) coremlc compiles keeps a clean run inside
    // the test timeout.
    let mut lo = 1usize;
    let mut hi;
    // establish the invariant "prefix hi fails": find the largest
    // prefix ending in an fp16 tensor output and test it.
    let mut top = n_ops;
    loop {
        let built = build();
        let op = &built.block.ops[top - 1];
        let is_f16 = matches!(
            &op.outputs[0].ty,
            mil_spec::ValueType::Tensor(t) if t.dtype == mil_spec::DType::Fp16
        );
        let is_terminal = matches!(
            op.ty.as_str(),
            "const" | "constexpr_blockwise_shift_scale" | "read_state" | "write_state"
        );
        if is_f16 && !is_terminal {
            break;
        }
        top -= 1;
    }
    match try_prefix(top) {
        Ok(true) => {
            eprintln!("prefix {top} (last fp16 op) passes — no failing prefix found");
            if std::env::var("MIL_KEEP_ARTIFACTS").is_err() {
                let _ = std::fs::remove_dir_all(&root);
            }
            return;
        }
        Ok(false) => unreachable!("top is fp16 by construction"),
        Err(e) => eprintln!("prefix {top} fails: {e} — bisecting"),
    }
    hi = top;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        match try_prefix(mid) {
            Ok(true) => {
                eprintln!("prefix {mid} ok");
                lo = mid;
            }
            Ok(false) => {
                // skipped output type — probe one op lower instead
                eprintln!("prefix {mid}: skipped, probing {mid} - 1");
                match try_prefix(mid - 1) {
                    Ok(true) | Ok(false) => lo = mid,
                    Err(_) => hi = mid - 1,
                }
            }
            Err(e) => {
                eprintln!("prefix {mid} FAILS: {e}");
                hi = mid;
            }
        }
    }
    {
        let built = build();
        let op = &built.block.ops[hi - 1];
        eprintln!(
            "FIRST FAILING prefix {hi}: op '{}' -> {}",
            op.ty, op.outputs[0].name
        );
        for (n, _) in &op.inputs {
            eprintln!("   input: {n}");
        }
    }
    if std::env::var("MIL_KEEP_ARTIFACTS").is_err() {
        let _ = std::fs::remove_dir_all(&root);
    }
}
