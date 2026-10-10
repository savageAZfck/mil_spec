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
        for k in 0..=p.min(max_kv - 1) {
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
        let mut flex = built.flex.take().unwrap_or_default();
        if let Some(marks) = built.syms.get(&oname) {
            let s0 = opts.seq_lens[0];
            flex.insert(
                oname.clone(),
                mil_spec::Flex::Enumerated(mil_spec::EnumeratedShapes {
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
                }),
            );
        }
        let spec = mil_spec::encode_model_flexible(
            &built.inputs,
            &outputs,
            &built.states,
            &built.block,
            &built.fn_inputs,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
            &flex,
            &built.syms,
        )
        .unwrap();
        let dir = root.join(format!("p{upto}"));
        let pkg = dir.join("m.mlpackage");
        std::fs::create_dir_all(&dir).unwrap();
        mil_spec::write_mlpackage_stream(&pkg, &spec, Some(&root.join("w.bin")))
            .map_err(|e| format!("pkg: {e}"))?;
        let comp = mil_compile::compile(&pkg, &dir.join("c")).map_err(|e| format!("cc: {e}"))?;
        let cpu = std::env::var("MIL_BISECT_CPU").is_ok();
        let units = if cpu {
            mil_infer::ComputeUnits::CpuOnly
        } else {
            mil_infer::ComputeUnits::All
        };
        let m = mil_infer::Model::load(&comp.path, units).map_err(|e| format!("load: {e}"))?;
        if cpu {
            let _ = std::fs::remove_dir_all(&dir);
            return Ok(true);
        }
        let st = m.new_state().map_err(|e| format!("state: {e}"))?;
        m.predict_with_state(Some(&st), &inputs_at(1, 16, true))
            .map_err(|e| format!("s=1 predict: {e}"))?;
        let _ = std::fs::remove_dir_all(&dir);
        Ok(true)
    };

    if let Ok(n) = std::env::var("MIL_BISECT_LINEAR") {
        let n: usize = n.parse().unwrap();
        for upto in 1..=n {
            let built0 = build();
            let op = &built0.block.ops[upto - 1];
            let verdict = match try_prefix(upto) {
                Ok(true) => "ok".to_string(),
                Ok(false) => "skip".to_string(),
                Err(e) => format!("FAIL {}", e.chars().take(40).collect::<String>()),
            };
            eprintln!(
                "LINEAR prefix {upto:>3} ({} -> {}): {verdict}",
                op.ty, op.outputs[0].name
            );
        }
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

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

fn tiny_opts(seq: i64) -> Options {
    Options {
        seq,
        max_kv: 16,
        quant: Quant::Int8,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..Options::default()
    }
}

/// Temp root that self-cleans unless `MIL_KEEP_ARTIFACTS=1`.
struct Root(std::path::PathBuf);
impl Root {
    fn new(tag: &str) -> Root {
        let p = std::env::temp_dir().join(format!("enum_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Root(p)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn load(pkg: &Path, work: &Path, units: mil_infer::ComputeUnits) -> mil_infer::Model {
    let c = mil_compile::compile(pkg, work).unwrap();
    mil_infer::Model::load(&c.path, units).unwrap()
}

/// `--seq-range 1..16`: predict at lengths that are in no enumerated
/// list (3, 7, 16) and compare with fixed-shape packages converted at
/// exactly those lengths. Out-of-range s=17 must error.
#[test]
#[ignore = "e2e: needs coremlc"]
fn range_shapes_match_fixed_packages() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = Root::new("range");
    let model = root.0.join("model");
    tiny_qwen3(&model);

    let range_pkg = root.0.join("range.mlpackage");
    convert(
        &model,
        &range_pkg,
        &Options {
            seq_range: Some((1, 16)),
            ..tiny_opts(1)
        },
    )
    .unwrap();
    let mr = load(
        &range_pkg,
        &root.0.join("c_range"),
        mil_infer::ComputeUnits::All,
    );

    let mut worst = 0f32;
    for s in [3i64, 7, 16] {
        let fixed_pkg = root.0.join(format!("fixed{s}.mlpackage"));
        convert(&model, &fixed_pkg, &tiny_opts(s)).unwrap();
        let mf = load(
            &fixed_pkg,
            &root.0.join(format!("c{s}")),
            mil_infer::ComputeUnits::All,
        );
        let want = predict(&mf, &inputs_at(s, 16, false));
        let got = predict(&mr, &inputs_at(s, 16, true));
        assert_eq!(got.len(), want.len());
        let max_d = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        println!("seq={s}: range vs fixed max|Δ| = {max_d}");
        worst = worst.max(max_d);
        assert!(got.iter().all(|v| v.is_finite()));
    }
    println!("range vs fixed worst max|Δ| = {worst}");
    assert_eq!(worst, 0.0, "range output differs from fixed packages");

    // default (lower bound) also runs
    let g1 = predict(&mr, &inputs_at(1, 16, true));
    assert_eq!(g1.len(), 64);

    // CPU-only loading of flexible-seq packages (informational)
    for (n, cu) in [
        ("cpu", mil_infer::ComputeUnits::CpuOnly),
        ("cpu+gpu", mil_infer::ComputeUnits::CpuAndGpu),
    ] {
        let c = mil_compile::compile(&range_pkg, &root.0.join(format!("c_{n}"))).unwrap();
        match mil_infer::Model::load(&c.path, cu) {
            Ok(m) => {
                let got = predict(&m, &inputs_at(7, 16, true));
                println!("tiny range pkg on {n}: ok, {} values", got.len());
            }
            Err(e) => println!("tiny range pkg on {n}: load failed: {e}"),
        }
    }

    // out of range: an error, not garbage
    let st = mr.new_state().unwrap();
    let r = mr.predict_with_state(Some(&st), &inputs_at(17, 16, true));
    assert!(r.is_err(), "s=17 is outside 1..16 and must be rejected");
    println!("s=17 error: {}", r.err().unwrap());
}

#[test]
fn seq_range_option_validation() {
    let base = |r: Option<(i64, i64)>, lens: Vec<i64>| Options {
        seq_range: r,
        seq_lens: lens,
        max_kv: 16,
        ..Options::default()
    };
    assert!(base(Some((1, 16)), vec![]).check_seq_flex().is_ok());
    assert!(base(Some((1, 16)), vec![4]).check_seq_flex().is_ok());
    assert!(base(None, vec![1, 4]).check_seq_flex().is_ok());
    let e = base(Some((1, 16)), vec![1, 4])
        .check_seq_flex()
        .unwrap_err();
    assert!(e.contains("mutually exclusive"), "{e}");
    let e = base(Some((1, 17)), vec![]).check_seq_flex().unwrap_err();
    assert!(e.contains("exceeds --max-kv"), "{e}");
    let e = base(Some((1, -1)), vec![]).check_seq_flex().unwrap_err();
    assert!(e.contains("unbounded"), "{e}");
    let e = base(Some((0, 8)), vec![]).check_seq_flex().unwrap_err();
    assert!(e.contains(">= 1"), "{e}");
    let e = base(Some((9, 8)), vec![]).check_seq_flex().unwrap_err();
    assert!(e.contains("below"), "{e}");
}

#[test]
fn convert_rejects_bad_seq_range_before_touching_disk() {
    let root = Root::new("range_err");
    let model = root.0.join("model");
    tiny_qwen3(&model);
    let out = root.0.join("o.mlpackage");
    let e = convert(
        &model,
        &out,
        &Options {
            seq_range: Some((1, 99)),
            ..tiny_opts(1)
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("--max-kv"), "{e}");
    assert!(!out.exists());
    let e = convert(
        &model,
        &out,
        &Options {
            seq_range: Some((1, 8)),
            seq_lens: vec![1, 4],
            ..tiny_opts(1)
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("mutually exclusive"), "{e}");
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// enumerated → range → enumerated via `reshape_flex`: each rewritten
/// package compiles and predicts at a length its source didn't allow.
#[test]
#[ignore = "e2e: needs coremlc"]
fn reshape_enum_range_round_trip() {
    use mil_convert::surgery::{reshape_flex, SeqFlex};
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = Root::new("reshape_rt");
    let model = root.0.join("model");
    tiny_qwen3(&model);

    // enumerated {1,4} → range 1..16, predict at s=7
    let enum_pkg = root.0.join("enum.mlpackage");
    convert(
        &model,
        &enum_pkg,
        &Options {
            seq_lens: vec![1, 4],
            ..tiny_opts(1)
        },
    )
    .unwrap();
    let ranged = root.0.join("ranged.mlpackage");
    let r = reshape_flex(&enum_pkg, &SeqFlex::Range(1, 16), Some(&ranged)).unwrap();
    println!("{}", r.lines.join("\n"));
    let fixed7 = root.0.join("fixed7.mlpackage");
    convert(&model, &fixed7, &tiny_opts(7)).unwrap();
    let want = predict(
        &load(&fixed7, &root.0.join("c7"), mil_infer::ComputeUnits::All),
        &inputs_at(7, 16, false),
    );
    let mr = load(&ranged, &root.0.join("cr"), mil_infer::ComputeUnits::All);
    let got = predict(&mr, &inputs_at(7, 16, true));
    let d = max_abs_diff(&got, &want);
    println!("enum→range, s=7: max|Δ| vs fixed = {d}");
    assert_eq!(d, 0.0);

    // range → enumerated {3, 11}: 11 is newly allowed, 5 is no longer
    let back = root.0.join("back.mlpackage");
    let r = reshape_flex(&ranged, &SeqFlex::Lens(vec![3, 11]), Some(&back)).unwrap();
    println!("{}", r.lines.join("\n"));
    let fixed11 = root.0.join("fixed11.mlpackage");
    convert(&model, &fixed11, &tiny_opts(11)).unwrap();
    let want11 = predict(
        &load(&fixed11, &root.0.join("c11"), mil_infer::ComputeUnits::All),
        &inputs_at(11, 16, false),
    );
    let mb = load(&back, &root.0.join("cb"), mil_infer::ComputeUnits::All);
    let got11 = predict(&mb, &inputs_at(11, 16, true));
    let d = max_abs_diff(&got11, &want11);
    println!("range→enum, s=11: max|Δ| vs fixed = {d}");
    assert_eq!(d, 0.0);
    let st = mb.new_state().unwrap();
    assert!(
        mb.predict_with_state(Some(&st), &inputs_at(5, 16, true))
            .is_err(),
        "s=5 is not in the new enumerated set {{3,11}}"
    );

    // a single-sequence flexible rewrite of a fixed-shape package is refused
    let e = reshape_flex(&fixed7, &SeqFlex::Range(1, 8), None)
        .err()
        .map(|e| e.to_string())
        .unwrap();
    assert!(e.contains("no sequence flexibility"), "{e}");
}

// ---------------- CpuOnly diagnostic matrix ----------------

fn free_gb() -> String {
    let o = std::process::Command::new("df")
        .args(["-h", "/"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .nth(1)
        .and_then(|l| l.split_whitespace().nth(3).map(String::from))
        .unwrap_or_default()
}

/// Try loading (and predicting at `s`) a compiled bundle on CpuOnly and
/// All; returns a one-line verdict.
fn verdict(pkg: &Path, work: &Path, s: i64, with_seq: bool) -> String {
    let c = match mil_compile::compile(pkg, work) {
        Ok(c) => c,
        Err(e) => {
            return format!(
                "coremlc REJECTS: {}",
                e.to_string().lines().next().unwrap_or("")
            )
        }
    };
    let mut parts = Vec::new();
    for (n, cu) in [
        ("cpu", mil_infer::ComputeUnits::CpuOnly),
        ("all", mil_infer::ComputeUnits::All),
    ] {
        parts.push(match mil_infer::Model::load(&c.path, cu) {
            Err(e) => format!(
                "{n}: LOAD FAIL ({})",
                e.message.rsplit("error code").next().unwrap_or("").trim()
            ),
            Ok(m) => {
                let st = m.new_state().ok();
                match m.predict_with_state(st.as_ref(), &inputs_at(s, 16, with_seq)) {
                    Ok(_) => format!("{n}: load+predict ok"),
                    Err(e) => format!("{n}: loads, PREDICT FAIL ({e})"),
                }
            }
        });
    }
    parts.join(" | ")
}

/// Evidence matrix for "flexible packages fail to load on CpuOnly
/// (-14)". Prints `DIAG` lines; asserts nothing.
#[test]
#[ignore = "diagnostic: needs coremlc; prints a matrix"]
fn cpu_only_flex_matrix() {
    use mil_convert::builder::{self, WeightEmitter};
    use mil_convert::safetensors;
    use mil_convert::{plan, ModelConfig};
    use mil_spec::{Flex, ShapeRange};
    use std::collections::BTreeMap;
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = Root::new("diag");
    let model = root.0.join("model");
    tiny_qwen3(&model);
    let cfg = ModelConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let say = |what: &str, v: &str| println!("DIAG [free {}] {what}: {v}", free_gb());

    // (a)-(c): full converter packages
    for (tag, o, s, seq) in [
        ("a fixed --seq 4", tiny_opts(4), 4, false),
        (
            "b --seq-lens 1,4",
            Options {
                seq_lens: vec![1, 4],
                ..tiny_opts(1)
            },
            4,
            true,
        ),
        (
            "c --seq-range 1..4",
            Options {
                seq_range: Some((1, 4)),
                ..tiny_opts(1)
            },
            4,
            true,
        ),
    ] {
        let pkg = root.0.join(format!("{}.mlpackage", &tag[..1]));
        convert(&model, &pkg, &o).unwrap();
        let v = verdict(&pkg, &root.0.join(format!("c_{}", &tag[..1])), s, seq);
        say(tag, &v);
    }

    // (d): hybrids built from the real builder output
    let build = |o: &Options| -> builder::Built {
        let src = safetensors::open_dir(&model).unwrap();
        let plan = plan::Plan::uniform(o.quant);
        let tmp_w = root.0.join("w.bin");
        let mut writer = mil_spec::BlobWriter::create(&tmp_w).unwrap();
        let mut em = WeightEmitter {
            w: &mut writer,
            quant: o.quant,
            file: "@model_path/weights/weight.bin".into(),
            src: &src,
            plan: &plan,
        };
        let b = builder::build(&cfg, o, &mut em).unwrap();
        writer.finish().unwrap();
        b
    };
    let emit = |tag: &str,
                b: &builder::Built,
                flex: &BTreeMap<String, Flex>,
                syms: &BTreeMap<String, Vec<Option<String>>>|
     -> std::path::PathBuf {
        let spec = mil_spec::encode_model_flexible(
            &b.inputs,
            &b.outputs,
            &b.states,
            &b.block,
            &b.fn_inputs,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
            flex,
            syms,
        )
        .unwrap();
        let pkg = root.0.join(format!("{tag}.mlpackage"));
        mil_spec::write_mlpackage_stream(&pkg, &spec, Some(&root.0.join("w.bin"))).unwrap();
        pkg
    };
    let fixed4 = build(&tiny_opts(4));
    let flex_o = Options {
        seq_lens: vec![1, 4],
        ..tiny_opts(1)
    };
    let flexb = build(&flex_o);
    let none: BTreeMap<String, Vec<Option<String>>> = BTreeMap::new();
    let nflex: BTreeMap<String, Flex> = BTreeMap::new();
    let flex_map = flexb.flex.clone().unwrap();

    // d1: flex feature descriptions, static program (no `seq` input)
    let mut m1 = flex_map.clone();
    m1.retain(|k, _| k != "seq");
    let p = emit("d1", &fixed4, &m1, &none);
    say(
        "d1 flex descriptions + STATIC program",
        &verdict(&p, &root.0.join("c_d1"), 4, false),
    );
    // d2: flex program (symbolic dims, runtime seq, dyn reshapes) with
    // STATIC feature descriptions
    let p = emit("d2", &flexb, &nflex, &flexb.syms);
    say(
        "d2 static descriptions + FLEX program",
        &verdict(&p, &root.0.join("c_d2"), 1, true),
    );
    // d3: flex program, symbolic dims dropped from the types
    let p = emit("d3", &flexb, &flex_map, &none);
    say(
        "d3 flex descriptions + flex program w/o symbolic dim marks",
        &verdict(&p, &root.0.join("c_d3"), 4, true),
    );
    // d4: full flex (control = row b)
    let p = emit("d4", &flexb, &flex_map, &flexb.syms);
    say(
        "d4 flex descriptions + FLEX program (= b)",
        &verdict(&p, &root.0.join("c_d4"), 4, true),
    );

    // (e): minimal hand-built flexible model: y = add(x, x)
    let tiny = |tag: &str, flex: Option<Flex>, sym: bool| {
        let mut b = mil_spec::Block::new();
        let y = b.add("x", "x", &[1, 4, 2, 1], "y");
        b.outputs = vec![y];
        let fin = vec![mil_spec::NVT {
            name: "x".into(),
            ty: mil_spec::ValueType::Tensor(mil_spec::TensorType::f16(&[1, 4, 2, 1])),
        }];
        let f = |n: &str| mil_spec::Feature {
            name: n.into(),
            shape: vec![1, 4, 2, 1],
            dtype: mil_spec::DType::Fp16,
            is_state: false,
        };
        let mut fl = BTreeMap::new();
        let mut sy = BTreeMap::new();
        if let Some(fx) = flex {
            fl.insert("x".to_string(), fx.clone());
            fl.insert("y".to_string(), fx);
        }
        if sym {
            for n in ["x", "y"] {
                sy.insert(n.to_string(), vec![None, None, Some("s".to_string()), None]);
            }
        }
        let spec = mil_spec::encode_model_flexible(
            &[f("x")],
            &[f("y")],
            &[],
            &b,
            &fin,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
            &fl,
            &sy,
        )
        .unwrap();
        let pkg = root.0.join(format!("{tag}.mlpackage"));
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
        let c = match mil_compile::compile(&pkg, &root.0.join(format!("c_{tag}"))) {
            Ok(c) => c,
            Err(e) => {
                return format!(
                    "coremlc REJECTS: {}",
                    e.to_string().lines().next().unwrap_or("")
                )
            }
        };
        let x: Vec<u8> = (0..4 * 2)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();
        let sh: [i64; 4] = [1, 4, 2, 1];
        let mut out = Vec::new();
        for (n, cu) in [
            ("cpu", mil_infer::ComputeUnits::CpuOnly),
            ("all", mil_infer::ComputeUnits::All),
        ] {
            out.push(match mil_infer::Model::load(&c.path, cu) {
                Err(e) => format!(
                    "{n}: LOAD FAIL ({})",
                    e.message.rsplit("error code").next().unwrap_or("").trim()
                ),
                Ok(m) => match m.predict(&[mil_infer::Input {
                    name: "x",
                    shape: &sh,
                    data: &x,
                    dtype: mil_spec::DType::Fp16,
                }]) {
                    Ok(_) => format!("{n}: load+predict ok"),
                    Err(e) => format!("{n}: loads, PREDICT FAIL ({e})"),
                },
            });
        }
        out.join(" | ")
    };
    let es = Flex::Enumerated(mil_spec::EnumeratedShapes {
        shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 5, 1]],
    });
    let rg = Flex::Range(ShapeRange {
        dims: vec![(1, 1), (4, 4), (1, 8), (1, 1)],
    });
    say("e0 1-op add, fixed (control)", &tiny("e0", None, false));
    say(
        "e1 1-op add, enumerated, no symbolic dims",
        &tiny("e1", Some(es.clone()), false),
    );
    say(
        "e2 1-op add, enumerated + symbolic dim",
        &tiny("e2", Some(es), true),
    );
    say(
        "e3 1-op add, range, no symbolic dims",
        &tiny("e3", Some(rg.clone()), false),
    );
    say(
        "e4 1-op add, range + symbolic dim",
        &tiny("e4", Some(rg), true),
    );

    // (e5/e6): minimal reshape variants on a seq-flexible input
    //   x (1,s,8,1) -> reshape -> (1,s,2,4), s in {2,5}
    let resh = |tag: &str, dynamic: bool, s0: i64| -> String {
        let mut b = mil_spec::Block::new();
        let sh = if dynamic {
            let p0 = b.konst_i32("p0", &[1]);
            let p2 = b.konst_i32("p2", &[2]);
            let p3 = b.konst_i32("p3", &[4]);
            let ax = b.konst_scalar_i32("ax", 0);
            let il = b.konst_bool("il", false);
            b.o1(
                "concat",
                vec![
                    (
                        "values".into(),
                        mil_spec::bind_many(&[&p0, "seq", &p2, &p3]),
                    ),
                    ("axis".into(), mil_spec::bind(&ax).1),
                    ("interleave".into(), mil_spec::bind(&il).1),
                ],
                "shp",
                mil_spec::ValueType::Tensor(mil_spec::TensorType {
                    dtype: mil_spec::DType::Int32,
                    shape: vec![4],
                }),
            )
        } else {
            b.konst_i32("shp", &[1, -1, 2, 4])
        };
        let y = b.o1(
            "reshape",
            vec![
                ("x".into(), mil_spec::bind("x").1),
                ("shape".into(), mil_spec::bind(&sh).1),
            ],
            "y",
            mil_spec::ValueType::Tensor(mil_spec::TensorType::f16(&[1, s0, 2, 4])),
        );
        b.outputs = vec![y];
        let mut fin = vec![mil_spec::NVT {
            name: "x".into(),
            ty: mil_spec::ValueType::Tensor(mil_spec::TensorType::f16(&[1, s0, 8, 1])),
        }];
        let feat = |n: &str, shape: &[i64], dt| mil_spec::Feature {
            name: n.into(),
            shape: shape.to_vec(),
            dtype: dt,
            is_state: false,
        };
        let mut ins = vec![feat("x", &[1, s0, 8, 1], mil_spec::DType::Fp16)];
        if dynamic {
            fin.push(mil_spec::NVT {
                name: "seq".into(),
                ty: mil_spec::ValueType::Tensor(mil_spec::TensorType {
                    dtype: mil_spec::DType::Int32,
                    shape: vec![1],
                }),
            });
            ins.push(feat("seq", &[1], mil_spec::DType::Int32));
        }
        let mut fl = BTreeMap::new();
        fl.insert(
            "x".to_string(),
            Flex::Enumerated(mil_spec::EnumeratedShapes {
                shapes: vec![vec![1, s0, 8, 1], vec![1, 5, 8, 1]],
            }),
        );
        fl.insert(
            "y".to_string(),
            Flex::Enumerated(mil_spec::EnumeratedShapes {
                shapes: vec![vec![1, s0, 2, 4], vec![1, 5, 2, 4]],
            }),
        );
        let mut sy = BTreeMap::new();
        sy.insert(
            "x".to_string(),
            vec![None, Some("s".to_string()), None, None],
        );
        sy.insert(
            "y".to_string(),
            vec![None, Some("s".to_string()), None, None],
        );
        let spec = mil_spec::encode_model_flexible(
            &ins,
            &[feat("y", &[1, s0, 2, 4], mil_spec::DType::Fp16)],
            &[],
            &b,
            &fin,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
            &fl,
            &sy,
        )
        .unwrap();
        let pkg = root.0.join(format!("{tag}.mlpackage"));
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
        let c = match mil_compile::compile(&pkg, &root.0.join(format!("c_{tag}"))) {
            Ok(c) => c,
            Err(e) => {
                return format!(
                    "coremlc REJECTS: {}",
                    e.to_string().lines().next().unwrap_or("")
                )
            }
        };
        let mut out = Vec::new();
        for (n, cu) in [
            ("cpu", mil_infer::ComputeUnits::CpuOnly),
            ("all", mil_infer::ComputeUnits::All),
        ] {
            out.push(match mil_infer::Model::load(&c.path, cu) {
                Err(e) => format!(
                    "{n}: LOAD FAIL ({})",
                    e.message.rsplit("error code").next().unwrap_or("").trim()
                ),
                Ok(m) => {
                    let mut res = Vec::new();
                    for s in [s0, 5] {
                        let x: Vec<u8> = (0..8 * s)
                            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
                            .collect();
                        let shp = [1, s, 8, 1];
                        let sv = (s as i32).to_le_bytes();
                        let shs = [1i64];
                        let mut inp = vec![mil_infer::Input {
                            name: "x",
                            shape: &shp,
                            data: &x,
                            dtype: mil_spec::DType::Fp16,
                        }];
                        if dynamic {
                            inp.push(mil_infer::Input {
                                name: "seq",
                                shape: &shs,
                                data: &sv,
                                dtype: mil_spec::DType::Int32,
                            });
                        }
                        res.push(match m.predict(&inp) {
                            Ok(p) => format!("s={s} ok(out {:?})", p.outputs[0].shape),
                            Err(e) => format!("s={s} PREDICT FAIL ({e})"),
                        });
                    }
                    format!("{n}: loads, {}", res.join(", "))
                }
            });
        }
        out.join(" | ")
    };
    say(
        "e5 reshape to runtime concat([1],seq,[2],[4]), s0=2",
        &resh("e5", true, 2),
    );
    say("e5b same, s0=1", &resh("e5b", true, 1));
    say(
        "e6 reshape to const [1,-1,2,4], s0=2",
        &resh("e6", false, 2),
    );
    say("e6b same, s0=1", &resh("e6b", false, 1));
}
