//! LoRA bake-in end-to-end: synthetic adapters over a tiny Qwen3
//! checkpoint prove load → validate → fuse → emit → compile → predict,
//! and that the fused weights land *before* fp16/int8 emission. The
//! real `dream` adapter is exercised when present.

use mil_convert::lora::{FusedSource, Lora};
use mil_convert::{convert, Options, Quant, WeightSource};
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------- fixture writers ----------

/// Write a `.safetensors` with F32 tensors (adapter files ship f32).
fn write_safetensors_f32(path: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
    let mut header = String::from("{");
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape, vals) in tensors {
        let off = data.len() as u64;
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{off},{}]}}",
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

fn write_safetensors_f16(path: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
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

/// Minimal `.npy` v1.0 f32 blob.
fn npy_f32(shape: &[i64], data: &[f32]) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}{}), }}",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        if shape.len() == 1 { "," } else { "" }
    );
    let pad = (64 - ((10 + h.len() + 1) % 64)) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Minimal stored-method ZIP → `.npz` (no compression).
fn npz_stored(members: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cds: Vec<(String, u32, usize)> = Vec::new();
    for (name, data) in members {
        let off = out.len() as u32;
        let n = name.as_bytes();
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags+method(stored)
        out.extend_from_slice(&0u32.to_le_bytes()); // time/date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(n);
        out.extend_from_slice(data);
        cds.push((name.to_string(), off, data.len()));
    }
    let cd = out.len();
    for (name, off, len) in &cds {
        let n = name.as_bytes();
        out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags+method
        out.extend_from_slice(&0u32.to_le_bytes()); // time/date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc
        out.extend_from_slice(&(*len as u32).to_le_bytes());
        out.extend_from_slice(&(*len as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // extra+comment+disk+int_attr
        out.extend_from_slice(&0u32.to_le_bytes()); // ext attr
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(n);
    }
    let cdlen = out.len() - cd;
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(cds.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cds.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cdlen as u32).to_le_bytes());
    out.extend_from_slice(&(cd as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

/// Same tiny Qwen3 as `pipeline.rs` — d=16, 2 layers, vocab 64.
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
    write_safetensors_f16(&dir.join("model.safetensors"), &refs);
}

/// Deterministic MLX-layout adapter for `tiny_qwen3`: rank 2,
/// `a` `[in, r]`, `b` `[r, out]`, on two projections. Small values keep
/// the delta well under base magnitude (cosine > 0.999) yet far above
/// fp16 noise.
fn write_adapter(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 2, "scale": 0.5, "dropout": 0}}"#,
    )
    .unwrap();
    let mk = |rows: i64, cols: i64, phase: f32| -> Vec<f32> {
        (0..rows * cols)
            .map(|i| phase * 0.01 + (i as f32 % 5.0) * 0.002)
            .collect()
    };
    let tensors: Vec<(String, Vec<i64>, Vec<f32>)> = vec![
        // q_proj W is [out=16, in=16] → a [16,2], b [2,16]
        (
            "model.layers.0.self_attn.q_proj.lora_a".into(),
            vec![16, 2],
            mk(16, 2, 1.0),
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_b".into(),
            vec![2, 16],
            mk(2, 16, 2.0),
        ),
        // down_proj W is [out=16, in=32] → a [32,2], b [2,16]
        (
            "model.layers.1.mlp.down_proj.lora_a".into(),
            vec![32, 2],
            mk(32, 2, 3.0),
        ),
        (
            "model.layers.1.mlp.down_proj.lora_b".into(),
            vec![2, 16],
            mk(2, 16, 4.0),
        ),
    ];
    let refs: Vec<(&str, Vec<i64>, Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
        .collect();
    write_safetensors_f32(&dir.join("adapters.safetensors"), &refs);
}

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lora_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn opts(quant: Quant, lora: Option<PathBuf>) -> Options {
    Options {
        seq: 1,
        max_kv: 16,
        quant,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ---------- adapter loading ----------

#[test]
fn adapter_loads_safetensors() {
    let root = tmp("load_st");
    let adir = root.join("adapter");
    write_adapter(&adir);
    let l = Lora::load(&adir).unwrap();
    assert_eq!(l.scale, 0.5);
    let mut targets = l.targets();
    targets.sort();
    assert_eq!(
        targets,
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn adapter_loads_npz() {
    let root = tmp("load_npz");
    let adir = root.join("adapter");
    std::fs::create_dir_all(&adir).unwrap();
    std::fs::write(
        adir.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 2, "scale": 2.0}}"#,
    )
    .unwrap();
    // q_proj W [16,16]: a [16,2], b [2,16] — MLX layout inside npz.
    let a: Vec<f32> = (0..32).map(|i| 0.001 * (i as f32 + 1.0)).collect();
    let b: Vec<f32> = (0..32).map(|i| 0.002 * (i as f32 + 1.0)).collect();
    let npz = npz_stored(&[
        (
            "model.layers.0.self_attn.q_proj.lora_a.npy",
            npy_f32(&[16, 2], &a),
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_b.npy",
            npy_f32(&[2, 16], &b),
        ),
    ]);
    std::fs::write(adir.join("adapters.npz"), npz).unwrap();
    let l = Lora::load(&adir).unwrap();
    assert_eq!(l.scale, 2.0);
    assert_eq!(l.targets(), vec!["model.layers.0.self_attn.q_proj.weight"]);
    let _ = std::fs::remove_dir_all(&root);
}

// ---------- validation failures ----------

#[test]
fn missing_target_is_a_clear_error() {
    let root = tmp("missing");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model);
    std::fs::create_dir_all(&adir).unwrap();
    std::fs::write(
        adir.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 2, "scale": 1.0}}"#,
    )
    .unwrap();
    let t: Vec<(&str, Vec<i64>, Vec<f32>)> = vec![
        (
            "model.layers.9.self_attn.q_proj.lora_a",
            vec![16, 2],
            vec![0.0; 32],
        ),
        (
            "model.layers.9.self_attn.q_proj.lora_b",
            vec![2, 16],
            vec![0.0; 32],
        ),
    ];
    write_safetensors_f32(&adir.join("adapters.safetensors"), &t);
    let e = convert(
        &model,
        &root.join("x.mlpackage"),
        &opts(Quant::Fp16, Some(adir)),
    )
    .unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("model.layers.9.self_attn.q_proj.weight"),
        "expected missing-target error, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn geometry_mismatch_is_a_clear_error() {
    let root = tmp("geom");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model);
    std::fs::create_dir_all(&adir).unwrap();
    std::fs::write(
        adir.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 4, "scale": 1.0}}"#,
    )
    .unwrap();
    // q_proj W is [16,16]; a [16,4] b [3,16] — inner dims 4≠3 can't fuse.
    let t: Vec<(&str, Vec<i64>, Vec<f32>)> = vec![
        (
            "model.layers.0.self_attn.q_proj.lora_a",
            vec![16, 4],
            vec![0.0; 64],
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_b",
            vec![3, 16],
            vec![0.0; 48],
        ),
    ];
    write_safetensors_f32(&adir.join("adapters.safetensors"), &t);
    let e = convert(
        &model,
        &root.join("x.mlpackage"),
        &opts(Quant::Fp16, Some(adir)),
    )
    .unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("cannot fuse"),
        "expected geometry error, got: {msg}"
    );
    // and it must fail before the package is produced
    assert!(!root.join("x.mlpackage").exists());
    let _ = std::fs::remove_dir_all(&root);
}

// ---------- per-tensor fused-value proof on real checkpoint weights ----------

#[test]
fn fused_values_match_hand_computed() {
    let root = tmp("pertensor");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model);
    write_adapter(&adir);
    let l = Lora::load(&adir).unwrap();
    let src: Vec<mil_convert::safetensors::Safetensors> =
        mil_convert::safetensors::open_dir(&model).unwrap();
    l.validate(&src).unwrap();
    let fused = FusedSource::new(&src, &l);

    for wname in l.targets() {
        let (shape, base_f32) = src.tensor_f32(wname).unwrap();
        let (_, got16) = fused.tensor_f16(wname).unwrap();
        let pair = l.pair(wname).unwrap();
        let (out, in_) = (shape[0] as usize, shape[1] as usize);
        let r = pair.a_shape[1] as usize; // MLX: a [in, r]
        for o in 0..out {
            for i in 0..in_ {
                let mut delta = 0f32;
                for k in 0..r {
                    delta += pair.b[k * out + o] * pair.a[i * r + k];
                }
                let expect = base_f32[o * in_ + i] + l.scale * delta;
                let got = half::f16::from_le_bytes([
                    got16[(o * in_ + i) * 2],
                    got16[(o * in_ + i) * 2 + 1],
                ])
                .to_f32();
                assert!(
                    (got - expect).abs() < 1e-3 + expect.abs() * 1e-3,
                    "{wname}[{o},{i}]: got {got}, expect {expect}"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

// ---------- full pipeline: fused logits vs base logits ----------

/// Run one decode step on a compiled tiny model; returns 64 logits.
fn predict_logits(compiled: &Path) -> Vec<f32> {
    use mil_infer::{ComputeUnits, Input, Model};
    let m = Model::load(compiled, ComputeUnits::All).unwrap();
    let state = m.new_state().expect("stateful model");
    let f16s = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    };
    let x: Vec<f32> = (0..16).map(|i| 0.05 * (i as f32 + 1.0)).collect();
    let xb = f16s(&x);
    let cos = f16s(&[1.0; 8]);
    let sin = f16s(&[0.0; 8]);
    let mk = f16s(&[0.0; 16]); // pos 0: everything visible
    let pos: i32 = 0;
    let p = m
        .predict_with_state(
            Some(&state),
            &[
                Input {
                    name: "x",
                    shape: &[1, 16, 1, 1],
                    data: &xb,
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
                    data: &mk,
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
    p.outputs[0].values()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// The proof: same graph, same inputs, adapter on → logits shift by
/// exactly the small fused delta — top-1 unchanged, cosine ≈ 1, but
/// provably different from the unfused run.
fn fused_logits_case(tag: &str, quant: Quant) {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp(tag);
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model);
    write_adapter(&adir);

    let pkg_base = root.join("base.mlpackage");
    let pkg_lora = root.join("lora.mlpackage");
    convert(&model, &pkg_base, &opts(quant, None)).unwrap();
    convert(&model, &pkg_lora, &opts(quant, Some(adir))).unwrap();

    let c_base = mil_compile::compile(&pkg_base, &root.join("base.c")).unwrap();
    let c_lora = mil_compile::compile(&pkg_lora, &root.join("lora.c")).unwrap();
    let l_base = predict_logits(&c_base.path);
    let l_lora = predict_logits(&c_lora.path);

    // fusion provably reached the emitted weights
    assert_ne!(
        l_base, l_lora,
        "adapter had no effect — fused weights did not reach emission"
    );
    // …but by only the small low-rank delta
    let cos = cosine(&l_base, &l_lora);
    assert!(
        cos > 0.999,
        "fused logits diverged too far from base: cosine {cos}"
    );
    assert_eq!(
        argmax(&l_base),
        argmax(&l_lora),
        "top-1 token changed under a small adapter delta"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fused_logits_fp16() {
    fused_logits_case("fp16", Quant::Fp16);
}

/// Same proof through the int8 path — fused values must be what
/// `quantize_int8` saw (i.e. fusion happens *before* quant emission).
#[test]
fn fused_logits_int8() {
    fused_logits_case("int8", Quant::Int8);
}

// ---------- synthetic adapter on the real Qwen3-0.6B ----------

/// A synthetic rank-2 MLX adapter over real 0.6B weights: fuse →
/// compile → same hidden-state input → logits cosine > 0.999, top-1
/// identical, provably changed. This is the full-scale version of
/// `fused_logits_fp16`.
#[test]
#[ignore = "needs ~/.cache/mil_gguf_test/models/qwen3-hf + coremlc"]
fn qwen3_0p6b_lora_fused_logits() {
    let hf =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models/qwen3-hf");
    if !hf.exists() || mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp("qwen3");
    let adir = root.join("adapter");
    std::fs::create_dir_all(&adir).unwrap();
    std::fs::write(
        adir.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 2, "scale": 0.05}}"#,
    )
    .unwrap();
    // Qwen3-0.6B: q_proj W [2048,1024]; down_proj W [1024,3072].
    // a [in, r], b [r, out] (MLX layout).
    let mk = |n: usize, phase: f32| -> Vec<f32> {
        (0..n)
            .map(|i| phase * 1e-3 + ((i * 7919) % 13) as f32 * 2e-4)
            .collect()
    };
    let tensors: Vec<(String, Vec<i64>, Vec<f32>)> = vec![
        (
            "model.layers.0.self_attn.q_proj.lora_a".into(),
            vec![1024, 2],
            mk(2048, 1.0),
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_b".into(),
            vec![2, 2048],
            mk(4096, 2.0),
        ),
        (
            "model.layers.27.mlp.down_proj.lora_a".into(),
            vec![3072, 2],
            mk(6144, 3.0),
        ),
        (
            "model.layers.27.mlp.down_proj.lora_b".into(),
            vec![2, 1024],
            mk(2048, 4.0),
        ),
    ];
    let refs: Vec<(&str, Vec<i64>, Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
        .collect();
    write_safetensors_f32(&adir.join("adapters.safetensors"), &refs);

    let pkg_base = root.join("base.mlpackage");
    let pkg_lora = root.join("lora.mlpackage");
    convert(&hf, &pkg_base, &opts(Quant::Fp16, None)).unwrap();
    convert(&hf, &pkg_lora, &opts(Quant::Fp16, Some(adir))).unwrap();
    let c_base = mil_compile::compile(&pkg_base, &root.join("base.c")).unwrap();
    let c_lora = mil_compile::compile(&pkg_lora, &root.join("lora.c")).unwrap();

    // Feed a deterministic hidden state through pos 0 of each.
    use mil_infer::{ComputeUnits, Input, Model};
    let run = |path: &Path| -> Vec<f32> {
        let m = Model::load(path, ComputeUnits::All).unwrap();
        let state = m.new_state().unwrap();
        let f16s = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let x: Vec<f32> = (0..1024)
            .map(|i| ((i * 31) % 17) as f32 * 0.01 - 0.08)
            .collect();
        let xb = f16s(&x);
        let cos = f16s(&[1.0; 128]);
        let sin = f16s(&[0.0; 128]);
        let mk_ = f16s(&[0.0; 16]);
        let pos: i32 = 0;
        m.predict_with_state(
            Some(&state),
            &[
                Input {
                    name: "x",
                    shape: &[1, 1024, 1, 1],
                    data: &xb,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "cos",
                    shape: &[1, 1, 1, 128],
                    data: &cos,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "sin",
                    shape: &[1, 1, 1, 128],
                    data: &sin,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "mask",
                    shape: &[1, 1, 1, 16],
                    data: &mk_,
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
        .unwrap()
        .outputs[0]
            .values()
    };
    let l_base = run(&c_base.path);
    let l_lora = run(&c_lora.path);
    assert_ne!(l_base, l_lora, "fused weights never reached emission");
    let cos = cosine(&l_base, &l_lora);
    assert!(cos > 0.999, "cosine {cos}");
    assert_eq!(argmax(&l_base), argmax(&l_lora));
    println!("0.6B lora cosine: {cos:.6}");
    let _ = std::fs::remove_dir_all(&root);
}

// ---------- the real dream adapter ----------

/// Load the trained MLX adapter: pairing, scale, and the clear
/// missing-target error it produces against the wrong-size checkpoint.
#[test]
#[ignore = "needs /var/lib/bad_apple adapter + ~/.cache/mil_gguf_test models"]
fn dream_adapter_loads_and_rejects_wrong_checkpoint() {
    let adir = PathBuf::from("/var/lib/bad_apple/lora_adapters/dream-qwen3-8b");
    if !adir.exists() {
        return;
    }
    let l = Lora::load(&adir).unwrap();
    assert_eq!(l.scale, 10.0);
    assert_eq!(l.targets().len(), 16 * 7, "16 layers × 7 projections");

    // Against qwen3-0.6B (28 layers, hidden 1024): layers 20–27 exist
    // but at 8B geometry (hidden 4096) so the first failure is a
    // geometry mismatch; layers 28+ are missing weights outright.
    // Either way the error must name the offending target.
    let hf =
        PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models/qwen3-hf");
    if hf.exists() {
        let sts = mil_convert::safetensors::open_dir(&hf).unwrap();
        let e = l.validate(&sts).unwrap_err();
        assert!(e.to_string().contains("model.layers.2"), "{e}");
        let r = convert(
            &hf,
            &std::env::temp_dir().join("dream_x.mlpackage"),
            &opts(Quant::Fp16, Some(adir)),
        );
        assert!(r.is_err());
    }
}
