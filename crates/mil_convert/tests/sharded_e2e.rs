//! `sharded_e2e` — `.milshards` bundles on the real runtime.
//!
//! The ANE execution-plan builder rejects monolithic programs past
//! ~3.2k ops (error -14, measured: d=576 fails at 27 layers / 3,284
//! ops and passes at 26 / 3,164). `--shard N` is the fix: every member
//! package stays small, so real models reach the Neural Engine one
//! shard at a time. These tests prove a bundle (a) loads on every
//! compute-unit set including CPU+NeuralEngine, and (b) computes the
//! same logits as the monolithic package.

#![cfg(target_os = "macos")]

use mil_convert::builder::Options;
use mil_convert::{convert, Quant};
use mil_infer::{ComputeUnits, Input, Model, ShardedModel};
use std::path::{Path, PathBuf};

fn write_safetensors(path: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
    let mut header = String::from("{");
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape, vals) in tensors {
        let off = data.len() as u64;
        for v in vals {
            data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F16\",\"shape\":{shape:?},\"data_offsets\":[{off},{}]}}",
            data.len()
        ));
    }
    header.push_str(&entries.join(","));
    header.push('}');
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// qwen3 synthetic checkpoint: 6 layers, d=32, 4 q-heads / 2 kv-heads,
/// hd=8, inter=64, vocab=96 — small enough for instant conversion but
/// big enough to span 3 layer shards + 3 head shards.
fn synth(dir: &Path) {
    let (nl, d, qh, kvh, hd, inter, vocab) = (6usize, 32i64, 4i64, 2i64, 8i64, 64i64, 96i64);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"model_type":"qwen3","hidden_size":{d},"num_hidden_layers":{nl},
            "num_attention_heads":{qh},"num_key_value_heads":{kvh},"head_dim":{hd},
            "intermediate_size":{inter},"vocab_size":{vocab},"rms_norm_eps":1e-6,
            "rope_theta":1000000.0,"max_position_embeddings":512,
            "tie_word_embeddings":false}}"#
        ),
    )
    .unwrap();
    let mut t: Vec<(String, Vec<i64>, Vec<f32>)> = Vec::new();
    let mut push = |n: String, shape: Vec<i64>, fill: f32| {
        let cnt: i64 = shape.iter().product();
        t.push((
            n,
            shape,
            (0..cnt).map(|i| fill + (i % 7) as f32 * 0.001).collect(),
        ));
    };
    for l in 0..nl {
        let p = format!("model.layers.{l}");
        push(format!("{p}.input_layernorm.weight"), vec![d], 1.0);
        push(format!("{p}.post_attention_layernorm.weight"), vec![d], 1.0);
        push(
            format!("{p}.self_attn.q_proj.weight"),
            vec![qh * hd, d],
            0.02,
        );
        push(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kvh * hd, d],
            0.02,
        );
        push(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kvh * hd, d],
            0.02,
        );
        push(
            format!("{p}.self_attn.o_proj.weight"),
            vec![d, qh * hd],
            0.02,
        );
        push(format!("{p}.self_attn.q_norm.weight"), vec![hd], 1.0);
        push(format!("{p}.self_attn.k_norm.weight"), vec![hd], 1.0);
        push(format!("{p}.mlp.gate_proj.weight"), vec![inter, d], 0.02);
        push(format!("{p}.mlp.up_proj.weight"), vec![inter, d], 0.02);
        push(format!("{p}.mlp.down_proj.weight"), vec![d, inter], 0.02);
    }
    push("model.norm.weight".into(), vec![d], 1.0);
    push("model.embed_tokens.weight".into(), vec![vocab, d], 0.02);
    push("lm_head.weight".into(), vec![vocab, d], 0.02);
    let refs: Vec<(&str, Vec<i64>, Vec<f32>)> = t
        .iter()
        .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
        .collect();
    write_safetensors(&dir.join("model.safetensors"), &refs);
}

fn opts(seq: i64) -> Options {
    Options {
        seq,
        max_kv: 16,
        quant: Quant::Fp16,
        lm_head: true,
        embed: false,
        ..Options::default()
    }
}

struct Root(PathBuf);
impl Root {
    fn new(tag: &str) -> Root {
        let p = std::env::temp_dir().join(format!("sharded_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Root(p)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        if std::env::var("MIL_KEEP_ARTIFACTS").is_err() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn coremlc() -> bool {
    mil_compile::coremlc_path().is_some()
}

fn feeds(
    cfg: &mil_convert::ModelConfig,
    model_dir: &Path,
) -> (Vec<mil_verify::golden::PreparedInput>, Vec<Vec<u8>>) {
    let shards = mil_convert::safetensors::open_dir(model_dir).unwrap();
    let f = mil_verify::golden::package_inputs(cfg, &shards, &[3, 1, 4, 1], 16).unwrap();
    (f, Vec::new())
}

fn as_inputs<'a>(feeds: &'a [mil_verify::golden::PreparedInput]) -> Vec<Input<'a>> {
    feeds
        .iter()
        .map(|p| Input {
            name: &p.name,
            shape: &p.shape,
            data: &p.data,
            dtype: p.dtype,
        })
        .collect()
}

/// The bundle contract: N-layer stateful shards + vocab-sliced heads +
/// manifest, all members load on every unit set, chained prediction
/// matches the monolithic package's logits.
#[test]
fn sharded_bundle_matches_monolithic() {
    if !coremlc() {
        return;
    }
    let root = Root::new("e2e");
    let model_dir = root.0.join("model");
    synth(&model_dir);
    let cfg =
        mil_convert::ModelConfig::from_json(&std::fs::read(model_dir.join("config.json")).unwrap())
            .unwrap();

    // Monolithic reference package.
    let mono = root.0.join("mono.mlpackage");
    convert(&model_dir, &mono, &opts(4)).unwrap();

    // 3 layer shards + 3 head shards.
    let bundle = root.0.join("s.milshards");
    convert(
        &model_dir,
        &bundle,
        &Options {
            shard_layers: 2,
            head_shards: 3,
            ..opts(4)
        },
    )
    .unwrap();
    for f in [
        "manifest.json",
        "layer_00-02.mlpackage",
        "layer_02-04.mlpackage",
        "layer_04-06.mlpackage",
        "head_v0-32.mlpackage",
        "head_v32-64.mlpackage",
        "head_v64-96.mlpackage",
    ] {
        assert!(bundle.join(f).exists(), "missing bundle member {f}");
    }
    let manifest = mil_infer::ShardManifest::load(&bundle).unwrap();
    assert_eq!(manifest.layer_shards.len(), 3);
    assert_eq!(manifest.head_shards.len(), 3);
    assert_eq!(manifest.vocab_size, 96);

    let (f, _) = feeds(&cfg, &model_dir);
    let inputs = as_inputs(&f);

    // Monolithic logits on CPU.
    let c = mil_compile::compile(&mono, &root.0.join("cm")).unwrap();
    let m = Model::load(&c.path, ComputeUnits::CpuOnly).unwrap();
    let st = m.new_state().unwrap();
    let mono_logits = m.predict_with_state(Some(&st), &inputs).unwrap().outputs[0].values();

    // Bundle logits on CPU — same feeds chained through the shards.
    let sm = ShardedModel::load(&bundle, ComputeUnits::CpuOnly, &root.0.join("cs")).unwrap();
    let p = sm.predict(&inputs).unwrap();
    let out = &p.outputs[0];
    assert_eq!(out.shape, vec![1, 96, 4, 1]);
    let shard_logits = out.values();
    assert_eq!(shard_logits.len(), mono_logits.len());
    let max_diff = shard_logits
        .iter()
        .zip(&mono_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 0.05, "sharded vs monolithic max diff {max_diff}");

    // Same top-1 at every position.
    for pos in 0..4 {
        let col = |v: &[f32]| -> usize {
            (0..96)
                .map(|c| (v[c * 4 + pos], c))
                .fold((f32::NEG_INFINITY, 0), |a, x| if x.0 > a.0 { x } else { a })
                .1
        };
        assert_eq!(col(&shard_logits), col(&mono_logits), "top-1 at pos {pos}");
    }
}

/// Every member package must load on CPU+NeuralEngine — the whole
/// reason bundles exist. Printed as a matrix for the record.
#[test]
fn sharded_members_load_on_ane() {
    if !coremlc() {
        return;
    }
    let root = Root::new("anemat");
    let model_dir = root.0.join("model");
    synth(&model_dir);
    let bundle = root.0.join("s.milshards");
    convert(
        &model_dir,
        &bundle,
        &Options {
            shard_layers: 2,
            head_shards: 3,
            ..opts(4)
        },
    )
    .unwrap();
    let manifest = mil_infer::ShardManifest::load(&bundle).unwrap();
    let units = [
        ("cpu", ComputeUnits::CpuOnly),
        ("gpu", ComputeUnits::CpuAndGpu),
        ("all", ComputeUnits::All),
        ("ane", ComputeUnits::CpuAndNeuralEngine),
    ];
    let files: Vec<String> = manifest
        .layer_shards
        .iter()
        .map(|(f, _, _)| f.clone())
        .chain(manifest.head_shards.iter().map(|(f, _, _)| f.clone()))
        .collect();
    for file in &files {
        let c =
            mil_compile::compile(&bundle.join(file), &root.0.join(format!("c_{file}"))).unwrap();
        let verdict: Vec<String> = units
            .iter()
            .map(|(n, u)| {
                if Model::load(&c.path, *u).is_ok() {
                    format!("{n}:ok")
                } else {
                    format!("{n}:FAIL")
                }
            })
            .collect();
        println!("SHARDMAT {file:<28} {}", verdict.join(" "));
        for (n, u) in &units {
            assert!(
                Model::load(&c.path, *u).is_ok(),
                "{file} failed to load on {n} — shard still over the ANE limit"
            );
        }
    }
}

/// `--embed` actually emits the gather: the package takes `ids`
/// instead of `x`, and predicting through it must match the
/// host-side embed + `x` package on the same tokens.
#[test]
fn embed_package_takes_ids_and_matches() {
    if !coremlc() {
        return;
    }
    let root = Root::new("embed");
    let model_dir = root.0.join("model");
    synth(&model_dir);
    let cfg =
        mil_convert::ModelConfig::from_json(&std::fs::read(model_dir.join("config.json")).unwrap())
            .unwrap();

    // embed package: ids (1,S) int32 in.
    let epkg = root.0.join("embed.mlpackage");
    convert(
        &model_dir,
        &epkg,
        &Options {
            embed: true,
            ..opts(4)
        },
    )
    .unwrap();
    let spec = std::fs::read(epkg.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    let summary = mil_verify::summarize(&spec).unwrap();
    let names: Vec<&str> = summary.inputs.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"ids"), "embed package inputs: {names:?}");
    assert!(!names.contains(&"x"), "embed package kept x: {names:?}");

    // ids input: [3,1,4,1].
    let mut idb = Vec::new();
    for id in [3i32, 1, 4, 1] {
        idb.extend_from_slice(&id.to_le_bytes());
    }
    let (f, _) = feeds(&cfg, &model_dir);
    let rest: Vec<&mil_verify::golden::PreparedInput> =
        f.iter().filter(|p| p.name != "x").collect();
    let mut bufs: Vec<Vec<u8>> = vec![idb];
    for p in &rest {
        bufs.push(p.data.clone());
    }
    let mut inputs = vec![Input {
        name: "ids",
        shape: &[1, 4],
        data: &bufs[0],
        dtype: mil_spec::DType::Int32,
    }];
    for (i, p) in rest.iter().enumerate() {
        inputs.push(Input {
            name: &p.name,
            shape: &p.shape,
            data: &bufs[i + 1],
            dtype: p.dtype,
        });
    }

    let ce = mil_compile::compile(&epkg, &root.0.join("ce")).unwrap();
    let em = Model::load(&ce.path, ComputeUnits::CpuOnly).unwrap();
    let es = em.new_state().unwrap();
    let embed_logits = em.predict_with_state(Some(&es), &inputs).unwrap().outputs[0].values();

    // x package on the same tokens (host-side embed via package_inputs).
    let xpkg = root.0.join("x.mlpackage");
    convert(&model_dir, &xpkg, &opts(4)).unwrap();
    let cx = mil_compile::compile(&xpkg, &root.0.join("cx")).unwrap();
    let xm = Model::load(&cx.path, ComputeUnits::CpuOnly).unwrap();
    let xs = xm.new_state().unwrap();
    let x_inputs = as_inputs(&f);
    let x_logits = xm.predict_with_state(Some(&xs), &x_inputs).unwrap().outputs[0].values();

    assert_eq!(embed_logits.len(), x_logits.len());
    let max_diff = embed_logits
        .iter()
        .zip(&x_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 0.05, "embed vs host-embed max diff {max_diff}");
}
