//! Post-conversion package surgery end-to-end: convert a tiny Qwen3,
//! then prove `fuse-lora`, `requant`, `graft`, `reshape`, and `attest`
//! produce packages that still compile and predict when `coremlc` is
//! available (the structural checks always run). Temp dirs self-clean
//! unless `MIL_KEEP_ARTIFACTS=1`.

use mil_convert::surgery::{self, GroupKind, PackageEdit, TargetQuant};
use mil_convert::{attest, convert, Options, Quant};
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------- fixtures ----------

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("surg_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}
fn cleanup(p: &Path) {
    if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
        let _ = std::fs::remove_dir_all(p);
    }
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

/// Tiny qwen3 (d=16, 2 layers, vocab 64) with a per-weight `seed` so
/// donor/base checkpoints differ deterministically.
fn tiny_qwen3(dir: &Path, seed: f32) {
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
            (0..n)
                .map(|i| fill + seed + (i as f32 % 7.0) * 0.01)
                .collect(),
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

/// Rank-2 MLX adapter on `l0 q_proj` + `l1 down_proj` (same fixture as
/// mil_verify's lora_e2e).
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

fn opts(quant: Quant) -> Options {
    Options {
        seq: 1,
        max_kv: 16,
        quant,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        ..Options::default()
    }
}

fn group(pe: &PackageEdit, name: &str) -> surgery::WeightGroup {
    pe.weight_groups()
        .unwrap()
        .into_iter()
        .find(|g| g.name == name)
        .unwrap_or_else(|| panic!("no weight group {name}"))
}

fn vals_of(pe: &PackageEdit, name: &str) -> Vec<f32> {
    pe.group_values(&group(pe, name)).unwrap()
}

fn wsize(pkg: &Path) -> u64 {
    std::fs::metadata(pkg.join("Data/com.apple.CoreML/weights/weight.bin"))
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Provenance (weights + program) must verify after a surgery edit.
fn assert_attests(pkg: &Path, what: &str) {
    let r = attest::verify(pkg, None).unwrap();
    assert!(r.has_provenance, "{what}: provenance lost");
    assert!(r.ok, "{what}: attest failed after edit: {:?}", r.lines);
    assert!(
        r.lines
            .iter()
            .any(|l| l.contains("program") && l.contains("OK")),
        "{what}: program hash not verified: {:?}",
        r.lines
    );
}

/// Replace the first occurrence of `from` with the same-length `to`
/// in the package's `model.mlmodel` (an in-place tamper).
fn patch_spec(pkg: &Path, from: &[u8], to: &[u8]) {
    assert_eq!(from.len(), to.len());
    let p = pkg.join("Data/com.apple.CoreML/model.mlmodel");
    let mut b = std::fs::read(&p).unwrap();
    let at = b
        .windows(from.len())
        .position(|w| w == from)
        .unwrap_or_else(|| panic!("{:?} not found in spec", String::from_utf8_lossy(from)));
    b[at..at + to.len()].copy_from_slice(to);
    std::fs::write(&p, &b).unwrap();
}

#[test]
fn attest_program_tamper_detected() {
    let root = tmp("attest_prog");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("m.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();
    assert_attests(&pkg, "fresh convert");

    // (c) shortDescription is covered — only userDefined is excluded
    let c = root.join("c.mlpackage");
    copy_dir(&pkg, &c);
    patch_spec(&c, b"converted by mil_convert", b"converted bY mil_convert");
    let r = attest::verify(&c, None).unwrap();
    assert!(!r.ok, "{:?}", r.lines);
    assert!(r.lines.iter().any(|l| l.contains("TAMPERED program")));

    // (b) one op changed (softmax → sigmoid, same length)
    let b = root.join("b.mlpackage");
    copy_dir(&pkg, &b);
    patch_spec(&b, b"softmax", b"sigmoid");
    let r = attest::verify(&b, None).unwrap();
    assert!(!r.ok, "{:?}", r.lines);
    assert!(r.lines.iter().any(|l| l.contains("TAMPERED program")));

    // editing a userDefined value is NOT a program change
    let u = root.join("u.mlpackage");
    copy_dir(&pkg, &u);
    patch_spec(&u, b"mil_convert 0.", b"mil_convert 1.");
    let r = attest::verify(&u, None).unwrap();
    assert!(r.ok, "{:?}", r.lines);

    // provenance without the program key → reported, not a failure
    let old = root.join("old.mlpackage");
    copy_dir(&pkg, &old);
    {
        let sp = old.join("Data/com.apple.CoreML/model.mlmodel");
        let mut m = mil_spec::proto::decode(&std::fs::read(&sp).unwrap()).unwrap();
        let mut desc = m.msg(2).unwrap();
        let mut meta = desc.msg(100).unwrap();
        meta.fields.retain(|f| {
            !(f.num == 16
                && f.val.as_msg().and_then(|e| e.str(1)).as_deref() == Some("mil.prov.program"))
        });
        desc.set_msg(100, &meta);
        m.set_msg(2, &desc);
        std::fs::write(&sp, mil_spec::proto::encode(&m)).unwrap();
    }
    let r = attest::verify(&old, None).unwrap();
    assert!(r.ok, "{:?}", r.lines);
    assert!(r
        .lines
        .iter()
        .any(|l| l.contains("spec not covered (provenance predates mil.prov.program)")));
    cleanup(&root);
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let (s, d) = (e.path(), to.join(e.file_name()));
        if s.is_dir() {
            copy_dir(&s, &d);
        } else {
            std::fs::copy(&s, &d).unwrap();
        }
    }
}
// ---------- attest ----------

#[test]
fn attest_clean_tampered_and_missing() {
    let root = tmp("attest");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("m.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();

    // clean package verifies, including source fingerprints
    let r = attest::verify(&pkg, Some(&model)).unwrap();
    assert!(r.has_provenance);
    assert!(r.ok, "{:?}", r.lines);
    assert!(r.lines.iter().any(|l| l.contains("mil.prov.weights")));
    assert!(r.lines.iter().any(|l| l.contains("1/1 source")));

    // tamper one byte of weight.bin → must fail
    let wpath = pkg.join("Data/com.apple.CoreML/weights/weight.bin");
    let mut w = std::fs::read(&wpath).unwrap();
    let n = w.len();
    w[n - 1] ^= 0xff;
    std::fs::write(&wpath, &w).unwrap();
    let r2 = attest::verify(&pkg, None).unwrap();
    assert!(!r2.ok, "tampered weight.bin verified clean: {:?}", r2.lines);
    assert!(r2.lines.iter().any(|l| l.contains("TAMPERED")));

    // a package with no provenance → reported, not an error
    let noprov = root.join("noprov.mlpackage");
    {
        // Re-emit the same spec minus userDefined by clearing metadata:
        // rewrite the package through PackageEdit after stripping the
        // metadata message (description field 100) entirely.
        let spec_path = pkg.join("Data/com.apple.CoreML/model.mlmodel");
        let mut model = mil_spec::proto::decode(&std::fs::read(&spec_path).unwrap()).unwrap();
        let mut desc = model.msg(2).unwrap();
        desc.remove_all(100);
        model.set_msg(2, &desc);
        std::fs::create_dir_all(noprov.join("Data/com.apple.CoreML/weights")).unwrap();
        std::fs::write(
            noprov.join("Data/com.apple.CoreML/model.mlmodel"),
            mil_spec::proto::encode(&model),
        )
        .unwrap();
        std::fs::copy(
            &wpath,
            noprov.join("Data/com.apple.CoreML/weights/weight.bin"),
        )
        .unwrap();
        std::fs::write(noprov.join("Manifest.json"), "{}").unwrap();
    }
    let r3 = attest::verify(&noprov, None).unwrap();
    assert!(!r3.has_provenance);
    assert!(r3.ok);
    cleanup(&root);
}

// ---------- requant ----------

#[test]
fn requant_fp16_to_int8_shrinks_and_roundtrips() {
    let root = tmp("requant");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("fp16.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();
    let base_size = wsize(&pkg);

    let pe = PackageEdit::open(&pkg).unwrap();
    let base_wq = vals_of(&pe, "l0_wq");
    let n_fp16 = pe
        .weight_groups()
        .unwrap()
        .iter()
        .filter(|g| g.kind == GroupKind::Fp16)
        .count();
    assert!(n_fp16 > 0);

    let q8 = root.join("int8.mlpackage");
    let rep = surgery::requant(&pkg, TargetQuant::Int8, Some(&q8)).unwrap();
    assert_attests(&q8, "requant");
    assert!(rep.written.as_ref().unwrap().weight_bytes < base_size);
    let qe = PackageEdit::open(&q8).unwrap();
    for g in qe.weight_groups().unwrap() {
        if g.is_conv_weight() {
            assert_eq!(g.kind, GroupKind::Int8, "{} not requantized", g.name);
        } else {
            // norms/biases stay fp16 — the converter never quantizes them
            assert_eq!(g.kind, GroupKind::Fp16, "{} should stay fp16", g.name);
        }
    }
    // decoded values within int8 tolerance of the fp16 originals
    let q_wq = vals_of(&qe, "l0_wq");
    for (a, b) in base_wq.iter().zip(q_wq.iter()) {
        assert!((a - b).abs() < 0.01, "l0_wq int8: {a} vs {b}");
    }

    // int8 → palette4 shrinks further
    let p4 = root.join("pal4.mlpackage");
    let _rep4 = surgery::requant(&q8, TargetQuant::Palette4, Some(&p4)).unwrap();
    let pe4 = PackageEdit::open(&p4).unwrap();
    for g in pe4.weight_groups().unwrap() {
        if g.is_conv_weight() {
            assert_eq!(g.kind, GroupKind::Palette4);
        } else {
            assert_eq!(g.kind, GroupKind::Fp16);
        }
    }
    let p4_wq = vals_of(&pe4, "l0_wq");
    for (a, b) in base_wq.iter().zip(p4_wq.iter()) {
        assert!((a - b).abs() < 0.05, "l0_wq pal4: {a} vs {b}");
    }
    // NB: for these tiny tensors the 16-entry fp16 LUT per row outweighs
    // the nibble savings — palette4 only wins at real model widths. The
    // requant is still applied and decodable; no size assertion here.

    // in-place write (no -o) keeps the package readable
    surgery::requant(&p4, TargetQuant::Fp16, None).unwrap();
    let pe5 = PackageEdit::open(&p4).unwrap();
    for g in pe5.weight_groups().unwrap() {
        assert_eq!(g.kind, GroupKind::Fp16);
        let _ = g;
    }
    cleanup(&root);
}

/// Full pipeline proof: fp16 → int8 package still compiles and its
/// logits track the fp16 model's.
#[test]
fn requant_compile_and_predict() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp("requant_e2e");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("fp16.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();
    let q8 = root.join("int8.mlpackage");
    surgery::requant(&pkg, TargetQuant::Int8, Some(&q8)).unwrap();

    let c0 = mil_compile::compile(&pkg, &root.join("c0")).unwrap();
    let c1 = mil_compile::compile(&q8, &root.join("c1")).unwrap();
    let l0 = predict_logits(&c0.path);
    let l1 = predict_logits(&c1.path);
    let cos = cosine(&l0, &l1);
    assert!(cos > 0.99, "requantized logits diverged: cosine {cos}");
    cleanup(&root);
}

// ---------- graft ----------

#[test]
fn graft_splices_donor_layers() {
    let root = tmp("graft");
    let dm = root.join("donor_model");
    let bm = root.join("base_model");
    tiny_qwen3(&dm, 0.3);
    tiny_qwen3(&bm, 0.0);
    let donor = root.join("donor.mlpackage");
    let base = root.join("base.mlpackage");
    convert(&dm, &donor, &opts(Quant::Int8)).unwrap();
    convert(&bm, &base, &opts(Quant::Int8)).unwrap();

    let out = root.join("grafted.mlpackage");
    surgery::graft(&donor, &base, (0, 0), &out).unwrap();
    assert_attests(&out, "graft");

    let dpe = PackageEdit::open(&donor).unwrap();
    let gpe = PackageEdit::open(&out).unwrap();
    let bpe = PackageEdit::open(&base).unwrap();
    // layer 0 groups now equal donor's; layer 1 still base's
    for name in [
        "l0_wq", "l0_wk", "l0_wv", "l0_wo", "l0_wg", "l0_wu", "l0_wd",
    ] {
        let d = vals_of(&dpe, name);
        let g = vals_of(&gpe, name);
        assert_eq!(d, g, "{name} was not grafted losslessly");
    }
    let b1 = vals_of(&bpe, "l1_wq");
    let g1 = vals_of(&gpe, "l1_wq");
    assert_eq!(b1, g1, "l1_wq should come from base");

    // mismatch rejection: same-arch package vs a 1-layer model
    let one_layer = root.join("one_model");
    tiny_qwen3(&one_layer, 0.0);
    // shrink to 1 layer
    std::fs::write(
        one_layer.join("config.json"),
        std::fs::read_to_string(one_layer.join("config.json"))
            .unwrap()
            .replace("num_hidden_layers\": 2", "num_hidden_layers\": 1"),
    )
    .unwrap();
    let onepkg = root.join("one.mlpackage");
    convert(&one_layer, &onepkg, &opts(Quant::Int8)).unwrap();
    let bad = root.join("bad.mlpackage");
    assert!(surgery::graft(&donor, &onepkg, (0, 0), &bad).is_err());
    cleanup(&root);
}

#[test]
fn graft_compile_and_predict() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp("graft_e2e");
    let dm = root.join("donor_model");
    let bm = root.join("base_model");
    tiny_qwen3(&dm, 0.3);
    tiny_qwen3(&bm, 0.0);
    let donor = root.join("donor.mlpackage");
    let base = root.join("base.mlpackage");
    convert(&dm, &donor, &opts(Quant::Fp16)).unwrap();
    convert(&bm, &base, &opts(Quant::Fp16)).unwrap();
    let out = root.join("grafted.mlpackage");
    surgery::graft(&donor, &base, (0, 1), &out).unwrap();
    let c = mil_compile::compile(&out, &root.join("c")).unwrap();
    let lg = predict_logits(&c.path);
    let cd = mil_compile::compile(&donor, &root.join("cd")).unwrap();
    let ld = predict_logits(&cd.path);
    // all transformer layers came from the donor → logits match its
    // (the head/norm still come from base but donor ≈ base + 0.3 shift)
    let cos = cosine(&lg, &ld);
    assert!(cos > 0.9, "grafted model far from donor: cosine {cos}");
    cleanup(&root);
}

// ---------- reshape ----------

#[test]
fn reshape_rewrites_enumerated_shapes() {
    let root = tmp("reshape");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("flex.mlpackage");
    let mut o = opts(Quant::Fp16);
    o.seq = 4;
    o.seq_lens = vec![1, 4];
    convert(&model, &pkg, &o).unwrap();

    let out = root.join("flex2.mlpackage");
    surgery::reshape(&pkg, &[2, 8], Some(&out)).unwrap();
    assert_attests(&out, "reshape");

    // read the new EnumeratedShapes out of the rewritten spec
    let spec = std::fs::read(out.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    let m = mil_spec::proto::decode(&spec).unwrap();
    let desc = m.msg(2).unwrap();
    let mut saw_enum = 0;
    for f in desc.msgs(1) {
        if let Some(ma) = f.msg(3).and_then(|t| t.msg(5)) {
            if let Some(es) = ma.msg(21) {
                saw_enum += 1;
                let entries: Vec<Vec<i64>> = es
                    .msgs(1)
                    .iter()
                    .map(|s| {
                        s.get_all(1)
                            .iter()
                            .map(|v| v.as_varint().unwrap_or(0) as i64)
                            .collect()
                    })
                    .collect();
                assert_eq!(
                    entries.len(),
                    2,
                    "{}: entries {entries:?}",
                    f.str(1).unwrap_or_default()
                );
            }
        }
    }
    assert!(saw_enum > 0, "no enumeratedShapes after reshape");

    // a package without EnumeratedShapes is rejected honestly
    let fixed = root.join("fixed.mlpackage");
    convert(&model, &fixed, &opts(Quant::Fp16)).unwrap();
    assert!(surgery::reshape(&fixed, &[2, 4], None).is_err());
    cleanup(&root);
}

#[test]
fn reshape_compile_and_predict() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp("reshape_e2e");
    let model = root.join("model");
    tiny_qwen3(&model, 0.0);
    let pkg = root.join("flex.mlpackage");
    let mut o = opts(Quant::Fp16);
    o.seq = 4;
    o.seq_lens = vec![1, 4];
    convert(&model, &pkg, &o).unwrap();
    let out = root.join("flex2.mlpackage");
    surgery::reshape(&pkg, &[2, 8], Some(&out)).unwrap();
    assert_attests(&out, "reshape");
    let c = mil_compile::compile(&out, &root.join("c")).unwrap();
    // predict at the new default seq len (8) — output is vocab×seq
    // logits, so 64*8 values.
    let logits = predict_seq(&c.path, 8);
    assert_eq!(logits.len(), 64 * 8);
    assert!(logits.iter().all(|v| v.is_finite()));
    cleanup(&root);
}

// ---------- fuse-lora ----------

#[test]
fn fuse_lora_matches_convert_lora() {
    let root = tmp("fusing");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model, 0.0);
    write_adapter(&adir);
    let pkg = root.join("base.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();

    let fused_pkg = root.join("fused.mlpackage");
    let rep = surgery::fuse_lora(&pkg, &adir, &fused_pkg).unwrap();
    assert_attests(&fused_pkg, "fuse-lora");
    assert!(rep.lines.iter().any(|l| l.contains("l0_wq")));
    assert!(rep.lines.iter().any(|l| l.contains("l1_wd")));

    // reference: convert --lora fused values
    let ref_pkg = root.join("ref.mlpackage");
    let mut o = opts(Quant::Fp16);
    o.lora = Some(adir.clone());
    convert(&model, &ref_pkg, &o).unwrap();

    let fpe = PackageEdit::open(&fused_pkg).unwrap();
    let rpe = PackageEdit::open(&ref_pkg).unwrap();
    for name in ["l0_wq", "l1_wd"] {
        let f = vals_of(&fpe, name);
        let r = vals_of(&rpe, name);
        for (a, b) in f.iter().zip(r.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "{name}: package-fused {a} vs convert-fused {b}"
            );
        }
    }
    // untouched weights identical to base
    let bpe = PackageEdit::open(&pkg).unwrap();
    assert_eq!(vals_of(&fpe, "l0_wk"), vals_of(&bpe, "l0_wk"));

    // int8 package: fusion decodes → fuses → requantizes in place
    let pkg8 = root.join("base8.mlpackage");
    convert(&model, &pkg8, &opts(Quant::Int8)).unwrap();
    let fused8 = root.join("fused8.mlpackage");
    surgery::fuse_lora(&pkg8, &adir, &fused8).unwrap();
    assert_attests(&fused8, "fuse-lora int8");
    let g = PackageEdit::open(&fused8).unwrap();
    assert_eq!(group(&g, "l0_wq").kind, GroupKind::Int8);
    cleanup(&root);
}

#[test]
fn fuse_lora_rejects_wrong_checkpoint() {
    let root = tmp("lora_bad");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model, 0.0);
    write_adapter(&adir);
    // adapter with a target that doesn't exist in this package
    let bad = root.join("bad_adapter");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(
        bad.join("adapter_config.json"),
        r#"{"lora_parameters": {"rank": 2, "scale": 0.5, "dropout": 0}}"#,
    )
    .unwrap();
    let mk = |rows: i64, cols: i64| -> Vec<f32> { vec![0.01; (rows * cols) as usize] };
    write_safetensors_f32(
        &bad.join("adapters.safetensors"),
        &[
            (
                "model.layers.9.self_attn.q_proj.lora_a",
                vec![16, 2],
                mk(16, 2),
            ),
            (
                "model.layers.9.self_attn.q_proj.lora_b",
                vec![2, 16],
                mk(2, 16),
            ),
        ],
    );
    let pkg = root.join("base.mlpackage");
    convert(&model, &pkg, &opts(Quant::Fp16)).unwrap();
    let out = root.join("out.mlpackage");
    let e = surgery::fuse_lora(&pkg, &bad, &out).unwrap_err();
    assert!(
        e.contains("no weight const") || e.contains("unsupported"),
        "{e}"
    );
    assert!(!out.exists(), "partial output written on failure");
    let _ = adir;
    cleanup(&root);
}

#[test]
fn fuse_lora_compile_and_predict() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = tmp("fusing_e2e");
    let model = root.join("model");
    let adir = root.join("adapter");
    tiny_qwen3(&model, 0.0);
    write_adapter(&adir);
    let pkg = root.join("base.mlpackage");
    convert(&model, &pkg, &opts(Quant::Int8)).unwrap();
    let fused = root.join("fused.mlpackage");
    surgery::fuse_lora(&pkg, &adir, &fused).unwrap();

    let c0 = mil_compile::compile(&pkg, &root.join("c0")).unwrap();
    let c1 = mil_compile::compile(&fused, &root.join("c1")).unwrap();
    let l0 = predict_logits(&c0.path);
    let l1 = predict_logits(&c1.path);
    assert_ne!(l0, l1, "fused adapter had no effect");
    let cos = cosine(&l0, &l1);
    assert!(cos > 0.9, "fused logits diverged wildly: cosine {cos}");
    cleanup(&root);
}

// ---------- prediction helpers ----------

fn f16s(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
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

/// One decode step (seq=1) on a compiled tiny model → 64 logits.
#[cfg(target_os = "macos")]
fn predict_logits(compiled: &Path) -> Vec<f32> {
    use mil_infer::{ComputeUnits, Input, Model};
    let m = Model::load(compiled, ComputeUnits::All).unwrap();
    let state = m.new_state().expect("stateful model");
    let x: Vec<f32> = (0..16).map(|i| 0.05 * (i as f32 + 1.0)).collect();
    let xb = f16s(&x);
    let cos = f16s(&[1.0; 8]);
    let sin = f16s(&[0.0; 8]);
    let mk = f16s(&[0.0; 16]);
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

#[cfg(not(target_os = "macos"))]
fn predict_logits(_compiled: &Path) -> Vec<f32> {
    unreachable!("predict only runs under coremlc gate on macOS")
}

/// Predict at sequence length `s` on a flexible-shape compiled model:
/// reads input shapes from the spec, fills f16 features with a ramp,
/// `pos`/`seq` int32 inputs with sane values.
#[cfg(target_os = "macos")]
fn predict_seq(compiled: &Path, s: i64) -> Vec<f32> {
    use mil_infer::{ComputeUnits, Input, Model};
    // Flexible packages can't plan on `all` (the GPU/scheduler path
    // rejects dynamic dims — measured: enum=cpu+ane only, range=cpu
    // only). CpuOnly is the deterministic choice here.
    let m = Model::load(compiled, ComputeUnits::CpuOnly).unwrap();
    let state = m.new_state().expect("stateful model");
    // input feature list: name → shape (defaults = first enum entry)
    let spec = std::fs::read(
        compiled
            .parent()
            .unwrap()
            .join("../flex2.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
    );
    let _ = spec; // shapes supplied below instead — derive per name
    let mk_f16 = |shape: &[i64]| -> Vec<u8> {
        let n: usize = shape.iter().map(|d| d.max(&1)).product::<i64>() as usize;
        f16s(&(0..n).map(|i| 0.01 * (i % 8) as f32).collect::<Vec<_>>())
    };
    let x = mk_f16(&[1, 16, s, 1]);
    let cos = mk_f16(&[1, 1, s, 8]);
    let sin = mk_f16(&[1, 1, s, 8]);
    let mk = mk_f16(&[1, 1, s, 16]);
    let pos: i32 = 0;
    let seq: i32 = s as i32;
    let p = m
        .predict_with_state(
            Some(&state),
            &[
                Input {
                    name: "x",
                    shape: &[1, 16, s, 1],
                    data: &x,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "cos",
                    shape: &[1, 1, s, 8],
                    data: &cos,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "sin",
                    shape: &[1, 1, s, 8],
                    data: &sin,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "mask",
                    shape: &[1, 1, s, 16],
                    data: &mk,
                    dtype: mil_spec::DType::Fp16,
                },
                Input {
                    name: "pos",
                    shape: &[1],
                    data: &pos.to_le_bytes(),
                    dtype: mil_spec::DType::Int32,
                },
                Input {
                    name: "seq",
                    shape: &[1],
                    data: &seq.to_le_bytes(),
                    dtype: mil_spec::DType::Int32,
                },
            ],
        )
        .unwrap();
    p.outputs[0].values()
}

#[cfg(not(target_os = "macos"))]
fn predict_seq(_compiled: &Path, _s: i64) -> Vec<f32> {
    unreachable!("predict only runs under coremlc gate on macOS")
}
