//! Reference-forward proofs — `#[ignore]`d: they need the model files
//! under `~/.cache/mil_gguf_test/models/`. Run explicitly:
//!
//! ```sh
//! cargo test -p mil_verify --test reference_e2e --release -- --ignored --nocapture
//! ```
//!
//! 1. **Anchor**: the pure-Rust reference forward, run on HF
//!    safetensors, must greedily continue "The capital of France is"
//!    with "ĠParis" — a semantic check that can't pass by accident.
//! 2. **GGUF isolation**: reference on GGUF weights vs reference on HF
//!    weights — same engine, only the weight source differs, so any
//!    divergence is provably in the GGUF read/dequant/map path.
//! 3. **Converter vs reference**: the converted+compiled mlpackage vs
//!    the reference on identical token ids — the full pipeline check.

use mil_convert::gguf::Gguf;
use mil_convert::safetensors::{self, Safetensors};
use mil_convert::{ModelConfig, WeightSource};
use std::path::{Path, PathBuf};

fn cache() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models")
}

/// Look up a token string's id in a HF `tokenizer.json` by scanning
/// `"<token>":<id>` pairs — the vocab object is flat.
fn token_id(tok_json: &[u8], tok: &str) -> u32 {
    let needle = format!("\"{tok}\":");
    let n = needle.as_bytes();
    let mut i = 0usize;
    while i + n.len() < tok_json.len() {
        if &tok_json[i..i + n.len()] == n {
            let mut v = 0u32;
            let mut j = i + n.len();
            while tok_json[j].is_ascii_whitespace() {
                j += 1;
            }
            while tok_json[j].is_ascii_digit() {
                v = v * 10 + (tok_json[j] - b'0') as u32;
                j += 1;
            }
            return v;
        }
        i += 1;
    }
    panic!("token {tok:?} not found in tokenizer.json");
}

/// Prompt ids for "The capital of France is" — every word must be a
/// single BPE token in the model's vocab or the test fails loudly.
fn anchor_ids(hf_dir: &Path) -> Vec<u32> {
    let tj = std::fs::read(hf_dir.join("tokenizer.json")).expect("tokenizer.json");
    let paris = token_id(&tj, "ĠParis");
    let ids: Vec<u32> = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
        .iter()
        .map(|t| token_id(&tj, t))
        .collect();
    println!("anchor ids: {ids:?} (ĠParis={paris})");
    ids
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal));
    ix[..k].to_vec()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    dot / (na * nb + 1e-30)
}

fn hf_source(dir: &Path) -> Vec<Safetensors> {
    safetensors::open_dir(dir).expect("open safetensors")
}

fn anchor(tag: &str, hf_dir: &Path, expect_tok: &str) -> Vec<u32> {
    let cfg = ModelConfig::from_json(&std::fs::read(hf_dir.join("config.json")).unwrap()).unwrap();
    let w = hf_source(hf_dir);
    let ids = anchor_ids(hf_dir);
    let logits = mil_verify::reference::forward(&cfg, &w, &ids).expect("forward");
    let last = logits.last().unwrap();
    let next = argmax(last);
    let tj = std::fs::read(hf_dir.join("tokenizer.json")).unwrap();
    let want = token_id(&tj, expect_tok) as usize;
    let t5 = top_k(last, 5);
    println!(
        "{tag} anchor: next={next} (want {want}={expect_tok:?}) top5={t5:?} last_logits[..5]={:?}",
        &last[..5]
    );
    assert_eq!(
        next, want,
        "{tag}: reference must greedily pick {expect_tok:?}"
    );
    ids
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn anchor_qwen3() {
    anchor("qwen3", &cache().join("qwen3-hf"), "ĠParis");
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn anchor_smollm2() {
    anchor("smollm2", &cache().join("smollm2-hf"), "ĠParis");
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn anchor_qwen25() {
    anchor("qwen25", &cache().join("qwen25-hf"), "ĠParis");
}

/// Multiplier formula for the Q8_0 block quantizer at the pinned
/// llama.cpp commit: `quantize_row_q8_0_ref` and the NEON path use
/// `id = 1/d` with `d = amax/127`; the AVX path uses `id = 127/amax`
/// (one ULP can differ). Producers may have used either.
#[derive(Clone, Copy)]
enum IdMode {
    RecipD,
    Div127,
}

/// `roundf` rounds half away from zero (reference path); SIMD
/// (`vcvtnq`, `_mm256_round_ps`) rounds half to even.
#[derive(Clone, Copy)]
enum Round {
    HalfAway,
    TiesEven,
}

/// Q8_0 block-quantize `x` exactly as `quantize_row_q8_0` at the
/// pinned commit: 32-element blocks, fp16 scale `d = amax/127`, int8
/// codes `round(x*id)`. Returns the raw stored bytes (fp16 d + 32 i8).
fn quantize_q8_0(x: &[f32], id: IdMode, rnd: Round) -> Vec<u8> {
    const QK: usize = 32;
    let mut out = Vec::with_capacity(x.len() / QK * (QK + 2));
    for blk in x.chunks_exact(QK) {
        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = match (id, amax) {
            (_, 0.0) => 0.0,
            (IdMode::RecipD, _) => 1.0 / d,
            (IdMode::Div127, a) => 127.0 / a,
        };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in blk {
            let q = match rnd {
                Round::HalfAway => (v * id).round(),
                Round::TiesEven => (v * id).round_ties_even(),
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

/// HF tensor elements as f32. `via_f16` rounds through fp16 first
/// (pipelines that quantize from an F16 GGUF see those values).
fn st_elems_f32(srcs: &[Safetensors], name: &str, via_f16: bool) -> Option<(Vec<i64>, Vec<f32>)> {
    let st = srcs.iter().find(|s| s.has(name))?;
    if via_f16 {
        let (shape, bytes) = st.tensor_f16(name).ok()?;
        let xs = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        return Some((shape, xs));
    }
    let info = st.info(name)?;
    let raw = st.tensor_bytes(name).ok()?;
    let xs: Vec<f32> = match info.dtype {
        safetensors::StDType::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::StDType::F16 => raw
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::StDType::Bf16 => raw
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        other => panic!("{name}: unsupported HF dtype {other:?}"),
    };
    Some((info.shape.clone(), xs))
}

/// Forward llama Q/K head interleave, `conversion/llama.py` at the
/// pinned commit: `reshape(nh, 2, rows/nh/2, cols).swapaxes(1, 2)` —
/// `out[(h*hd2+j)*2+i] = in[(h*2+i)*hd2+j]`. Not an involution for
/// `hd2 != 2`; `Gguf`'s `unpermute` applies the inverse on load.
fn permute_heads_fwd(w: &mut [f32], n_head: usize, rows: usize, cols: usize) {
    let hd2 = rows / n_head / 2;
    let src = w.to_vec();
    for h in 0..n_head {
        for i in 0..2 {
            for j in 0..hd2 {
                let dst_base = ((h * hd2 + j) * 2 + i) * cols;
                let src_base = ((h * 2 + i) * hd2 + j) * cols;
                w[dst_base..dst_base + cols].copy_from_slice(&src[src_base..src_base + cols]);
            }
        }
    }
}

/// GGUF frontend isolation, proved exactly: requantize every Q8_0
/// tensor from the HF weights with `quantize_row_q8_0` semantics at
/// the pinned commit and compare stored blocks bit-for-bit (fp16
/// scale + int8 codes). The source path is tried both direct
/// (bf16->f32) and through an fp16 intermediate (bf16->f16->f32), each
/// under the reference, NEON and AVX id/rounding variants; at least
/// one combination must reproduce every stored byte. The
/// reference-vs-reference logits comparison below is kept as a
/// reported metric gated on top-1 equality only: its cosine quantifies
/// the quantization error baked into the file (already verified
/// per-tensor by the strict bound tests), not frontend correctness.
fn gguf_isolation(tag: &str, hf_dir: &Path, gguf_path: &Path) {
    use mil_convert::gguf::{names, GgufType};

    let cfg = ModelConfig::from_json(&std::fs::read(hf_dir.join("config.json")).unwrap()).unwrap();
    let w_hf = hf_source(hf_dir);
    let g = Gguf::open(gguf_path).expect("open gguf");
    let llama = g.arch.as_deref() == Some("llama");

    let paths = [(false, "src->f32"), (true, "src->f16->f32")];
    let variants = [
        (IdMode::RecipD, Round::HalfAway, "ref"),
        (IdMode::RecipD, Round::TiesEven, "neon"),
        (IdMode::Div127, Round::TiesEven, "avx"),
    ];
    let mut combo_mismatches = vec![0u64; paths.len() * variants.len()];
    let mut n_q8 = 0usize;
    for t in g.tensors() {
        if t.gguf_type != GgufType::Q8_0 {
            continue;
        }
        n_q8 += 1;
        let hf_name =
            names::ggml_to_hf(&t.name).unwrap_or_else(|| panic!("{}: no HF mapping", t.name));
        let stored = g.tensor_bytes(&t.name).expect("stored bytes");
        let (heads, kind) = match names::permute_kind(&t.name) {
            Some(names::PermuteKind::Q) => (cfg.num_heads, true),
            Some(names::PermuteKind::K) => (cfg.num_kv_heads, true),
            None => (0, false),
        };
        let mut per_tensor = Vec::new();
        for (pi, &(via_f16, path_name)) in paths.iter().enumerate() {
            let (shape, mut xs) = st_elems_f32(&w_hf, &hf_name, via_f16)
                .unwrap_or_else(|| panic!("{hf_name}: not in HF shards"));
            assert_eq!(xs.len() as i64, t.nelements(), "{hf_name}: element count");
            if llama && kind {
                let cols = *shape.last().unwrap() as usize;
                let rows = xs.len() / cols;
                permute_heads_fwd(&mut xs, heads as usize, rows, cols);
            }
            for (vi, &(id, rnd, vname)) in variants.iter().enumerate() {
                let ours = quantize_q8_0(&xs, id, rnd);
                let mism = ours.iter().zip(&stored).filter(|(a, b)| a != b).count() as u64;
                combo_mismatches[pi * variants.len() + vi] += mism;
                per_tensor.push(format!("{path_name}/{vname}={mism}"));
            }
        }
        eprintln!(
            "[{tag} iso] {} q8_0 mismatches: {}",
            t.name,
            per_tensor.join(" ")
        );
    }
    for (pi, &(_, path_name)) in paths.iter().enumerate() {
        for (vi, &(_, _, vname)) in variants.iter().enumerate() {
            eprintln!(
                "[{tag} iso] combo {path_name}/{vname}: {n_q8} Q8_0 tensors, mismatched bytes = {}",
                combo_mismatches[pi * variants.len() + vi]
            );
        }
    }
    assert!(n_q8 > 0, "{tag}: no Q8_0 tensors to prove");
    assert!(
        combo_mismatches.contains(&0),
        "{tag}: no quantize path reproduces the stored Q8_0 blocks bit-for-bit"
    );

    // Quantization-effect metric: same reference forward, HF vs GGUF
    // weights. Gate on top-1 equality only; cosine quantifies the
    // rounding the exact proof above already accounted for.
    let ids = anchor_ids(hf_dir);
    let la = mil_verify::reference::forward(&cfg, &w_hf, &ids).unwrap();
    let lb = mil_verify::reference::forward(&cfg, &g, &ids).unwrap();
    assert_eq!(la.len(), lb.len());
    let mut ok = true;
    for (p, (a, b)) in la.iter().zip(&lb).enumerate() {
        let c = cosine(a, b);
        let ta = argmax(a);
        let tb = argmax(b);
        let t5 = top_k(b, 5)
            .iter()
            .filter(|i| top_k(a, 5).contains(i))
            .count();
        println!(
            "{tag} pos {p}: cosine={c:.6} top1 hf={ta} gg={tb} top5 overlap={t5}/5 max|d|={:.4}",
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        );
        if ta != tb {
            ok = false;
        }
    }
    assert!(ok, "{tag}: top-1 diverges between HF and GGUF weights");
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn gguf_isolation_qwen3() {
    gguf_isolation(
        "qwen3",
        &cache().join("qwen3-hf"),
        &cache().join("qwen3-gguf-unsloth/Qwen3-0.6B-Q8_0.gguf"),
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn gguf_isolation_smollm2() {
    gguf_isolation(
        "smollm2",
        &cache().join("smollm2-hf"),
        &cache().join("smollm2-gguf/SmolLM2-135M-Instruct-Q8_0.gguf"),
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn gguf_isolation_qwen25() {
    gguf_isolation(
        "qwen25",
        &cache().join("qwen25-hf"),
        &cache().join("qwen25-gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
    );
}

/// Temp work dir that self-cleans unless `MIL_KEEP_ARTIFACTS=1`.
struct WorkDir(PathBuf);
impl WorkDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ref_e2e_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        WorkDir(p)
    }
}
impl Drop for WorkDir {
    fn drop(&mut self) {
        if std::env::var("MIL_KEEP_ARTIFACTS").is_err() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            println!("artifacts kept: {}", self.0.display());
        }
    }
}

/// `d`-dim embedding rows for `ids`, laid out for the package's `x`
/// input `(1, d, s, 1)` conv layout: `x[c*s + p] = emb[ids[p]][c]`.
fn embed_rows(w: &dyn mil_convert::WeightSource, ids: &[u32], d: usize) -> Vec<u8> {
    let (shape, bytes) = w.tensor_f16("model.embed_tokens.weight").unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[1] as usize, d);
    let s = ids.len();
    let mut x = vec![0u8; s * d * 2];
    for (p, &id) in ids.iter().enumerate() {
        let off = id as usize * d * 2;
        for c in 0..d {
            x[(c * s + p) * 2] = bytes[off + c * 2];
            x[(c * s + p) * 2 + 1] = bytes[off + c * 2 + 1];
        }
    }
    x
}

/// Convert → compile → run one s-token prefill on `ids`, compare
/// package logits to the reference at every position.
/// Pass: top-1 equal, top-5 overlap >= `min_top5`, and
/// cosine > `min_cos` at every position.
fn package_vs_reference(
    tag: &str,
    hf_dir: &Path,
    from_gguf: Option<&Path>,
    min_cos: f64,
    min_top5: usize,
) {
    let cfg = ModelConfig::from_json(&std::fs::read(hf_dir.join("config.json")).unwrap()).unwrap();
    let w_hf = hf_source(hf_dir);
    let ids = anchor_ids(hf_dir);
    let s = ids.len() as i64;

    let work = WorkDir::new(tag);
    let opts = mil_convert::Options {
        seq: s,
        max_kv: 64,
        quant: mil_convert::Quant::Fp16,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..mil_convert::Options::default()
    };
    let pkg = work.0.join("model.mlpackage");
    let compiled = work.0.join("model.compiled");
    match from_gguf {
        Some(g) => {
            mil_convert::convert_gguf(g, &pkg, &opts).expect("convert gguf");
        }
        None => {
            mil_convert::convert(hf_dir, &pkg, &opts).expect("convert hf");
        }
    }
    let mc = mil_compile::compile(&pkg, &compiled).expect("compile");

    // reference logits on HF weights (for the HF package) or on the
    // GGUF (for the GGUF package) — either way it isolates converter
    // error from weight-source difference. The `x` input comes from the
    // SAME weight source the reference reads, so a Q8_0 embed table is
    // fed identically to both sides.
    let gguf;
    let (ref_logits, x_src): (_, &dyn WeightSource) = match from_gguf {
        Some(g) => {
            gguf = Gguf::open(g).unwrap();
            (
                mil_verify::reference::forward(&cfg, &gguf, &ids).unwrap(),
                &gguf,
            )
        }
        None => (
            mil_verify::reference::forward(&cfg, &w_hf, &ids).unwrap(),
            &w_hf,
        ),
    };

    // ---- inputs per the builder contract ----
    let d = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let max_kv = opts.max_kv as usize;
    let x = embed_rows(x_src, &ids, d);
    // cos/sin (1,1,s,hd): HF rotate-half layout, freq = theta^{-2(c%hd/2)/hd}
    let mut cos = Vec::with_capacity(s as usize * hd * 2);
    let mut sin = Vec::with_capacity(s as usize * hd * 2);
    for p in 0..s as usize {
        for c in 0..hd {
            let f =
                (p as f64) * (cfg.rope_theta as f64).powf(-2.0 * (c % (hd / 2)) as f64 / hd as f64);
            cos.extend_from_slice(&half::f16::from_f64(f.cos()).to_le_bytes());
            sin.extend_from_slice(&half::f16::from_f64(f.sin()).to_le_bytes());
        }
    }
    // mask (1,1,s,max_kv): 0 where allowed (j <= p), -1e4 elsewhere
    let mut mask = Vec::with_capacity(s as usize * max_kv * 2);
    for p in 0..s as usize {
        for j in 0..max_kv {
            let v = if j <= p { 0.0f32 } else { -1e4 };
            mask.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    let pos = 0i32.to_le_bytes();
    let sh_x: [i64; 4] = [1, d as i64, s, 1];
    let sh_cs: [i64; 4] = [1, 1, s, hd as i64];
    let sh_m: [i64; 4] = [1, 1, s, max_kv as i64];
    let sh_p: [i64; 1] = [1];
    let inputs = vec![
        mil_infer::Input {
            name: "x",
            shape: &sh_x,
            data: &x,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "cos",
            shape: &sh_cs,
            data: &cos,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "sin",
            shape: &sh_cs,
            data: &sin,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "mask",
            shape: &sh_m,
            data: &mask,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "pos",
            shape: &sh_p,
            data: &pos,
            dtype: mil_spec::DType::Int32,
        },
    ];
    // The strict gate runs on CpuOnly — deterministic fp16 semantics —
    // while ComputeUnits::All is reported for context: the ANE adds its
    // own fp16 rounding on top of the same graph.
    let mut cpu_ok = true;
    for (cu_name, cu, strict) in [
        ("cpu", mil_infer::ComputeUnits::CpuOnly, true),
        ("all", mil_infer::ComputeUnits::All, false),
    ] {
        let m = mil_infer::Model::load(&mc.path, cu).expect("load");
        let st = m.new_state().expect("state");
        let pr = m.predict_with_state(Some(&st), &inputs).expect("predict");
        let out = pr
            .outputs
            .iter()
            .find(|o| o.name.contains("logits"))
            .unwrap_or(&pr.outputs[0]);
        if cu_name == "cpu" {
            println!("{tag}: output {} shape={:?}", out.name, out.shape);
        }
        let flat = out.values();
        let vocab = flat.len() / s as usize;
        // output (1, vocab, s, 1): element (v, p) at v*s + p
        let mut ok = true;
        for p in 0..s as usize {
            let col: Vec<f32> = (0..vocab).map(|v| flat[v * s as usize + p]).collect();
            let nan = col.iter().filter(|v| v.is_nan()).count();
            if nan > 0 {
                println!("{tag} {cu_name} pos {p}: {nan}/{vocab} NaN");
                ok = false;
                continue;
            }
            let c = cosine(&col, &ref_logits[p]);
            let ta = argmax(&col);
            let tr = argmax(&ref_logits[p]);
            let t5 = top_k(&col, 5)
                .iter()
                .filter(|i| top_k(&ref_logits[p], 5).contains(i))
                .count();
            println!(
                "{tag} {cu_name} pos {p}: cosine={c:.6} top1 pkg={ta} ref={tr} top5={t5}/5 max|d|={:.4}",
                col.iter()
                    .zip(&ref_logits[p])
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            );
            if ta != tr || t5 < min_top5 || c <= min_cos {
                ok = false;
            }
        }
        if strict {
            cpu_ok = ok;
        }
    }
    assert!(cpu_ok, "{tag}: package logits diverged from reference");
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn package_vs_reference_smollm2_hf() {
    package_vs_reference("smollm2-hf", &cache().join("smollm2-hf"), None, 0.999, 0);
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn package_vs_reference_smollm2_gguf() {
    package_vs_reference(
        "smollm2-gguf",
        &cache().join("smollm2-hf"),
        Some(&cache().join("smollm2-gguf/SmolLM2-135M-Instruct-Q8_0.gguf")),
        0.999,
        0,
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn package_vs_reference_qwen3_gguf() {
    package_vs_reference(
        "qwen3-gguf",
        &cache().join("qwen3-hf"),
        Some(&cache().join("qwen3-gguf-unsloth/Qwen3-0.6B-Q8_0.gguf")),
        0.999,
        0,
    );
}

// Qwen2.5 gate is evidence-based, not loosened blindly. Measured in
// `qwen25_attention_probe`: max|bias| q=79.0, k=214.0, v=11.4 (other
// models carry none); max |q.k/sqrt(hd)| at positions 0-1 is 980.7
// (layer 0) and 595.8 (layer 8), ~25x the qwen3/smollm2 maxima.
// fp16 rounding of q/k/scores alone produces only ~1e-5 deviation,
// but modeling the package's fp16 conv accumulation reproduces the
// observed ~1.3e-3 dip with the same early-position pattern (pos 0:
// 0.9943). A package depth-bisect shows the same deviation at depth
// 1 (~1.4e-3), and its worst position shifts with depth — magnitude-
// driven fp16 accumulation noise in an unusually-hot model, not a
// deterministic graph bug. Gate: top-1 equal everywhere, top-5
// overlap >= 4, cosine > 0.998.
#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn package_vs_reference_qwen25_hf() {
    package_vs_reference("qwen25-hf", &cache().join("qwen25-hf"), None, 0.998, 4);
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn package_vs_reference_qwen25_gguf() {
    package_vs_reference(
        "qwen25-gguf",
        &cache().join("qwen25-hf"),
        Some(&cache().join("qwen25-gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf")),
        0.998,
        4,
    );
}

/// Input-contract probe: feed every position the same embedding row.
/// Whatever the mask/rope do, identical hidden states → identical
/// attention output at every position → identical logits columns.
/// Any column difference proves an input-layout or graph bug.
#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn contract_probe_smollm2() {
    contract_probe("smollm2", &cache().join("smollm2-hf"));
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn contract_probe_qwen25() {
    contract_probe("qwen25", &cache().join("qwen25-hf"));
}

/// Input-contract invariant: feed every position the SAME embedding row.
/// V is never rotated, so all value vectors are identical; softmax
/// weights sum to 1, so every position's attention output equals that
/// same vector regardless of mask or rope — and every downstream column
/// must be identical. Any per-position difference proves an
/// input-layout, state-write, mask, or rope bug in the graph.
/// Pass: pairwise cosine > 0.999 and equal argmax at every position.
fn contract_probe(tag: &str, hf_dir: &Path) {
    let cfg = ModelConfig::from_json(&std::fs::read(hf_dir.join("config.json")).unwrap()).unwrap();
    let w_hf = hf_source(hf_dir);
    let s: i64 = 5;
    let work = WorkDir::new(&format!("probe_{tag}"));
    let opts = mil_convert::Options {
        seq: s,
        max_kv: 64,
        quant: mil_convert::Quant::Fp16,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..mil_convert::Options::default()
    };
    let pkg = work.0.join("m.mlpackage");
    let compiled = work.0.join("m.compiled");
    mil_convert::convert(hf_dir, &pkg, &opts).unwrap();
    let cm = mil_compile::compile(&pkg, &compiled).unwrap();

    let d = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let max_kv = opts.max_kv as usize;
    let (shape, emb) = w_hf.tensor_f16("model.embed_tokens.weight").unwrap();
    assert_eq!(shape[1] as usize, d);
    let row = &emb[123 * d * 2..124 * d * 2]; // token 123's embedding

    // x in the package's (1, d, s, 1) channel layout: x[c*s + p]
    let mut x = vec![0u8; s as usize * d * 2];
    for p in 0..s as usize {
        for c in 0..d {
            x[(c * s as usize + p) * 2] = row[c * 2];
            x[(c * s as usize + p) * 2 + 1] = row[c * 2 + 1];
        }
    }
    let mut cos = Vec::with_capacity(s as usize * hd * 2);
    let mut sin = Vec::with_capacity(s as usize * hd * 2);
    for p in 0..s as usize {
        for c in 0..hd {
            let f =
                (p as f64) * (cfg.rope_theta as f64).powf(-2.0 * (c % (hd / 2)) as f64 / hd as f64);
            cos.extend_from_slice(&half::f16::from_f64(f.cos()).to_le_bytes());
            sin.extend_from_slice(&half::f16::from_f64(f.sin()).to_le_bytes());
        }
    }
    let mut mask = Vec::with_capacity(s as usize * max_kv * 2);
    for p in 0..s as usize {
        for j in 0..max_kv {
            let v = if j <= p { 0.0f32 } else { -1e4 };
            mask.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    let pos = 0i32.to_le_bytes();
    let sh_x: [i64; 4] = [1, d as i64, s, 1];
    let sh_cs: [i64; 4] = [1, 1, s, hd as i64];
    let sh_m: [i64; 4] = [1, 1, s, max_kv as i64];
    let sh_p: [i64; 1] = [1];
    let inputs = vec![
        mil_infer::Input {
            name: "x",
            shape: &sh_x,
            data: &x,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "cos",
            shape: &sh_cs,
            data: &cos,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "sin",
            shape: &sh_cs,
            data: &sin,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "mask",
            shape: &sh_m,
            data: &mask,
            dtype: mil_spec::DType::Fp16,
        },
        mil_infer::Input {
            name: "pos",
            shape: &sh_p,
            data: &pos,
            dtype: mil_spec::DType::Int32,
        },
    ];
    let mut ok = true;
    // Strict gate on CpuOnly (deterministic fp16 — a column difference
    // there is a graph bug); All is reported for context since the ANE
    // adds its own rounding on top of the same graph.
    for (cu_name, cu, strict) in [
        ("cpu", mil_infer::ComputeUnits::CpuOnly, true),
        ("all", mil_infer::ComputeUnits::All, false),
    ] {
        let mc = mil_infer::Model::load(&cm.path, cu).unwrap();
        let st = mc.new_state().unwrap();
        let pr = mc.predict_with_state(Some(&st), &inputs).unwrap();
        let out = pr
            .outputs
            .iter()
            .find(|o| o.name.contains("logits"))
            .unwrap();
        let flat = out.values();
        let vocab = flat.len() / s as usize;
        let cols: Vec<Vec<f32>> = (0..s as usize)
            .map(|p| (0..vocab).map(|v| flat[v * s as usize + p]).collect())
            .collect();
        for (p, col) in cols.iter().enumerate().skip(1) {
            let c = cosine(col, &cols[0]);
            let (t0, tp) = (argmax(&cols[0]), argmax(col));
            println!("{cu_name} col{p} vs col0: cosine={c:.6} top1 {tp} vs {t0}");
            // NaN cosine is not a pass — it means the graph produced NaNs.
            if strict && (!c.is_finite() || c <= 0.999 || tp != t0) {
                ok = false;
            }
        }
        let nans: usize = cols.iter().flatten().filter(|v| v.is_nan()).count();
        assert_eq!(nans, 0, "identical inputs produced NaN logits");
    }
    assert!(ok, "identical inputs produced different columns");
}

/// fp16 precision probe for the package-vs-reference cosine dip.
/// Measures max|q/k/v bias|, per-layer max |q.k/sqrt(hd)| at query
/// positions 0-1, and the cosine of three fp16 reference variants
/// against the f32 reference: `fp16attn` (q/k/scores rounded only),
/// `fp16mac` (+ fp16 attention accumulation), `fp16all` (+ fp16
/// accumulation in every projection/MLP conv). If a variant
/// reproduces the ~1e-3 package dip, the dip is model-side fp16
/// sensitivity, not a converter bug.
fn attention_probe(tag: &str, hf_dir: &Path) {
    let cfg = ModelConfig::from_json(&std::fs::read(hf_dir.join("config.json")).unwrap()).unwrap();
    let w = hf_source(hf_dir);
    let ids = anchor_ids(hf_dir);

    for proj in ["q_proj", "k_proj", "v_proj"] {
        let mut mx = 0f32;
        for l in 0..cfg.num_layers {
            let name = format!("model.layers.{l}.self_attn.{proj}.bias");
            if !w.has(&name) {
                continue;
            }
            let (_, bytes) = w.tensor_f16(&name).unwrap();
            for b in bytes.chunks_exact(2) {
                mx = mx.max(half::f16::from_le_bytes([b[0], b[1]]).to_f32().abs());
            }
        }
        println!("{tag} max|{proj}.bias| = {mx:.4}");
    }

    let (la, qk) = mil_verify::reference::forward_probe_qk(&cfg, &w, &ids).unwrap();
    let mut worst = 0f64;
    for (l, m) in qk.iter().enumerate() {
        worst = worst.max(*m);
        println!("{tag} layer {l}: max|q.k| pos0-1 = {m:.3}");
    }
    println!("{tag} max|q.k| pos0-1 over layers = {worst:.3}");

    type Fwd = fn(&ModelConfig, &dyn WeightSource, &[u32]) -> std::io::Result<Vec<Vec<f32>>>;
    let variants: [(&str, Fwd); 3] = [
        ("fp16attn", mil_verify::reference::forward_attn_fp16),
        ("fp16mac", mil_verify::reference::forward_attn_fp16_mac),
        ("fp16all", mil_verify::reference::forward_fp16_all),
    ];
    for (name, f) in variants {
        let lb = f(&cfg, &w, &ids).unwrap();
        for (p, (a, b)) in la.iter().zip(&lb).enumerate() {
            let c = cosine(a, b);
            let (ta, tb) = (argmax(a), argmax(b));
            println!("{tag} {name} pos {p}: cosine vs f32 ref = {c:.6} top1 {ta} vs {tb}");
        }
    }

    // Depth scaling of the fp16all variant — compare against the
    // package depth-bisection to test whether per-layer fp16
    // accumulation predicts the same error growth.
    for nl in [1usize, 4] {
        let mut c1 = cfg.clone();
        c1.num_layers = nl;
        let (la1, _) = mil_verify::reference::forward_probe_qk(&c1, &w, &ids).unwrap();
        let lb1 = mil_verify::reference::forward_fp16_all(&c1, &w, &ids).unwrap();
        for (p, (a, b)) in la1.iter().zip(&lb1).enumerate() {
            let c = cosine(a, b);
            println!("{tag} fp16all-L{nl} pos {p}: cosine = {c:.6}");
        }
    }
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn qwen25_attention_probe() {
    attention_probe("qwen25", &cache().join("qwen25-hf"));
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn qwen3_attention_probe() {
    attention_probe("qwen3", &cache().join("qwen3-hf"));
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn smollm2_attention_probe() {
    attention_probe("smollm2", &cache().join("smollm2-hf"));
}
