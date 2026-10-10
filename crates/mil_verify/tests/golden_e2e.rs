//! Golden-vector e2e — `#[ignore]`d: needs the cached models under
//! `~/.cache/mil_gguf_test/models/` and `coremlc`. Run explicitly:
//!
//! ```sh
//! cargo test -p mil_verify --test golden_e2e --release -- --ignored --nocapture
//! ```
//!
//! Flow per model: generate golden records from the reference forward
//! (seq-1 and a 5-token prompt), convert + compile a package at the
//! same sequence lengths, predict on `CpuOnly`, and check the package
//! against the record. Then the two differentials:
//!
//! - **CPU vs default units** — the same package on `CpuOnly` vs
//!   `ComputeUnits::All`; per-position deltas are reported and top-1
//!   agreement is asserted.
//! - **Corruption control** — the same package with one flipped
//!   `weight.bin` byte must FAIL the golden check. If it ever passed,
//!   the check would be decorative.
//!
//! Disk discipline: an fp16 qwen3 package + compiled pair is ~2.5 GB,
//! so tests serialize on `SERIAL`, delete each `.mlpackage` as soon as
//! its `.mlmodelc` exists (the compiled form embeds the weights), and
//! delete compiled dirs once their predictions are taken.

use mil_convert::safetensors::{self, Safetensors};
use mil_convert::{ModelConfig, WeightSource};
use mil_verify::golden::{self, Golden, Tolerances};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One heavyweight artifact chain at a time — parallel converts would
/// need ~5 GB of transient package+compiled space.
static SERIAL: Mutex<()> = Mutex::new(());

fn cache() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models")
}

/// Temp work dir that self-cleans unless `MIL_KEEP_ARTIFACTS=1`.
struct WorkDir(PathBuf);
impl WorkDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("golden_e2e_{tag}_{}", std::process::id()));
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

fn keep_artifacts() -> bool {
    std::env::var("MIL_KEEP_ARTIFACTS").is_ok()
}

/// Same tokenizer scan as `reference_e2e`: `"<token>":<id>` in a flat
/// HF `tokenizer.json` vocab object.
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

/// "The capital of France is" — the shared multi-token prompt.
fn prompt_ids(hf_dir: &Path) -> Vec<u32> {
    let tj = std::fs::read(hf_dir.join("tokenizer.json")).expect("tokenizer.json");
    ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
        .iter()
        .map(|t| token_id(&tj, t))
        .collect()
}

fn hf_source(dir: &Path) -> Vec<Safetensors> {
    safetensors::open_dir(dir).expect("open safetensors")
}

fn hf_config(dir: &Path) -> ModelConfig {
    ModelConfig::from_json(&std::fs::read(dir.join("config.json")).unwrap()).unwrap()
}

/// Convert an HF dir at sequence length `seq` and compile. The
/// `.mlpackage` is deleted once the `.mlmodelc` exists — the compiled
/// form embeds the weights, and keeping both doubles the disk cost.
/// (`MIL_KEEP_ARTIFACTS=1` keeps it.)
fn build_pkg(tag: &str, hf_dir: &Path, seq: i64, work: &WorkDir) -> PathBuf {
    let opts = mil_convert::Options {
        seq,
        max_kv: 64,
        quant: mil_convert::Quant::Fp16,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..mil_convert::Options::default()
    };
    let pkg = work.0.join(format!("{tag}_s{seq}.mlpackage"));
    mil_convert::convert(hf_dir, &pkg, &opts).expect("convert");
    let compiled = work.0.join(format!("{tag}_s{seq}.compiled"));
    mil_compile::compile(&pkg, &compiled).expect("compile");
    if !keep_artifacts() {
        let _ = std::fs::remove_dir_all(&pkg);
    }
    compiled.join(format!("{tag}_s{seq}.mlmodelc"))
}

/// Predict `ids` through a compiled package under `cu`; per-position
/// logit columns out.
fn predict_cols(
    compiled: &Path,
    cu: mil_infer::ComputeUnits,
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
    max_kv: usize,
) -> Vec<Vec<f32>> {
    let cols = try_predict_cols(compiled, cu, cfg, w, ids, max_kv).expect("predict");
    assert_eq!(cols.len(), ids.len());
    cols
}

/// Fallible form of [`predict_cols`] — lets the units differential
/// cascade past a plan-build failure on one compute-unit set.
fn try_predict_cols(
    compiled: &Path,
    cu: mil_infer::ComputeUnits,
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
    max_kv: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let prepped = golden::package_inputs(cfg, w, ids, max_kv).map_err(|e| e.to_string())?;
    let inputs: Vec<mil_infer::Input> = prepped
        .iter()
        .map(|p| mil_infer::Input {
            name: &p.name,
            shape: &p.shape,
            data: &p.data,
            dtype: p.dtype,
        })
        .collect();
    let model = mil_infer::Model::load(compiled, cu).map_err(|e| e.to_string())?;
    let state = model.new_state().map_err(|e| e.to_string())?;
    let pr = model
        .predict_with_state(Some(&state), &inputs)
        .map_err(|e| e.to_string())?;
    let out = pr
        .outputs
        .iter()
        .find(|o| o.name.contains("logits"))
        .unwrap_or(&pr.outputs[0]);
    Ok(golden::columns_from_output(&out.values(), ids.len()))
}

/// First unit set in `candidates` that loads and predicts on
/// `compiled`, with its results — `None` if every one fails.
fn first_working_units(
    compiled: &Path,
    candidates: &[(mil_infer::ComputeUnits, &'static str)],
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Option<(mil_infer::ComputeUnits, &'static str, Vec<Vec<f32>>)> {
    for &(cu, name) in candidates {
        match try_predict_cols(compiled, cu, cfg, w, ids, 64) {
            Ok(cols) if !cols.is_empty() => return Some((cu, name, cols)),
            Ok(_) => eprintln!("units {name}: empty output"),
            Err(e) => eprintln!("units {name}: {e}"),
        }
    }
    None
}

/// Release a compiled dir once its predictions are taken.
fn drop_compiled(path: &Path) {
    if !keep_artifacts() {
        let _ = std::fs::remove_dir_all(path);
    }
}

/// Generate a TopK-32 golden for `ids` from the HF weight source.
fn make_golden(
    tag: &str,
    hf_dir: &Path,
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Golden {
    let (sha, desc) = golden::hash_model_source(hf_dir).expect("hash weights");
    let prov = golden::Provenance::new(
        tag,
        desc,
        sha,
        format!("mil_verify {}", env!("CARGO_PKG_VERSION")),
    );
    Golden::from_reference(&prov, cfg, w, ids.to_vec(), golden::DEFAULT_TOP_K)
        .expect("reference forward")
}

/// The shared gate — same numbers as `Tolerances::default`, asserted
/// here so the e2e and the API can't drift apart.
fn gate() -> Tolerances {
    Tolerances::default()
}

/// One model end to end: goldens at seq 1 and the 5-token prompt,
/// packages at the same seqs, golden check on CpuOnly, then the
/// CPU-vs-default differential on the multi-token package.
fn golden_flow(tag: &str, hf_dir: &Path) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = hf_config(hf_dir);
    let w = hf_source(hf_dir);
    let work = WorkDir::new(tag);
    let prompt = prompt_ids(hf_dir);
    let seq1 = [prompt[0]];

    // ---- generate ----
    let g1 = make_golden(&format!("{tag}-s1"), hf_dir, &cfg, &w, &seq1);
    let g5 = make_golden(&format!("{tag}-s5"), hf_dir, &cfg, &w, &prompt);
    // Persist + reload: the on-disk form is the artifact the harness
    // actually checks against.
    let p1 = work.0.join(format!("{tag}-s1.mgv"));
    let p5 = work.0.join(format!("{tag}-s5.mgv"));
    g1.write(&p1).unwrap();
    g5.write(&p5).unwrap();
    println!("{} -> {}", g1.summary(), p1.display());
    println!("{} -> {}", g5.summary(), p5.display());
    let g1 = Golden::read(&p1).unwrap();
    let g5 = Golden::read(&p5).unwrap();
    println!(
        "record sizes: {} B (seq1), {} B (seq5)",
        std::fs::metadata(&p1).unwrap().len(),
        std::fs::metadata(&p5).unwrap().len()
    );

    // ---- check converted packages ----
    // seq1 first, compiled dir released before the seq5 build starts —
    // each compiled model is ~1.2 GB on qwen3.
    let compiled1 = build_pkg(tag, hf_dir, 1, &work);
    let cols = predict_cols(
        &compiled1,
        mil_infer::ComputeUnits::CpuOnly,
        &cfg,
        &w,
        &seq1,
        64,
    );
    let rep = g1.check(&cols, &gate());
    println!("{tag} seq1 cpu:\n{rep}");
    assert!(rep.is_ok(), "{tag} seq1: golden check failed");
    drop_compiled(&compiled1);
    let compiled5 = build_pkg(tag, hf_dir, 5, &work);
    let cols = predict_cols(
        &compiled5,
        mil_infer::ComputeUnits::CpuOnly,
        &cfg,
        &w,
        &prompt,
        64,
    );
    let rep = g5.check(&cols, &gate());
    println!("{tag} seq5 cpu:\n{rep}");
    assert!(rep.is_ok(), "{tag} seq5: golden check failed");

    // ---- CPU vs default-units differential on the 5-token package ----
    // `All` is the intended "default" column; when the ANE plan won't
    // build on this machine (error -14 on the ~1.2 GB fp16 package)
    // the differential falls back to GPU then ANE-only, so it always
    // measures two real backends instead of skipping. The assert is
    // the same either way: greedy top-1 must agree per position.
    let cpu = cols; // the CpuOnly columns already taken for the golden check
    let (other_cu, other_name, other) = first_working_units(
        &compiled5,
        &[
            (mil_infer::ComputeUnits::All, "all"),
            (mil_infer::ComputeUnits::CpuAndGpu, "gpu"),
            (mil_infer::ComputeUnits::CpuAndNeuralEngine, "ane"),
        ],
        &cfg,
        &w,
        &prompt,
    )
    .unwrap_or_else(|| panic!("{tag}: no non-CPU unit could run the package"));
    let _ = other_cu;
    let diffs = golden::diff_columns(&cpu, &other);
    let mut worst = 0f32;
    for d in &diffs {
        worst = worst.max(d.max_abs);
        println!(
            "{tag} cpu-vs-{other_name} pos {}: max|d|={:.5} mean|d|={:.6} cosine={:.6} top1 {}vs{}",
            d.pos, d.max_abs, d.mean_abs, d.cosine, d.top1_a, d.top1_b
        );
        assert_eq!(
            d.top1_a, d.top1_b,
            "{tag} pos {}: CPU and {other_name} disagree on top-1",
            d.pos
        );
    }
    println!("{tag} cpu-vs-{other_name} worst max|d| = {worst:.5}");
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn golden_qwen3() {
    golden_flow("qwen3", &cache().join("qwen3-hf"));
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn golden_smollm2() {
    golden_flow("smollm2", &cache().join("smollm2-hf"));
}

/// Planted control: corrupt one byte inside the first `weight.bin`
/// blob (the high byte of its first fp16 element — a wild value, not
/// a quiet ULP), rename the package so the compiled-model cache can't
/// serve the pre-corruption build, recompile, and require the golden
/// check to fail. If this test ever passes trivially the check is
/// broken.
#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads + coremlc"]
fn corrupt_package_fails_golden() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let tag = "qwen3-corrupt";
    let hf_dir = cache().join("qwen3-hf");
    let cfg = hf_config(&hf_dir);
    let w = hf_source(&hf_dir);
    let work = WorkDir::new(tag);
    let ids = [prompt_ids(&hf_dir)[0]];

    let g = make_golden(tag, &hf_dir, &cfg, &w, &ids);

    // Convert + compile, keeping the package this time (it gets
    // corrupted in place — renaming the dir beats copying ~1.2 GB).
    let opts = mil_convert::Options {
        seq: 1,
        max_kv: 64,
        quant: mil_convert::Quant::Fp16,
        lm_head: true,
        embed: false,
        spec_version: 10,
        opset: "CoreML9".into(),
        lora: None,
        ..mil_convert::Options::default()
    };
    let good_pkg = work.0.join("good.mlpackage");
    mil_convert::convert(&hf_dir, &good_pkg, &opts).expect("convert");
    let good_out = work.0.join("good.compiled");
    let good_compiled = good_out.join("good.mlmodelc");
    mil_compile::compile(&good_pkg, &good_out).expect("compile");

    // Baseline must pass first — otherwise the control proves nothing.
    let cols = predict_cols(
        &good_compiled,
        mil_infer::ComputeUnits::CpuOnly,
        &cfg,
        &w,
        &ids,
        64,
    );
    let rep = g.check(&cols, &gate());
    println!("baseline:\n{rep}");
    assert!(rep.is_ok(), "baseline package failed the golden check");
    drop_compiled(&good_compiled);

    // Flip the high byte of the first fp16 element in the first blob's
    // data region: storage header 64 B, then a 64 B record whose
    // data_off field (record+16) points at the payload.
    let wpath = good_pkg.join("Data/com.apple.CoreML/weights/weight.bin");
    let mut wbytes = std::fs::read(&wpath).unwrap();
    let data_off = u64::from_le_bytes(wbytes[80..88].try_into().unwrap()) as usize;
    assert!(data_off + 1 < wbytes.len());
    let victim = data_off + 1;
    let before = wbytes[victim];
    wbytes[victim] ^= 0xFF;
    std::fs::write(&wpath, &wbytes).unwrap();
    drop(wbytes);
    println!(
        "corrupt: weight.bin[{victim}] {before:#04x} -> {:#04x}",
        before ^ 0xFF
    );

    // Rename (not copy): a new package URL means CoreML's compiled-model
    // cache cannot hand back the pre-corruption build.
    let bad_pkg = work.0.join("corrupt.mlpackage");
    std::fs::rename(&good_pkg, &bad_pkg).unwrap();
    let bad_out = work.0.join("corrupt.compiled");
    mil_compile::compile(&bad_pkg, &bad_out).expect("compile corrupt");
    if !keep_artifacts() {
        let _ = std::fs::remove_dir_all(&bad_pkg);
    }
    let bad_cols = predict_cols(
        &bad_out.join("corrupt.mlmodelc"),
        mil_infer::ComputeUnits::CpuOnly,
        &cfg,
        &w,
        &ids,
        64,
    );
    let rep = g.check(&bad_cols, &gate());
    println!("corrupted:\n{rep}");
    assert!(
        !rep.is_ok(),
        "corrupted package PASSED the golden check — the check is decorative"
    );
}
