//! `golden` — CLI harness for the golden-vector workflow (the library
//! API is `mil_verify::golden`; milc wiring lives elsewhere).
//!
//! ```sh
//! cargo run -p mil_verify --release --example golden -- \
//!   gen <weights> <ids-csv|@file> <out.mgv> [--k N | --full] [--id NAME]
//! cargo run -p mil_verify --release --example golden -- \
//!   check <compiled.mlmodelc> <golden.mgv> <weights>
//!   [--units cpu|all|gpu|ane] [--max-kv N]
//!   [--tol-delta F] [--min-overlap N] [--no-top1]
//! cargo run -p mil_verify --release --example golden -- \
//!   diff <compiled.mlmodelc> <weights> <ids-csv|@file> [--max-kv N]
//! ```
//!
//! `<weights>` is an HF model dir (`config.json` + `*.safetensors`)
//! or a `.gguf` file — the same source the golden was generated from,
//! since `gen` records its SHA-256. `check` needs it to rebuild the
//! package inputs; `diff` predicts the same compiled package on
//! `CpuOnly` and `All` and reports the per-position delta.

use mil_convert::config::ModelConfig;
use mil_convert::gguf::Gguf;
use mil_convert::safetensors;
use mil_convert::WeightSource;
use mil_verify::golden::{self, Golden, Tolerances};
use std::path::{Path, PathBuf};
use std::process::exit;

fn usage() -> ! {
    eprintln!(
        "usage:\n  golden gen <weights> <ids-csv|@file> <out.mgv> [--k N|--full] [--id NAME]\n  \
         golden check <compiled.mlmodelc> <golden.mgv> <weights> [--units U] [--max-kv N] [--tol-delta F] [--min-overlap N] [--no-top1]\n  \
         golden diff <compiled.mlmodelc> <weights> <ids-csv|@file> [--max-kv N]"
    );
    exit(2);
}

/// `(config, weight source)` for an HF dir or a `.gguf` file.
fn open_weights(path: &Path) -> (ModelConfig, Box<dyn WeightSource>) {
    if path.is_file() {
        let g = Gguf::open(path).unwrap_or_else(|e| {
            eprintln!("open {}: {e}", path.display());
            exit(1)
        });
        let cfg = mil_convert::gguf::config::model_config(&g).unwrap_or_else(|e| {
            eprintln!("gguf config: {e}");
            exit(1)
        });
        (cfg, Box::new(g))
    } else {
        let cfg = ModelConfig::from_json(
            &std::fs::read(path.join("config.json")).expect("read config.json"),
        )
        .expect("parse config.json");
        let st = safetensors::open_dir(path).expect("open safetensors");
        (cfg, Box::new(st))
    }
}

fn parse_ids(arg: &str) -> Vec<u32> {
    let text = match arg.strip_prefix('@') {
        Some(f) => std::fs::read_to_string(f).expect("read ids file"),
        None => arg.to_string(),
    };
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("token id"))
        .collect()
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn units(s: &str) -> mil_infer::ComputeUnits {
    match s {
        "cpu" => mil_infer::ComputeUnits::CpuOnly,
        "all" => mil_infer::ComputeUnits::All,
        "gpu" => mil_infer::ComputeUnits::CpuAndGpu,
        "ane" => mil_infer::ComputeUnits::CpuAndNeuralEngine,
        _ => usage(),
    }
}

/// Predict `ids` through a compiled package, returning per-position
/// logit columns.
fn predict_cols(
    compiled: &Path,
    cu: mil_infer::ComputeUnits,
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
    max_kv: usize,
) -> Vec<Vec<f32>> {
    let prepped = golden::package_inputs(cfg, w, ids, max_kv).expect("package inputs");
    let inputs: Vec<mil_infer::Input> = prepped
        .iter()
        .map(|p| mil_infer::Input {
            name: &p.name,
            shape: &p.shape,
            data: &p.data,
            dtype: p.dtype,
        })
        .collect();
    let model = mil_infer::Model::load(compiled, cu).expect("load compiled model");
    let state = model.new_state().expect("new state");
    let pr = model
        .predict_with_state(Some(&state), &inputs)
        .expect("predict");
    let out = pr
        .outputs
        .iter()
        .find(|o| o.name.contains("logits"))
        .unwrap_or(&pr.outputs[0]);
    let cols = golden::columns_from_output(&out.values(), ids.len());
    if cols.is_empty() {
        eprintln!(
            "output {} shape {:?} doesn't split into {} positions",
            out.name,
            out.shape,
            ids.len()
        );
        exit(1);
    }
    cols
}

fn cmd_gen(args: &[String]) {
    if args.len() < 3 {
        usage();
    }
    let weights = PathBuf::from(&args[0]);
    let ids = parse_ids(&args[1]);
    let out = PathBuf::from(&args[2]);
    let k: usize = flag(args, "--k")
        .map(|s| s.parse().unwrap())
        .unwrap_or(golden::DEFAULT_TOP_K);
    let full = has(args, "--full");
    let id = flag(args, "--id").map(String::from).unwrap_or_else(|| {
        weights
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    });
    let (cfg, w) = open_weights(&weights);
    let (sha, desc) = golden::hash_model_source(&weights).expect("hash weights");
    eprintln!(
        "hashing {} → {} ({desc})",
        weights.display(),
        golden::hex(&sha)
    );
    let toolchain = format!("mil_verify {}", env!("CARGO_PKG_VERSION"));
    let prov = golden::Provenance::new(&id, desc, sha, &toolchain);
    let g = if full {
        let logits = mil_verify::reference::forward(&cfg, &*w, &ids).expect("reference forward");
        Golden::from_logits_full(&prov, &cfg, ids, logits)
    } else {
        Golden::from_reference(&prov, &cfg, &*w, ids, k).expect("reference forward")
    };
    g.write(&out).expect("write golden");
    println!("{} -> {}", g.summary(), out.display());
}

fn cmd_check(args: &[String]) {
    if args.len() < 3 {
        usage();
    }
    let compiled = PathBuf::from(&args[0]);
    let g = Golden::read(Path::new(&args[1])).expect("read golden");
    let weights = PathBuf::from(&args[2]);
    let cu = flag(args, "--units")
        .map(units)
        .unwrap_or(mil_infer::ComputeUnits::CpuOnly);
    let max_kv: usize = flag(args, "--max-kv")
        .map(|s| s.parse().unwrap())
        .unwrap_or(64);
    let mut tol = Tolerances::default();
    if let Some(v) = flag(args, "--tol-delta") {
        tol.max_val_delta = v.parse().unwrap();
    }
    if let Some(v) = flag(args, "--min-overlap") {
        tol.min_overlap = v.parse().unwrap();
    }
    if has(args, "--no-top1") {
        tol.top1 = false;
    }
    eprintln!("golden: {}", g.summary());
    let (cfg, w) = open_weights(&weights);
    if let Some(desc) = weights.file_name().and_then(|s| s.to_str()) {
        if !g.source_desc.contains(desc) {
            eprintln!(
                "warning: weight source {desc} not in golden source_desc {:?} — check hashes",
                g.source_desc
            );
        }
    }
    let (sha, _) = golden::hash_model_source(&weights).expect("hash weights");
    if sha != g.source_sha256 {
        eprintln!("warning: weight source SHA-256 differs from the record's");
    }
    let cols = predict_cols(&compiled, cu, &cfg, &*w, &g.token_ids, max_kv);
    let rep = g.check(&cols, &tol);
    println!("{rep}");
    exit(if rep.is_ok() { 0 } else { 1 });
}

fn cmd_diff(args: &[String]) {
    if args.len() < 3 {
        usage();
    }
    let compiled = PathBuf::from(&args[0]);
    let weights = PathBuf::from(&args[1]);
    let ids = parse_ids(&args[2]);
    let max_kv: usize = flag(args, "--max-kv")
        .map(|s| s.parse().unwrap())
        .unwrap_or(64);
    let (cfg, w) = open_weights(&weights);
    let cpu = predict_cols(
        &compiled,
        mil_infer::ComputeUnits::CpuOnly,
        &cfg,
        &*w,
        &ids,
        max_kv,
    );
    let all = predict_cols(
        &compiled,
        mil_infer::ComputeUnits::All,
        &cfg,
        &*w,
        &ids,
        max_kv,
    );
    let diffs = golden::diff_columns(&cpu, &all);
    let mut agree = true;
    for d in &diffs {
        let same = if d.top1_a == d.top1_b { "same" } else { "DIFF" };
        if d.top1_a != d.top1_b {
            agree = false;
        }
        println!(
            "pos {}: max|d|={:.5} mean|d|={:.6} cosine={:.6} top1 {}vs{} {same}",
            d.pos, d.max_abs, d.mean_abs, d.cosine, d.top1_a, d.top1_b
        );
    }
    println!("top-1 agreement: {}", if agree { "yes" } else { "NO" });
    exit(if agree { 0 } else { 1 });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "gen" => cmd_gen(&args[1..]),
        "check" => cmd_check(&args[1..]),
        "diff" => cmd_diff(&args[1..]),
        _ => usage(),
    }
}
