//! ANE-vs-CPU numerics on truncated SmolLM2 packages: is the Neural
//! Engine's answer the CPU's answer? Prints one row per
//! (layers, head) with NaN counts, max|Δ| and cosine between the
//! CpuOnly and CpuAndNeuralEngine outputs of the *same* package.
//!
//! `cargo test -p mil_verify --test ane_numerics --release -- --ignored --nocapture`

use mil_convert::safetensors;
use mil_convert::{ModelConfig, Options, Quant};
use mil_infer::{ComputeUnits, Input, Model};
use std::path::{Path, PathBuf};

fn smol() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models/smollm2-hf")
}

fn run(
    pkg_dir: &Path,
    compiled: &Path,
    ins: &[mil_verify::golden::PreparedInput],
    units: ComputeUnits,
) -> Result<Vec<f32>, String> {
    let m = Model::load(compiled, units).map_err(|e| e.to_string())?;
    let st = m.new_state().map_err(|e| e.to_string())?;
    let feeds: Vec<Input> = ins
        .iter()
        .map(|p| Input {
            name: &p.name,
            shape: &p.shape,
            data: &p.data,
            dtype: p.dtype,
        })
        .collect();
    let _ = pkg_dir;
    let pr = m
        .predict_with_state(Some(&st), &feeds)
        .map_err(|e| e.to_string())?;
    Ok(pr.outputs[0].values())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt() + 1e-30)
}

#[test]
#[ignore = "diagnostic: needs coremlc + ~/.cache/mil_gguf_test models"]
fn ane_vs_cpu_truncated_smollm2() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("ane_num_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let rows: Vec<(usize, bool)> = std::env::var("MIL_ANE_ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|t| {
                    let (l, h) = t.split_once(':').unwrap();
                    (l.parse().unwrap(), h == "h")
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(1, false), (1, true), (4, false), (4, true), (30, true)]);
    for (nl, head) in rows {
        let dir = root.join(format!("m{nl}"));
        std::fs::create_dir_all(&dir).unwrap();
        for e in std::fs::read_dir(smol()).unwrap().flatten() {
            if e.path().extension().and_then(|x| x.to_str()) == Some("safetensors") {
                std::os::unix::fs::symlink(e.path(), dir.join(e.file_name())).unwrap();
            }
        }
        let cfg_txt = std::fs::read_to_string(smol().join("config.json")).unwrap();
        let cfg_txt = regex_free_replace_layers(&cfg_txt, nl);
        std::fs::write(dir.join("config.json"), &cfg_txt).unwrap();
        let cfg = ModelConfig::from_json(cfg_txt.as_bytes()).unwrap();
        let w = safetensors::open_dir(&dir).unwrap();
        let all_ids: Vec<u32> = (0..64u32).map(|i| 500 + i * 37 % 4000).collect();
        let seq: usize = std::env::var("MIL_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let ids = &all_ids[..seq];
        let opts = Options {
            seq: std::env::var("MIL_SEQ")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            max_kv: 64,
            quant: Quant::Fp16,
            lm_head: head,
            embed: false,
            ..Options::default()
        };
        let pkg = root.join("p.mlpackage");
        mil_convert::convert(&dir, &pkg, &opts).unwrap();
        let comp = mil_compile::compile(&pkg, &root.join("c")).unwrap();
        let _ = std::fs::remove_dir_all(&pkg);
        let ins = mil_verify::golden::package_inputs(&cfg, &w, &ids, 64).unwrap();
        let cpu = run(&pkg, &comp.path, &ins, ComputeUnits::CpuOnly);
        let ane = run(&pkg, &comp.path, &ins, ComputeUnits::CpuAndNeuralEngine);
        match (cpu, ane) {
            (Ok(c), Ok(a)) => {
                let nan_a = a.iter().filter(|v| !v.is_finite()).count();
                let nan_c = c.iter().filter(|v| !v.is_finite()).count();
                let maxd = c
                    .iter()
                    .zip(&a)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                let amax = a.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
                println!(
                    "ANENUM L={nl:<2} head={head:<5} n={} nonfinite cpu={nan_c} ane={nan_a} max|Δ|={maxd:.4} ane max|v|={amax:.3} cosine={:.6}",
                    c.len(),
                    cosine(&c, &a)
                );
            }
            (c, a) => println!("ANENUM L={nl} head={head}: cpu={c:?} ane={:?}", a.err()),
        }
        let _ = std::fs::remove_dir_all(root.join("c"));
        let _ = std::fs::remove_dir_all(&dir);
    }
    let _ = std::fs::remove_dir_all(&root);
}

fn regex_free_replace_layers(cfg: &str, n: usize) -> String {
    let key = "\"num_hidden_layers\"";
    let at = cfg.find(key).unwrap() + key.len();
    let colon = cfg[at..].find(':').unwrap() + at + 1;
    let mut s = colon;
    while cfg.as_bytes()[s] == b' ' {
        s += 1;
    }
    let mut e = s;
    while cfg.as_bytes()[e].is_ascii_digit() {
        e += 1;
    }
    format!("{}{}{}", &cfg[..s], n, &cfg[e..])
}
