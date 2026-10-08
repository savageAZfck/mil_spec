//! `milc` — the mil_spec toolchain CLI.
//!
//! ```text
//! milc convert <model_dir> -o <pkg> [--seq N] [--max-kv N] [--fp16]
//! milc lint    <pkg>                  # ANE-placement report
//! milc inspect <pkg>                  # spec summary
//! milc diff    <a> <b>                # structural spec diff
//! milc compile <pkg> [-o <dir>]       # → .mlmodelc
//! milc run     <mlmodelc> [--units ane|gpu|all|cpu] [--steps N]
//! milc verify                         # conformance battery
//! milc check   <pkg>                  # structural verify + lint
//! ```
//!
//! Every subcommand is a thin shell over a library crate — the binary
//! holds no logic the libraries don't expose.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "convert" => cmd_convert(&args[1..]),
        "lint" => cmd_lint(&args[1..]),
        "inspect" => cmd_inspect(&args[1..]),
        "diff" => cmd_diff(&args[1..]),
        "compile" => cmd_compile(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "verify" => ExitCode::from(mil_verify::run_battery() as u8),
        "check" => cmd_check(&args[1..]),
        "help" | "-h" | "--help" => {
            usage();
            ExitCode::SUCCESS
        }
        "-V" | "--version" | "version" => {
            println!("milc {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("milc: unknown command {other}");
            usage();
            ExitCode::from(2)
        }
    }
}

fn usage() {
    eprintln!(
        "milc — the mil_spec toolchain\n\
         \n\
         usage:\n\
         \x20 milc convert <model_dir> -o <pkg.mlpackage> [--seq N] [--max-kv N] [--fp16]\n\
         \x20 milc lint    <pkg.mlpackage|file.mlmodel>\n\
         \x20 milc inspect <pkg.mlpackage|file.mlmodel>\n\
         \x20 milc diff    <a.mlmodel|pkg> <b.mlmodel|pkg>\n\
         \x20 milc compile <pkg.mlpackage> [-o <out_dir>]\n\
         \x20 milc run     <model.mlmodelc> [--units ane|gpu|all|cpu] [--steps N]\n\
         \x20 milc verify\n\
         \x20 milc check   <pkg.mlpackage>\n"
    );
}

fn spec_bytes(path: &Path) -> Result<Vec<u8>, String> {
    if path.is_dir() {
        let p = path.join("Data/com.apple.CoreML/model.mlmodel");
        std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
    } else {
        std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

fn cmd_convert(args: &[String]) -> ExitCode {
    let mut model_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut opts = mil_convert::Options::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--seq" => {
                opts.seq = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1);
                i += 2;
            }
            "--max-kv" => {
                opts.max_kv = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2048);
                i += 2;
            }
            "--fp16" => {
                opts.quant = mil_convert::Quant::Fp16;
                i += 1;
            }
            "--no-lm-head" => {
                opts.lm_head = false;
                i += 1;
            }
            "--embed" => {
                opts.embed = true;
                i += 1;
            }
            "--spec" => {
                opts.spec_version = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
                i += 2;
            }
            "--opset" => {
                opts.opset = args.get(i + 1).cloned().unwrap_or_else(|| "CoreML9".into());
                i += 2;
            }
            other if !other.starts_with('-') => {
                if model_dir.is_none() {
                    model_dir = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc convert: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (model_dir, out) = match (model_dir, out) {
        (Some(m), Some(o)) => (m, o),
        _ => {
            eprintln!("milc convert: needs <model_dir> and -o <pkg>");
            return ExitCode::from(2);
        }
    };
    match mil_convert::convert(&model_dir, &out, &opts) {
        Ok(r) => {
            println!("wrote {}", r.package.display());
            println!("{} ops, {} weight bytes", r.op_count, r.weight_bytes);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("milc convert: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_lint(args: &[String]) -> ExitCode {
    let Some(path) = args.first().map(PathBuf::from) else {
        eprintln!("milc lint: needs <pkg|mlmodel>");
        return ExitCode::from(2);
    };
    let spec = match spec_bytes(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("milc lint: {e}");
            return ExitCode::FAILURE;
        }
    };
    match mil_verify::lint_spec(&spec) {
        Some(r) => {
            print!("{}", r.render());
            if r.cpu_ops > 0
                || r.findings
                    .iter()
                    .any(|f| f.severity == mil_lint::Severity::Error)
            {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        None => {
            eprintln!("milc lint: could not decode spec");
            ExitCode::FAILURE
        }
    }
}

fn cmd_inspect(args: &[String]) -> ExitCode {
    let Some(path) = args.first().map(PathBuf::from) else {
        eprintln!("milc inspect: needs <pkg|mlmodel>");
        return ExitCode::from(2);
    };
    let spec = match spec_bytes(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("milc inspect: {e}");
            return ExitCode::FAILURE;
        }
    };
    match mil_verify::summarize(&spec) {
        Some(s) => {
            println!("spec version: {}", s.spec_version);
            if let Some(op) = &s.opset {
                println!("opset: {op}");
            }
            println!("inputs:");
            for f in &s.inputs {
                println!("  {:<24} {:?} dtype={}", f.name, f.shape, f.dtype);
            }
            println!("outputs:");
            for f in &s.outputs {
                println!("  {:<24} {:?} dtype={}", f.name, f.shape, f.dtype);
            }
            if !s.states.is_empty() {
                println!("states:");
                for f in &s.states {
                    println!("  {:<24} {:?} dtype={}", f.name, f.shape, f.dtype);
                }
            }
            println!("ops ({}):", s.total_ops);
            for (t, n) in &s.op_counts {
                println!("  {:<32} {}", t, n);
            }
            if !s.block_outputs.is_empty() {
                println!("block outputs: {}", s.block_outputs.join(", "));
            }
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("milc inspect: could not decode spec");
            ExitCode::FAILURE
        }
    }
}

fn cmd_diff(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("milc diff: needs <a> <b>");
        return ExitCode::from(2);
    }
    let (a, b) = match (
        spec_bytes(&PathBuf::from(&args[0])),
        spec_bytes(&PathBuf::from(&args[1])),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("milc diff: {e}");
            return ExitCode::FAILURE;
        }
    };
    let diffs = mil_verify::diff_specs(&a, &b);
    if diffs.is_empty() {
        println!("identical");
    } else {
        for d in &diffs {
            println!("{d}");
        }
        println!("{} difference(s)", diffs.len());
    }
    ExitCode::SUCCESS
}

fn cmd_compile(args: &[String]) -> ExitCode {
    let mut pkg: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if pkg.is_none() {
                    pkg = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc compile: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(pkg) = pkg else {
        eprintln!("milc compile: needs <pkg>");
        return ExitCode::from(2);
    };
    let out = out.unwrap_or_else(|| pkg.with_extension("compiled"));
    match mil_compile::compile(&pkg, &out) {
        Ok(m) => {
            println!("{} via {}", m.path.display(), m.backend);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc run` — predict on a compiled model with synthetic inputs.
/// The units flag is the point: `--units ane` pins the run to the
/// neural engine — the latency number the lint report predicts.
#[cfg(target_os = "macos")]
fn cmd_run(args: &[String]) -> ExitCode {
    use mil_infer::{ComputeUnits, Input, Model};

    let mut path: Option<PathBuf> = None;
    let mut units = ComputeUnits::All;
    let mut steps = 8usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--units" => {
                units = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("ane") => ComputeUnits::CpuAndNeuralEngine,
                    Some("gpu") => ComputeUnits::CpuAndGpu,
                    Some("cpu") => ComputeUnits::CpuOnly,
                    _ => ComputeUnits::All,
                };
                i += 2;
            }
            "--steps" => {
                steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc run: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("milc run: needs <pkg.mlpackage>");
        return ExitCode::from(2);
    };
    let spec_path = path.join("Data/com.apple.CoreML/model.mlmodel");
    let spec = match std::fs::read(&spec_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("milc run: {}: {e}", spec_path.display());
            return ExitCode::FAILURE;
        }
    };
    // compile (or reuse) the .mlmodelc next to the package
    let comp_dir = path.with_extension("compiled");
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let compiled_path = comp_dir.join(format!("{stem}.mlmodelc"));
    if !compiled_path.exists() {
        match mil_compile::compile(&path, &comp_dir) {
            Ok(m) => println!("compiled {} via {}", m.path.display(), m.backend),
            Err(e) => {
                eprintln!("milc run: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let path = compiled_path;
    let summary = match mil_verify::summarize(&spec) {
        Some(s) => s,
        None => {
            eprintln!("milc run: could not decode spec");
            return ExitCode::FAILURE;
        }
    };
    let model = match Model::load(&path, units) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("milc run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let state = if summary.states.is_empty() {
        None
    } else {
        match model.new_state() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("milc run: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    // dtype enum → element size / generation
    let dtype_of = |code: u64| -> mil_spec::DType {
        match code {
            16 => mil_spec::DType::Fp16,
            5 => mil_spec::DType::Int32,
            8 => mil_spec::DType::Int64,
            _ => mil_spec::DType::Fp32,
        }
    };
    let mut lat = Vec::with_capacity(steps);
    for step in 0..steps {
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut inputs: Vec<Input> = Vec::new();
        for f in &summary.inputs {
            let n: i64 = f.shape.iter().product();
            let n = n.max(1) as usize;
            let dt = dtype_of(f.dtype);
            let buf = match dt {
                mil_spec::DType::Int32 | mil_spec::DType::Int64 => {
                    // scalar-ish int inputs: feed the step index (pos-style)
                    let mut v = Vec::with_capacity(n * 4);
                    for _ in 0..n {
                        v.extend_from_slice(&(step as i32).to_le_bytes());
                    }
                    v
                }
                _ => {
                    let mut v = Vec::with_capacity(n * 2);
                    if f.name.contains("mask") {
                        // causal: 0 for seen slots, -1e4 for future
                        let seen = (step + 1).min(n);
                        for j in 0..n {
                            let x = if j < seen { 0.0 } else { -1e4 };
                            v.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
                        }
                    } else if f.name.contains("cos") {
                        for _ in 0..n {
                            v.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                        }
                    } else if f.name.contains("sin") {
                        v.resize(n * 2, 0);
                    } else {
                        for j in 0..n {
                            let x = 0.001 * ((j + step) % 17) as f32;
                            v.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
                        }
                    }
                    v
                }
            };
            bufs.push(buf);
        }
        for (f, b) in summary.inputs.iter().zip(bufs.iter()) {
            inputs.push(Input {
                name: &f.name,
                shape: &f.shape,
                data: b,
                dtype: dtype_of(f.dtype),
            });
        }
        let p = match &state {
            Some(s) => model.predict_with_state(Some(s), &inputs),
            None => model.predict(&inputs),
        };
        match p {
            Ok(p) => lat.push(p.latency),
            Err(e) => {
                eprintln!("milc run: step {step}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    lat.sort();
    let total: std::time::Duration = lat.iter().sum();
    let med = lat[lat.len() / 2];
    let unit_name = match units {
        ComputeUnits::All => "all",
        ComputeUnits::CpuOnly => "cpu",
        ComputeUnits::CpuAndGpu => "cpu+gpu",
        ComputeUnits::CpuAndNeuralEngine => "cpu+ane",
    };
    println!(
        "{} steps on {unit_name}: median {:.2?}  min {:.2?}  max {:.2?}  total {:.2?}",
        lat.len(),
        med,
        lat[0],
        lat[lat.len() - 1],
        total
    );
    ExitCode::SUCCESS
}

#[cfg(not(target_os = "macos"))]
fn cmd_run(_args: &[String]) -> ExitCode {
    eprintln!("milc run requires macOS");
    ExitCode::FAILURE
}

fn cmd_check(args: &[String]) -> ExitCode {
    let Some(path) = args.first().map(PathBuf::from) else {
        eprintln!("milc check: needs <pkg>");
        return ExitCode::from(2);
    };
    let r = mil_verify::verify_package(&path);
    println!("{r}");
    if r.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
