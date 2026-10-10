//! `milc` — the mil_spec toolchain CLI.
//!
//! ```text
//! milc convert <model_dir> -o <pkg> [--seq N] [--max-kv N] [--fp16] [--lora <adapter>]
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
        "gguf" => cmd_gguf(&args[1..]),
        "onnx" => cmd_onnx(&args[1..]),
        "lint" => cmd_lint(&args[1..]),
        "inspect" => cmd_inspect(&args[1..]),
        "diff" => cmd_diff(&args[1..]),
        "compile" => cmd_compile(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "plan" => cmd_plan(&args[1..]),
        "verify" => ExitCode::from(mil_verify::run_battery() as u8),
        "check" => cmd_check(&args[1..]),
        "fuse-lora" => cmd_fuse_lora(&args[1..]),
        "requant" => cmd_requant(&args[1..]),
        "graft" => cmd_graft(&args[1..]),
        "reshape" => cmd_reshape(&args[1..]),
        "attest" => cmd_attest(&args[1..]),
        "machine" => cmd_machine(&args[1..]),
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
         \x20 milc convert <model_dir|model.gguf> -o <pkg.mlpackage> [--seq N] [--seq-lens a,b,c | --seq-range LO..HI] [--max-kv N] [--fp16] [--lora <adapter>] [--embed] [--shard N] [--head-shards M]\n\
         \x20   --shard N → <pkg> is a .milshards bundle dir: N-layer stateful shards + vocab-sliced heads + manifest.json (the layout the ANE plan builder accepts — monolithic >~3.2k ops fails -14)\n\
         \x20                    [--plan] [--quant-policy uniform|placement|error] [--plan-file <json>]\n\
         \x20 milc gguf    <file.gguf> [--json] [--tokenizer <out.json>]   # header, metadata, tensor table\n\
         \x20 milc onnx    <model.onnx|model.json> -o <pkg.mlpackage> [--dim name=N]... [--inline-max BYTES] [--classical]\n\
         \x20 milc lint    <pkg.mlpackage|file.mlmodel>\n\
         \x20 milc inspect <pkg.mlpackage|file.mlmodel>\n\
         \x20 milc diff    <a.mlmodel|pkg> <b.mlmodel|pkg>\n\
         \x20 milc compile <pkg.mlpackage> [-o <out_dir>]\n\
         \x20 milc run     <model.mlmodelc|bundle.milshards> [--units ane|gpu|all|cpu] [--steps N]\n\
         \x20 milc plan    <pkg|model.mlmodelc> [--units all|ane|gpu|cpu] [--json]   # Core ML compute plan: per-op device + cost\n\
         \x20 milc verify\n\
         \x20 milc check   <pkg.mlpackage>\n\
         \x20 milc fuse-lora <pkg> --lora <adapter> -o <pkg>        # fuse adapter into package weights\n\
         \x20 milc requant  <pkg> --to <int8|fp16|palette4> [-o <pkg>]\n\
         \x20 milc graft    <donor> --layers a..b --onto <base> -o <pkg>\n\
         \x20 milc reshape  <pkg> --seq-lens a,b,c | --seq-range LO..HI [-o <pkg>]   # rewrite enumerated shapes / shape range\n\
         \x20 milc attest   <pkg> [--source <dir|file>]                 # verify embedded provenance\n\
         \x20 milc machine dfa <patterns.txt> -o <pkg.mlpackage>   # blocklist → stateful DFA\n\
         \x20 milc machine sentinel -o <pkg.mlpackage> [--alpha A] [--eps E] [--thresh T]\n\
         \x20 milc machine memory -o <pkg.mlpackage> [--slots N] [--dim D]\n"
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

/// `LO..HI` → `(lo, hi)`; a missing/negative HI parses as `-1`
/// (unbounded) so the option check can reject it with a clear message.
fn parse_seq_range(s: &str) -> Result<(i64, i64), String> {
    let (lo, hi) = s
        .split_once("..")
        .ok_or_else(|| format!("--seq-range {s:?}: expected LO..HI"))?;
    let lo = lo
        .trim()
        .parse::<i64>()
        .map_err(|_| format!("--seq-range {s:?}: LO must be an integer"))?;
    let hi = if hi.trim().is_empty() {
        -1
    } else {
        hi.trim()
            .parse::<i64>()
            .map_err(|_| format!("--seq-range {s:?}: HI must be an integer"))?
    };
    Ok((lo, hi))
}

fn cmd_convert(args: &[String]) -> ExitCode {
    let mut model_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut opts = mil_convert::Options::default();
    let mut print_plan = false;
    let mut plan_file: Option<PathBuf> = None;
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
            "--seq-lens" => {
                // EnumeratedShapes list — first entry is the default.
                // Emits a flexible-shape program with a runtime `seq`
                // int32 input instead of a fixed-length graph.
                opts.seq_lens = args
                    .get(i + 1)
                    .map(|s| {
                        s.split(',')
                            .filter_map(|t| t.trim().parse::<i64>().ok())
                            .collect()
                    })
                    .unwrap_or_default();
                if opts.seq_lens.is_empty() {
                    eprintln!("milc convert: --seq-lens needs a,b,c");
                    return ExitCode::from(2);
                }
                i += 2;
            }
            "--seq-range" => {
                // ShapeRange on the sequence dim — continuous lengths
                // LO..HI instead of an enumerated list.
                match args.get(i + 1).map(|s| parse_seq_range(s)) {
                    Some(Ok(r)) => opts.seq_range = Some(r),
                    Some(Err(e)) => {
                        eprintln!("milc convert: {e}");
                        return ExitCode::from(2);
                    }
                    None => {
                        eprintln!("milc convert: --seq-range needs LO..HI");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            "--plan" => {
                // Print the planner's decisions as JSON on stdout —
                // no package is written.
                print_plan = true;
                i += 1;
            }
            "--quant-policy" => {
                let v = args.get(i + 1).map(String::as_str).unwrap_or("");
                match mil_convert::plan::QuantPolicy::parse(v) {
                    Some(p) => opts.quant_policy = Some(p),
                    None => {
                        eprintln!(
                            "milc convert: --quant-policy uniform|placement|error, got {v:?}"
                        );
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            "--plan-file" => {
                plan_file = args.get(i + 1).map(PathBuf::from);
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
            "--shard" => {
                // Layers per package — emits a .milshards bundle of
                // small packages the ANE plan builder accepts (the
                // monolithic program dies at ~3.2k ops, error -14).
                opts.shard_layers = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--head-shards" => {
                opts.head_shards = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                i += 2;
            }
            "--spec" => {
                opts.spec_version = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
                i += 2;
            }
            "--opset" => {
                opts.opset = args.get(i + 1).cloned().unwrap_or_else(|| "CoreML9".into());
                i += 2;
            }
            "--lora" => {
                // Adapter dir (adapter_config.json + adapters.safetensors
                // / *.npz) or a bare safetensors/npz file. Baked into the
                // emitted weights as W + scale·(B @ A).
                opts.lora = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--updatable" => {
                // Metadata marker only — mlProgram has no real updatable
                // field (that's a NeuralNetwork-spec feature). Names go
                // into `mil.updatable` userDefined metadata.
                opts.updatable = args
                    .get(i + 1)
                    .map(|s| {
                        s.split(',')
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
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
    // --plan-file loads a hand-edited plan; it wins over --quant-policy.
    if let Some(pf) = &plan_file {
        let json = match std::fs::read_to_string(pf) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("milc convert: cannot read {}: {e}", pf.display());
                return ExitCode::FAILURE;
            }
        };
        match mil_convert::plan::Plan::from_json(&json) {
            Ok(p) => opts.plan = Some(p),
            Err(e) => {
                eprintln!("milc convert: bad plan file {}: {e}", pf.display());
                return ExitCode::FAILURE;
            }
        }
    }
    let model_dir = match model_dir {
        Some(m) => m,
        None => {
            eprintln!("milc convert: needs <model_dir|model.gguf>");
            return ExitCode::from(2);
        }
    };
    if print_plan {
        // Dry-run: an explicit plan file is echoed back; otherwise the
        // policy's computed plan (default `error` — the informative one).
        let plan = match &opts.plan {
            Some(p) => p.clone(),
            None => {
                let policy = opts
                    .quant_policy
                    .unwrap_or(mil_convert::plan::QuantPolicy::Error);
                match mil_convert::compute_plan(&model_dir, policy) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("milc convert --plan: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        };
        print!("{}", plan.to_json());
        return ExitCode::SUCCESS;
    }
    let out = match out {
        Some(o) => o,
        None => {
            eprintln!("milc convert: needs -o <pkg>");
            return ExitCode::from(2);
        }
    };
    // GGUF input auto-detects by magic bytes (file, not directory)
    let is_gguf = model_dir.is_file()
        && std::fs::File::open(&model_dir)
            .and_then(|mut f| {
                let mut m = [0u8; 4];
                std::io::Read::read_exact(&mut f, &mut m).map(|_| m == *b"GGUF")
            })
            .unwrap_or(false);
    let res = if is_gguf {
        mil_convert::convert_gguf(&model_dir, &out, &opts)
    } else {
        mil_convert::convert(&model_dir, &out, &opts)
    };
    match res {
        Ok(r) => {
            println!("wrote {}", r.package.display());
            println!("{} ops, {} weight bytes", r.op_count, r.weight_bytes);
            if opts.shard_layers == 0 && r.op_count > 3000 {
                eprintln!(
                    "note: {} ops is past the ANE plan-builder limit (~3.2k ops / ~51 MB \
                     per program — the model will fail to load with error -14 on cpu+ane \
                     and all). Re-convert with --shard N (e.g. --shard 8) to emit a \
                     .milshards bundle that runs on the Neural Engine.",
                    r.op_count
                );
            }
            if opts.seq_range.is_some() {
                eprintln!(
                    "note: --seq-range gives continuous sequence lengths, but Apple documents EnumeratedShapes \
                     as the performance/Neural Engine option and range-flexible models may not run on the ANE; \
                     use --seq-lens a,b,c for ANE workloads (milc reshape switches an existing package either way)"
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("milc convert: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc gguf <file> [--json] [--tokenizer <out.json>]` — header,
/// metadata, tensor table, optional tokenizer export.
fn cmd_gguf(args: &[String]) -> ExitCode {
    let mut path: Option<PathBuf> = None;
    let mut json = false;
    let mut tok_out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--tokenizer" => {
                tok_out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc gguf: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("milc gguf: needs <file.gguf>");
        return ExitCode::from(2);
    };
    let g = match mil_convert::Gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(out) = tok_out {
        match mil_convert::gguf::tokenizer::to_tokenizer_json(&g) {
            Ok(j) => {
                if let Err(e) = std::fs::write(&out, j) {
                    eprintln!("{}: {e}", out.display());
                    return ExitCode::FAILURE;
                }
                println!("wrote {}", out.display());
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if json {
        print_gguf_json(&g);
    } else {
        print_gguf(&g);
    }
    ExitCode::SUCCESS
}

fn fmt_value(v: &mil_convert::gguf::Value) -> String {
    use mil_convert::gguf::Value;
    match v {
        Value::Arr(_, vals) => {
            let head: Vec<String> = vals.iter().take(4).map(fmt_value).collect();
            format!(
                "array[{}]{}",
                vals.len(),
                if vals.is_empty() {
                    String::new()
                } else {
                    format!(
                        " ({}{})",
                        head.join(", "),
                        if vals.len() > 4 { ", …" } else { "" }
                    )
                }
            )
        }
        Value::Str(s) => {
            if s.chars().count() > 72 {
                format!("\"{}…\"", s.chars().take(72).collect::<String>())
            } else {
                format!("\"{s}\"")
            }
        }
        other => format!("{other:?}"),
    }
}

fn print_gguf(g: &mil_convert::Gguf) {
    println!("version:   {}", g.version);
    println!("alignment: {}", g.alignment);
    if let Some(a) = &g.arch {
        println!("arch:      {a}");
    }
    println!("shards:    {}", g.paths().len());
    println!("metadata ({}):", g.metadata().len());
    for (k, v) in g.metadata() {
        println!("  {k:<48} {}", fmt_value(v));
    }
    let tensors: Vec<_> = g.tensors().collect();
    println!("tensors ({}):", tensors.len());
    println!(
        "  {:<52} {:<10} {:<20} {:>12}",
        "name", "type", "shape", "bytes"
    );
    for t in tensors {
        let shape = t
            .hf_shape()
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let name = match mil_convert::gguf::names::ggml_to_hf(&t.name) {
            Some(hf) => format!("{} ({})", t.name, hf),
            None => t.name.clone(),
        };
        println!(
            "  {:<52} {:<10} {:<20} {:>12}",
            name, t.gguf_type, shape, t.nbytes
        );
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn print_gguf_json(g: &mil_convert::Gguf) {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"version\": {},\n", g.version));
    s.push_str(&format!("  \"alignment\": {},\n", g.alignment));
    if let Some(a) = &g.arch {
        s.push_str(&format!("  \"architecture\": \"{}\",\n", json_escape(a)));
    }
    s.push_str("  \"metadata\": {");
    let mut first = true;
    for (k, v) in g.metadata() {
        if !first {
            s.push(',');
        }
        first = false;
        s.push_str(&format!(
            "\n    \"{}\": \"{}\"",
            json_escape(k),
            json_escape(&fmt_value(v))
        ));
    }
    s.push_str("\n  },\n  \"tensors\": [");
    let mut first = true;
    for t in g.tensors() {
        if !first {
            s.push(',');
        }
        first = false;
        s.push_str(&format!(
            "\n    {{\"name\": \"{}\", \"type\": \"{}\", \"shape\": [{}], \"bytes\": {}}}",
            json_escape(&t.name),
            t.gguf_type,
            t.hf_shape()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            t.nbytes
        ));
    }
    s.push_str("\n  ]\n}\n");
    print!("{s}");
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
            // userDefined metadata (provenance, updatable markers) —
            // packages only; bare .mlmodel files have no manifest dir.
            if path.is_dir() {
                if let Ok(ud) = mil_convert::attest::read_user_defined(&path) {
                    if !ud.is_empty() {
                        println!("metadata ({}):", ud.len());
                        for (k, v) in &ud {
                            println!("  {k} = {v}");
                        }
                    }
                }
                if let Ok(pe) = mil_convert::surgery::PackageEdit::open(&path) {
                    if let Ok(groups) = pe.weight_groups() {
                        if !groups.is_empty() {
                            println!("weight groups ({}):", groups.len());
                            for g in &groups {
                                println!(
                                    "  {:<16} {:<8} {:?} ({} member(s))",
                                    g.name,
                                    g.kind.as_str(),
                                    g.shape,
                                    g.members.len()
                                );
                            }
                        }
                    }
                }
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
    use mil_infer::{ComputeUnits, Model};

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
    if path.join("manifest.json").exists() {
        return run_sharded(&path, units, steps);
    }
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
    let mut lat = Vec::with_capacity(steps);
    for step in 0..steps {
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let inputs = synth_inputs(&summary.inputs, step, &mut bufs);
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

/// dtype enum → element dtype for feed synthesis.
#[cfg(target_os = "macos")]
fn dtype_of(code: u64) -> mil_spec::DType {
    match code {
        16 => mil_spec::DType::Fp16,
        5 => mil_spec::DType::Int32,
        8 => mil_spec::DType::Int64,
        _ => mil_spec::DType::Fp32,
    }
}

/// Synthesize plausible feeds from a spec's input features: causal
/// mask (0 seen / -1e4 future), cos=1, sin=0, pos/seq=step, small
/// ramp for everything else. Buffers land in `bufs` so the returned
/// `Input`s borrow them.
#[cfg(target_os = "macos")]
fn synth_inputs<'a>(
    feats: &'a [mil_verify::FeatureInfo],
    step: usize,
    bufs: &'a mut Vec<Vec<u8>>,
) -> Vec<mil_infer::Input<'a>> {
    use mil_infer::Input;
    for f in feats {
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
    feats
        .iter()
        .zip(bufs.iter())
        .map(|(f, b)| Input {
            name: &f.name,
            shape: &f.shape,
            data: b,
            dtype: dtype_of(f.dtype),
        })
        .collect()
}

/// `milc run <bundle.milshards>` — compile + load every member package
/// and time chained predictions, all on the requested units.
#[cfg(target_os = "macos")]
fn run_sharded(dir: &Path, units: mil_infer::ComputeUnits, steps: usize) -> ExitCode {
    use mil_infer::ShardedModel;
    let manifest = match mil_infer::ShardManifest::load(dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("milc run: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Feed shapes come from the first layer shard's spec — every layer
    // shard shares the input contract.
    let first = match manifest.layer_shards.first() {
        Some((f, _, _)) => dir.join(f),
        None => {
            eprintln!("milc run: bundle has no layer shards");
            return ExitCode::FAILURE;
        }
    };
    let spec = match std::fs::read(first.join("Data/com.apple.CoreML/model.mlmodel")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("milc run: {}: {e}", first.display());
            return ExitCode::FAILURE;
        }
    };
    let summary = match mil_verify::summarize(&spec) {
        Some(s) => s,
        None => {
            eprintln!("milc run: could not decode shard spec");
            return ExitCode::FAILURE;
        }
    };
    let comp_dir = dir.with_extension("compiled");
    let sm = match ShardedModel::load(dir, units, &comp_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("milc run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut lat = Vec::with_capacity(steps);
    for step in 0..steps {
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let inputs = synth_inputs(&summary.inputs, step, &mut bufs);
        match sm.predict(&inputs) {
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
        mil_infer::ComputeUnits::All => "all",
        mil_infer::ComputeUnits::CpuOnly => "cpu",
        mil_infer::ComputeUnits::CpuAndGpu => "cpu+gpu",
        mil_infer::ComputeUnits::CpuAndNeuralEngine => "cpu+ane",
    };
    println!(
        "{} steps on {unit_name}: median {:.2?}  min {:.2?}  max {:.2?}  total {:.2?}  ({} layer + {} head shards)",
        lat.len(),
        med,
        lat[0],
        lat[lat.len() - 1],
        total,
        sm.manifest.layer_shards.len(),
        sm.manifest.head_shards.len(),
    );
    ExitCode::SUCCESS
}

#[cfg(not(target_os = "macos"))]
fn cmd_run(_args: &[String]) -> ExitCode {
    eprintln!("milc run requires macOS");
    ExitCode::FAILURE
}

/// `milc plan <pkg|mlmodelc> [--units all|ane|gpu|cpu] [--json]` —
/// where Core ML's compute plan (`MLComputePlan`) puts every op of an
/// ML Program, with a per-device summary of op count and estimated cost.
#[cfg(target_os = "macos")]
fn cmd_plan(args: &[String]) -> ExitCode {
    use mil_infer::{summarize_plan, ComputeUnits, Device};
    let mut path: Option<PathBuf> = None;
    let mut units = ComputeUnits::All;
    let mut json = false;
    let mut vs_lint = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--units" => {
                units = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("ane") => ComputeUnits::CpuAndNeuralEngine,
                    Some("gpu") => ComputeUnits::CpuAndGpu,
                    Some("cpu") => ComputeUnits::CpuOnly,
                    Some("all") => ComputeUnits::All,
                    other => {
                        eprintln!("milc plan: --units all|ane|gpu|cpu, got {other:?}");
                        return ExitCode::from(2);
                    }
                };
                i += 2;
            }
            "--vs-lint" => {
                vs_lint = true;
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            other if !other.starts_with('-') => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc plan: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("milc plan: needs <pkg.mlpackage|model.mlmodelc>");
        return ExitCode::from(2);
    };
    // .mlpackage → compile into a scratch dir that is removed afterwards
    let scratch = std::env::temp_dir().join(format!("milc_plan_{}", std::process::id()));
    let compiled = if path.extension().and_then(|e| e.to_str()) == Some("mlmodelc") {
        path.clone()
    } else {
        match mil_compile::compile(&path, &scratch) {
            Ok(m) => m.path,
            Err(e) => {
                eprintln!("milc plan: compile: {e}");
                let _ = std::fs::remove_dir_all(&scratch);
                return ExitCode::FAILURE;
            }
        }
    };
    let res = mil_infer::compute_plan(&compiled, units);
    let _ = std::fs::remove_dir_all(&scratch);
    let ops = match res {
        Ok(o) => o,
        Err(e) => {
            eprintln!("milc plan: {e}");
            return ExitCode::FAILURE;
        }
    };
    if vs_lint {
        let spec = match std::fs::read(path.join("Data/com.apple.CoreML/model.mlmodel")) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("milc plan --vs-lint: needs an .mlpackage ({e})");
                return ExitCode::FAILURE;
            }
        };
        let Some(report) = mil_verify::lint_spec(&spec) else {
            eprintln!("milc plan --vs-lint: spec decode failed");
            return ExitCode::FAILURE;
        };
        let predicted: std::collections::HashMap<&str, mil_lint::Unit> = report
            .verdicts
            .iter()
            .map(|v| (v.name.as_str(), v.unit))
            .collect();
        let dev_of = |u: mil_lint::Unit| match u {
            mil_lint::Unit::Ane => Device::NeuralEngine,
            mil_lint::Unit::Gpu | mil_lint::Unit::Unknown => Device::Gpu,
            mil_lint::Unit::Cpu => Device::Cpu,
        };
        let (mut agree, mut total) = (0usize, 0usize);
        let mut miss: std::collections::BTreeMap<(String, &str, &str), usize> =
            std::collections::BTreeMap::new();
        for o in &ops {
            let (Some(actual), Some(&pu)) = (o.preferred, predicted.get(o.output.as_str())) else {
                continue;
            };
            total += 1;
            let pd = dev_of(pu);
            if pd == actual {
                agree += 1;
            } else {
                *miss
                    .entry((o.op_type.clone(), pd.label(), actual.label()))
                    .or_insert(0) += 1;
            }
        }
        println!(
            "lint vs plan under {units:?}: {agree}/{total} ops agree ({:.1}%)",
            100.0 * agree as f64 / total.max(1) as f64
        );
        for ((op, p, a), n) in &miss {
            println!("  {n:>5}x {op:<28} lint {p:<4} plan {a}");
        }
        return ExitCode::SUCCESS;
    }
    let sum = summarize_plan(&ops);
    let placed: usize = sum.values().map(|v| v.0).sum();
    let total_cost: f64 = sum.values().map(|v| v.1).sum();
    let label = |d: Option<Device>| d.map(|d| d.label()).unwrap_or("-");
    if json {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        println!("{{\"units\":\"{units:?}\",\"ops\":[");
        for (k, o) in ops.iter().enumerate() {
            let sup: Vec<String> = o
                .supported
                .iter()
                .map(|d| format!("\"{}\"", d.label()))
                .collect();
            println!(
                "{{\"function\":\"{}\",\"op\":\"{}\",\"output\":\"{}\",\"preferred\":{},\"supported\":[{}],\"cost\":{}}}{}",
                esc(&o.function),
                esc(&o.op_type),
                esc(&o.output),
                o.preferred
                    .map(|d| format!("\"{}\"", d.label()))
                    .unwrap_or_else(|| "null".into()),
                sup.join(","),
                o.cost
                    .map(|c| format!("{c:e}"))
                    .unwrap_or_else(|| "null".into()),
                if k + 1 < ops.len() { "," } else { "" }
            );
        }
        println!("],\"summary\":{{");
        let parts: Vec<String> = sum
            .iter()
            .map(|(d, (n, c))| {
                format!(
                    "\"{}\":{{\"ops\":{n},\"op_pct\":{:.2},\"cost_pct\":{:.2}}}",
                    d.label(),
                    100.0 * *n as f64 / placed.max(1) as f64,
                    100.0 * c / total_cost.max(f64::MIN_POSITIVE)
                )
            })
            .collect();
        println!("{}}}}}", parts.join(","));
        return ExitCode::SUCCESS;
    }
    println!(
        "{:<5} {:<28} {:<6} {:<10} {}",
        "dev", "op", "cost%", "supported", "output"
    );
    for o in &ops {
        if o.preferred.is_none() {
            continue;
        }
        let sup: Vec<&str> = o.supported.iter().map(|d| d.label()).collect();
        println!(
            "{:<5} {:<28} {:<6.2} {:<10} {}",
            label(o.preferred),
            o.op_type,
            100.0 * o.cost.unwrap_or(0.0) / total_cost.max(f64::MIN_POSITIVE),
            sup.join("+"),
            o.output
        );
    }
    println!(
        "\nplan under {units:?}: {} ops placed ({} const/no-device ops skipped)",
        placed,
        ops.len() - placed
    );
    for (d, (n, c)) in &sum {
        println!(
            "  {:<4} {:>6} ops ({:>5.1}%)   est. cost {:>5.1}%",
            d.label(),
            n,
            100.0 * *n as f64 / placed.max(1) as f64,
            100.0 * c / total_cost.max(f64::MIN_POSITIVE)
        );
    }
    ExitCode::SUCCESS
}

#[cfg(not(target_os = "macos"))]
fn cmd_plan(_args: &[String]) -> ExitCode {
    eprintln!("milc plan requires macOS");
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

// ======== package surgery + provenance commands ========

fn surg_out(r: mil_convert::surgery::SurgReport) -> ExitCode {
    for l in &r.lines {
        println!("{l}");
    }
    if let Some(w) = &r.written {
        println!(
            "wrote {} ({} ops, {} weight bytes, {} blobs)",
            w.package.display(),
            w.op_count,
            w.weight_bytes,
            w.blob_count
        );
    }
    ExitCode::SUCCESS
}

/// `milc fuse-lora <pkg> --lora <adapter> -o <pkg>` — fuse an adapter
/// into a built package's weight blobs (same W + scale·(B@A) math as
/// `convert --lora`, verified by the deterministic const names).
fn cmd_fuse_lora(args: &[String]) -> ExitCode {
    let mut pkg: Option<PathBuf> = None;
    let mut lora: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lora" => {
                lora = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
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
                eprintln!("milc fuse-lora: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(pkg), Some(lora), Some(out)) = (pkg, lora, out) else {
        eprintln!("milc fuse-lora: needs <pkg> --lora <adapter> -o <pkg>");
        return ExitCode::from(2);
    };
    match mil_convert::surgery::fuse_lora(&pkg, &lora, &out) {
        Ok(r) => surg_out(r),
        Err(e) => {
            eprintln!("milc fuse-lora: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc requant <pkg> --to <int8|fp16|palette4> [-o <pkg>]` — re-encode
/// eligible weight groups. In-place when `-o` is omitted.
fn cmd_requant(args: &[String]) -> ExitCode {
    let mut pkg: Option<PathBuf> = None;
    let mut to: Option<mil_convert::surgery::TargetQuant> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--to" => {
                let v = args.get(i + 1).map(String::as_str).unwrap_or("");
                match mil_convert::surgery::TargetQuant::parse(v) {
                    Some(t) => to = Some(t),
                    None => {
                        eprintln!("milc requant: --to int8|fp16|palette4, got {v:?}");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
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
                eprintln!("milc requant: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(pkg), Some(t)) = (pkg, to) else {
        eprintln!("milc requant: needs <pkg> --to <int8|fp16|palette4>");
        return ExitCode::from(2);
    };
    match mil_convert::surgery::requant(&pkg, t, out.as_deref()) {
        Ok(r) => surg_out(r),
        Err(e) => {
            eprintln!("milc requant: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc graft <donor> --layers a..b --onto <base> -o <pkg>` — splice
/// donor layer weights into a same-architecture base.
fn cmd_graft(args: &[String]) -> ExitCode {
    let mut donor: Option<PathBuf> = None;
    let mut base: Option<PathBuf> = None;
    let mut layers: Option<(usize, usize)> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--layers" => {
                let v = args.get(i + 1).map(String::as_str).unwrap_or("");
                match mil_convert::surgery::parse_layer_range(v) {
                    Ok(r) => layers = Some(r),
                    Err(e) => {
                        eprintln!("milc graft: {e}");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            "--onto" => {
                base = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if donor.is_none() {
                    donor = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc graft: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(donor), Some(layers), Some(base), Some(out)) = (donor, layers, base, out) else {
        eprintln!("milc graft: needs <donor> --layers a..b --onto <base> -o <pkg>");
        return ExitCode::from(2);
    };
    match mil_convert::surgery::graft(&donor, &base, layers, &out) {
        Ok(r) => surg_out(r),
        Err(e) => {
            eprintln!("milc graft: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc reshape <pkg> --seq-lens a,b,c | --seq-range LO..HI [-o <pkg>]`
/// — rewrite the sequence flexibility (enumerated shapes or shape
/// range) of a flexible-shape package. In-place without -o.
fn cmd_reshape(args: &[String]) -> ExitCode {
    let mut pkg: Option<PathBuf> = None;
    let mut target: Option<mil_convert::surgery::SeqFlex> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seq-lens" => {
                target = args.get(i + 1).map(|s| {
                    mil_convert::surgery::SeqFlex::Lens(
                        s.split(',')
                            .filter_map(|t| t.trim().parse::<i64>().ok())
                            .collect(),
                    )
                });
                i += 2;
            }
            "--seq-range" => {
                match args.get(i + 1).map(|s| parse_seq_range(s)) {
                    Some(Ok((lo, hi))) => {
                        target = Some(mil_convert::surgery::SeqFlex::Range(lo, hi));
                    }
                    Some(Err(e)) => {
                        eprintln!("milc reshape: {e}");
                        return ExitCode::from(2);
                    }
                    None => {
                        eprintln!("milc reshape: --seq-range needs LO..HI");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
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
                eprintln!("milc reshape: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(pkg), Some(target)) = (pkg, target) else {
        eprintln!("milc reshape: needs <pkg> and --seq-lens a,b,c or --seq-range LO..HI");
        return ExitCode::from(2);
    };
    match mil_convert::surgery::reshape_flex(&pkg, &target, out.as_deref()) {
        Ok(r) => surg_out(r),
        Err(e) => {
            eprintln!("milc reshape: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc attest <pkg> [--source <dir>]` — verify embedded provenance:
/// weight.bin hash always; per-source-file hashes with --source.
fn cmd_attest(args: &[String]) -> ExitCode {
    let mut pkg: Option<PathBuf> = None;
    let mut source: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                source = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if pkg.is_none() {
                    pkg = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc attest: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(pkg) = pkg else {
        eprintln!("milc attest: needs <pkg>");
        return ExitCode::from(2);
    };
    match mil_convert::attest::verify(&pkg, source.as_deref()) {
        Ok(r) => {
            if r.has_provenance {
                println!("provenance:");
            }
            for l in &r.lines {
                println!("  {l}");
            }
            if r.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("milc attest: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `milc onnx <model.onnx|model.json> -o <pkg> [--dim name=N]...
/// [--inline-max BYTES] [--classical]` — ONNX / classical-ML JSON →
/// `.mlpackage` with embedded provenance.
fn cmd_onnx(args: &[String]) -> ExitCode {
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut opts = mil_onnx::ConvertOptions::default();
    let mut classical = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--dim" => {
                let Some((k, v)) = args.get(i + 1).and_then(|s| s.split_once('=')) else {
                    eprintln!("milc onnx: --dim needs name=N");
                    return ExitCode::from(2);
                };
                let Ok(n) = v.trim().parse::<i64>() else {
                    eprintln!("milc onnx: --dim {k}={v}: value must be an integer");
                    return ExitCode::from(2);
                };
                if opts.dims.insert(k.trim().to_string(), n).is_some() {
                    eprintln!("milc onnx: --dim {k} given twice");
                    return ExitCode::from(2);
                }
                i += 2;
            }
            "--inline-max" => {
                let Some(n) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) else {
                    eprintln!("milc onnx: --inline-max needs BYTES");
                    return ExitCode::from(2);
                };
                opts.inline_max_bytes = n;
                i += 2;
            }
            "--classical" => {
                classical = true;
                i += 1;
            }
            other if !other.starts_with('-') => {
                if input.is_none() {
                    input = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc onnx: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(input), Some(out)) = (input, out) else {
        eprintln!(
            "usage: milc onnx <model.onnx|model.json> -o <pkg.mlpackage> [--dim name=N]... [--inline-max BYTES] [--classical]"
        );
        return ExitCode::from(2);
    };
    let classical = classical
        || input
            .extension()
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false);
    let bytes = match std::fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("milc onnx: {}: {e}", input.display());
            return ExitCode::FAILURE;
        }
    };
    if classical && !opts.dims.is_empty() {
        eprintln!("milc onnx: --dim applies to ONNX inputs, not classical JSON");
        return ExitCode::from(2);
    }
    let built = if classical {
        mil_onnx::convert_classical_json(&bytes)
    } else {
        mil_onnx::ModelProto::decode(&bytes).and_then(|m| mil_onnx::convert_with(&m, &opts))
    };
    let built = match built {
        Ok(b) => b,
        Err(e) => {
            eprintln!("milc onnx: {e}");
            return ExitCode::FAILURE;
        }
    };

    let weight_hash = built
        .weight_bin
        .as_deref()
        .map(mil_spec::sha256::sha256_hex);
    let options_desc = format!(
        "{}|inline_max={}|dims={:?}",
        if classical { "classical" } else { "onnx" },
        opts.inline_max_bytes,
        opts.dims
    );
    let prov = match mil_convert::attest::provenance_map_files(
        &format!("mil_onnx {}", mil_onnx::VERSION),
        &options_desc,
        std::slice::from_ref(&input),
        weight_hash.as_deref(),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("milc onnx: provenance: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut meta = mil_spec::ModelMeta::new(10, "CoreML9")
        .creator("mil_onnx")
        .description(&format!(
            "converted by mil_onnx from {}",
            input
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
    for (k, v) in prov {
        meta = meta.user_meta(&k, &v);
    }
    let spec =
        match mil_convert::attest::encode_with_program_hash(meta, |m| Ok(built.encode_spec(m))) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("milc onnx: provenance: {e}");
                return ExitCode::FAILURE;
            }
        };
    if let Err(e) = built.write_spec(&out, &spec) {
        eprintln!("milc onnx: write {}: {e}", out.display());
        return ExitCode::FAILURE;
    }

    println!("{} → {}", input.display(), out.display());
    for (what, fs) in [("input", &built.inputs), ("output", &built.outputs)] {
        for f in fs.iter() {
            println!("  {what:<6} {} {:?} {:?}", f.name, f.shape, f.dtype);
        }
    }
    if !opts.dims.is_empty() {
        let d: Vec<String> = opts.dims.iter().map(|(k, v)| format!("{k}={v}")).collect();
        println!("  dims bound: {}", d.join(", "));
    }
    let total: usize = built.op_histogram.values().sum();
    let hist: Vec<String> = built
        .op_histogram
        .iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect();
    println!("  ops    {total}: {}", hist.join(" "));
    match &built.weight_bin {
        Some(w) => println!("  weight.bin {} bytes", w.len()),
        None => println!("  weight.bin none (all weights inlined)"),
    }
    ExitCode::SUCCESS
}

fn cmd_machine(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("milc machine: needs dfa|sentinel|memory");
        return ExitCode::from(2);
    }
    let mut out: Option<PathBuf> = None;
    let mut path: Option<PathBuf> = None;
    let (mut alpha, mut eps, mut thresh) = (0.1f32, 1e-3f32, 3.0f32);
    let (mut slots, mut dim) = (64i64, 32i64);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--alpha" => {
                alpha = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(alpha);
                i += 2;
            }
            "--eps" => {
                eps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(eps);
                i += 2;
            }
            "--thresh" => {
                thresh = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(thresh);
                i += 2;
            }
            "--slots" => {
                slots = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(slots);
                i += 2;
            }
            "--dim" => {
                dim = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(dim);
                i += 2;
            }
            other if !other.starts_with('-') => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                }
                i += 1;
            }
            other => {
                eprintln!("milc machine: unknown flag {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("milc machine: needs -o <pkg.mlpackage>");
        return ExitCode::from(2);
    };
    let res = match args[0].as_str() {
        "dfa" => {
            let Some(p) = path else {
                eprintln!("milc machine dfa: needs <patterns.txt>");
                return ExitCode::from(2);
            };
            let text = match std::fs::read_to_string(&p) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}: {e}", p.display());
                    return ExitCode::FAILURE;
                }
            };
            let pats: Vec<&[u8]> = text
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.as_bytes())
                .collect();
            if pats.is_empty() {
                eprintln!("milc machine dfa: no patterns in {}", p.display());
                return ExitCode::FAILURE;
            }
            let ac = mil_machines::Automaton::build(&pats);
            let (n, np) = (ac.states(), pats.len());
            mil_machines::write_dfa_package(&out, &ac)
                .map_err(|e| e.to_string())
                .map(|()| println!("{n} states from {np} patterns → {}", out.display()))
        }
        "sentinel" => mil_machines::write_sentinel_package(&out, alpha, eps, thresh)
            .map_err(|e| e.to_string())
            .map(|()| println!("sentinel a={alpha} e={eps} z>{thresh} → {}", out.display())),
        "memory" => mil_machines::write_memory_package(&out, slots, dim)
            .map_err(|e| e.to_string())
            .map(|()| println!("memory [{slots}x{dim}] fp16 → {}", out.display())),
        other => {
            eprintln!("milc machine: unknown kind {other} (dfa|sentinel|memory)");
            return ExitCode::from(2);
        }
    };
    match res {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: {e}", out.display());
            ExitCode::FAILURE
        }
    }
}
