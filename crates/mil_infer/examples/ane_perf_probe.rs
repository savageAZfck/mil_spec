//! `ane_perf_probe` — discover where `kANEFPerformanceStatsMask`
//! surfaces counters: loads a precompiled HWX, evaluates once, then
//! dumps every stat-shaped property/dict-key that responds on the
//! model, the `_ANEClient`, and the post-eval `_ANERequest`.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bench_steps: Option<u32> = args
        .iter()
        .position(|a| a == "--bench")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .or(Some(100));
    let hwx = args
        .iter()
        .find(|a| !a.starts_with('-') && a.parse::<u32>().is_err())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/chain/c64k1.hwx")
        });
    let prog = match mil_infer::EspressoProgram::load_precompiled(&hwx, &[]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("load {}: {e}", hwx.display());
            std::process::exit(1);
        }
    };
    println!(
        "loaded {}: {} in / {} out, {} proc(s)",
        hwx.display(),
        prog.num_inputs,
        prog.num_outputs,
        prog.procedures.len()
    );
    if std::env::var_os("ANE_DUMP_METHODS").is_some() {
        for (cls, methods) in prog.method_inventory() {
            println!("\n== {cls} ==");
            let mut m = methods;
            m.sort();
            for name in m {
                println!("  {name}");
            }
        }
    }
    if let Some(steps) = bench_steps {
        match prog.bench(0, steps) {
            Ok(b) => {
                println!(
                    "bench {steps} evals: min {:.1}µs  mean {:.1}µs  p99 {:.1}µs  io {}+{}B  ~{:.2} GB/s",
                    b.min_us,
                    b.mean_us,
                    b.p99_us,
                    b.in_bytes,
                    b.out_bytes,
                    b.gbps(),
                );
            }
            Err(e) => eprintln!("bench: {e}"),
        }
    }
    match prog.perf_probe() {
        Ok(hits) => {
            if hits.is_empty() {
                println!("no stats surface responded");
            }
            for (t, k, v) in hits {
                if v.is_empty() {
                    println!("{t:>14} {k}");
                } else {
                    println!("{t:>14} {k} = {v}");
                }
            }
        }
        Err(e) => {
            eprintln!("perf_probe: {e}");
            std::process::exit(1);
        }
    }
}
