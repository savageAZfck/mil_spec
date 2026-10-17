//! e5_bw — run the lowered 8-layer transformer shard on the ANE.
//!
//! Ports: live inputs (cos, sin, mask, x) + 8 KV stripe states
//! (kv_0..kv_7) + output l7_rm. States are in-out ports — the engine
//! reads and writes them in place across calls.

use std::path::PathBuf;

fn main() {
    let mil = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bw_val/model.mil"));
    let steps: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let live: Vec<(&str, u64)> = vec![
        ("cos", 5 * 64),
        ("mask", 5 * 64),
        ("sin", 5 * 64),
        ("x", (576 * 5)),
    ];
    let mut inouts: Vec<(&str, u64)> = vec![];
    if std::env::var("E5_NOSTATE").is_err() {
        for i in 0..8 {
            inouts.push((
                match i {
                    0 => "kv_0",
                    1 => "kv_1",
                    2 => "kv_2",
                    3 => "kv_3",
                    4 => "kv_4",
                    5 => "kv_5",
                    6 => "kv_6",
                    _ => "kv_7",
                },
                2 * 3 * 64 * 64,
            ));
        }
    }
    let outs: Vec<(&str, u64)> = vec![("l7_rm", (576 * 5))];

    eprintln!(
        "[e5] compiling {} ({} in, {} inout, {} out)",
        mil.display(),
        live.len(),
        inouts.len(),
        outs.len()
    );
    let t0 = std::time::Instant::now();
    let prog = match mil_infer::e5rt::Program::compile(
        &mil,
        mil_infer::e5rt::DeviceMask::Ane,
        &live,
        &outs,
        &inouts,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[e5] compile failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "[e5] compiled in {:?} — functions {:?}",
        t0.elapsed(),
        prog.functions
    );

    // Seed live inputs with deterministic data; states start zeroed.
    for (n, e) in &live {
        let v: Vec<u16> = (0..*e)
            .map(|i| half::f16::from_f32((i % 97) as f32 * 0.01).to_bits())
            .collect();
        prog.set_input_f16(n, &v).unwrap();
    }
    for s in 0..8 {
        let n = [
            "kv_0", "kv_1", "kv_2", "kv_3", "kv_4", "kv_5", "kv_6", "kv_7",
        ][s];
        let _ = prog.set_input_f16(n, &vec![0u16; 2 * 3 * 64 * 64]);
    }

    for i in 0..steps {
        let t = std::time::Instant::now();
        prog.execute().unwrap();
        eprintln!("[e5] step {i}: {:?}", t.elapsed());
    }
    let out = prog.output_f16("l7_rm").unwrap();
    let nz = out.iter().filter(|b| **b != 0).count();
    let first: Vec<f32> = out
        .iter()
        .take(8)
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    eprintln!(
        "[e5] l7_rm: {} elems, {} nonzero, head {:?}",
        out.len(),
        nz,
        first
    );
    // Sanity on state write-back: kv_0 should no longer be all-zero.
    let kv = prog.output_f16("kv_0");
    match kv {
        Ok(v) => {
            let nz = v.iter().filter(|b| **b != 0).count();
            eprintln!("[e5] kv_0 nonzero: {}/{}", nz, v.len());
        }
        Err(e) => eprintln!("[e5] kv_0 readback: {e}"),
    }
}
