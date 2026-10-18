//! par_probe — do concurrent evals on separate programs overlap in the
//! daemon? N threads each load avw(128,1024) and eval 25x; compare to
//! the same total done sequentially.
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_avw;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn load_and_eval(dir: &std::path::Path, tag: usize, iters: u32) -> f64 {
    let mil = dir.join("x.mil");
    std::fs::write(&mil, b"x").unwrap();
    let p = dir.join(format!("avw{tag}.macho"));
    let img = emit_avw(128, 1024, &mil.to_string_lossy(), &p.to_string_lossy());
    std::fs::write(&p, &img).unwrap();
    let mut m = EspressoProgram::load_precompiled(&p, &[]).unwrap();
    {
        let b = m.input_mut(0);
        for i in 0..64 {
            b[i * 2..i * 2 + 2].copy_from_slice(&f16(0.01).to_le_bytes());
        }
    }
    let t = std::time::Instant::now();
    for _ in 0..iters {
        m.eval().unwrap();
    }
    t.elapsed().as_secs_f64() * 1e6
}

fn main() {
    let dir = std::env::temp_dir().join("parprobe");
    std::fs::create_dir_all(&dir).unwrap();

    // warmup + sequential baseline: 100 evals
    let seq_us = load_and_eval(&dir, 0, 100);
    println!(
        "sequential 100 evals: {seq_us:.0}us ({:.0}us/eval)",
        seq_us / 100.0
    );

    // 4 threads x 25 evals on distinct programs (identical bytes ->
    // dedup means they may share a resident model; use distinct paths
    // anyway since dedup keys on content/path either way this tests
    // the eval pipeline, not the loader)
    let t = std::time::Instant::now();
    let mut hs = Vec::new();
    for tag in 1..5 {
        let d = dir.clone();
        hs.push(std::thread::spawn(move || load_and_eval(&d, tag, 25)));
    }
    let per: Vec<f64> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    let wall = t.elapsed().as_secs_f64() * 1e6;
    println!("parallel 4x25: wall {wall:.0}us (per-thread {per:?})");
    println!(
        "=> if serialized would be ~{:.0}us; overlap factor {:.2}x",
        4.0 * seq_us / 4.0,
        (per.iter().sum::<f64>() / 4.0) / (wall / 4.0)
    );
}
