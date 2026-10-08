//! ANE × GPU concurrency probe — experiment 2 of the silicon-duel series.
//!
//! Loads one compiled stateful decoder twice, pinned to different compute
//! units, and measures whether ANE-pinned and GPU-pinned inference actually
//! run on separate silicon:
//!
//!   phase A  solo medians per engine
//!   phase B  both engines hammered concurrently for --hold seconds
//!   phase C  K ANE programs resident at once, round-robin predicts
//!
//! If ANE work truly runs on the Neural Engine, phase B degrades far less
//! than a same-engine duel would. Phase C reveals the multi-program
//! residency behavior (swap-in latency spikes).
//!
//! Usage: cargo run -p mil_infer --example duel --release -- \
//!            ~/models/drafter/l4f16v2.mlpackage --steps 24 --hold 10

#[cfg(target_os = "macos")]
mod imp {
    use mil_infer::{ComputeUnits, Input, Model};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    struct Spec {
        inputs: Vec<(String, Vec<i64>, u64)>,
        has_state: bool,
    }

    fn dtype_of(code: u64) -> mil_spec::DType {
        match code {
            16 => mil_spec::DType::Fp16,
            5 => mil_spec::DType::Int32,
            8 => mil_spec::DType::Int64,
            _ => mil_spec::DType::Fp32,
        }
    }

    fn load_spec(pkg: &Path) -> Result<Spec, String> {
        let spec_path = pkg.join("Data/com.apple.CoreML/model.mlmodel");
        let bytes =
            std::fs::read(&spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?;
        let s = mil_verify::summarize(&bytes).ok_or("could not decode spec")?;
        Ok(Spec {
            inputs: s
                .inputs
                .iter()
                .map(|f| (f.name.clone(), f.shape.clone(), f.dtype))
                .collect(),
            has_state: !s.states.is_empty(),
        })
    }

    fn make_inputs(spec: &Spec, step: usize) -> (Vec<Vec<u8>>, Vec<String>) {
        let mut bufs = Vec::new();
        let mut names = Vec::new();
        for (name, shape, dt_code) in &spec.inputs {
            let n = shape.iter().product::<i64>().max(1) as usize;
            let buf = match dtype_of(*dt_code) {
                mil_spec::DType::Int32 | mil_spec::DType::Int64 => {
                    let mut v = Vec::with_capacity(n * 4);
                    for _ in 0..n {
                        v.extend_from_slice(&(step as i32).to_le_bytes());
                    }
                    v
                }
                _ => {
                    let mut v = Vec::with_capacity(n * 2);
                    if name.contains("mask") {
                        let seen = (step + 1).min(n);
                        for j in 0..n {
                            let x = if j < seen { 0.0 } else { -1e4 };
                            v.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
                        }
                    } else if name.contains("cos") {
                        for _ in 0..n {
                            v.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                        }
                    } else if name.contains("sin") {
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
            names.push(name.clone());
        }
        (bufs, names)
    }

    fn predict_once(
        model: &Model,
        state: &Option<mil_infer::State>,
        spec: &Spec,
        step: usize,
    ) -> Result<Duration, String> {
        let (bufs, _names) = make_inputs(spec, step);
        let inputs: Vec<Input> = spec
            .inputs
            .iter()
            .zip(bufs.iter())
            .map(|((name, shape, dt), b)| Input {
                name,
                shape,
                data: b,
                dtype: dtype_of(*dt),
            })
            .collect();
        let t = Instant::now();
        let r = match state {
            Some(s) => model.predict_with_state(Some(s), &inputs),
            None => model.predict(&inputs),
        };
        r.map(|_| t.elapsed()).map_err(|e| e.to_string())
    }

    fn median(v: &mut Vec<Duration>) -> f64 {
        v.sort();
        v[v.len() / 2].as_secs_f64() * 1e3
    }

    fn bench_solo(
        compiled: &Path,
        units: ComputeUnits,
        spec: &Spec,
        steps: usize,
    ) -> Result<(f64, f64), String> {
        let model = Model::load(compiled, units).map_err(|e| e.to_string())?;
        let state = spec
            .has_state
            .then(|| model.new_state())
            .transpose()
            .map_err(|e| e.to_string())?;
        let mut lat = Vec::with_capacity(steps);
        for s in 0..steps {
            lat.push(predict_once(&model, &state, spec, s)?);
        }
        let max = lat.iter().max().unwrap().as_secs_f64() * 1e3;
        Ok((median(&mut lat), max))
    }

    fn bench_concurrent(
        compiled: &Path,
        _spec: &Spec,
        hold: Duration,
    ) -> Result<(f64, f64, usize, usize), String> {
        let stop = Arc::new(AtomicBool::new(false));
        let go = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        let compiled = compiled.to_path_buf();
        for units in [ComputeUnits::CpuAndNeuralEngine, ComputeUnits::CpuAndGpu] {
            let (stop, go, c) = (stop.clone(), go.clone(), compiled.clone());
            let spec_inputs = load_spec(&pkg_for(&c))?;
            handles.push(thread::spawn(move || -> Result<Vec<Duration>, String> {
                let model = Model::load(&c, units).map_err(|e| e.to_string())?;
                let state = spec_inputs
                    .has_state
                    .then(|| model.new_state())
                    .transpose()
                    .map_err(|e| e.to_string())?;
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut lat = Vec::new();
                let mut step = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    lat.push(predict_once(&model, &state, &spec_inputs, step)?);
                    step += 1;
                }
                Ok(lat)
            }));
        }
        thread::sleep(Duration::from_millis(300)); // both models loaded + primed
        go.store(true, Ordering::Release);
        thread::sleep(hold);
        stop.store(true, Ordering::Relaxed);
        let mut a = handles.remove(0).join().unwrap()?;
        let mut b = handles.remove(0).join().unwrap()?;
        let (na, nb) = (a.len(), b.len());
        Ok((median(&mut a), median(&mut b), na, nb))
    }

    fn bench_residency(
        compiled: &Path,
        spec: &Spec,
        k: usize,
        steps: usize,
    ) -> Result<Vec<f64>, String> {
        let models: Result<Vec<Model>, _> = (0..k)
            .map(|_| Model::load(compiled, ComputeUnits::CpuAndNeuralEngine))
            .collect();
        let models = models.map_err(|e| e.to_string())?;
        let states: Result<Vec<_>, _> = models
            .iter()
            .map(|m| {
                if spec.has_state {
                    m.new_state().map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect();
        let states = states.map_err(|e| e.to_string())?;
        let mut lat = Vec::new();
        for s in 0..steps {
            let i = s % k;
            lat.push(predict_once(&models[i], &states[i], spec, s)?.as_secs_f64() * 1e3);
        }
        Ok(lat)
    }

    fn pkg_for(compiled: &Path) -> PathBuf {
        // compiled path is <stem>.compiled/<stem>.mlmodelc → sibling .mlpackage
        let stem = compiled.file_stem().unwrap().to_str().unwrap();
        compiled
            .parent()
            .and_then(|d| d.parent())
            .map(|d| d.join(format!("{stem}.mlpackage")))
            .unwrap_or_default()
    }

    pub fn run(args: &[String]) -> Result<(), String> {
        let pkg = PathBuf::from(
            args.first()
                .ok_or("usage: duel <pkg.mlpackage> [--steps N] [--hold SECS] [--residents K]")?,
        );
        let mut steps = 24usize;
        let mut hold = 10u64;
        let mut k = 4usize;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--steps" => {
                    steps = args[i + 1].parse().unwrap_or(24);
                    i += 2;
                }
                "--hold" => {
                    hold = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                }
                "--residents" => {
                    k = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                }
                _ => i += 1,
            }
        }
        let stem = pkg.file_stem().unwrap().to_str().unwrap().to_string();
        let compiled = pkg
            .with_extension("compiled")
            .join(format!("{stem}.mlmodelc"));
        if !compiled.exists() {
            return Err(format!(
                "no compiled model at {} (run milc run/compile first)",
                compiled.display()
            ));
        }
        let spec = load_spec(&pkg)?;

        println!("=== phase A: solo medians ===");
        for (name, units) in [
            ("ane", ComputeUnits::CpuAndNeuralEngine),
            ("gpu", ComputeUnits::CpuAndGpu),
            ("cpu", ComputeUnits::CpuOnly),
        ] {
            let (med, max) = bench_solo(&compiled, units, &spec, steps)?;
            println!("{name:4} solo: median {med:.2}ms  max {max:.2}ms");
        }

        println!("=== phase B: ane × gpu concurrent ({hold}s) ===");
        let (a, g, na, nb) = bench_concurrent(&compiled, &spec, Duration::from_secs(hold))?;
        println!("ane under load: median {a:.2}ms  ({na} predicts)");
        println!("gpu under load: median {g:.2}ms  ({nb} predicts)");

        println!("=== phase C: {k} ane programs resident, round-robin ===");
        let lat = bench_residency(&compiled, &spec, k, steps)?;
        let med = {
            let mut v = lat.clone();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let max = lat.iter().cloned().fold(0.0, f64::max);
        println!("median {med:.2}ms  max {max:.2}ms");
        Ok(())
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let Err(e) = imp::run(&args) {
            eprintln!("duel: {e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("duel: macOS only");
}
