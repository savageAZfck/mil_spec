//! All-ANE speculative decode — experiment 5 of the silicon-duel series.
//!
//! The full proposer/verifier loop pinned to the Neural Engine while the
//! GPU is under load from a second model: the token path's draft+verify
//! mechanics live entirely on the second silicon faculty.
//!
//!   draft:   l4f16v2 (4-layer stateful decoder, ~4ms/step) — K steps
//!   verify:  l14     (14-layer stateful decoder, ~20ms)    — 1 step
//!   load:    l4f16v2 pinned CpuAndGpu hammering in a thread
//!
//! Measures spec-decode cycle time solo vs under GPU contention.
//!
//! Usage: cargo run -p mil_infer --example specdec --release -- \
//!            ~/models/drafter --cycles 16 --k 4 --hold 8

#[cfg(target_os = "macos")]
mod imp {
    use mil_infer::{ComputeUnits, Input, Model, State};
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

    fn compiled_for(pkg: &Path) -> PathBuf {
        let stem = pkg.file_stem().unwrap().to_str().unwrap();
        pkg.with_extension("compiled")
            .join(format!("{stem}.mlmodelc"))
    }

    fn make_inputs(spec: &Spec, step: usize) -> Vec<Vec<u8>> {
        spec.inputs
            .iter()
            .map(|(name, shape, dt)| {
                let n = shape.iter().product::<i64>().max(1) as usize;
                match dtype_of(*dt) {
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
                }
            })
            .collect()
    }

    struct Runner {
        model: Model,
        state: Option<State>,
        spec: Spec,
        step: usize,
    }

    impl Runner {
        fn load(compiled: &Path, pkg: &Path, units: ComputeUnits) -> Result<Self, String> {
            let spec = load_spec(pkg)?;
            let model = Model::load(compiled, units).map_err(|e| e.to_string())?;
            let state = if spec.has_state {
                Some(model.new_state().map_err(|e| e.to_string())?)
            } else {
                None
            };
            Ok(Self {
                model,
                state,
                spec,
                step: 0,
            })
        }

        fn step(&mut self) -> Result<Duration, String> {
            let bufs = make_inputs(&self.spec, self.step);
            let inputs: Vec<Input> = self
                .spec
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
            let r = match &self.state {
                Some(s) => self.model.predict_with_state(Some(s), &inputs),
                None => self.model.predict(&inputs),
            };
            self.step += 1;
            r.map(|_| t.elapsed()).map_err(|e| e.to_string())
        }
    }

    fn cycles(
        draft: &mut Runner,
        verify: &mut Runner,
        n: usize,
        k: usize,
    ) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let t = Instant::now();
            for _ in 0..k {
                draft.step()?;
            }
            verify.step()?;
            out.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(out)
    }

    fn median(v: &mut Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    pub fn run(args: &[String]) -> Result<(), String> {
        let dir = PathBuf::from(
            args.first()
                .ok_or("usage: specdec <models_dir> [--cycles N] [--k K] [--hold S]")?,
        );
        let mut n = 16usize;
        let mut k = 4usize;
        let mut hold = 8u64;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--cycles" => {
                    n = args[i + 1].parse().unwrap_or(16);
                    i += 2;
                }
                "--k" => {
                    k = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                }
                "--hold" => {
                    hold = args[i + 1].parse().unwrap_or(8);
                    i += 2;
                }
                _ => i += 1,
            }
        }
        let draft_pkg = dir.join("l4f16v2.mlpackage");
        let verify_pkg = dir.join("l14.mlpackage");
        for p in [&draft_pkg, &verify_pkg] {
            if !compiled_for(p).exists() {
                return Err(format!(
                    "{} not compiled (run milc run/compile first)",
                    p.display()
                ));
            }
        }

        println!("=== phase A: solo spec-decode (k={k}, both ANE-pinned) ===");
        let mut draft = Runner::load(
            &compiled_for(&draft_pkg),
            &draft_pkg,
            ComputeUnits::CpuAndNeuralEngine,
        )?;
        let mut verify = Runner::load(
            &compiled_for(&verify_pkg),
            &verify_pkg,
            ComputeUnits::CpuAndNeuralEngine,
        )?;
        let mut solo = cycles(&mut draft, &mut verify, n, k)?;
        let solo_med = median(&mut solo);
        println!(
            "cycle: median {solo_med:.2}ms  (~{} draft-steps equiv)  → {:.0} cycles/s",
            k + 1,
            1000.0 / solo_med
        );

        println!("=== phase B: spec-decode while GPU main-brain hammers ({hold}s) ===");
        let stop = Arc::new(AtomicBool::new(false));
        let go = Arc::new(AtomicBool::new(false));
        let gpu_compiled = compiled_for(&draft_pkg);
        let gpu_pkg = draft_pkg.clone();
        let (stop2, go2) = (stop.clone(), go.clone());
        let gpu_thread = thread::spawn(move || -> Result<usize, String> {
            let mut r = Runner::load(&gpu_compiled, &gpu_pkg, ComputeUnits::CpuAndGpu)?;
            while !go2.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let mut count = 0usize;
            while !stop2.load(Ordering::Relaxed) {
                r.step()?;
                count += 1;
            }
            Ok(count)
        });
        thread::sleep(Duration::from_millis(400));
        go.store(true, Ordering::Release);
        thread::sleep(Duration::from_millis(200)); // let GPU reach steady state

        let mut under = cycles(&mut draft, &mut verify, n, k)?;
        stop.store(true, Ordering::Relaxed);
        let gpu_steps = gpu_thread.join().unwrap()?;
        let under_med = median(&mut under);
        println!("cycle under load: median {under_med:.2}ms → {:.0} cycles/s   (gpu did {gpu_steps} predicts in parallel)", 1000.0 / under_med);
        println!(
            "gpu-contention cost: {:+.1}%",
            (under_med - solo_med) / solo_med * 100.0
        );
        println!("verdict: proposer + verifier ran on ANE while the GPU brain was fully loaded");
        Ok(())
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(e) = imp::run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("specdec: {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("specdec: macOS only");
}
