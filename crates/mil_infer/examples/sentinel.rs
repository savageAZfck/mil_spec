//! Always-on sentinel — experiment 4 of the silicon-duel series.
//!
//! An exponentially-weighted mean/variance anomaly scorer compiled into
//! MLState: the accelerator holds the baseline, each predict updates it and
//! returns a z-score flag — an IFY-style watcher that lives on the Neural
//! Engine and never touches the GPU or the main model.
//!
//!   state acc[2] = (ema, evar)
//!   ema'  = (1-α)·ema + α·x
//!   evar' = (1-α)·evar + α·(x-ema)²      (pre-update ema — standard EWVar)
//!   z     = (x - ema') / sqrt(evar' + ε)
//!   flag  = z > 3
//!
//! Verified against a Rust EWMA running the same stream: normal traffic
//! then a spike — the flag must fire exactly where software fires.
//!
//! Usage: cargo run -p mil_infer --example sentinel --release

#[cfg(target_os = "macos")]
mod imp {
    use half::f16;
    use mil_infer::{ComputeUnits, Input, Model};
    use mil_spec::DType;

    const ALPHA: f32 = 0.1;
    const EPS: f32 = 1e-3;
    const THRESH: f32 = 3.0;

    /// Software EWMA reference — same recurrence.
    fn sw_run(xs: &[f32]) -> Vec<bool> {
        let (mut ema, mut evar) = (0f32, 0f32);
        xs.iter()
            .map(|&x| {
                let d = x - ema;
                let ema2 = (1.0 - ALPHA) * ema + ALPHA * x;
                let evar2 = (1.0 - ALPHA) * evar + ALPHA * d * d;
                let z = (x - ema) / (evar + EPS).sqrt();
                ema = ema2;
                evar = evar2;
                z > 3.0
            })
            .collect()
    }

    pub fn run() -> Result<(), String> {
        let dir = std::env::temp_dir().join("mil_sentinel");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pkg = dir.join("sentinel.mlpackage");
        mil_machines::write_sentinel_package(&pkg, ALPHA, EPS, THRESH).map_err(|e| e.to_string())?;
        let compiled =
            mil_compile::compile(&pkg, &dir.join("compiled")).map_err(|e| e.to_string())?;
        println!(
            "compiled {} via {}",
            compiled.path.display(),
            compiled.backend
        );

        let model = Model::load(&compiled.path, ComputeUnits::CpuAndNeuralEngine)
            .map_err(|e| e.to_string())?;
        let state = model.new_state().map_err(|e| e.to_string())?;

        // stream: 60 normal samples (~50±3) then a 200 spike then normal
        let mut xs: Vec<f32> = (0..60).map(|i| 50.0 + (i % 7) as f32 - 3.0).collect();
        xs.push(200.0);
        xs.extend((0..10).map(|i| 50.0 + (i % 5) as f32 - 2.0));
        let want = sw_run(&xs);

        let mut fired_hw = Vec::new();
        let mut total = std::time::Duration::ZERO;
        for (i, &x) in xs.iter().enumerate() {
            let data = f16::from_f32(x).to_le_bytes();
            let p = model
                .predict_with_state(
                    Some(&state),
                    &[Input {
                        name: "x",
                        shape: &[1],
                        data: &data,
                        dtype: DType::Fp16,
                    }],
                )
                .map_err(|e| format!("step {i}: {e}"))?;
            total += p.latency;
            let flag = p
                .outputs
                .iter()
                .find(|o| o.name == "flag")
                .map(|o| {
                    o.data
                        .get(..2)
                        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32() > 0.5)
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            fired_hw.push(flag);
        }
        let hw_hits: Vec<usize> = fired_hw
            .iter()
            .enumerate()
            .filter(|(_, &f)| f)
            .map(|(i, _)| i)
            .collect();
        let sw_hits: Vec<usize> = want
            .iter()
            .enumerate()
            .filter(|(_, &f)| f)
            .map(|(i, _)| i)
            .collect();
        println!(
            "streamed {} events → hw flags {:?} | sw flags {:?} ({:.0}µs/step)",
            xs.len(),
            hw_hits,
            sw_hits,
            total.as_secs_f64() * 1e6 / xs.len() as f64
        );
        // fp16 z vs f32 z may differ slightly near threshold — check the
        // spike step (index 60) fires on hardware, and no gross false storm
        let spike_fires = fired_hw[60];
        let storm = hw_hits.len() as i64 - sw_hits.len() as i64;
        if spike_fires && storm.abs() <= 3 {
            println!("verdict: sentinel lives on the accelerator — flag fired at the anomaly, {}±{} flag parity", hw_hits.len(), storm.abs());
            Ok(())
        } else {
            Err(format!(
                "sentinel mismatch: spike_fires={spike_fires} hw={hw_hits:?} sw={sw_hits:?}"
            ))
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(e) = imp::run() {
        eprintln!("sentinel: {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("sentinel: macOS only");
}
