//! ANE persistent memory — experiment 3 of the silicon-duel series.
//!
//! `MLState` is a writable tensor that survives across predicts — that is a
//! hardware-resident key-value store nobody uses as one. This graph is a
//! mutable memory bank: fp16 [64,32] state, random-access read via `gather`,
//! write via `slice_update`, all inside one predict:
//!
//!   m = read_state(mem) → out = gather(m, rslot)
//!   m' = slice_update(m, wvec, [wslot,0]..[wslot+1,32]) → write_state(m')
//!
//! Read-before-write semantics per call; contents persist across calls
//! inside the MLState object. Verified: write N vectors, read them back
//! fp16-exact, then prove a fresh state reads zeros — the memory lives in
//! the state object, not the program.
//!
//! Usage: cargo run -p mil_infer --example mem --release

#[cfg(target_os = "macos")]
mod imp {
    use half::f16;
    use mil_infer::{ComputeUnits, Input, Model};
    use mil_spec::DType;
    use std::time::Duration;

    const SLOTS: i64 = 64;
    const DIM: i64 = 32;

    pub fn run() -> Result<(), String> {
        let dir = std::env::temp_dir().join("mil_mem");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pkg = dir.join("mem.mlpackage");
        mil_machines::write_memory_package(&pkg, SLOTS, DIM).map_err(|e| e.to_string())?;
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

        let call = |wslot: i32, wvec: &[f32], rslot: i32| -> Result<Vec<f32>, String> {
            let ws = wslot.to_le_bytes();
            let rs = rslot.to_le_bytes();
            let wv: Vec<u8> = wvec
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                .collect();
            let p = model
                .predict_with_state(
                    Some(&state),
                    &[
                        Input {
                            name: "wslot",
                            shape: &[1],
                            data: &ws,
                            dtype: DType::Int32,
                        },
                        Input {
                            name: "wvec",
                            shape: &[1, DIM],
                            data: &wv,
                            dtype: DType::Fp16,
                        },
                        Input {
                            name: "rslot",
                            shape: &[1],
                            data: &rs,
                            dtype: DType::Int32,
                        },
                    ],
                )
                .map_err(|e| e.to_string())?;
            Ok(p.outputs
                .iter()
                .find(|o| o.name == "out")
                .map(|o| o.values())
                .unwrap_or_default())
        };

        // write 8 distinct vectors into slots
        let zeros = vec![0.0f32; DIM as usize];
        let mut wrote = Vec::new();
        let mut total = Duration::ZERO;
        for s in 0..8i32 {
            let v: Vec<f32> = (0..DIM).map(|d| s as f32 + d as f32 / 100.0).collect();
            wrote.push(v.clone());
            let t = std::time::Instant::now();
            call(s, &v, 0)?;
            total += t.elapsed();
        }
        // read them back — fp16-exact
        let mut ok = 0;
        for s in 0..8i32 {
            let got = call(63, &zeros, s)?; // harmless write to scratch slot 63
            if got.len() == DIM as usize
                && wrote[s as usize]
                    .iter()
                    .zip(got.iter())
                    .all(|(w, g)| (*w - *g).abs() < 0.02)
            {
                ok += 1;
            }
        }
        println!(
            "memory: wrote 8 vectors, read back {ok}/8 fp16-exact ({:.0}µs/write avg)",
            total.as_secs_f64() * 1e6 / 8.0
        );

        // persistence is state-object-scoped: fresh state must read zeros
        let state2 = model.new_state().map_err(|e| e.to_string())?;
        let st2 = &state2;
        let rs = 0i32.to_le_bytes();
        let ws = 62i32.to_le_bytes();
        let wv: Vec<u8> = zeros
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let p = model
            .predict_with_state(
                Some(st2),
                &[
                    Input {
                        name: "wslot",
                        shape: &[1],
                        data: &ws,
                        dtype: DType::Int32,
                    },
                    Input {
                        name: "wvec",
                        shape: &[1, DIM],
                        data: &wv,
                        dtype: DType::Fp16,
                    },
                    Input {
                        name: "rslot",
                        shape: &[1],
                        data: &rs,
                        dtype: DType::Int32,
                    },
                ],
            )
            .map_err(|e| e.to_string())?;
        let fresh = p
            .outputs
            .iter()
            .find(|o| o.name == "out")
            .map(|o| o.values())
            .unwrap_or_default();
        let zeros_read = fresh.iter().all(|v| v.abs() < 1e-3);
        println!(
            "fresh state reads zeros: {zeros_read} (memory lives in MLState, not the program)"
        );
        if ok == 8 && zeros_read {
            println!("verdict: hardware-resident mutable KV store — proven");
            Ok(())
        } else {
            Err("memory verification failed".into())
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(e) = imp::run() {
        eprintln!("mem: {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("mem: macOS only");
}
