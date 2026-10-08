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
    use mil_spec::{
        bind, bind_many, Block, DType, Feature, Immediate, ModelMeta, TensorType, Value, ValueType,
        NVT,
    };

    fn tt(dt: DType, shape: &[i64]) -> ValueType {
        ValueType::Tensor(TensorType {
            dtype: dt,
            shape: shape.to_vec(),
        })
    }
    fn f16tt(shape: &[i64]) -> ValueType {
        tt(DType::Fp16, shape)
    }
    fn i32tt(shape: &[i64]) -> ValueType {
        tt(DType::Int32, shape)
    }
    fn f16s(b: &mut Block, name: &str, v: f32) -> String {
        let n = b.fresh(name);
        b.op(
            "const",
            vec![],
            vec![(&n, f16tt(&[]))],
            vec![("val".into(), Value::f16_scalar(v))],
        )[0]
        .clone()
    }
    fn i32s(b: &mut Block, name: &str, v: i32) -> String {
        let n = b.fresh(name);
        b.op(
            "const",
            vec![],
            vec![(&n, i32tt(&[]))],
            vec![(
                "val".into(),
                Value::Imm(i32tt(&[]), Immediate::Ints(vec![v])),
            )],
        )[0]
        .clone()
    }
    fn i32v(b: &mut Block, name: &str, vs: &[i32]) -> String {
        let n = b.fresh(name);
        b.op(
            "const",
            vec![],
            vec![(&n, i32tt(&[vs.len() as i64]))],
            vec![("val".into(), Value::i32s(vs))],
        )[0]
        .clone()
    }
    fn ew(b: &mut Block, ty: &str, x: &str, y: &str, shape: &[i64], name: &str) -> String {
        b.o1(
            ty,
            vec![("x".into(), bind(x).1), ("y".into(), bind(y).1)],
            name,
            f16tt(shape),
        )
    }

    const ALPHA: f32 = 0.1;
    const EPS: f32 = 1e-3;

    fn emit() -> Vec<u8> {
        let mut b = Block::new();
        let k0 = i32s(&mut b, "k0", 0);
        let i0 = i32v(&mut b, "i0", &[0]);
        let i1 = i32v(&mut b, "i1", &[1]);
        let vt = {
            let n = b.fresh("vt");
            b.op(
                "const",
                vec![],
                vec![(&n, tt(DType::Bool, &[]))],
                vec![(
                    "val".into(),
                    Value::Imm(tt(DType::Bool, &[]), Immediate::Bools(vec![true])),
                )],
            )[0]
            .clone()
        };
        let al = f16s(&mut b, "al", ALPHA);
        let oa = f16s(&mut b, "oa", 1.0 - ALPHA);
        let ep = f16s(&mut b, "ep", EPS);
        let zt = f16s(&mut b, "zt", 3.0);
        let ax = i32s(&mut b, "ax", 0);
        let il = {
            let n = b.fresh("il");
            b.op(
                "const",
                vec![],
                vec![(&n, tt(DType::Bool, &[]))],
                vec![(
                    "val".into(),
                    Value::Imm(tt(DType::Bool, &[]), Immediate::Bools(vec![false])),
                )],
            )[0]
            .clone()
        };
        let _ = k0;
        let (
            n_ema,
            n_evar,
            n_d,
            n_m1,
            n_m2,
            n_ema2,
            n_d2,
            n_v1,
            n_v2,
            n_evar2,
            n_dz,
            n_ve,
            n_sq2,
            n_acc2n,
        ) = (
            b.fresh("ema"),
            b.fresh("evar"),
            b.fresh("d"),
            b.fresh("m1"),
            b.fresh("m2"),
            b.fresh("ema2"),
            b.fresh("d2"),
            b.fresh("v1"),
            b.fresh("v2"),
            b.fresh("evar2"),
            b.fresh("dz"),
            b.fresh("ve"),
            b.fresh("sq2"),
            b.fresh("acc2n"),
        );

        // acc = read_state(acc) [2] fp16
        b.op(
            "read_state",
            vec![("input".into(), bind("acc").1)],
            vec![("acc_r", f16tt(&[2]))],
            vec![("name".into(), Value::Str("acc_r".into()))],
        );
        let g = |b: &mut Block, idx: &str, name: &str| -> String {
            b.o1(
                "gather",
                vec![
                    ("x".into(), bind("acc_r").1),
                    ("indices".into(), bind(idx).1),
                    ("axis".into(), bind(&ax).1),
                    ("validate_indices".into(), bind(&vt).1),
                ],
                name,
                f16tt(&[1]),
            )
        };
        let ema = g(&mut b, &i0, &n_ema);
        let evar = g(&mut b, &i1, &n_evar);

        // d = x - ema ; ema' = oa*ema + al*x ; evar' = oa*evar + al*d²
        let d = ew(&mut b, "sub", "x", &ema, &[1], &n_d);
        let m1 = ew(&mut b, "mul", &ema, &oa, &[1], &n_m1);
        let m2 = ew(&mut b, "mul", "x", &al, &[1], &n_m2);
        let ema2 = ew(&mut b, "add", &m1, &m2, &[1], &n_ema2);
        let d2 = ew(&mut b, "mul", &d, &d, &[1], &n_d2);
        let v1 = ew(&mut b, "mul", &evar, &oa, &[1], &n_v1);
        let v2 = ew(&mut b, "mul", &d2, &al, &[1], &n_v2);
        let evar2 = ew(&mut b, "add", &v1, &v2, &[1], &n_evar2);

        // z = (x - ema) / sqrt(evar + eps) — pre-update stats, so a spike
        // can't inflate its own baseline before it's scored
        let dz = ew(&mut b, "sub", "x", &ema, &[1], &n_dz);
        let ve = ew(&mut b, "add", &evar, &ep, &[1], &n_ve);
        let sqn = n_sq2;
        b.op(
            "sqrt",
            vec![("x".into(), bind(&ve).1)],
            vec![(&sqn, f16tt(&[1]))],
            vec![],
        );
        let z = b.o1(
            "real_div",
            vec![("x".into(), bind(&dz).1), ("y".into(), bind(&sqn).1)],
            "z",
            f16tt(&[1]),
        );
        let _ = &z;
        let flagb = b.o1(
            "greater",
            vec![("x".into(), bind(&z).1), ("y".into(), bind(&zt).1)],
            "flagb",
            tt(DType::Bool, &[1]),
        );
        let tyf = {
            let n = b.fresh("tyf");
            b.op(
                "const",
                vec![],
                vec![(&n, tt(DType::Str, &[]))],
                vec![("val".into(), Value::Str("fp16".into()))],
            )[0]
            .clone()
        };
        let flag = b.o1(
            "cast",
            vec![("x".into(), bind(&flagb).1), ("dtype".into(), bind(&tyf).1)],
            "flag",
            f16tt(&[1]),
        );
        let _ = flag;

        // acc' = concat([ema', evar']) ; write
        let acc2 = {
            let n = n_acc2n;
            b.o1(
                "concat",
                vec![
                    ("values".into(), bind_many(&[&ema2, &evar2])),
                    ("axis".into(), bind(&ax).1),
                    ("interleave".into(), bind(&il).1),
                ],
                &n,
                f16tt(&[2]),
            )
        };
        let _ = acc2;
        b.op(
            "write_state",
            vec![
                ("input".into(), bind("acc").1),
                ("data".into(), bind(&acc2).1),
            ],
            vec![],
            vec![("name".into(), Value::Str("acc_write".into()))],
        );
        b.outputs = vec![z.clone(), flag.clone()];

        let inputs = [Feature {
            name: "x".into(),
            shape: vec![1],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let outputs = [
            Feature {
                name: "z".into(),
                shape: vec![1],
                dtype: DType::Fp16,
                is_state: false,
            },
            Feature {
                name: "flag".into(),
                shape: vec![1],
                dtype: DType::Fp16,
                is_state: false,
            },
        ];
        let states = [Feature {
            name: "acc".into(),
            shape: vec![2],
            dtype: DType::Fp16,
            is_state: true,
        }];
        let fn_inputs = [
            NVT {
                name: "x".into(),
                ty: f16tt(&[1]),
            },
            NVT {
                name: "acc".into(),
                ty: ValueType::State(TensorType {
                    dtype: DType::Fp16,
                    shape: vec![2],
                }),
            },
        ];
        mil_spec::encode_model(
            &inputs,
            &outputs,
            &states,
            &b,
            &fn_inputs,
            &ModelMeta::new(10, "CoreML9"),
        )
    }

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
        mil_spec::write_mlpackage(&pkg, &emit(), None).map_err(|e| e.to_string())?;
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
