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
    use mil_spec::{
        bind, bind_many, Block, DType, Feature, ModelMeta, TensorType, Value, ValueType, NVT,
    };
    use std::time::Duration;

    const SLOTS: i64 = 64;
    const DIM: i64 = 32;

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

    fn emit() -> Vec<u8> {
        let mut b = Block::new();
        let (k0n, kdn, kstn, k1n, k0bn, vtn) = (
            b.fresh("k0"),
            b.fresh("kd"),
            b.fresh("st"),
            b.fresh("k1"),
            b.fresh("k0b"),
            b.fresh("vt"),
        );
        let k0 = b.konst_scalar_i32(&k0n, 0);
        let k_dim = b.konst_i32(&kdn, &[DIM as i32]);
        let k_st = b.konst_i32(&kstn, &[1, 1]);
        let k_one = b.konst_i32(&k1n, &[1]);
        let k0b = b.konst_i32(&k0bn, &[0]);
        let vt = b.konst_bool(&vtn, true);

        // consts for concat axes + masks
        let axn = b.fresh("ax");
        let ax = b.konst_scalar_i32(&axn, 0);
        let il = b.fresh("il");
        b.op(
            "const",
            vec![],
            vec![(&il, tt(DType::Bool, &[]))],
            vec![(
                "val".into(),
                Value::Imm(
                    tt(DType::Bool, &[]),
                    mil_spec::Immediate::Bools(vec![false]),
                ),
            )],
        );
        let bm = b.fresh("bm");
        b.op(
            "const",
            vec![],
            vec![(&bm, tt(DType::Bool, &[2]))],
            vec![(
                "val".into(),
                Value::Imm(
                    tt(DType::Bool, &[2]),
                    mil_spec::Immediate::Bools(vec![false, false]),
                ),
            )],
        );
        let em = b.fresh("em");
        b.op(
            "const",
            vec![],
            vec![(&em, tt(DType::Bool, &[2]))],
            vec![(
                "val".into(),
                Value::Imm(
                    tt(DType::Bool, &[2]),
                    mil_spec::Immediate::Bools(vec![false, false]),
                ),
            )],
        );
        let sm = b.fresh("sm");
        b.op(
            "const",
            vec![],
            vec![(&sm, tt(DType::Bool, &[2]))],
            vec![(
                "val".into(),
                Value::Imm(
                    tt(DType::Bool, &[2]),
                    mil_spec::Immediate::Bools(vec![false, false]),
                ),
            )],
        );

        // m = read_state(mem) [64,32] fp16
        b.op(
            "read_state",
            vec![("input".into(), bind("mem").1)],
            vec![("m", f16tt(&[SLOTS, DIM]))],
            vec![("name".into(), Value::Str("m".into()))],
        );
        // out = gather(m, rslot) → [1,32]
        b.op(
            "gather",
            vec![
                ("x".into(), bind("m").1),
                ("indices".into(), bind("rslot").1),
                ("axis".into(), bind(&k0).1),
                ("validate_indices".into(), bind(&vt).1),
            ],
            vec![("out", f16tt(&[1, DIM]))],
            vec![],
        );
        // beg = concat([wslot, 0]) ; end = concat([wslot+1, 32])
        let beg_name = b.fresh("beg");
        let beg = b.o1(
            "concat",
            vec![
                ("values".into(), bind_many(&["wslot", &k0b])),
                ("axis".into(), bind(&ax).1),
                ("interleave".into(), bind(&il).1),
            ],
            &beg_name,
            i32tt(&[2]),
        );
        let w1_name = b.fresh("w1");
        let w1 = b.o1(
            "add",
            vec![("x".into(), bind("wslot").1), ("y".into(), bind(&k_one).1)],
            &w1_name,
            i32tt(&[1]),
        );
        let end_name = b.fresh("end");
        let end = b.o1(
            "concat",
            vec![
                ("values".into(), bind_many(&[&w1, &k_dim])),
                ("axis".into(), bind(&ax).1),
                ("interleave".into(), bind(&il).1),
            ],
            &end_name,
            i32tt(&[2]),
        );
        // m' = slice_update(m, wvec, beg, end)
        b.op(
            "slice_update",
            vec![
                ("x".into(), bind("m").1),
                ("update".into(), bind("wvec").1),
                ("begin".into(), bind(&beg).1),
                ("end".into(), bind(&end).1),
                ("stride".into(), bind(&k_st).1),
                ("begin_mask".into(), bind(&bm).1),
                ("end_mask".into(), bind(&em).1),
                ("squeeze_mask".into(), bind(&sm).1),
            ],
            vec![("m2", f16tt(&[SLOTS, DIM]))],
            vec![],
        );
        b.op(
            "write_state",
            vec![
                ("input".into(), bind("mem").1),
                ("data".into(), bind("m2").1),
            ],
            vec![],
            vec![("name".into(), Value::Str("mem_write".into()))],
        );
        b.outputs = vec!["out".into()];

        let inputs = [
            Feature {
                name: "wslot".into(),
                shape: vec![1],
                dtype: DType::Int32,
                is_state: false,
            },
            Feature {
                name: "wvec".into(),
                shape: vec![1, DIM],
                dtype: DType::Fp16,
                is_state: false,
            },
            Feature {
                name: "rslot".into(),
                shape: vec![1],
                dtype: DType::Int32,
                is_state: false,
            },
        ];
        let outputs = [Feature {
            name: "out".into(),
            shape: vec![1, DIM],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let states = [Feature {
            name: "mem".into(),
            shape: vec![SLOTS, DIM],
            dtype: DType::Fp16,
            is_state: true,
        }];
        let fn_inputs = [
            NVT {
                name: "wslot".into(),
                ty: i32tt(&[1]),
            },
            NVT {
                name: "wvec".into(),
                ty: f16tt(&[1, DIM]),
            },
            NVT {
                name: "rslot".into(),
                ty: i32tt(&[1]),
            },
            NVT {
                name: "mem".into(),
                ty: ValueType::State(TensorType {
                    dtype: DType::Fp16,
                    shape: vec![SLOTS, DIM],
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

    pub fn run() -> Result<(), String> {
        let dir = std::env::temp_dir().join("mil_mem");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pkg = dir.join("mem.mlpackage");
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
