//! Hardware automaton — experiment 1 of the silicon-duel series.
//!
//! Compiles an Aho-Corasick DFA into a *stateful* MIL graph and runs it on
//! Apple's runtime as a real model: the transition table is a const weight,
//! the DFA state is an `MLState`, each predict steps one byte:
//!
//!   read_state cur → idx = cur*256 + byte → gather(table, idx) = next
//!   → gather(verdict, next) = hit → write_state(next)
//!
//! A deterministic string matcher expressed as matmul-class silicon ops —
//! a workload the Neural Engine was never designed for, running on the
//! stack nobody else emits to.
//!
//! Usage: cargo run -p mil_infer --example dfa --release

#[cfg(target_os = "macos")]
mod imp {
    use mil_infer::{ComputeUnits, Input, Model};
    use mil_spec::{
        bind, Block, DType, Feature, Immediate, ModelMeta, TensorType, Value, ValueType, NVT,
    };
    use std::collections::VecDeque;

    // ---------- software Aho-Corasick (reference + table source) ----------

    struct Ac {
        next: Vec<[i32; 256]>,
        out: Vec<i32>,
    }

    fn build_ac(patterns: &[&[u8]]) -> Ac {
        let mut next = vec![[-1i32; 256]];
        let mut out = vec![0i32];
        for pat in patterns {
            let mut s = 0usize;
            for &c in pat.iter() {
                let n = next[s][c as usize];
                if n < 0 {
                    let ns = next.len() as i32;
                    next[s][c as usize] = ns;
                    next.push([-1; 256]);
                    out.push(0);
                    s = ns as usize;
                } else {
                    s = n as usize;
                }
            }
            out[s] = 1;
        }
        // failure links, then flatten goto into a full DFA
        let mut fail = vec![0usize; next.len()];
        let mut q = VecDeque::new();
        for c in 0..256 {
            let n = next[0][c];
            if n >= 0 {
                q.push_back(n as usize);
            } else {
                next[0][c] = 0;
            }
        }
        while let Some(u) = q.pop_front() {
            for c in 0..256 {
                let v = next[u][c];
                if v >= 0 {
                    let mut f = fail[u];
                    while next[f][c] < 0 {
                        f = fail[f];
                    }
                    fail[v as usize] = next[f][c] as usize;
                    out[v as usize] |= out[fail[v as usize]];
                    q.push_back(v as usize);
                }
            }
        }
        // complete the DFA: fill missing edges through failure chain
        for s in 0..next.len() {
            for c in 0..256 {
                if next[s][c] < 0 {
                    let mut f = fail[s];
                    while next[f][c] < 0 {
                        f = fail[f];
                    }
                    next[s][c] = next[f][c];
                }
            }
        }
        Ac { next, out }
    }

    fn ac_run(ac: &Ac, bytes: &[u8]) -> Vec<i32> {
        let mut s = 0usize;
        bytes
            .iter()
            .map(|&b| {
                s = ac.next[s][b as usize] as usize;
                ac.out[s]
            })
            .collect()
    }

    // ---------- MIL graph emission ----------

    fn i32tt(shape: &[i64]) -> ValueType {
        ValueType::Tensor(TensorType {
            dtype: DType::Int32,
            shape: shape.to_vec(),
        })
    }

    fn scalar_i32(v: i32) -> Value {
        Value::Imm(i32tt(&[]), Immediate::Ints(vec![v]))
    }

    fn emit(ac: &Ac) -> Vec<u8> {
        let n = ac.next.len();
        let table: Vec<i32> = ac.next.iter().flatten().copied().collect();
        let mut b = Block::new();

        // consts
        b.op(
            "const",
            vec![],
            vec![("table", i32tt(&[(n * 256) as i64]))],
            vec![("val".into(), Value::i32s(&table))],
        );
        b.op(
            "const",
            vec![],
            vec![("vtab", i32tt(&[n as i64]))],
            vec![("val".into(), Value::i32s(&ac.out))],
        );
        b.op(
            "const",
            vec![],
            vec![("k256", i32tt(&[]))],
            vec![("val".into(), scalar_i32(256))],
        );
        b.op(
            "const",
            vec![],
            vec![("k0", i32tt(&[]))],
            vec![("val".into(), scalar_i32(0))],
        );
        b.konst_bool("kv_true", true);

        // const str for cast targets
        b.op(
            "const",
            vec![],
            vec![(
                "ty_i32",
                ValueType::Tensor(TensorType {
                    dtype: DType::Str,
                    shape: vec![],
                }),
            )],
            vec![("val".into(), Value::Str("int32".into()))],
        );
        b.op(
            "const",
            vec![],
            vec![(
                "ty_f16",
                ValueType::Tensor(TensorType {
                    dtype: DType::Str,
                    shape: vec![],
                }),
            )],
            vec![("val".into(), Value::Str("fp16".into()))],
        );

        // read current state (fp16 [1] — CoreML requires float16 state)
        b.op(
            "read_state",
            vec![("input".into(), bind("cur").1)],
            vec![(
                "s_f",
                ValueType::Tensor(TensorType {
                    dtype: DType::Fp16,
                    shape: vec![1],
                }),
            )],
            vec![("name".into(), Value::Str("s_f".into()))],
        );
        // s = cast(s_f → int32)
        b.op(
            "cast",
            vec![
                ("x".into(), bind("s_f").1),
                ("dtype".into(), bind("ty_i32").1),
            ],
            vec![("s", i32tt(&[1]))],
            vec![],
        );
        // idx = cur * 256 + byte
        b.op(
            "mul",
            vec![("x".into(), bind("s").1), ("y".into(), bind("k256").1)],
            vec![("s256", i32tt(&[1]))],
            vec![],
        );
        b.op(
            "add",
            vec![("x".into(), bind("s256").1), ("y".into(), bind("byte").1)],
            vec![("idx", i32tt(&[1]))],
            vec![],
        );
        // next = table[idx]; hit = verdict[next]
        b.op(
            "gather",
            vec![
                ("x".into(), bind("table").1),
                ("indices".into(), bind("idx").1),
                ("axis".into(), bind("k0").1),
                ("validate_indices".into(), bind("kv_true").1),
            ],
            vec![("nxt", i32tt(&[1]))],
            vec![],
        );
        b.op(
            "gather",
            vec![
                ("x".into(), bind("vtab").1),
                ("indices".into(), bind("nxt").1),
                ("axis".into(), bind("k0").1),
                ("validate_indices".into(), bind("kv_true").1),
            ],
            vec![("hit", i32tt(&[1]))],
            vec![],
        );
        // cast next back to fp16 for the state slot
        b.op(
            "cast",
            vec![
                ("x".into(), bind("nxt").1),
                ("dtype".into(), bind("ty_f16").1),
            ],
            vec![(
                "nxt_f",
                ValueType::Tensor(TensorType {
                    dtype: DType::Fp16,
                    shape: vec![1],
                }),
            )],
            vec![],
        );
        b.op(
            "write_state",
            vec![
                ("input".into(), bind("cur").1),
                ("data".into(), bind("nxt_f").1),
            ],
            vec![],
            vec![("name".into(), Value::Str("cur_write".into()))],
        );
        b.outputs = vec!["hit".into()];

        let inputs = [Feature {
            name: "byte".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        }];
        let outputs = [Feature {
            name: "hit".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        }];
        let states = [Feature {
            name: "cur".into(),
            shape: vec![1],
            dtype: DType::Fp16,
            is_state: true,
        }];
        let fn_inputs = [
            NVT {
                name: "byte".into(),
                ty: i32tt(&[1]),
            },
            NVT {
                name: "cur".into(),
                ty: ValueType::State(TensorType {
                    dtype: DType::Fp16,
                    shape: vec![1],
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
        let patterns: Vec<&[u8]> = vec![
            b"password",
            b"api_key",
            b"secret",
            b"ssh-rsa",
            b"BEGIN PRIVATE",
        ];
        let ac = build_ac(&patterns);
        println!(
            "dfa: {} states, table {} B",
            ac.next.len(),
            ac.next.len() * 256 * 4
        );

        let dir = std::env::temp_dir().join("mil_dfa");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pkg = dir.join("dfa.mlpackage");
        mil_spec::write_mlpackage(&pkg, &emit(&ac), None).map_err(|e| e.to_string())?;
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

        let stream: Vec<u8> = b"hello world my password is hunter2 and the api_key=xyz not a secret, ssh-rsa ok BEGIN PRIVATE"
            .to_vec();
        let want = ac_run(&ac, &stream);
        let mut got = Vec::new();
        let mut t_all = std::time::Duration::ZERO;
        for (i, &byte) in stream.iter().enumerate() {
            let data = (byte as i32).to_le_bytes();
            let p = model
                .predict_with_state(
                    Some(&state),
                    &[Input {
                        name: "byte",
                        shape: &[1],
                        data: &data,
                        dtype: DType::Int32,
                    }],
                )
                .map_err(|e| format!("step {i}: {e}"))?;
            t_all += p.latency;
            let hit = p
                .outputs
                .iter()
                .find(|o| o.name == "hit")
                .and_then(|o| {
                    o.data
                        .get(..4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                })
                .unwrap_or(-1);
            got.push(hit);
        }
        let matches = got.iter().filter(|&&h| h == 1).count();
        let same = got == want;
        println!(
            "streamed {} bytes → {} hit positions (software: {}) — {} in {:.1}ms total ({:.0}µs/step)",
            stream.len(),
            matches,
            want.iter().filter(|&&h| h == 1).count(),
            if same { "MATCH" } else { "MISMATCH" },
            t_all.as_secs_f64() * 1e3,
            t_all.as_secs_f64() * 1e6 / stream.len() as f64,
        );
        if !same {
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                if g != w {
                    println!(
                        "  first mismatch at byte {} ('{}'): ane={} sw={}",
                        i, stream[i] as char, g, w
                    );
                    break;
                }
            }
            return Err("hardware automaton disagrees with software DFA".into());
        }
        println!("verdict: the automaton ran on the model runtime, byte-exact");
        let _ = pkg;
        Ok(())
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(e) = imp::run() {
        eprintln!("dfa: {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("dfa: macOS only");
}
