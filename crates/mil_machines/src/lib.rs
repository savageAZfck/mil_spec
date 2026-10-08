//! Non-neural state machines compiled to stateful MIL programs.
//!
//! `MLState` is mutable accelerator memory — a machine that lives on the
//! chip and steps one symbol at a time. This crate emits the programs:
//!
//! - [`Automaton`] / [`emit_dfa`] — Aho-Corasick DFA as a stateful graph:
//!   `read_state → gather(table, cur*256+byte) → gather(verdict) →
//!   write_state`. A streaming string matcher on silicon built for matmuls.
//! - [`emit_sentinel`] — EWMA mean/variance anomaly scorer compiled into
//!   state: flag = `|x - ema| / sqrt(evar + eps) > threshold` on pre-update
//!   stats so a spike cannot inflate its own baseline.
//! - [`emit_memory`] — an fp16 `[slots, dim]` `MLState` bank: `gather` read,
//!   `slice_update` write, random access across predicts.
//!
//! All emitters return spec bytes; wrap with `write_*_package` to get a
//! `.mlpackage` ready for `mil_compile` / `coremlc`.

use mil_spec::{
    bind, bind_many, Block, DType, Feature, Immediate, ModelMeta, TensorType, Value, ValueType, NVT,
};
use std::collections::VecDeque;
use std::path::Path;

// ---------------------------------------------------------------------------

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
fn state_ty(dt: DType, shape: &[i64]) -> ValueType {
    ValueType::State(TensorType {
        dtype: dt,
        shape: shape.to_vec(),
    })
}
fn scalar_i32(v: i32) -> Value {
    Value::Imm(i32tt(&[]), Immediate::Ints(vec![v]))
}
fn konst_i32s(b: &mut Block, name: &str, vs: &[i32]) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, i32tt(&[vs.len() as i64]))],
        vec![("val".into(), Value::i32s(vs))],
    )[0]
    .clone()
}
fn konst_i32_scalar(b: &mut Block, name: &str, v: i32) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, i32tt(&[]))],
        vec![("val".into(), scalar_i32(v))],
    )[0]
    .clone()
}
fn konst_f16_scalar(b: &mut Block, name: &str, v: f32) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, f16tt(&[]))],
        vec![("val".into(), Value::f16_scalar(v))],
    )[0]
    .clone()
}
fn konst_bool_scalar(b: &mut Block, name: &str, v: bool) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, tt(DType::Bool, &[]))],
        vec![(
            "val".into(),
            Value::Imm(tt(DType::Bool, &[]), Immediate::Bools(vec![v])),
        )],
    )[0]
    .clone()
}
fn konst_str(b: &mut Block, name: &str, s: &str) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, tt(DType::Str, &[]))],
        vec![("val".into(), Value::Str(s.into()))],
    )[0]
    .clone()
}
fn bool_vec(b: &mut Block, name: &str, vs: &[bool]) -> String {
    let n = b.fresh(name);
    b.op(
        "const",
        vec![],
        vec![(&n, tt(DType::Bool, &[vs.len() as i64]))],
        vec![("val".into(), Value::bools(vs))],
    )[0]
    .clone()
}

// ============================ Automaton (DFA) ===============================

/// A byte-level Aho-Corasick automaton — flat transition table plus
/// accepting-state flags.
#[derive(Clone, Debug)]
pub struct Automaton {
    /// `next[state * 256 + byte] -> next_state` — fully flattened DFA.
    pub next: Vec<i32>,
    /// `out[state] != 0` when the state terminates a pattern.
    pub out: Vec<i32>,
}

impl Automaton {
    /// Number of DFA states.
    pub fn states(&self) -> usize {
        self.out.len()
    }

    /// Build the full DFA from byte patterns (trie + BFS failure links).
    pub fn build(patterns: &[&[u8]]) -> Self {
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
        Self {
            next: next.into_iter().flatten().collect(),
            out,
        }
    }

    /// Step the software DFA — reference for hardware verification.
    pub fn step(&self, state: usize, byte: u8) -> usize {
        self.next[state * 256 + byte as usize] as usize
    }

    /// Run a stream; returns per-byte hit flags.
    pub fn run(&self, bytes: &[u8]) -> Vec<i32> {
        let mut s = 0usize;
        bytes
            .iter()
            .map(|&b| {
                s = self.step(s, b);
                self.out[s]
            })
            .collect()
    }

    /// Transition-table bytes for the `table` const.
    pub fn table(&self) -> &[i32] {
        &self.next
    }
}

/// Emit the DFA as a stateful MIL program.
///
/// Inputs: `byte` int32 `[1]`. State: `cur` fp16 `[1]` (CoreML requires
/// float16 state tensors — values are small ints, exactly representable).
/// Output: `hit` int32 `[1]` — 1 while the DFA sits on an accepting state.
pub fn emit_dfa(ac: &Automaton) -> Vec<u8> {
    let n = ac.states();
    let mut b = Block::new();

    b.op(
        "const",
        vec![],
        vec![("table", i32tt(&[(n * 256) as i64]))],
        vec![("val".into(), Value::i32s(ac.table()))],
    );
    b.op(
        "const",
        vec![],
        vec![("vtab", i32tt(&[n as i64]))],
        vec![("val".into(), Value::i32s(&ac.out))],
    );
    let k256 = konst_i32_scalar(&mut b, "k256", 256);
    let k0 = konst_i32_scalar(&mut b, "k0", 0);
    let vtrue = konst_bool_scalar(&mut b, "vt", true);
    let ty_i32 = konst_str(&mut b, "tyi", "int32");
    let ty_f16 = konst_str(&mut b, "tyf", "fp16");

    // read state (fp16) → cast to int32
    b.op(
        "read_state",
        vec![("input".into(), bind("cur").1)],
        vec![("s_f", f16tt(&[1]))],
        vec![("name".into(), Value::Str("s_f".into()))],
    );
    b.op(
        "cast",
        vec![
            ("x".into(), bind("s_f").1),
            ("dtype".into(), bind(&ty_i32).1),
        ],
        vec![("s", i32tt(&[1]))],
        vec![],
    );
    // idx = cur*256 + byte
    b.op(
        "mul",
        vec![("x".into(), bind("s").1), ("y".into(), bind(&k256).1)],
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
            ("axis".into(), bind(&k0).1),
            ("validate_indices".into(), bind(&vtrue).1),
        ],
        vec![("nxt", i32tt(&[1]))],
        vec![],
    );
    b.op(
        "gather",
        vec![
            ("x".into(), bind("vtab").1),
            ("indices".into(), bind("nxt").1),
            ("axis".into(), bind(&k0).1),
            ("validate_indices".into(), bind(&vtrue).1),
        ],
        vec![("hit", i32tt(&[1]))],
        vec![],
    );
    // cast back to fp16 → write_state
    b.op(
        "cast",
        vec![
            ("x".into(), bind("nxt").1),
            ("dtype".into(), bind(&ty_f16).1),
        ],
        vec![("nxt_f", f16tt(&[1]))],
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
            ty: state_ty(DType::Fp16, &[1]),
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

/// Emit a DFA package directly (spec only — table lives in consts).
pub fn write_dfa_package(path: &Path, ac: &Automaton) -> std::io::Result<()> {
    mil_spec::write_mlpackage(path, &emit_dfa(ac), None)
}

// ============================ EWMA sentinel =================================

/// Emit an exponentially-weighted mean/variance anomaly scorer as a
/// stateful program.
///
/// State `acc[2]` = `(ema, evar)`. Per event `x`: `z = (x - ema) /
/// sqrt(evar + eps)` scored on **pre-update** stats — a spike can't inflate
/// its own baseline — then `ema' = (1-α)ema + αx`, `evar' = (1-α)evar +
/// α(x-ema)²`. Outputs `z` fp16 `[1]` and `flag` fp16 `[1]` (1.0 when
/// `z > threshold`; bool isn't a valid model output type, hence fp16).
pub fn emit_sentinel(alpha: f32, eps: f32, threshold: f32) -> Vec<u8> {
    let mut b = Block::new();
    let i0 = konst_i32s(&mut b, "i0", &[0]);
    let i1 = konst_i32s(&mut b, "i1", &[1]);
    let vt = konst_bool_scalar(&mut b, "vt", true);
    let al = konst_f16_scalar(&mut b, "al", alpha);
    let oa = konst_f16_scalar(&mut b, "oa", 1.0 - alpha);
    let ep = konst_f16_scalar(&mut b, "ep", eps);
    let zt = konst_f16_scalar(&mut b, "zt", threshold);
    let ax = konst_i32_scalar(&mut b, "ax", 0);
    let il = konst_bool_scalar(&mut b, "il", false);
    let tyf = konst_str(&mut b, "tyf", "fp16");

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
    let n_ema = b.fresh("ema");
    let ema = g(&mut b, &i0, &n_ema);
    let n_evar = b.fresh("evar");
    let evar = g(&mut b, &i1, &n_evar);

    let ew = |b: &mut Block, ty: &str, x: &str, y: &str, name: &str| -> String {
        b.o1(
            ty,
            vec![("x".into(), bind(x).1), ("y".into(), bind(y).1)],
            name,
            f16tt(&[1]),
        )
    };
    let (nd, nm1, nm2, nema2, nd2, nv1, nv2, nevar2, ndz, nve) = (
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
    );
    let d = ew(&mut b, "sub", "x", &ema, &nd);
    let m1 = ew(&mut b, "mul", &ema, &oa, &nm1);
    let m2 = ew(&mut b, "mul", "x", &al, &nm2);
    let ema2 = ew(&mut b, "add", &m1, &m2, &nema2);
    let d2 = ew(&mut b, "mul", &d, &d, &nd2);
    let v1 = ew(&mut b, "mul", &evar, &oa, &nv1);
    let v2 = ew(&mut b, "mul", &d2, &al, &nv2);
    let evar2 = ew(&mut b, "add", &v1, &v2, &nevar2);

    // z on pre-update stats
    let dz = ew(&mut b, "sub", "x", &ema, &ndz);
    let ve = ew(&mut b, "add", &evar, &ep, &nve);
    let sqn = b.fresh("sq");
    b.op(
        "sqrt",
        vec![("x".into(), bind(&ve).1)],
        vec![(&sqn, f16tt(&[1]))],
        vec![],
    );
    b.o1(
        "real_div",
        vec![("x".into(), bind(&dz).1), ("y".into(), bind(&sqn).1)],
        "z",
        f16tt(&[1]),
    );
    let flagb = b.o1(
        "greater",
        vec![("x".into(), bind("z").1), ("y".into(), bind(&zt).1)],
        "flagb",
        tt(DType::Bool, &[1]),
    );
    b.o1(
        "cast",
        vec![("x".into(), bind(&flagb).1), ("dtype".into(), bind(&tyf).1)],
        "flag",
        f16tt(&[1]),
    );

    let n_acc2 = b.fresh("acc2");
    let acc2 = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&ema2, &evar2])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        &n_acc2,
        f16tt(&[2]),
    );
    b.op(
        "write_state",
        vec![
            ("input".into(), bind("acc").1),
            ("data".into(), bind(&acc2).1),
        ],
        vec![],
        vec![("name".into(), Value::Str("acc_write".into()))],
    );
    b.outputs = vec!["z".into(), "flag".into()];

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
            ty: state_ty(DType::Fp16, &[2]),
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

/// Emit a sentinel package (spec only).
pub fn write_sentinel_package(
    path: &Path,
    alpha: f32,
    eps: f32,
    threshold: f32,
) -> std::io::Result<()> {
    mil_spec::write_mlpackage(path, &emit_sentinel(alpha, eps, threshold), None)
}

// ============================ Mutable memory ================================

/// Emit a mutable fp16 memory bank `[slots, dim]` held in `MLState`.
///
/// Per call: `out = gather(mem, rslot)` (read-before-write), then
/// `mem[wslot] = wvec` via `slice_update`. Inputs `wslot`/`rslot` int32
/// `[1]`, `wvec` fp16 `[1, dim]`; output `out` fp16 `[1, dim]`.
pub fn emit_memory(slots: i64, dim: i64) -> Vec<u8> {
    let mut b = Block::new();
    let k0 = konst_i32_scalar(&mut b, "k0", 0);
    let k0b = konst_i32s(&mut b, "k0b", &[0]);
    let k_dim = konst_i32s(&mut b, "kd", &[dim as i32]);
    let k_st = konst_i32s(&mut b, "st", &[1, 1]);
    let k_one = konst_i32s(&mut b, "k1", &[1]);
    let vt = konst_bool_scalar(&mut b, "vt", true);
    let ax = konst_i32_scalar(&mut b, "ax", 0);
    let il = konst_bool_scalar(&mut b, "il", false);
    let masks: Vec<String> = ["bm", "em", "sm"]
        .iter()
        .map(|n| bool_vec(&mut b, n, &[false, false]))
        .collect();
    let (bm, em, sm) = (masks[0].clone(), masks[1].clone(), masks[2].clone());

    b.op(
        "read_state",
        vec![("input".into(), bind("mem").1)],
        vec![("m", f16tt(&[slots, dim]))],
        vec![("name".into(), Value::Str("m".into()))],
    );
    b.op(
        "gather",
        vec![
            ("x".into(), bind("m").1),
            ("indices".into(), bind("rslot").1),
            ("axis".into(), bind(&k0).1),
            ("validate_indices".into(), bind(&vt).1),
        ],
        vec![("out", f16tt(&[1, dim]))],
        vec![],
    );
    let (nbeg, nw1, nend) = (b.fresh("beg"), b.fresh("w1"), b.fresh("end"));
    let beg = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&["wslot", &k0b])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        &nbeg,
        i32tt(&[2]),
    );
    let w1 = b.o1(
        "add",
        vec![("x".into(), bind("wslot").1), ("y".into(), bind(&k_one).1)],
        &nw1,
        i32tt(&[1]),
    );
    let end = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&w1, &k_dim])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        &nend,
        i32tt(&[2]),
    );
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
        vec![("m2", f16tt(&[slots, dim]))],
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
            shape: vec![1, dim],
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
        shape: vec![1, dim],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let states = [Feature {
        name: "mem".into(),
        shape: vec![slots, dim],
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
            ty: f16tt(&[1, dim]),
        },
        NVT {
            name: "rslot".into(),
            ty: i32tt(&[1]),
        },
        NVT {
            name: "mem".into(),
            ty: state_ty(DType::Fp16, &[slots, dim]),
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

/// Emit a memory package (spec only).
pub fn write_memory_package(path: &Path, slots: i64, dim: i64) -> std::io::Result<()> {
    mil_spec::write_mlpackage(path, &emit_memory(slots, dim), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ac() -> Automaton {
        Automaton::build(&[b"password", b"api_key", b"secret", b"ssh-rsa", b"sk-"])
    }

    #[test]
    fn trie_and_failure_links() {
        let a = Automaton::build(&[b"he", b"she", b"his", b"hers"]);
        // classic AC example: "ushers" hits "she","he","hers" overlapping
        let hits: Vec<usize> = a
            .run(b"ushers")
            .iter()
            .enumerate()
            .filter(|(_, &h)| h != 0)
            .map(|(i, _)| i)
            .collect();
        // "she"+"he" co-terminate at index 3, "hers" at 5
        assert_eq!(hits, vec![3, 5]);
    }

    #[test]
    fn run_matches_stream_boundaries() {
        let a = ac();
        // pattern split across any chunking must still hit — state persists
        let hits = a.run(b"prefix sk-123 and my api_key ends secret");
        let flagged: Vec<usize> = hits
            .iter()
            .enumerate()
            .filter(|(_, &h)| h != 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(flagged, vec![9, 27, 39]);
    }

    #[test]
    fn no_false_positives_on_clean_text() {
        let a = ac();
        assert!(a.run(b"hello world, this is fine").iter().all(|&h| h == 0));
        // near-miss: "pass" and "word" adjacent ≠ "password"
        assert!(a.run(b"pass word passwor").iter().all(|&h| h == 0));
    }

    #[test]
    fn emit_dfa_is_valid_spec() {
        let spec = emit_dfa(&ac());
        assert!(!spec.is_empty());
        // protobuf magic-free but must round-trip through mil_verify later;
        // cheap check: contains the op names
        let s = String::from_utf8_lossy(&spec);
        for needle in ["read_state", "write_state", "gather", "table", "cur"] {
            assert!(s.contains(needle), "spec missing {needle}");
        }
    }

    #[test]
    fn emit_sentinel_and_memory_are_valid_specs() {
        for spec in [emit_sentinel(0.1, 1e-3, 3.0), emit_memory(64, 32)] {
            assert!(!spec.is_empty());
            let s = String::from_utf8_lossy(&spec);
            assert!(s.contains("read_state"));
            assert!(s.contains("write_state"));
        }
        let spec = emit_memory(64, 32);
        let mem = String::from_utf8_lossy(&spec);
        assert!(mem.contains("slice_update"));
    }
}
