//! End-to-end coverage for the `Block` typed op helpers: every helper
//! builds a single-op (or small composite) model, writes an `.mlpackage`,
//! compiles it through `mil_compile` (`coremlc`), runs it via `mil_infer`
//! on randomized fp16 input, and checks the output against an independent
//! Rust f64 reference.
//!
//! All tests early-return when `xcrun -f coremlc` is unavailable. Work
//! dirs are removed after each test unless `MIL_KEEP_ARTIFACTS=1`.

use mil_infer::{ComputeUnits, Input, Model, Output, Prediction};
use mil_spec::*;
use std::sync::OnceLock;

// ---------- harness ----------

static HAVE_COREMLC: OnceLock<bool> = OnceLock::new();

fn coremlc() -> bool {
    *HAVE_COREMLC.get_or_init(|| mil_compile::coremlc_path().is_some())
}

fn keep_artifacts() -> bool {
    std::env::var("MIL_KEEP_ARTIFACTS").as_deref() == Ok("1")
}

/// Write `b` as a model, compile, load, predict. Returns `None` when no
/// `coremlc` — the test skips silently.
fn run(
    tag: &str,
    b: &Block,
    ins: &[(&str, &[i64], DType)],
    outs: &[(&str, &[i64], DType)],
    feeds: &[Input<'_>],
) -> Option<Prediction> {
    if !coremlc() {
        eprintln!("coremlc unavailable — skipping {tag}");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("mil_ops_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let inputs: Vec<Feature> = ins
        .iter()
        .map(|(n, s, d)| Feature {
            name: n.to_string(),
            shape: s.to_vec(),
            dtype: *d,
            is_state: false,
        })
        .collect();
    let outputs: Vec<Feature> = outs
        .iter()
        .map(|(n, s, d)| Feature {
            name: n.to_string(),
            shape: s.to_vec(),
            dtype: *d,
            is_state: false,
        })
        .collect();
    let fn_in: Vec<NVT> = ins
        .iter()
        .map(|(n, s, d)| NVT {
            name: n.to_string(),
            ty: ValueType::Tensor(TensorType {
                dtype: *d,
                shape: s.to_vec(),
            }),
        })
        .collect();
    let spec = encode_model(
        &inputs,
        &outputs,
        &[],
        b,
        &fn_in,
        &ModelMeta::new(10, "CoreML9"),
    );
    let pkg = dir.join("m.mlpackage");
    write_mlpackage(&pkg, &spec, None).expect("write package");
    let cm = mil_compile::compile(&pkg, &dir.join("compiled"))
        .unwrap_or_else(|e| panic!("coremlc rejected {tag} model:\n{e}"));
    // `All` lets CoreML pick ANE where eligible. Note: `CpuAndGpu` is
    // also exercised — GPU-produced outputs arrive with padded strides
    // and mil_infer honours them (see output marshalling).
    let units = if std::env::var("MIL_OPS_CU").as_deref() == Ok("gpu") {
        ComputeUnits::CpuAndGpu
    } else {
        ComputeUnits::All
    };
    let m = Model::load(&cm.path, units).expect("load compiled model");
    let p = m
        .predict(feeds)
        .unwrap_or_else(|e| panic!("predict failed for {tag}: {e}"));
    if !keep_artifacts() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Some(p)
}

/// Shaped fp16 const in `b`.
fn k16(b: &mut Block, name: &str, shape: &[i64], vs: &[f32]) -> String {
    assert_eq!(
        vs.len(),
        shape.iter().map(|d| *d as usize).product::<usize>()
    );
    b.op(
        "const",
        vec![],
        vec![(name, ValueType::Tensor(TensorType::f16(shape)))],
        vec![("val".into(), Value::f16s(shape, vs))],
    )[0]
    .clone()
}

/// Shaped int32 const in `b`.
fn k32(b: &mut Block, name: &str, shape: &[i64], vs: &[i32]) -> String {
    assert_eq!(
        vs.len(),
        shape.iter().map(|d| *d as usize).product::<usize>()
    );
    let t = TensorType {
        dtype: DType::Int32,
        shape: shape.to_vec(),
    };
    b.op(
        "const",
        vec![],
        vec![(name, ValueType::Tensor(t))],
        vec![(
            "val".into(),
            Value::Imm(
                ValueType::Tensor(TensorType {
                    dtype: DType::Int32,
                    shape: shape.to_vec(),
                }),
                Immediate::Ints(vs.to_vec()),
            ),
        )],
    )[0]
    .clone()
}

/// Shaped bool const in `b`.
fn kbool(b: &mut Block, name: &str, shape: &[i64], vs: &[bool]) -> String {
    assert_eq!(
        vs.len(),
        shape.iter().map(|d| *d as usize).product::<usize>()
    );
    let t = TensorType {
        dtype: DType::Bool,
        shape: shape.to_vec(),
    };
    b.op(
        "const",
        vec![],
        vec![(name, ValueType::Tensor(t))],
        vec![(
            "val".into(),
            Value::Imm(
                ValueType::Tensor(TensorType {
                    dtype: DType::Bool,
                    shape: shape.to_vec(),
                }),
                Immediate::Bools(vs.to_vec()),
            ),
        )],
    )[0]
    .clone()
}

fn f16_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// xorshift64 deterministic values in `[lo, hi)`.
fn randv(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            (lo as f64 + (hi - lo) as f64 * u) as f32
        })
        .collect()
}

fn out<'p>(p: &'p Prediction, name: &str) -> &'p Output {
    p.outputs
        .iter()
        .find(|o| o.name == name)
        .unwrap_or_else(|| panic!("missing output {name}"))
}

/// Assert `got` ≈ `want` elementwise: `|got - want| <= atol + rtol*|want|`.
fn check(got: &[f32], want: &[f64], rtol: f64, atol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (*g as f64 - *w).abs();
        let tol = atol + rtol * w.abs();
        assert!(
            g.is_finite() && d <= tol,
            "{what}[{i}]: got {g}, want {w} (|Δ| {d} > tol {tol})"
        );
    }
}

fn erf64(x: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26 (|ε| < 1.5e-7) — plenty for fp16 comparison.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf64(x / std::f64::consts::SQRT_2))
}

fn gelu_tanh64(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// f64 view of an fp16-rounded f32 slice (what CoreML actually consumed).
fn qh(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|v| half::f16::from_f32(*v).to_f32() as f64)
        .collect()
}

fn shape_num(s: &[i64]) -> usize {
    s.iter().map(|d| *d as usize).product()
}

// ---------- elementwise unary ----------

#[test]
fn unary_pos_e2e() {
    let shape = [2, 24];
    let x = randv(shape_num(&shape), 0.05, 4.0, 0xA51E);
    let mut b = Block::new();
    b.exp("x", &shape, "o_exp");
    b.exp2("x", &shape, "o_exp2");
    b.log("x", 0.0, &shape, "o_log");
    b.sqrt("x", &shape, "o_sqrt");
    b.rsqrt("x", 0.0, &shape, "o_rsqrt");
    b.inverse("x", 0.0, &shape, "o_inv");
    b.threshold("x", 0.5, &shape, "o_thr");
    let names = [
        "o_exp", "o_exp2", "o_log", "o_sqrt", "o_rsqrt", "o_inv", "o_thr",
    ];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let p = run(
        "unary_pos",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let w: Vec<f64> = xq.iter().map(|v| v.exp()).collect();
    check(&out(&p, "o_exp").values(), &w, 5e-3, 1e-3, "exp");
    let w: Vec<f64> = xq.iter().map(|v| v.exp2()).collect();
    check(&out(&p, "o_exp2").values(), &w, 5e-3, 1e-3, "exp2");
    let w: Vec<f64> = xq.iter().map(|v| v.ln()).collect();
    check(&out(&p, "o_log").values(), &w, 5e-3, 2e-3, "log");
    let w: Vec<f64> = xq.iter().map(|v| v.sqrt()).collect();
    check(&out(&p, "o_sqrt").values(), &w, 5e-3, 1e-3, "sqrt");
    let w: Vec<f64> = xq.iter().map(|v| 1.0 / v.sqrt()).collect();
    check(&out(&p, "o_rsqrt").values(), &w, 5e-3, 2e-3, "rsqrt");
    let w: Vec<f64> = xq.iter().map(|v| 1.0 / v).collect();
    check(&out(&p, "o_inv").values(), &w, 5e-3, 5e-3, "inverse");
    let w: Vec<f64> = xq.iter().map(|v| v.max(0.5)).collect();
    check(&out(&p, "o_thr").values(), &w, 5e-3, 1e-3, "threshold");
}

#[test]
fn unary_signed_e2e() {
    let shape = [2, 24];
    let x = randv(shape_num(&shape), -3.0, 3.0, 0xB02A);
    let mut b = Block::new();
    b.abs("x", &shape, "o_abs");
    b.floor("x", &shape, "o_floor");
    b.ceil("x", &shape, "o_ceil");
    b.round("x", &shape, "o_round");
    b.neg("x", &shape, "o_neg");
    b.sign("x", &shape, "o_sign");
    b.square("x", &shape, "o_sq");
    let names = [
        "o_abs", "o_floor", "o_ceil", "o_round", "o_neg", "o_sign", "o_sq",
    ];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let p = run(
        "unary_signed",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let w: Vec<f64> = xq.iter().map(|v| v.abs()).collect();
    check(&out(&p, "o_abs").values(), &w, 0.0, 1e-3, "abs");
    let w: Vec<f64> = xq.iter().map(|v| v.floor()).collect();
    check(&out(&p, "o_floor").values(), &w, 0.0, 1e-3, "floor");
    let w: Vec<f64> = xq.iter().map(|v| v.ceil()).collect();
    check(&out(&p, "o_ceil").values(), &w, 0.0, 1e-3, "ceil");
    let w: Vec<f64> = xq.iter().map(|v| v.round()).collect();
    check(&out(&p, "o_round").values(), &w, 0.0, 1e-3, "round");
    let w: Vec<f64> = xq.iter().map(|v| -*v).collect();
    check(&out(&p, "o_neg").values(), &w, 0.0, 1e-3, "neg");
    let w: Vec<f64> = xq.iter().map(|v| v.signum()).collect();
    check(&out(&p, "o_sign").values(), &w, 0.0, 1e-3, "sign");
    let w: Vec<f64> = xq.iter().map(|v| v * v).collect();
    check(&out(&p, "o_sq").values(), &w, 5e-3, 1e-3, "square");
}

#[test]
fn unary_trig_erf_e2e() {
    let shape = [2, 24];
    // |x| < 1 keeps asin/acos/atanh in domain; tan stays far from π/2.
    let x = randv(shape_num(&shape), -0.95, 0.95, 0xC0DE);
    let mut b = Block::new();
    b.sin("x", &shape, "o_sin");
    b.cos("x", &shape, "o_cos");
    b.tan("x", &shape, "o_tan");
    b.sinh("x", &shape, "o_sinh");
    b.cosh("x", &shape, "o_cosh");
    b.tanh("x", &shape, "o_tanh");
    b.erf("x", &shape, "o_erf");
    b.asin("x", &shape, "o_asin");
    b.acos("x", &shape, "o_acos");
    b.atan("x", &shape, "o_atan");
    b.atanh("x", &shape, "o_atanh");
    let names = [
        "o_sin", "o_cos", "o_tan", "o_sinh", "o_cosh", "o_tanh", "o_erf", "o_asin", "o_acos",
        "o_atan", "o_atanh",
    ];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let p = run(
        "unary_trig",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    type Ref = fn(f64) -> f64;
    let cases: [(&str, Ref); 11] = [
        ("o_sin", f64::sin),
        ("o_cos", f64::cos),
        ("o_tan", f64::tan),
        ("o_sinh", f64::sinh),
        ("o_cosh", f64::cosh),
        ("o_tanh", f64::tanh),
        ("o_erf", erf64),
        ("o_asin", f64::asin),
        ("o_acos", f64::acos),
        ("o_atan", f64::atan),
        ("o_atanh", f64::atanh),
    ];
    for (n, f) in cases {
        let w: Vec<f64> = xq.iter().map(|v| f(*v)).collect();
        check(&out(&p, n).values(), &w, 1e-2, 5e-3, n);
    }
}

#[test]
fn activations_e2e() {
    let shape = [2, 24];
    let x = randv(shape_num(&shape), -3.0, 3.0, 0xD1CE);
    let mut b = Block::new();
    b.sigmoid("x", &shape, "o_sig");
    b.relu("x", &shape, "o_relu");
    b.relu6("x", &shape, "o_relu6");
    b.leaky_relu("x", 0.1, &shape, "o_lrelu");
    b.gelu("x", false, &shape, "o_gelu");
    b.gelu("x", true, &shape, "o_gelut");
    b.clip("x", -1.0, 1.0, &shape, "o_clip");
    b.sigmoid_hard("x", 0.2, 0.5, &shape, "o_sigh");
    b.elu("x", 0.5, &shape, "o_elu");
    b.clamped_relu("x", -0.5, 2.0, &shape, "o_crelu");
    b.softplus("x", &shape, "o_splus");
    b.softsign("x", &shape, "o_ssign");
    let names = [
        "o_sig", "o_relu", "o_relu6", "o_lrelu", "o_gelu", "o_gelut", "o_clip", "o_sigh", "o_elu",
        "o_crelu", "o_splus", "o_ssign",
    ];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let p = run(
        "activ",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let w: Vec<f64> = xq.iter().map(|v| sigmoid64(*v)).collect();
    check(&out(&p, "o_sig").values(), &w, 5e-3, 2e-3, "sigmoid");
    let w: Vec<f64> = xq.iter().map(|v| v.max(0.0)).collect();
    check(&out(&p, "o_relu").values(), &w, 0.0, 1e-3, "relu");
    let w: Vec<f64> = xq.iter().map(|v| v.clamp(0.0, 6.0)).collect();
    check(&out(&p, "o_relu6").values(), &w, 5e-3, 1e-3, "relu6");
    let w: Vec<f64> = xq
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.1 * *v })
        .collect();
    check(&out(&p, "o_lrelu").values(), &w, 5e-3, 2e-3, "leaky_relu");
    let w: Vec<f64> = xq.iter().map(|v| gelu64(*v)).collect();
    check(&out(&p, "o_gelu").values(), &w, 5e-3, 2e-3, "gelu exact");
    let w: Vec<f64> = xq.iter().map(|v| gelu_tanh64(*v)).collect();
    check(&out(&p, "o_gelut").values(), &w, 1e-2, 5e-3, "gelu tanh");
    let w: Vec<f64> = xq.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    check(&out(&p, "o_clip").values(), &w, 0.0, 1e-3, "clip");
    let w: Vec<f64> = xq
        .iter()
        .map(|v| (0.2 * *v + 0.5).clamp(0.0, 1.0))
        .collect();
    check(&out(&p, "o_sigh").values(), &w, 5e-3, 2e-3, "sigmoid_hard");
    let w: Vec<f64> = xq
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.5 * (v.exp() - 1.0) })
        .collect();
    check(&out(&p, "o_elu").values(), &w, 5e-3, 3e-3, "elu");
    // MIL: x >= 0 → min(beta, x); x < 0 → min(beta, alpha*x)
    let w: Vec<f64> = xq
        .iter()
        .map(|v| {
            if *v >= 0.0 {
                v.min(2.0)
            } else {
                (-0.5 * *v).min(2.0)
            }
        })
        .collect();
    check(&out(&p, "o_crelu").values(), &w, 5e-3, 1e-3, "clamped_relu");
    let w: Vec<f64> = xq.iter().map(|v| (1.0 + v.exp()).ln()).collect();
    check(&out(&p, "o_splus").values(), &w, 5e-3, 3e-3, "softplus");
    let w: Vec<f64> = xq.iter().map(|v| *v / (1.0 + v.abs())).collect();
    check(&out(&p, "o_ssign").values(), &w, 5e-3, 2e-3, "softsign");
}

// ---------- elementwise binary ----------

#[test]
fn binary_e2e() {
    let shape = [2, 24];
    let x = randv(shape_num(&shape), 0.2, 3.0, 0xBEEF);
    let y = randv(shape_num(&shape), 0.2, 2.0, 0xF00D);
    let mut b = Block::new();
    b.pow("x", "y", &shape, "o_pow");
    b.maximum("x", "y", &shape, "o_max");
    b.minimum("x", "y", &shape, "o_min");
    b.real_div("x", "y", &shape, "o_div");
    b.floor_div("x", "y", &shape, "o_fdiv");
    b.modulo("x", "y", &shape, "o_mod");
    let names = ["o_pow", "o_max", "o_min", "o_div", "o_fdiv", "o_mod"];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let yb = f16_bytes(&y);
    let p = run(
        "binary",
        &b,
        &[("x", &shape, DType::Fp16), ("y", &shape, DType::Fp16)],
        &outs,
        &[
            Input {
                name: "x",
                shape: &shape,
                data: &xb,
                dtype: DType::Fp16,
            },
            Input {
                name: "y",
                shape: &shape,
                data: &yb,
                dtype: DType::Fp16,
            },
        ],
    );
    let Some(p) = p else { return };
    let (xq, yq) = (qh(&x), qh(&y));
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| a.powf(*b)).collect();
    check(&out(&p, "o_pow").values(), &w, 1e-2, 1e-2, "pow");
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| a.max(*b)).collect();
    check(&out(&p, "o_max").values(), &w, 0.0, 1e-3, "maximum");
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| a.min(*b)).collect();
    check(&out(&p, "o_min").values(), &w, 0.0, 1e-3, "minimum");
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| a / b).collect();
    check(&out(&p, "o_div").values(), &w, 5e-3, 5e-3, "real_div");
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| (a / b).floor()).collect();
    check(&out(&p, "o_fdiv").values(), &w, 0.0, 2e-3, "floor_div");
    // positive operands only — fp mod agrees with np.mod / C fmod there.
    let w: Vec<f64> = xq.iter().zip(&yq).map(|(a, b)| a % b).collect();
    check(&out(&p, "o_mod").values(), &w, 5e-3, 5e-3, "mod");
}

// ---------- reductions ----------

#[test]
fn reduce_e2e() {
    let shape = [2, 3, 4];
    let x = randv(shape_num(&shape), 0.1, 2.0, 0xED00);
    let keep = [2, 1, 4];
    let mut b = Block::new();
    b.reduce_sum("x", &[1], true, &keep, "o_sum");
    b.reduce_mean("x", &[1], true, &keep, "o_mean");
    b.reduce_max("x", &[1], true, &keep, "o_max");
    b.reduce_min("x", &[1], true, &keep, "o_min");
    b.reduce_prod("x", &[1], true, &keep, "o_prod");
    b.reduce_l1_norm("x", &[1], true, &keep, "o_l1");
    b.reduce_l2_norm("x", &[1], true, &keep, "o_l2");
    b.reduce_sum_square("x", &[1], true, &keep, "o_ssq");
    b.reduce_log_sum("x", &[1], true, &keep, "o_lsum");
    b.reduce_log_sum_exp("x", &[1], true, &keep, "o_lse");
    b.cumsum("x", 2, false, false, &shape, "o_cs");
    b.cumsum("x", 2, true, false, &shape, "o_cse");
    b.cumsum("x", 2, false, true, &shape, "o_csr");
    b.log_softmax("x", -1, &shape, "o_lsm");
    b.reduce_argmax("x", 1, false, &[2, 4], "o_amax");
    b.reduce_argmin("x", 1, false, &[2, 4], "o_amin");
    let f16outs = [
        ("o_sum", &keep[..]),
        ("o_mean", &keep[..]),
        ("o_max", &keep[..]),
        ("o_min", &keep[..]),
        ("o_prod", &keep[..]),
        ("o_l1", &keep[..]),
        ("o_l2", &keep[..]),
        ("o_ssq", &keep[..]),
        ("o_lsum", &keep[..]),
        ("o_lse", &keep[..]),
        ("o_cs", &shape[..]),
        ("o_cse", &shape[..]),
        ("o_csr", &shape[..]),
        ("o_lsm", &shape[..]),
    ];
    b.outputs = f16outs
        .iter()
        .map(|(n, _)| n.to_string())
        .chain(["o_amax".to_string(), "o_amin".to_string()])
        .collect();
    let mut outs: Vec<(&str, &[i64], DType)> =
        f16outs.iter().map(|(n, s)| (*n, *s, DType::Fp16)).collect();
    outs.push(("o_amax", &[2, 4], DType::Int32));
    outs.push(("o_amin", &[2, 4], DType::Int32));
    let xb = f16_bytes(&x);
    let p = run(
        "reduce",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    // [b0, c, s] → reduced over c (axis 1)
    let red = |f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
        let mut outv = Vec::new();
        for b0 in 0..2 {
            for s in 0..4 {
                let lane: Vec<f64> = (0..3).map(|c| xq[b0 * 12 + c * 4 + s]).collect();
                outv.push(f(&lane));
            }
        }
        outv
    };
    check(
        &out(&p, "o_sum").values(),
        &red(&|l: &[f64]| l.iter().sum()),
        5e-3,
        2e-3,
        "reduce_sum",
    );
    check(
        &out(&p, "o_mean").values(),
        &red(&|l: &[f64]| l.iter().sum::<f64>() / l.len() as f64),
        5e-3,
        2e-3,
        "reduce_mean",
    );
    check(
        &out(&p, "o_max").values(),
        &red(&|l: &[f64]| l.iter().cloned().fold(f64::MIN, f64::max)),
        0.0,
        1e-3,
        "reduce_max",
    );
    check(
        &out(&p, "o_min").values(),
        &red(&|l: &[f64]| l.iter().cloned().fold(f64::MAX, f64::min)),
        0.0,
        1e-3,
        "reduce_min",
    );
    check(
        &out(&p, "o_prod").values(),
        &red(&|l: &[f64]| l.iter().product()),
        1e-2,
        1e-2,
        "reduce_prod",
    );
    check(
        &out(&p, "o_l1").values(),
        &red(&|l: &[f64]| l.iter().map(|v| v.abs()).sum()),
        5e-3,
        2e-3,
        "reduce_l1",
    );
    check(
        &out(&p, "o_l2").values(),
        &red(&|l: &[f64]| l.iter().map(|v| v * v).sum::<f64>().sqrt()),
        5e-3,
        2e-3,
        "reduce_l2",
    );
    check(
        &out(&p, "o_ssq").values(),
        &red(&|l: &[f64]| l.iter().map(|v| v * v).sum()),
        5e-3,
        2e-3,
        "reduce_sum_square",
    );
    check(
        &out(&p, "o_lsum").values(),
        &red(&|l: &[f64]| l.iter().sum::<f64>().ln()),
        5e-3,
        5e-3,
        "reduce_log_sum",
    );
    check(
        &out(&p, "o_lse").values(),
        &red(&|l: &[f64]| l.iter().map(|v| v.exp()).sum::<f64>().ln()),
        1e-2,
        5e-3,
        "reduce_log_sum_exp",
    );
    // cumsum inclusive / exclusive / reverse along last dim
    let mut cs = vec![0.0; 24];
    let mut cse = vec![0.0; 24];
    let mut csr = vec![0.0; 24];
    for i in 0..6 {
        for s in 0..4 {
            let lane: Vec<f64> = (0..4).map(|k| xq[i * 4 + k]).collect();
            cs[i * 4 + s] = lane[..=s].iter().sum();
            cse[i * 4 + s] = lane[..s].iter().sum();
            csr[i * 4 + s] = lane[s..].iter().sum();
        }
    }
    check(&out(&p, "o_cs").values(), &cs, 5e-3, 3e-3, "cumsum");
    check(&out(&p, "o_cse").values(), &cse, 5e-3, 3e-3, "cumsum excl");
    check(&out(&p, "o_csr").values(), &csr, 5e-3, 3e-3, "cumsum rev");
    let mut lsm = vec![0.0; 24];
    for i in 0..6 {
        let lane: Vec<f64> = (0..4).map(|k| xq[i * 4 + k]).collect();
        let lse = lane.iter().map(|v| v.exp()).sum::<f64>().ln();
        for s in 0..4 {
            lsm[i * 4 + s] = lane[s] - lse;
        }
    }
    check(&out(&p, "o_lsm").values(), &lsm, 1e-2, 5e-3, "log_softmax");
    // argmax/argmin over axis 1 → [2,4]; compare the selected *values*.
    let amax: Vec<f32> = out(&p, "o_amax").values();
    let amin: Vec<f32> = out(&p, "o_amin").values();
    for b0 in 0..2 {
        for s in 0..4 {
            let lane: Vec<f64> = (0..3).map(|c| xq[b0 * 12 + c * 4 + s]).collect();
            let mx = lane.iter().cloned().fold(f64::MIN, f64::max);
            let mn = lane.iter().cloned().fold(f64::MAX, f64::min);
            let ai = amax[b0 * 4 + s] as usize;
            let ii = amin[b0 * 4 + s] as usize;
            assert!(
                ai < 3 && (lane[ai] - mx).abs() < 1e-3,
                "argmax wrong at [{b0},{s}]"
            );
            assert!(
                ii < 3 && (lane[ii] - mn).abs() < 1e-3,
                "argmin wrong at [{b0},{s}]"
            );
        }
    }
}

// ---------- indexing ----------

#[test]
fn indexing_e2e() {
    let shape = [3, 4];
    let x = randv(shape_num(&shape), -2.0, 2.0, 0x1D00);
    let mut b = Block::new();
    let idx01 = b.konst_i32("idx01", &[0, 2]);
    b.gather("x", &idx01, 0, 0, &[2, 4], "o_g0");
    let idx1 = b.konst_i32("idx1", &[0, 3, 1]);
    b.gather("x", &idx1, 1, 0, &[3, 3], "o_g1");
    let idxaa = k32(&mut b, "idxaa", &[3, 2], &[0, 3, 2, 1, 3, 0]);
    b.gather_along_axis("x", &idxaa, 1, &[3, 2], "o_gaa");
    let upd = k16(&mut b, "upd", &[2, 4], &randv(8, -1.0, 1.0, 5));
    let sidx = b.konst_i32("sidx", &[0, 2]);
    b.scatter("x", &sidx, &upd, 0, "update", &[3, 4], "o_su");
    let upd2 = k16(&mut b, "upd2", &[1, 4], &randv(4, -1.0, 1.0, 9));
    let sidx2 = b.konst_i32("sidx2", &[1]);
    b.scatter("x", &sidx2, &upd2, 0, "add", &[3, 4], "o_sa");
    let saa_i = k32(&mut b, "saa_i", &[3, 2], &[0, 3, 1, 0, 2, 1]);
    let saa_u = k16(&mut b, "saa_u", &[3, 2], &randv(6, -1.0, 1.0, 3));
    b.scatter_along_axis("x", &saa_i, &saa_u, 1, "update", &[3, 4], "o_saa");
    let gnd_i = k32(&mut b, "gnd_i", &[3, 2], &[0, 1, 2, 3, 1, 0]);
    b.gather_nd("x", &gnd_i, &[3], "o_gnd");
    let snd_i = k32(&mut b, "snd_i", &[2, 2], &[0, 0, 2, 3]);
    let snd_u = k16(&mut b, "snd_u", &[2], &[9.0, 8.0]);
    b.scatter_nd("x", &snd_i, &snd_u, "update", &[3, 4], "o_snd");
    let (tkv, tki) = b.topk("x", 2, 1, false, &[3, 2], "o_tk");
    let sp = b.split("x", 1, &[("o_sp0", &[3, 2]), ("o_sp1", &[3, 2])]);
    assert_eq!(sp, ["o_sp0", "o_sp1"]);
    b.argsort("x", 1, true, &[3, 4], "o_as");
    let oh_i = k32(&mut b, "oh_i", &[3], &[0, 2, 3]);
    b.one_hot(&oh_i, 4, 1, 1.0, 0.0, &[3, 4], "o_oh");
    let _ = (tkv, tki);
    let outs: Vec<(&str, &[i64], DType)> = vec![
        ("o_g0", &[2, 4], DType::Fp16),
        ("o_g1", &[3, 3], DType::Fp16),
        ("o_gaa", &[3, 2], DType::Fp16),
        ("o_su", &[3, 4], DType::Fp16),
        ("o_sa", &[3, 4], DType::Fp16),
        ("o_saa", &[3, 4], DType::Fp16),
        ("o_gnd", &[3], DType::Fp16),
        ("o_snd", &[3, 4], DType::Fp16),
        ("o_tk_val", &[3, 2], DType::Fp16),
        ("o_tk_idx", &[3, 2], DType::Int32),
        ("o_sp0", &[3, 2], DType::Fp16),
        ("o_sp1", &[3, 2], DType::Fp16),
        ("o_as", &[3, 4], DType::Int32),
        ("o_oh", &[3, 4], DType::Fp16),
    ];
    b.outputs = outs.iter().map(|(n, _, _)| n.to_string()).collect();
    let xb = f16_bytes(&x);
    let p = run(
        "index",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let get2 = |r: usize, c: usize| xq[r * 4 + c];
    // gather axis 0 [0,2] → rows 0 and 2 (bound)
    let w: Vec<f64> = [0usize, 2]
        .iter()
        .flat_map(|r| (0..4).map(move |c| get2(*r, c)))
        .collect();
    check(&out(&p, "o_g0").values(), &w, 0.0, 1e-3, "gather axis0");
    let w: Vec<f64> = (0..3)
        .flat_map(|r| [0usize, 3, 1].iter().map(move |c| get2(r, *c)))
        .collect();
    check(
        &out(&p, "o_g1").values(),
        &w,
        0.0,
        1e-3,
        "gather axis1 bounds",
    );
    let w: Vec<f64> = (0..3)
        .flat_map(|r| {
            [[0, 3], [2, 1], [3, 0]][r]
                .iter()
                .map(move |c| get2(r, *c as usize))
        })
        .collect();
    check(
        &out(&p, "o_gaa").values(),
        &w,
        0.0,
        1e-3,
        "gather_along_axis",
    );
    // scatter update rows 0,2 with upd
    let uq = qh(&randv(8, -1.0, 1.0, 5));
    let mut w: Vec<f64> = xq.clone();
    w[0..4].copy_from_slice(&uq[0..4]);
    w[8..12].copy_from_slice(&uq[4..8]);
    check(&out(&p, "o_su").values(), &w, 0.0, 2e-3, "scatter update");
    let u2 = qh(&randv(4, -1.0, 1.0, 9));
    let mut w: Vec<f64> = xq.clone();
    for c in 0..4 {
        w[4 + c] += u2[c];
    }
    check(&out(&p, "o_sa").values(), &w, 5e-3, 2e-3, "scatter add");
    let su = qh(&randv(6, -1.0, 1.0, 3));
    let mut w: Vec<f64> = xq.clone();
    let si = [[0usize, 3], [1, 0], [2, 1]];
    for r in 0..3 {
        for (j, c) in si[r].iter().enumerate() {
            w[r * 4 + c] = su[r * 2 + j];
        }
    }
    check(
        &out(&p, "o_saa").values(),
        &w,
        0.0,
        2e-3,
        "scatter_along_axis",
    );
    let w: Vec<f64> = [[0usize, 1], [2, 3], [1, 0]]
        .iter()
        .map(|[r, c]| get2(*r, *c))
        .collect();
    check(&out(&p, "o_gnd").values(), &w, 0.0, 1e-3, "gather_nd");
    let mut w: Vec<f64> = xq.clone();
    w[0] = 9.0;
    w[11] = 8.0;
    check(&out(&p, "o_snd").values(), &w, 0.0, 2e-3, "scatter_nd");
    // topk k=2 along axis 1: the two largest values per row, descending.
    let tv = out(&p, "o_tk_val").values();
    for r in 0..3 {
        let mut lane: Vec<f64> = (0..4).map(|c| get2(r, c)).collect();
        lane.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for k in 0..2 {
            let g = tv[r * 2 + k] as f64;
            assert!(
                (g - lane[k]).abs() < 2e-3,
                "topk[{r}][{k}]: got {g}, want {}",
                lane[k]
            );
        }
    }
    // split axis 1 in halves
    for (o_i, base) in [(0usize, 0usize), (1usize, 2usize)] {
        let w: Vec<f64> = (0..3)
            .flat_map(|r| (0..2).map(move |c| get2(r, base + c)))
            .collect();
        check(
            &out(&p, &format!("o_sp{o_i}")).values(),
            &w,
            0.0,
            1e-3,
            "split",
        );
    }
    // argsort ascending along axis 1: selected values must be non-decreasing
    let asort: Vec<f32> = out(&p, "o_as").values();
    for r in 0..3 {
        let vals: Vec<f64> = (0..4).map(|c| get2(r, asort[r * 4 + c] as usize)).collect();
        for w_ in vals.windows(2) {
            assert!(w_[0] <= w_[1] + 1e-6, "argsort row {r} not sorted");
        }
    }
    // one_hot depth 4 on [0,2,3]
    let oh: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    check(&out(&p, "o_oh").values(), &oh, 0.0, 1e-3, "one_hot");
}

// ---------- shape / data movement ----------

#[test]
fn shape_e2e() {
    let shape = [3, 4];
    let x = randv(shape_num(&shape), -2.0, 2.0, 0x5EED);
    let mut b = Block::new();
    b.tile("x", &[2, 1], &[6, 4], "o_tile");
    b.pad("x", &[1, 0, 0, 2], "constant", 0.0, &[4, 6], "o_pad");
    b.broadcast_to("x", &[5, 3, 4], "o_bc");
    b.flatten2d("x", 1, &[3, 4], "o_f2d");
    let yk = k16(&mut b, "yk", &[3, 4], &randv(12, -1.0, 1.0, 11));
    b.stack(&["x".to_string(), yk], 0, &[2, 3, 4], "o_stk");
    b.reverse("x", &[0], &[3, 4], "o_rev");
    let cond = kbool(
        &mut b,
        "cond",
        &[3, 4],
        &(0..12).map(|i| i % 3 == 0).collect::<Vec<_>>(),
    );
    let bk = k16(&mut b, "bk", &[3, 4], &randv(12, -1.0, 1.0, 13));
    b.select(&cond, "x", &bk, &[3, 4], "o_sel");
    b.slice_by_size("x", &[1, 0], &[2, 3], &[2, 3], "o_sbs");
    b.sliding_windows("x", 1, 2, 1, &[3, 3, 2], "o_sw");
    b.outputs = [
        "o_tile", "o_pad", "o_bc", "o_f2d", "o_stk", "o_rev", "o_sel", "o_sbs", "o_sw",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let outs: Vec<(&str, &[i64], DType)> = vec![
        ("o_tile", &[6, 4], DType::Fp16),
        ("o_pad", &[4, 6], DType::Fp16),
        ("o_bc", &[5, 3, 4], DType::Fp16),
        ("o_f2d", &[3, 4], DType::Fp16),
        ("o_stk", &[2, 3, 4], DType::Fp16),
        ("o_rev", &[3, 4], DType::Fp16),
        ("o_sel", &[3, 4], DType::Fp16),
        ("o_sbs", &[2, 3], DType::Fp16),
        ("o_sw", &[3, 3, 2], DType::Fp16),
    ];
    let xb = f16_bytes(&x);
    let p = run(
        "shape",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let w: Vec<f64> = xq.iter().cycle().take(24).cloned().collect();
    check(&out(&p, "o_tile").values(), &w, 0.0, 1e-3, "tile");
    let mut w = vec![0.0; 24]; // [4,6] zeros; x row r → row r+1, cols 0..4
    for r in 0..3 {
        for c in 0..4 {
            w[(r + 1) * 6 + c] = xq[r * 4 + c];
        }
    }
    check(&out(&p, "o_pad").values(), &w, 0.0, 1e-3, "pad");
    let w: Vec<f64> = xq.iter().cycle().take(60).cloned().collect();
    check(&out(&p, "o_bc").values(), &w, 0.0, 1e-3, "broadcast_to");
    check(&out(&p, "o_f2d").values(), &xq, 0.0, 1e-3, "flatten2d");
    let yq = qh(&randv(12, -1.0, 1.0, 11));
    let mut w = xq.clone();
    w.extend(&yq);
    check(&out(&p, "o_stk").values(), &w, 0.0, 1e-3, "stack");
    let w: Vec<f64> = (0..3)
        .flat_map(|r| {
            let xq = &xq;
            (0..4).map(move |c| xq[(2 - r) * 4 + c])
        })
        .collect();
    check(&out(&p, "o_rev").values(), &w, 0.0, 1e-3, "reverse");
    let bq = qh(&randv(12, -1.0, 1.0, 13));
    let w: Vec<f64> = (0..12)
        .map(|i| if i % 3 == 0 { xq[i] } else { bq[i] })
        .collect();
    check(&out(&p, "o_sel").values(), &w, 0.0, 1e-3, "select");
    let w: Vec<f64> = (1..3)
        .flat_map(|r| {
            let xq = &xq;
            (0..3).map(move |c| xq[r * 4 + c])
        })
        .collect();
    check(&out(&p, "o_sbs").values(), &w, 0.0, 1e-3, "slice_by_size");
    let w: Vec<f64> = (0..3)
        .flat_map(|r| {
            let xq = &xq;
            (0..3).flat_map(move |w0| (0..2).map(move |k| xq[r * 4 + w0 + k]))
        })
        .collect();
    check(&out(&p, "o_sw").values(), &w, 0.0, 1e-3, "sliding_windows");
}

// ---------- normalization ----------

#[test]
fn norm_e2e() {
    // x [2,4] with a deliberately all-zero second row (norm edge case).
    let mut x = randv(8, -2.0, 2.0, 0xA11C);
    for v in x.iter_mut().skip(4) {
        *v = 0.0;
    }
    // batch_norm needs rank 3–5 under the CoreML9 opset — a second input.
    let x4 = randv(16, -2.0, 2.0, 0xA11D);
    let mut b = Block::new();
    let g4 = k16(&mut b, "ln_g", &[4], &[1.0, 0.5, 2.0, -0.25]);
    let be4 = k16(&mut b, "ln_b", &[4], &[0.1, -0.2, 0.0, 0.3]);
    b.layer_norm("x", &[1], Some(&g4), Some(&be4), 1e-5, &[2, 4], "o_ln");
    let bmean = k16(&mut b, "bn_m", &[4], &[0.2, -0.1, 0.0, 0.4]);
    let bvar = k16(&mut b, "bn_v", &[4], &[0.5, 1.0, 2.0, 0.3]);
    b.batch_norm(
        "x4",
        &bmean,
        &bvar,
        Some(&g4),
        Some(&be4),
        1e-5,
        &[1, 4, 2, 2],
        "o_bn",
    );
    let names = ["o_ln", "o_bn"];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = vec![
        ("o_ln", &[2, 4], DType::Fp16),
        ("o_bn", &[1, 4, 2, 2], DType::Fp16),
    ];
    let xb = f16_bytes(&x);
    let x4b = f16_bytes(&x4);
    let p = run(
        "norm",
        &b,
        &[
            ("x", &[2, 4], DType::Fp16),
            ("x4", &[1, 4, 2, 2], DType::Fp16),
        ],
        &outs,
        &[
            Input {
                name: "x",
                shape: &[2, 4],
                data: &xb,
                dtype: DType::Fp16,
            },
            Input {
                name: "x4",
                shape: &[1, 4, 2, 2],
                data: &x4b,
                dtype: DType::Fp16,
            },
        ],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let gq = qh(&[1.0, 0.5, 2.0, -0.25]);
    let bq = qh(&[0.1, -0.2, 0.0, 0.3]);
    // layer_norm per row; the all-zero row → 0*g + b = b (no NaN).
    let mut w = vec![0.0; 8];
    for r in 0..2 {
        let row = &xq[r * 4..r * 4 + 4];
        let mean = row.iter().sum::<f64>() / 4.0;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        for c in 0..4 {
            w[r * 4 + c] = (row[c] - mean) / (var + 1e-5).sqrt() * gq[c] + bq[c];
        }
    }
    check(&out(&p, "o_ln").values(), &w, 1e-2, 5e-3, "layer_norm");
    let xq4 = qh(&x4);
    let mq = qh(&[0.2, -0.1, 0.0, 0.4]);
    let vq = qh(&[0.5, 1.0, 2.0, 0.3]);
    let w: Vec<f64> = (0..16)
        .map(|i| {
            let c = (i / 4) % 4;
            (xq4[i] - mq[c]) / (vq[c] + 1e-5).sqrt() * gq[c] + bq[c]
        })
        .collect();
    check(&out(&p, "o_bn").values(), &w, 1e-2, 5e-3, "batch_norm");
}

#[test]
fn norm4d_e2e() {
    let shape = [1, 4, 2, 2];
    let x = randv(16, -2.0, 2.0, 0x1E5E);
    let mut b = Block::new();
    let gc = k16(&mut b, "in_g", &[4], &[1.0, 0.5, 1.5, 0.8]);
    let bc = k16(&mut b, "in_b", &[4], &[0.0, 0.1, -0.1, 0.2]);
    b.instance_norm("x", Some(&gc), Some(&bc), 1e-5, &shape, "o_in");
    let gc2 = k16(&mut b, "gn_g", &[1, 4, 1, 1], &[1.0, 0.5, 1.5, 0.8]);
    let bc2 = k16(&mut b, "gn_b", &[1, 4, 1, 1], &[0.0, 0.1, -0.1, 0.2]);
    b.group_norm("x", 1, 4, 2, 2, 2, Some(&gc2), Some(&bc2), 1e-5, "gn");
    b.local_response_norm("x", 3, 1e-4, 0.75, 2.0, &shape, "o_lrn");
    b.l2_norm("x", 1e-6, &shape, "o_l2n");
    let pa = k16(&mut b, "pa", &[4], &[0.1, 0.2, 0.3, 0.4]);
    b.prelu("x", &pa, &shape, "o_prelu");
    let names = ["o_in", "gn_gn", "o_lrn", "o_l2n", "o_prelu"];
    b.outputs = names.iter().map(|s| s.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> = names
        .iter()
        .map(|n| (*n, &shape[..], DType::Fp16))
        .collect();
    let xb = f16_bytes(&x);
    let p = run(
        "norm4d",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let gq = qh(&[1.0, 0.5, 1.5, 0.8]);
    let bq = qh(&[0.0, 0.1, -0.1, 0.2]);
    // instance_norm: per channel over the 2×2 spatial extent
    let mut w = vec![0.0; 16];
    for c in 0..4 {
        let lane: Vec<f64> = (0..4).map(|i| xq[c * 4 + i]).collect();
        let mean = lane.iter().sum::<f64>() / 4.0;
        let var = lane.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        for i in 0..4 {
            w[c * 4 + i] = (lane[i] - mean) / (var + 1e-5).sqrt() * gq[c] + bq[c];
        }
    }
    check(&out(&p, "o_in").values(), &w, 1e-2, 5e-3, "instance_norm");
    // group_norm groups=2 over c: normalize channels {0,1} and {2,3}
    let mut w = vec![0.0; 16];
    for g in 0..2 {
        let lane: Vec<f64> = (0..8).map(|i| xq[g * 8 + i]).collect();
        let mean = lane.iter().sum::<f64>() / 8.0;
        let var = lane.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 8.0;
        for i in 0..8 {
            let c = g * 2 + i / 4;
            w[g * 8 + i] = (lane[i] - mean) / (var + 1e-5).sqrt() * gq[c] + bq[c];
        }
    }
    check(&out(&p, "gn_gn").values(), &w, 1e-2, 5e-3, "group_norm");
    // lrn across channels at each spatial pos
    let mut w = vec![0.0; 16];
    for c in 0..4usize {
        for i in 0..4usize {
            let lo = c.saturating_sub(1); // size 3 window centered, clamped
            let hi = (c + 2).min(4);
            let ss: f64 = (lo..hi).map(|c2| xq[c2 * 4 + i].powi(2)).sum();
            w[c * 4 + i] = xq[c * 4 + i] / (2.0f64 + (1e-4 / 3.0) * ss).powf(0.75);
        }
    }
    check(
        &out(&p, "o_lrn").values(),
        &w,
        1e-2,
        5e-3,
        "local_response_norm",
    );
    let ss: f64 = xq.iter().map(|v| v * v).sum();
    let w: Vec<f64> = xq.iter().map(|v| *v / (ss + 1e-6).sqrt()).collect();
    check(&out(&p, "o_l2n").values(), &w, 5e-3, 2e-3, "l2_norm");
    // prelu on channel dim 1: alpha[c] when x[c,…] <= 0
    let pa = qh(&[0.1, 0.2, 0.3, 0.4]);
    let w: Vec<f64> = (0..16)
        .map(|i| {
            let c = i / 4;
            if xq[i] > 0.0 {
                xq[i]
            } else {
                pa[c] * xq[i]
            }
        })
        .collect();
    check(&out(&p, "o_prelu").values(), &w, 5e-3, 2e-3, "prelu");
}

// ---------- pooling / resize ----------

#[test]
fn pool_e2e() {
    let shape = [1, 2, 3, 4];
    let x = randv(24, -2.0, 2.0, 0x90A1);
    let mut b = Block::new();
    b.avg_pool(
        "x",
        &[2, 2],
        &[1, 1],
        "valid",
        &[0, 0, 0, 0],
        &[1, 2, 2, 3],
        "o_ap",
    );
    b.max_pool(
        "x",
        &[2, 2],
        &[1, 1],
        "valid",
        &[0, 0, 0, 0],
        &[1, 2, 2, 3],
        "o_mp",
    );
    b.avg_pool_global("x", &shape, "o_apg");
    b.max_pool_global("x", &shape, "o_mpg");
    b.upsample_nearest("x", 2, 2, &[1, 2, 6, 8], "o_unn");
    b.upsample_bilinear("x", 2, 2, true, &[1, 2, 6, 8], "o_ub");
    let defs: [(&str, &[i64]); 6] = [
        ("o_ap", &[1, 2, 2, 3]),
        ("o_mp", &[1, 2, 2, 3]),
        ("o_apg", &[1, 2, 1, 1]),
        ("o_mpg", &[1, 2, 1, 1]),
        ("o_unn", &[1, 2, 6, 8]),
        ("o_ub", &[1, 2, 6, 8]),
    ];
    b.outputs = defs.iter().map(|(n, _)| n.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> =
        defs.iter().map(|(n, s)| (*n, *s, DType::Fp16)).collect();
    let xb = f16_bytes(&x);
    let p = run(
        "pool",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let at = |c: usize, h: usize, wd: usize| xq[c * 12 + h * 4 + wd];
    let mut wap = vec![0.0; 12];
    let mut wmp = vec![0.0; 12];
    for c in 0..2 {
        for h in 0..2 {
            for wd in 0..3 {
                let lane = [
                    at(c, h, wd),
                    at(c, h, wd + 1),
                    at(c, h + 1, wd),
                    at(c, h + 1, wd + 1),
                ];
                wap[c * 6 + h * 3 + wd] = lane.iter().sum::<f64>() / 4.0;
                wmp[c * 6 + h * 3 + wd] = lane.iter().cloned().fold(f64::MIN, f64::max);
            }
        }
    }
    check(&out(&p, "o_ap").values(), &wap, 5e-3, 2e-3, "avg_pool");
    check(&out(&p, "o_mp").values(), &wmp, 0.0, 1e-3, "max_pool");
    let mut wag = Vec::new();
    let mut wmg = Vec::new();
    for c in 0..2 {
        let lane: Vec<f64> = (0..12).map(|i| at(c, i / 4, i % 4)).collect();
        wag.push(lane.iter().sum::<f64>() / 12.0);
        wmg.push(lane.iter().cloned().fold(f64::MIN, f64::max));
    }
    check(
        &out(&p, "o_apg").values(),
        &wag,
        5e-3,
        2e-3,
        "avg_pool_global",
    );
    check(
        &out(&p, "o_mpg").values(),
        &wmg,
        0.0,
        1e-3,
        "max_pool_global",
    );
    // nearest: each input pixel → 2×2 block
    let mut w = vec![0.0; 96];
    for c in 0..2 {
        for h in 0..6 {
            for wd in 0..8 {
                w[c * 48 + h * 8 + wd] = at(c, h / 2, wd / 2);
            }
        }
    }
    check(
        &out(&p, "o_unn").values(),
        &w,
        0.0,
        1e-3,
        "upsample_nearest",
    );
    // bilinear align_corners: src = dst * (D-1)/(D'-1)
    let mut w = vec![0.0; 96];
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    for c in 0..2 {
        for h in 0..6 {
            let sh = h as f64 * 2.0 / 5.0;
            let h0 = sh.floor() as usize;
            let h1 = (h0 + 1).min(2);
            let th = sh - h0 as f64;
            for wd in 0..8 {
                let sw = wd as f64 * 3.0 / 7.0;
                let w0 = sw.floor() as usize;
                let w1 = (w0 + 1).min(3);
                let tw = sw - w0 as f64;
                let top = lerp(at(c, h0, w0), at(c, h0, w1), tw);
                let bot = lerp(at(c, h1, w0), at(c, h1, w1), tw);
                w[c * 48 + h * 8 + wd] = lerp(top, bot, th);
            }
        }
    }
    check(
        &out(&p, "o_ub").values(),
        &w,
        1e-2,
        5e-3,
        "upsample_bilinear",
    );
}

// ---------- conv / linear / matmul ----------

#[test]
fn conv_e2e() {
    let shape = [1, 2, 4, 5];
    let x = randv(40, -1.0, 1.0, 0xC0FF);
    let w3 = randv(54, -0.5, 0.5, 0xACE0); // [3,2,3,3]
    let bias = randv(3, -0.2, 0.2, 0xB1A5);
    let mut b = Block::new();
    let wk = k16(&mut b, "wk", &[3, 2, 3, 3], &w3);
    let bk = k16(&mut b, "bk", &[3], &bias);
    b.conv(
        "x",
        &wk,
        Some(&bk),
        &[1, 1],
        "custom",
        &[1, 1, 1, 1],
        &[1, 1],
        1,
        &[1, 3, 4, 5],
        "o_c",
    );
    // stride-2 valid conv, same weights
    b.conv(
        "x",
        &wk,
        None,
        &[2, 2],
        "valid",
        &[0, 0, 0, 0],
        &[1, 1],
        1,
        &[1, 3, 1, 2],
        "o_c2",
    );
    // grouped conv: cin=2, groups=2 → w [4,1,2,2]
    let wg = randv(16, -0.5, 0.5, 0x6A0B);
    let wgk = k16(&mut b, "wgk", &[4, 1, 2, 2], &wg);
    b.conv(
        "x",
        &wgk,
        None,
        &[1, 1],
        "valid",
        &[0, 0, 0, 0],
        &[1, 1],
        2,
        &[1, 4, 3, 4],
        "o_cg",
    );
    // conv_transpose 2×2 stride 2: w [cin=2, cout=2, 2, 2]
    let wt = randv(16, -0.5, 0.5, 0x7BCC);
    let wtk = k16(&mut b, "wtk", &[2, 2, 2, 2], &wt);
    b.conv_transpose(
        "x",
        &wtk,
        None,
        &[2, 2],
        "valid",
        &[0, 0, 0, 0],
        &[1, 1],
        1,
        None,
        &[1, 2, 8, 10],
        "o_ct",
    );
    let defs: [(&str, &[i64]); 4] = [
        ("o_c", &[1, 3, 4, 5]),
        ("o_c2", &[1, 3, 1, 2]),
        ("o_cg", &[1, 4, 3, 4]),
        ("o_ct", &[1, 2, 8, 10]),
    ];
    b.outputs = defs.iter().map(|(n, _)| n.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> =
        defs.iter().map(|(n, s)| (*n, *s, DType::Fp16)).collect();
    let xb = f16_bytes(&x);
    let p = run(
        "conv",
        &b,
        &[("x", &shape, DType::Fp16)],
        &outs,
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let wq = qh(&w3);
    let bq = qh(&bias);
    let at = |c: usize, h: i64, wd: i64| -> f64 {
        if !(0..=3).contains(&h) || !(0..=4).contains(&wd) {
            0.0
        } else {
            xq[c * 20 + h as usize * 5 + wd as usize]
        }
    };
    // full custom-pad conv, stride 1
    let mut w = vec![0.0; 60];
    for oc in 0..3 {
        for h in 0..4 {
            for wd in 0..5 {
                let mut acc = bq[oc];
                for ic in 0..2 {
                    for kh in 0..3i64 {
                        for kw in 0..3i64 {
                            acc += at(ic, h as i64 + kh - 1, wd as i64 + kw - 1)
                                * wq[oc * 18 + ic * 9 + kh as usize * 3 + kw as usize];
                        }
                    }
                }
                w[oc * 20 + h * 5 + wd] = acc;
            }
        }
    }
    check(&out(&p, "o_c").values(), &w, 1e-2, 1e-2, "conv 3x3 pad");
    // stride-2 valid 3×3 kernel on 4×5 → out [1,2] spatial (h only 0)
    let mut w = vec![0.0; 6];
    for oc in 0..3 {
        for w_ow in 0..2usize {
            let mut acc = 0.0;
            for ic in 0..2 {
                for kh in 0..3i64 {
                    for kw in 0..3i64 {
                        acc += at(ic, kh, 2 * w_ow as i64 + kw)
                            * wq[oc * 18 + ic * 9 + kh as usize * 3 + kw as usize];
                    }
                }
            }
            w[oc * 2 + w_ow] = acc;
        }
    }
    check(&out(&p, "o_c2").values(), &w, 1e-2, 1e-2, "conv stride2");
    // grouped: out c 0..2 ← group0 (in c0), c 2..4 ← group1 (in c1); kernel 2×2
    let wgq = qh(&wg);
    let mut w = vec![0.0; 48];
    for oc in 0..4 {
        let ic = oc / 2; // cout_per_group=2; oc/2 → 0,0,1,1
        for h in 0..3 {
            for wd in 0..4 {
                let mut acc = 0.0;
                for kh in 0..2i64 {
                    for kw in 0..2i64 {
                        acc += at(ic, h as i64 + kh, wd as i64 + kw)
                            * wgq[oc * 4 + kh as usize * 2 + kw as usize];
                    }
                }
                w[oc * 12 + h * 4 + wd] = acc;
            }
        }
    }
    check(&out(&p, "o_cg").values(), &w, 1e-2, 1e-2, "conv groups");
    // conv_transpose 2×2 stride 2 → each input element scales a 2×2 w block
    let wtq = qh(&wt);
    let mut w = vec![0.0; 160];
    for ic in 0..2 {
        for h in 0..4 {
            for wd in 0..5 {
                let v = at(ic, h as i64, wd as i64);
                for oc in 0..2 {
                    for kh in 0..2 {
                        for kw in 0..2 {
                            w[oc * 80 + (h * 2 + kh) * 10 + wd * 2 + kw] +=
                                v * wtq[ic * 8 + oc * 4 + kh * 2 + kw];
                        }
                    }
                }
            }
        }
    }
    check(&out(&p, "o_ct").values(), &w, 1e-2, 1e-2, "conv_transpose");
}

#[test]
fn linear_matmul_e2e() {
    let mut b = Block::new();
    let w4 = randv(12, -0.5, 0.5, 0xD00D);
    let wk = k16(&mut b, "wk", &[4, 3], &w4);
    let bk = k16(&mut b, "bk", &[4], &[0.1, -0.2, 0.3, 0.0]);
    b.linear("x", &wk, Some(&bk), &[2, 4], "o_lin");
    // matmul [2,3] @ [3,4] with the second operand a const (ty=false → no transpose)
    let w5 = randv(12, -0.5, 0.5, 0xD00E);
    let mk = k16(&mut b, "mk", &[3, 4], &w5);
    b.matmul("x", &mk, false, &[2, 4], "o_mm");
    // matmul transpose_y: [2,3] @ [4,3]^T
    let w6 = randv(12, -0.5, 0.5, 0xD00F);
    let mk2 = k16(&mut b, "mk2", &[4, 3], &w6);
    b.matmul("x", &mk2, true, &[2, 4], "o_mmt");
    // batched matmul: [2,2,3] @ [2,3,4]
    let w7 = randv(24, -0.5, 0.5, 0xD010);
    let mk3 = k16(&mut b, "mk3", &[2, 3, 4], &w7);
    b.matmul("y", &mk3, false, &[2, 2, 4], "o_mmb");
    let defs: [(&str, &[i64]); 4] = [
        ("o_lin", &[2, 4]),
        ("o_mm", &[2, 4]),
        ("o_mmt", &[2, 4]),
        ("o_mmb", &[2, 2, 4]),
    ];
    b.outputs = defs.iter().map(|(n, _)| n.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> =
        defs.iter().map(|(n, s)| (*n, *s, DType::Fp16)).collect();
    let x = randv(6, -1.0, 1.0, 0x1EAF);
    let y = randv(12, -1.0, 1.0, 0x1EB0);
    let xb = f16_bytes(&x);
    let yb = f16_bytes(&y);
    let p = run(
        "lin_mm",
        &b,
        &[("x", &[2, 3], DType::Fp16), ("y", &[2, 2, 3], DType::Fp16)],
        &outs,
        &[
            Input {
                name: "x",
                shape: &[2, 3],
                data: &xb,
                dtype: DType::Fp16,
            },
            Input {
                name: "y",
                shape: &[2, 2, 3],
                data: &yb,
                dtype: DType::Fp16,
            },
        ],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let mm = |xq: &[f64], wq: &[f64], k: usize, n: usize, ty: bool| -> Vec<f64> {
        let mut outv = vec![0.0; (xq.len() / k) * n];
        for r in 0..xq.len() / k {
            for c in 0..n {
                for j in 0..k {
                    let wv = if ty { wq[c * k + j] } else { wq[j * n + c] };
                    outv[r * n + c] += xq[r * k + j] * wv;
                }
            }
        }
        outv
    };
    let w4q = qh(&w4);
    let bq = qh(&[0.1, -0.2, 0.3, 0.0]);
    let mut w = mm(&xq, &w4q, 3, 4, true);
    for (i, v) in w.iter_mut().enumerate() {
        *v += bq[i % 4];
    }
    check(&out(&p, "o_lin").values(), &w, 1e-2, 5e-3, "linear");
    let w = mm(&xq, &qh(&w5), 3, 4, false);
    check(&out(&p, "o_mm").values(), &w, 1e-2, 5e-3, "matmul");
    let w = mm(&xq, &qh(&w6), 3, 4, true);
    check(&out(&p, "o_mmt").values(), &w, 1e-2, 5e-3, "matmul ty");
    let yq = qh(&y);
    let w7q = qh(&w7);
    let mut w = vec![0.0; 16];
    for bt in 0..2 {
        for r in 0..2 {
            for c in 0..4 {
                for j in 0..3 {
                    w[bt * 8 + r * 4 + c] += yq[bt * 6 + r * 3 + j] * w7q[bt * 12 + j * 4 + c];
                }
            }
        }
    }
    check(&out(&p, "o_mmb").values(), &w, 1e-2, 5e-3, "matmul batched");
}

// ---------- space transforms ----------

#[test]
fn space_e2e() {
    // depth_to_space: [1,8,2,2] → [1,2,4,4]; space_to_depth inverts it;
    // pixel_shuffle: [1,4,2,2] → [1,1,4,4]
    let mut b = Block::new();
    b.depth_to_space("x", 2, &[1, 2, 4, 4], "o_d2s");
    b.pixel_shuffle("x", 2, &[1, 2, 4, 4], "o_ps");
    b.space_to_depth("y", 2, &[1, 8, 2, 2], "o_s2d");
    let defs: [(&str, &[i64]); 3] = [
        ("o_d2s", &[1, 2, 4, 4]),
        ("o_ps", &[1, 2, 4, 4]),
        ("o_s2d", &[1, 8, 2, 2]),
    ];
    b.outputs = defs.iter().map(|(n, _)| n.to_string()).collect();
    let outs: Vec<(&str, &[i64], DType)> =
        defs.iter().map(|(n, s)| (*n, *s, DType::Fp16)).collect();
    let x = randv(32, -1.0, 1.0, 0x5DA0);
    let y = randv(32, -1.0, 1.0, 0x5DB0);
    let xb = f16_bytes(&x);
    let yb = f16_bytes(&y);
    let p = run(
        "space",
        &b,
        &[
            ("x", &[1, 8, 2, 2], DType::Fp16),
            ("y", &[1, 2, 4, 4], DType::Fp16),
        ],
        &outs,
        &[
            Input {
                name: "x",
                shape: &[1, 8, 2, 2],
                data: &xb,
                dtype: DType::Fp16,
            },
            Input {
                name: "y",
                shape: &[1, 2, 4, 4],
                data: &yb,
                dtype: DType::Fp16,
            },
        ],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let yq = qh(&y);
    // depth_to_space (CRD order — the sub-channel index is OUTER):
    //   out[c, hb+dh, wb+dw] = in[(dh*b + dw)*C_out + c, h, w]
    let mut w = vec![0.0; 32];
    for c in 0..2 {
        for h in 0..4 {
            for wd in 0..4 {
                let oc = ((h % 2) * 2 + wd % 2) * 2 + c;
                w[c * 16 + h * 4 + wd] = xq[oc * 4 + (h / 2) * 2 + wd / 2];
            }
        }
    }
    check(&out(&p, "o_d2s").values(), &w, 0.0, 1e-3, "depth_to_space");
    // pixel_shuffle (PyTorch order — channel is OUTER):
    //   out[c, h*r+dh, w*r+dw] = in[c*r² + dh*r + dw, h, w]
    let mut w = vec![0.0; 32];
    for c in 0..2 {
        for h in 0..4 {
            for wd in 0..4 {
                let oc = c * 4 + (h % 2) * 2 + wd % 2;
                w[c * 16 + h * 4 + wd] = xq[oc * 4 + (h / 2) * 2 + wd / 2];
            }
        }
    }
    check(&out(&p, "o_ps").values(), &w, 0.0, 1e-3, "pixel_shuffle");
    // space_to_depth: CRD inverse — out_ch = dh*b²*C_out? (dh*b+dw)*C_out+c
    let mut w = vec![0.0; 32];
    for oc in 0..8 {
        for h in 0..2 {
            for wd in 0..2 {
                let dh = oc / 4;
                let dw = (oc % 4) / 2;
                let c = oc % 2;
                w[oc * 4 + h * 2 + wd] = yq[c * 16 + (h * 2 + dh) * 4 + wd * 2 + dw];
            }
        }
    }
    check(&out(&p, "o_s2d").values(), &w, 0.0, 1e-3, "space_to_depth");
}

/// Every helper is exercised above; `Block::op` remains the raw escape
/// hatch — this test proves it still works end-to-end.
#[test]
fn raw_op_escape_hatch() {
    let shape = [2, 4];
    let mut b = Block::new();
    let m1 = b.konst_f16("m1", -1.0);
    let vt = ValueType::Tensor(TensorType::f16(&shape));
    let y = b.o1(
        "mul",
        vec![("x".into(), bind("x").1), ("y".into(), bind(&m1).1)],
        "o_neg",
        vt,
    );
    b.outputs = vec![y];
    let x = randv(8, -2.0, 2.0, 0xE5CA);
    let xb = f16_bytes(&x);
    let p = run(
        "raw_op",
        &b,
        &[("x", &shape, DType::Fp16)],
        &[("o_neg", &shape, DType::Fp16)],
        &[Input {
            name: "x",
            shape: &shape,
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    let Some(p) = p else { return };
    let w: Vec<f64> = qh(&x).iter().map(|v| -*v).collect();
    check(&out(&p, "o_neg").values(), &w, 0.0, 1e-3, "raw mul");
}
