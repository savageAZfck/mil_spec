//! Shared test utilities: a hand-rolled ONNX protobuf *encoder* (the
//! mirror of `mil_onnx::proto`), a `.mlpackage` write→compile→predict
//! harness, and f64 reference helpers.
//!
//! No Python, no onnx package — models are assembled field-by-field the
//! same way [`mil_spec::write_mlpackage`] tests assemble protos.
#![allow(dead_code)]

use mil_infer::{ComputeUnits, Input, Model, Output, Prediction};
use mil_onnx::{BuiltModel, OnnxError};
use std::sync::OnceLock;

// ================= protobuf encoder =================

/// ONNX `TensorProto.DataType` constants (mirror `model::elem`).
pub mod et {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const UINT16: i32 = 4;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const STRING: i32 = 8;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const UINT32: i32 = 12;
    pub const UINT64: i32 = 13;
    pub const BFLOAT16: i32 = 16;
}

/// Append a varint.
pub fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
    varint(((field << 3) | wire) as u64, out);
}

/// `field = v` varint (two's-complement i64).
pub fn f_i(field: u32, v: i64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(v as u64, out);
}

/// `field = v` fixed32 float.
pub fn f_f32(field: u32, v: f32, out: &mut Vec<u8>) {
    tag(field, 5, out);
    out.extend_from_slice(&v.to_le_bytes());
}

/// `field = bytes` length-delimited.
pub fn f_len(field: u32, b: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(b.len() as u64, out);
    out.extend_from_slice(b);
}

/// `field = "s"`.
pub fn f_str(field: u32, s: &str, out: &mut Vec<u8>) {
    f_len(field, s.as_bytes(), out);
}

/// `field = sub-message`.
pub fn f_msg(field: u32, msg: &[u8], out: &mut Vec<u8>) {
    f_len(field, msg, out);
}

/// `field = packed varints`.
pub fn f_packed_i64(field: u32, vs: &[i64], out: &mut Vec<u8>) {
    let mut b = Vec::new();
    for &v in vs {
        varint(v as u64, &mut b);
    }
    f_len(field, &b, out);
}

/// `field = unpacked varints` (proto2 default).
pub fn f_rep_i64(field: u32, vs: &[i64], out: &mut Vec<u8>) {
    for &v in vs {
        f_i(field, v, out);
    }
}

/// `field = packed fixed32`.
pub fn f_packed_f32(field: u32, vs: &[f32], out: &mut Vec<u8>) {
    let mut b = Vec::new();
    for &v in vs {
        b.extend_from_slice(&v.to_le_bytes());
    }
    f_len(field, &b, out);
}

/// `field = unpacked fixed32` (proto2 default).
pub fn f_rep_f32(field: u32, vs: &[f32], out: &mut Vec<u8>) {
    for &v in vs {
        f_f32(field, v, out);
    }
}

// ---------- typed proto builders ----------

/// `TensorProto` with `raw_data` fp32 payload.
pub fn t_f32(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut m = Vec::new();
    f_rep_i64(1, dims, &mut m); // dims (unpacked — proto2 style)
    f_i(2, et::FLOAT as i64, &mut m);
    f_str(8, name, &mut m);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    f_len(9, &raw, &mut m);
    m
}

/// `TensorProto` with unpacked `float_data` (no raw_data) — exercises
/// the non-raw decode path.
pub fn t_f32_list(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut m = Vec::new();
    f_packed_i64(1, dims, &mut m); // packed — exercise packed path
    f_i(2, et::FLOAT as i64, &mut m);
    f_rep_f32(4, data, &mut m);
    f_str(8, name, &mut m);
    m
}

/// `TensorProto` fp32 scalar.
pub fn t_scalar_f32(name: &str, v: f32) -> Vec<u8> {
    t_f32(name, &[], &[v])
}

/// `TensorProto` with `int64_data` (unpacked varints).
pub fn t_i64(name: &str, dims: &[i64], data: &[i64]) -> Vec<u8> {
    let mut m = Vec::new();
    f_rep_i64(1, dims, &mut m);
    f_i(2, et::INT64 as i64, &mut m);
    f_rep_i64(7, data, &mut m); // int64_data
    f_str(8, name, &mut m);
    m
}

/// `TensorProto` with `int32_data` (packed varints).
pub fn t_i32(name: &str, dims: &[i64], data: &[i32]) -> Vec<u8> {
    let mut m = Vec::new();
    f_rep_i64(1, dims, &mut m);
    f_i(2, et::INT32 as i64, &mut m);
    let i64s: Vec<i64> = data.iter().map(|&v| v as i64).collect();
    f_packed_i64(5, &i64s, &mut m); // int32_data packed
    f_str(8, name, &mut m);
    m
}

/// `TensorProto` scalar i64.
pub fn t_scalar_i64(name: &str, v: i64) -> Vec<u8> {
    t_i64(name, &[], &[v])
}

/// `ValueInfoProto` for a statically-shaped float tensor.
pub fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        f_i(1, d, &mut dim); // dim_value
        f_msg(1, &dim, &mut shape);
    }
    let mut tt = Vec::new();
    f_i(1, elem_type as i64, &mut tt); // elem_type
    f_msg(2, &shape, &mut tt);
    let mut ty = Vec::new();
    f_msg(1, &tt, &mut ty); // tensor_type
    let mut vi = Vec::new();
    f_str(1, name, &mut vi);
    f_msg(2, &ty, &mut vi);
    vi
}

/// `ValueInfoProto` with a `dim_param` (symbolic) dim.
pub fn value_info_param(name: &str, elem_type: i32, params: &[&str]) -> Vec<u8> {
    let mut shape = Vec::new();
    for p in params {
        let mut dim = Vec::new();
        if let Ok(d) = p.parse::<i64>() {
            f_i(1, d, &mut dim);
        } else {
            f_str(2, p, &mut dim); // dim_param
        }
        f_msg(1, &dim, &mut shape);
    }
    let mut tt = Vec::new();
    f_i(1, elem_type as i64, &mut tt);
    f_msg(2, &shape, &mut tt);
    let mut ty = Vec::new();
    f_msg(1, &tt, &mut ty);
    let mut vi = Vec::new();
    f_str(1, name, &mut vi);
    f_msg(2, &ty, &mut vi);
    vi
}

/// One `AttributeProto`.
#[derive(Clone)]
pub enum Attr {
    /// `i`
    I(&'static str, i64),
    /// `ints` (packed encoding — proto3-style).
    Ints(&'static str, &'static [i64]),
    /// `f`
    F(&'static str, f32),
    /// `floats` (unpacked — proto2 style).
    Fs(&'static str, &'static [f32]),
    /// `s`
    S(&'static str, &'static str),
    /// `strings`
    Ss(&'static str, &'static [&'static str]),
    /// `t` (serialized TensorProto)
    T(&'static str, Vec<u8>),
}

impl Attr {
    fn encode(&self) -> Vec<u8> {
        let mut m = Vec::new();
        match self {
            Attr::I(n, v) => {
                f_str(1, n, &mut m);
                f_i(3, *v, &mut m);
            }
            Attr::Ints(n, vs) => {
                f_str(1, n, &mut m);
                f_packed_i64(8, vs, &mut m);
            }
            Attr::F(n, v) => {
                f_str(1, n, &mut m);
                f_f32(2, *v, &mut m);
            }
            Attr::Fs(n, vs) => {
                f_str(1, n, &mut m);
                f_rep_f32(7, vs, &mut m);
            }
            Attr::S(n, s) => {
                f_str(1, n, &mut m);
                f_len(4, s.as_bytes(), &mut m);
            }
            Attr::Ss(n, ss) => {
                f_str(1, n, &mut m);
                for s in *ss {
                    f_len(9, s.as_bytes(), &mut m);
                }
            }
            Attr::T(n, t) => {
                f_str(1, n, &mut m);
                f_msg(5, t, &mut m);
            }
        }
        m
    }
}

/// `NodeProto`.
pub fn node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[Attr],
    name: &str,
) -> Vec<u8> {
    node_dom(op_type, inputs, outputs, attrs, name, "")
}

/// `NodeProto` with explicit `domain` (e.g. `"ai.onnx.ml"`).
pub fn node_dom(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[Attr],
    name: &str,
    domain: &str,
) -> Vec<u8> {
    let mut m = Vec::new();
    for i in inputs {
        f_str(1, i, &mut m);
    }
    for o in outputs {
        f_str(2, o, &mut m);
    }
    if !name.is_empty() {
        f_str(3, name, &mut m);
    }
    f_str(4, op_type, &mut m);
    for a in attrs {
        f_msg(5, &a.encode(), &mut m);
    }
    if !domain.is_empty() {
        f_str(7, domain, &mut m);
    }
    m
}

/// `GraphProto`.
pub struct Graph<'a> {
    pub nodes: Vec<Vec<u8>>,
    pub initializers: Vec<Vec<u8>>,
    pub inputs: Vec<&'a str>,
    pub outputs: Vec<&'a str>,
    pub name: &'a str,
}

/// Assemble a `GraphProto` from builder parts.
pub fn graph(
    name: &str,
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
) -> Vec<u8> {
    let mut m = Vec::new();
    for n in &nodes {
        f_msg(1, n, &mut m);
    }
    if !name.is_empty() {
        f_str(2, name, &mut m);
    }
    for i in &initializers {
        f_msg(5, i, &mut m);
    }
    for v in &inputs {
        f_msg(11, v, &mut m);
    }
    for v in &outputs {
        f_msg(12, v, &mut m);
    }
    m
}

/// `ModelProto` at opset 13 (`""` domain).
pub fn model(g: &[u8]) -> Vec<u8> {
    model_opset(g, 13)
}

/// `ModelProto` at a chosen opset.
pub fn model_opset(g: &[u8], opset: i64) -> Vec<u8> {
    let mut m = Vec::new();
    f_i(1, 8, &mut m); // ir_version
    f_str(2, "mil_onnx test encoder", &mut m);
    f_msg(7, g, &mut m);
    let mut os = Vec::new();
    f_str(1, "", &mut os);
    f_i(2, opset, &mut os);
    f_msg(8, &os, &mut m);
    let mut ml = Vec::new();
    f_str(1, "ai.onnx.ml", &mut ml);
    f_i(2, 3, &mut ml);
    f_msg(8, &ml, &mut m);
    m
}

// ================= e2e harness =================

static HAVE_COREMLC: OnceLock<bool> = OnceLock::new();

/// `xcrun -f coremlc` presence — memoized.
pub fn coremlc() -> bool {
    *HAVE_COREMLC.get_or_init(|| mil_compile::coremlc_path().is_some())
}

/// `MIL_KEEP_ARTIFACTS=1` keeps temp dirs for debugging.
pub fn keep_artifacts() -> bool {
    std::env::var("MIL_KEEP_ARTIFACTS").as_deref() == Ok("1")
}

/// Decode→convert ONNX bytes; unwrap with context.
pub fn convert(bytes: &[u8]) -> Result<BuiltModel, OnnxError> {
    mil_onnx::convert_bytes(bytes)
}

/// Write `built` as `.mlpackage`, compile, predict. `None` if no coremlc.
pub fn run(tag: &str, built: &BuiltModel, feeds: &[Input<'_>]) -> Option<Prediction> {
    if !coremlc() {
        eprintln!("coremlc unavailable — skipping {tag}");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("mil_onnx_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let pkg = dir.join("m.mlpackage");
    built
        .write_mlpackage(&pkg)
        .unwrap_or_else(|e| panic!("{tag}: write package: {e}"));
    let cm = mil_compile::compile(&pkg, &dir.join("compiled"))
        .unwrap_or_else(|e| panic!("coremlc rejected {tag} model:\n{e}"));
    let m = Model::load(&cm.path, ComputeUnits::All).expect("load compiled model");
    let p = m
        .predict(feeds)
        .unwrap_or_else(|e| panic!("predict failed for {tag}: {e}"));
    if !keep_artifacts() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Some(p)
}

/// fp16 LE bytes for feeds.
pub fn f16_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// xorshift64 deterministic values in `[lo, hi)`.
pub fn randv(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
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

/// Fetch an output by name.
pub fn out<'p>(p: &'p Prediction, name: &str) -> &'p Output {
    p.outputs
        .iter()
        .find(|o| o.name == name)
        .unwrap_or_else(|| panic!("missing output {name}"))
}

/// `|got - want| <= atol + rtol·|want|` elementwise.
pub fn check(got: &[f32], want: &[f64], rtol: f64, atol: f64, what: &str) {
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

/// f64 view of the fp16-rounded input (what CoreML actually saw).
pub fn qh(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|v| half::f16::from_f32(*v).to_f32() as f64)
        .collect()
}
