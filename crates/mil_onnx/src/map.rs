//! ONNX graph → [`mil_spec::Block`] lowering.
//!
//! The converter walks `graph.node` (ONNX guarantees topological order),
//! keeps a static `(shape, dtype)` environment, and emits MIL ops whose
//! output names are the (sanitized) ONNX value names. Weights enter as
//! `const` ops — small ones inline, tensors at or above
//! [`INLINE_MAX_BYTES`] as `weight.bin` blob references.
//!
//! Dtype policy: every float tensor lowers to fp16; int tensors to
//! int32; bools to bool. `DOUBLE`/`INT64`/`FLOAT16` sources are cast
//! into that lattice. String tensors are rejected.

use crate::classical;
use crate::model::{elem, Dim, ModelProto, NodeProto, TensorProto};
use crate::{BuiltModel, OnnxError};
use mil_spec::{
    bind, bind_many, Block, DType, Feature, Immediate, TensorType, Value, ValueType, WeightBin, NVT,
};
use std::collections::{BTreeMap, HashMap};

/// Initializers at or above this many payload bytes go to `weight.bin`;
/// smaller ones become inline `const` immediates.
pub const INLINE_MAX_BYTES: usize = 1024;

/// Conversion knobs.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// Inline-const threshold in bytes (see [`INLINE_MAX_BYTES`]).
    pub inline_max_bytes: usize,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            inline_max_bytes: INLINE_MAX_BYTES,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TInfo {
    pub shape: Vec<i64>,
    pub dtype: DType,
}

type Res<T> = Result<T, OnnxError>;

fn numel(shape: &[i64]) -> usize {
    shape.iter().map(|d| (*d).max(0) as usize).product()
}

/// Numpy-style broadcast of two shapes; `Err` when incompatible.
fn bcast(a: &[i64], b: &[i64]) -> Res<Vec<i64>> {
    let n = a.len().max(b.len());
    let mut out = vec![0i64; n];
    for i in 0..n {
        let da = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let db = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(OnnxError::BadShape(format!(
                "cannot broadcast {a:?} with {b:?}"
            )));
        };
    }
    Ok(out)
}

fn norm_axis(ax: i64, rank: usize, what: &str) -> Res<i64> {
    let r = rank as i64;
    let a = if ax < 0 { ax + r } else { ax };
    if a < 0 || a >= r {
        return Err(OnnxError::BadShape(format!(
            "{what}: axis {ax} out of range for rank {r}"
        )));
    }
    Ok(a)
}

/// Reduce `shape` over `axes` (normalized positions), `keep` dims.
fn reduce_shape(shape: &[i64], axes: &[i64], keep: bool) -> Vec<i64> {
    if keep {
        let mut s = shape.to_vec();
        for &a in axes {
            s[a as usize] = 1;
        }
        s
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(&(*i as i64)))
            .map(|(_, &d)| d)
            .collect()
    }
}

/// Converter state — `pub(crate)` so `classical` can lower `ai.onnx.ml`
/// nodes through the same emitters.
pub(crate) struct Ctx<'a> {
    pub b: Block,
    /// sanitized name → shape/dtype of produced values.
    pub env: HashMap<String, TInfo>,
    /// sanitized name → initializer tensor.
    pub init: HashMap<String, &'a TensorProto>,
    /// initializer sanitized name → produced const value name.
    pub made: HashMap<String, String>,
    /// value-name aliases (Identity/Dropout passthroughs).
    pub alias: HashMap<String, String>,
    /// ONNX name → sanitized name.
    pub names: HashMap<String, String>,
    pub used_names: std::collections::HashSet<String>,
    pub wb: WeightBin,
    pub has_blob: bool,
    pub inline_max: usize,
    pub opset: i64,
    pub hist: BTreeMap<String, usize>,
}

impl<'a> Ctx<'a> {
    /// A bare context for the classical-JSON path (no ONNX graph).
    pub(crate) fn empty(opset: i64) -> Ctx<'static> {
        Ctx {
            b: Block::new(),
            env: HashMap::new(),
            init: HashMap::new(),
            made: HashMap::new(),
            alias: HashMap::new(),
            names: HashMap::new(),
            used_names: std::collections::HashSet::new(),
            wb: WeightBin::new(),
            has_blob: false,
            inline_max: INLINE_MAX_BYTES,
            opset,
            hist: BTreeMap::new(),
        }
    }

    // ---------- names ----------

    /// Sanitized unique value name for an ONNX name.
    pub(crate) fn san(&mut self, n: &str) -> String {
        if let Some(s) = self.names.get(n) {
            return s.clone();
        }
        let mut s: String = n
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if s.is_empty() || s.chars().next().unwrap().is_ascii_digit() {
            s = format!("n_{s}");
        }
        if self.used_names.contains(&s) {
            let mut k = 1;
            while self.used_names.contains(&format!("{s}_{k}")) {
                k += 1;
            }
            s = format!("{s}_{k}");
        }
        self.used_names.insert(s.clone());
        self.names.insert(n.to_string(), s.clone());
        s
    }

    /// Resolve an ONNX value name to a bound MIL value name,
    /// materializing initializer consts lazily.
    pub(crate) fn resolve(&mut self, onnx: &str, node: &str) -> Res<String> {
        let sn = self.san(onnx);
        if let Some(a) = self.alias.get(&sn) {
            return Ok(a.clone());
        }
        if self.env.contains_key(&sn) {
            return Ok(sn);
        }
        if let Some(tp) = self.init.get(&sn) {
            return self.materialize(&sn, tp);
        }
        Err(OnnxError::MissingInput {
            name: onnx.to_string(),
            node: node.to_string(),
        })
    }

    /// Shape/dtype of a resolved *MIL* value name.
    pub(crate) fn info(&self, produced: &str) -> Res<TInfo> {
        self.env
            .get(produced)
            .cloned()
            .ok_or_else(|| OnnxError::MissingInput {
                name: produced.into(),
                node: "internal".into(),
            })
    }

    /// Info for an ONNX-named value (resolving aliases/inits).
    #[allow(dead_code)]
    pub(crate) fn info_of(&mut self, onnx: &str, node: &str) -> Res<TInfo> {
        let p = self.resolve(onnx, node)?;
        self.info(&p)
    }

    /// Static integer contents of an input (initializer required).
    pub(crate) fn static_i64(&mut self, onnx: &str, node: &str) -> Res<Vec<i64>> {
        let sn = self.san(onnx);
        if let Some(tp) = self.init.get(&sn) {
            return tp.as_i64();
        }
        if let Some(a) = self.alias.get(&sn) {
            if let Some(tp) = self.init.get(a.as_str()) {
                return tp.as_i64();
            }
        }
        Err(OnnxError::Dynamic(format!(
            "node '{node}': input '{onnx}' must be a static initializer"
        )))
    }

    /// Static float contents of an input.
    pub(crate) fn static_f32(&mut self, onnx: &str, node: &str) -> Res<Vec<f32>> {
        let sn = self.san(onnx);
        if let Some(tp) = self.init.get(&sn) {
            return tp.as_f32();
        }
        Err(OnnxError::Dynamic(format!(
            "node '{node}': input '{onnx}' must be a static initializer"
        )))
    }

    // ---------- const emitters ----------

    /// Register a produced name's static shape/dtype in env.
    fn env_set(&mut self, name: &str, shape: &[i64], dtype: DType) {
        self.env.insert(
            name.to_string(),
            TInfo {
                shape: shape.to_vec(),
                dtype,
            },
        );
    }

    pub(crate) fn k_i32v(&mut self, vs: &[i32]) -> String {
        let n = self.b.fresh("ci32");
        let r = self.b.konst_i32(&n, vs);
        self.env_set(&r, &[vs.len() as i64], DType::Int32);
        r
    }
    pub(crate) fn k_i32s(&mut self, v: i32) -> String {
        let n = self.b.fresh("ci32s");
        let r = self.b.konst_scalar_i32(&n, v);
        self.env_set(&r, &[], DType::Int32);
        r
    }
    pub(crate) fn k_f16(&mut self, v: f32) -> String {
        let n = self.b.fresh("cf16");
        let r = self.b.konst_f16(&n, v);
        self.env_set(&r, &[], DType::Fp16);
        r
    }
    pub(crate) fn k_bool(&mut self, v: bool) -> String {
        let n = self.b.fresh("cbool");
        let r = self.b.konst_bool(&n, v);
        self.env_set(&r, &[], DType::Bool);
        r
    }
    pub(crate) fn k_str(&mut self, v: &str) -> String {
        let n = self.b.fresh("cstr");
        let r = self.b.konst_str(&n, v);
        // strings never enter the tensor env; record anyway for info()
        self.env_set(&r, &[], DType::Str);
        r
    }
    pub(crate) fn k_bools(&mut self, vs: &[bool]) -> String {
        let n = self.b.fresh("cbv");
        let shape = vec![vs.len() as i64];
        let vt = ValueType::Tensor(TensorType {
            dtype: DType::Bool,
            shape: shape.clone(),
        });
        let r = self.b.op(
            "const",
            vec![],
            vec![(&n, vt)],
            vec![("val".into(), Value::bools(vs))],
        )[0]
        .clone();
        self.env_set(&r, &shape, DType::Bool);
        r
    }
    pub(crate) fn k_f16t(&mut self, shape: &[i64], vs: &[f32]) -> String {
        let n = self.b.fresh("cf16t");
        let vt = ValueType::Tensor(TensorType::f16(shape));
        let r = self.b.op(
            "const",
            vec![],
            vec![(&n, vt)],
            vec![("val".into(), Value::f16s(shape, vs))],
        )[0]
        .clone();
        self.env_set(&r, shape, DType::Fp16);
        r
    }
    pub(crate) fn k_i32t(&mut self, shape: &[i64], vs: &[i32]) -> String {
        let n = self.b.fresh("ci32t");
        let vt = ValueType::Tensor(TensorType {
            dtype: DType::Int32,
            shape: shape.to_vec(),
        });
        let r = self.b.op(
            "const",
            vec![],
            vec![(&n, vt)],
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
        .clone();
        self.env_set(&r, shape, DType::Int32);
        r
    }

    /// fp16 tensor const that goes to `weight.bin` at/above the inline
    /// threshold — for converter-generated weights (violation matrices).
    pub(crate) fn k_f16t_w(&mut self, shape: &[i64], vs: &[f32]) -> String {
        if vs.len() * 4 >= self.inline_max {
            let n = self.b.fresh("wf16");
            let raw: Vec<u8> = vs
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            let off = self.wb.put(&n, DType::Fp16, shape, &raw);
            self.has_blob = true;
            let r = self.b.konst_blob(
                &n,
                "@model_path/weights/weight.bin",
                off,
                DType::Fp16,
                shape,
            );
            self.env_set(&r, shape, DType::Fp16);
            r
        } else {
            self.k_f16t(shape, vs)
        }
    }

    /// Materialize an initializer as a `const` op (inline or blob).
    pub(crate) fn materialize(&mut self, sn: &str, tp: &TensorProto) -> Res<String> {
        if let Some(p) = self.made.get(sn) {
            return Ok(p.clone());
        }
        if tp.external {
            return Err(OnnxError::Unsupported(format!(
                "initializer '{sn}' uses external data"
            )));
        }
        let produced = match tp.data_type {
            elem::FLOAT | elem::FLOAT16 | elem::DOUBLE => {
                let vs = tp.as_f32()?;
                if vs.len() != tp.count() {
                    return Err(OnnxError::Malformed {
                        what: format!("initializer '{sn}'"),
                        detail: format!("{} elems declared, {} provided", tp.count(), vs.len()),
                    });
                }
                let bytes = vs.len() * 4;
                if bytes >= self.inline_max {
                    let raw: Vec<u8> = vs
                        .iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                        .collect();
                    let off = self.wb.put(sn, DType::Fp16, &tp.dims, &raw);
                    self.has_blob = true;
                    self.b.konst_blob(
                        sn,
                        "@model_path/weights/weight.bin",
                        off,
                        DType::Fp16,
                        &tp.dims,
                    )
                    // env registered below via materialize's epilogue
                } else {
                    let vt = ValueType::Tensor(TensorType::f16(&tp.dims));
                    self.b.op(
                        "const",
                        vec![],
                        vec![(sn, vt)],
                        vec![("val".into(), Value::f16s(&tp.dims, &vs))],
                    )[0]
                    .clone()
                }
            }
            elem::INT32
            | elem::INT64
            | elem::UINT8
            | elem::INT8
            | elem::UINT16
            | elem::INT16
            | elem::UINT32
            | elem::UINT64 => {
                let vs = tp.as_i64()?;
                if vs.len() != tp.count() {
                    return Err(OnnxError::Malformed {
                        what: format!("initializer '{sn}'"),
                        detail: format!("{} elems declared, {} provided", tp.count(), vs.len()),
                    });
                }
                let mut v32 = Vec::with_capacity(vs.len());
                for &v in &vs {
                    v32.push(i32::try_from(v).map_err(|_| {
                        OnnxError::Unsupported(format!(
                            "initializer '{sn}': int64 value {v} exceeds int32"
                        ))
                    })?);
                }
                let vt = ValueType::Tensor(TensorType {
                    dtype: DType::Int32,
                    shape: tp.dims.clone(),
                });
                self.b.op(
                    "const",
                    vec![],
                    vec![(sn, vt)],
                    vec![(
                        "val".into(),
                        Value::Imm(
                            ValueType::Tensor(TensorType {
                                dtype: DType::Int32,
                                shape: tp.dims.clone(),
                            }),
                            Immediate::Ints(v32),
                        ),
                    )],
                )[0]
                .clone()
            }
            elem::BOOL => {
                let vs: Vec<bool> = tp.as_i64()?.iter().map(|&v| v != 0).collect();
                let vt = ValueType::Tensor(TensorType {
                    dtype: DType::Bool,
                    shape: tp.dims.clone(),
                });
                self.b.op(
                    "const",
                    vec![],
                    vec![(sn, vt)],
                    vec![("val".into(), Value::bools(&vs))],
                )[0]
                .clone()
            }
            t => {
                return Err(OnnxError::Unsupported(format!(
                    "initializer '{sn}': dtype {t}"
                )))
            }
        };
        self.env.insert(
            produced.clone(),
            TInfo {
                shape: tp.dims.clone(),
                dtype: match tp.data_type {
                    elem::BOOL => DType::Bool,
                    elem::FLOAT | elem::FLOAT16 | elem::DOUBLE => DType::Fp16,
                    _ => DType::Int32,
                },
            },
        );
        self.made.insert(sn.to_string(), produced.clone());
        Ok(produced)
    }

    // ---------- op emission ----------

    pub(crate) fn tt(&self, dt: DType, shape: &[i64]) -> ValueType {
        ValueType::Tensor(TensorType {
            dtype: dt,
            shape: shape.to_vec(),
        })
    }

    /// Emit an op; `outs` are (produced name, dtype, shape).
    pub(crate) fn emit(
        &mut self,
        ty: &str,
        inputs: Vec<(String, mil_spec::Argument)>,
        outs: Vec<(&str, DType, &[i64])>,
    ) -> Vec<String> {
        *self.hist.entry(ty.to_string()).or_insert(0) += 1;
        let ov: Vec<(&str, ValueType)> =
            outs.iter().map(|(n, d, s)| (*n, self.tt(*d, s))).collect();
        let names = self.b.op(ty, inputs, ov, vec![]);
        for (i, (n, d, s)) in outs.iter().enumerate() {
            self.env.insert(
                names[i].clone(),
                TInfo {
                    shape: s.to_vec(),
                    dtype: *d,
                },
            );
            let _ = n;
        }
        names
    }

    pub(crate) fn e1(
        &mut self,
        ty: &str,
        inputs: Vec<(String, mil_spec::Argument)>,
        out: &str,
        dt: DType,
        shape: &[i64],
    ) -> String {
        self.emit(ty, inputs, vec![(out, dt, shape)])[0].clone()
    }

    /// Unary passthrough op.
    pub(crate) fn unary(&mut self, ty: &str, x: &str, out: &str) -> Res<String> {
        let info = self.info(x)?;
        Ok(self.e1(
            ty,
            vec![("x".into(), bind(x).1)],
            out,
            info.dtype,
            &info.shape.clone(),
        ))
    }

    pub(crate) fn binary(&mut self, ty: &str, x: &str, y: &str, out: &str) -> Res<String> {
        let (a, b) = (self.info(x)?, self.info(y)?);
        let shape = bcast(&a.shape, &b.shape)?;
        Ok(self.e1(
            ty,
            vec![("x".into(), bind(x).1), ("y".into(), bind(y).1)],
            out,
            a.dtype,
            &shape,
        ))
    }

    pub(crate) fn compare(&mut self, ty: &str, x: &str, y: &str, out: &str) -> Res<String> {
        let (a, b) = (self.info(x)?, self.info(y)?);
        let shape = bcast(&a.shape, &b.shape)?;
        Ok(self.e1(
            ty,
            vec![("x".into(), bind(x).1), ("y".into(), bind(y).1)],
            out,
            DType::Bool,
            &shape,
        ))
    }

    pub(crate) fn reduce(
        &mut self,
        ty: &str,
        x: &str,
        axes: &[i64],
        keep: bool,
        out: &str,
        dt: DType,
    ) -> Res<String> {
        let info = self.info(x)?;
        let rank = info.shape.len();
        let mut ax: Vec<i64> = axes.to_vec();
        if ax.is_empty() {
            ax = (0..rank as i64).collect();
        }
        let mut norm = Vec::with_capacity(ax.len());
        for &a in &ax {
            norm.push(norm_axis(a, rank, ty)?);
        }
        let out_shape = reduce_shape(&info.shape, &norm, keep);
        let a = self.k_i32v(&norm.iter().map(|v| *v as i32).collect::<Vec<_>>());
        let kd = self.k_bool(keep);
        Ok(self.e1(
            ty,
            vec![
                ("x".into(), bind(x).1),
                ("axes".into(), bind(&a).1),
                ("keep_dims".into(), bind(&kd).1),
            ],
            out,
            dt,
            &out_shape,
        ))
    }
}

// ---------- attribute helpers ----------

fn as_i32s(vs: &[i64]) -> Res<Vec<i32>> {
    vs.iter()
        .map(|&v| {
            i32::try_from(v)
                .map_err(|_| OnnxError::Unsupported(format!("attr int {v} exceeds int32")))
        })
        .collect()
}

// ---------- the converter ----------

/// Convert a decoded ONNX model into a [`BuiltModel`].
pub fn convert_graph(model: &ModelProto, opts: &ConvertOptions) -> Res<BuiltModel> {
    let g = &model.graph;
    let mut cx = Ctx {
        b: Block::new(),
        env: HashMap::new(),
        init: HashMap::new(),
        made: HashMap::new(),
        alias: HashMap::new(),
        names: HashMap::new(),
        used_names: std::collections::HashSet::new(),
        wb: WeightBin::new(),
        has_blob: false,
        inline_max: opts.inline_max_bytes,
        opset: model.opset(""),
        hist: BTreeMap::new(),
    };

    // Seed initializers (sanitized names).
    for tp in &g.initializers {
        if tp.name.is_empty() {
            return Err(OnnxError::Malformed {
                what: "initializer".into(),
                detail: "unnamed tensor".into(),
            });
        }
        let sn = cx.san(&tp.name);
        cx.init.insert(sn, tp);
    }

    // Graph inputs that aren't initializers become features + env.
    let mut inputs = Vec::new();
    let mut fn_inputs = Vec::new();
    for vi in &g.inputs {
        let sn = cx.san(&vi.name);
        if cx.init.contains_key(&sn) {
            continue; // opset≥9: initializers may be listed as inputs
        }
        let shape = static_shape(vi, "input")?;
        let dtype = feature_dtype(vi.elem_type, &vi.name)?;
        cx.env.insert(
            sn.clone(),
            TInfo {
                shape: shape.clone(),
                dtype,
            },
        );
        inputs.push(Feature {
            name: sn.clone(),
            shape: shape.clone(),
            dtype,
            is_state: false,
        });
        fn_inputs.push(NVT {
            name: sn,
            ty: ValueType::Tensor(TensorType { dtype, shape }),
        });
    }

    // Nodes.
    for (i, node) in g.nodes.iter().enumerate() {
        map_node(&mut cx, node, i).map_err(|e| match e {
            // re-tag bare errors with the node label
            OnnxError::Malformed { what, detail } => OnnxError::Malformed {
                what,
                detail: format!(
                    "node {} ({label}): {detail}",
                    node.op_type,
                    label = node.label(i)
                ),
            },
            other => other,
        })?;
    }

    // Outputs — materialize identity ops when the produced name differs.
    let mut outputs = Vec::new();
    let mut block_outs = Vec::new();
    for vi in &g.outputs {
        let sn = cx.san(&vi.name);
        let resolved =
            cx.resolve(&vi.name, "<graph output>")
                .map_err(|_| OnnxError::MissingInput {
                    name: vi.name.clone(),
                    node: "graph output".into(),
                })?;
        let shape = match &vi.shape {
            Some(dims) => dims
                .iter()
                .map(|d| match d {
                    Dim::Value(v) => Ok(*v),
                    _ => Err(OnnxError::Unsupported(format!(
                        "output '{}' has non-static dims",
                        vi.name
                    ))),
                })
                .collect::<Res<Vec<i64>>>()?,
            None => cx.info(&resolved)?.shape.clone(),
        };
        let dtype = feature_dtype(vi.elem_type, &vi.name)?;
        let produced = if resolved == sn {
            resolved
        } else {
            cx.e1(
                "identity",
                vec![("x".into(), bind(&resolved).1)],
                &sn,
                dtype,
                &shape,
            )
        };
        // trust the computed env shape over the declaration if absent
        let info = cx.info(&produced)?;
        outputs.push(Feature {
            name: produced.clone(),
            shape: info.shape,
            dtype: info.dtype,
            is_state: false,
        });
        block_outs.push(produced);
    }
    cx.b.outputs = block_outs;

    let wb = if cx.has_blob {
        Some(cx.wb.finish())
    } else {
        None
    };
    Ok(BuiltModel {
        block: cx.b,
        inputs,
        outputs,
        fn_inputs,
        weight_bin: wb,
        op_histogram: cx.hist,
    })
}

fn static_shape(vi: &crate::model::ValueInfo, what: &str) -> Res<Vec<i64>> {
    let dims = vi.shape.as_ref().ok_or_else(|| {
        OnnxError::Unsupported(format!("{what} '{}' has no declared shape", vi.name))
    })?;
    dims.iter()
        .map(|d| match d {
            Dim::Value(v) if *v >= 0 => Ok(*v),
            Dim::Value(v) => Err(OnnxError::Unsupported(format!(
                "{what} '{}': dim {v} (unspecified)",
                vi.name
            ))),
            Dim::Param(p) => Err(OnnxError::Unsupported(format!(
                "{what} '{}': symbolic dim '{p}' — static shapes only",
                vi.name
            ))),
            Dim::Unknown => Err(OnnxError::Unsupported(format!(
                "{what} '{}': unspecified dim",
                vi.name
            ))),
        })
        .collect()
}

/// ONNX element type → feature dtype (float lattice → fp16).
fn feature_dtype(t: i32, name: &str) -> Res<DType> {
    match t {
        elem::FLOAT | elem::FLOAT16 | elem::DOUBLE => Ok(DType::Fp16),
        elem::INT32
        | elem::INT64
        | elem::INT8
        | elem::UINT8
        | elem::INT16
        | elem::UINT16
        | elem::UINT32
        | elem::UINT64 => Ok(DType::Int32),
        elem::BOOL => Ok(DType::Bool),
        0 => Ok(DType::Fp16), // undeclared — assume float
        t => Err(OnnxError::Unsupported(format!(
            "feature '{name}': element type {t}"
        ))),
    }
}

/// Resolve all present inputs of a node (skipping "" placeholders).
fn ins(cx: &mut Ctx, node: &NodeProto, label: &str) -> Res<Vec<String>> {
    let mut v = Vec::with_capacity(node.inputs.len());
    for name in &node.inputs {
        if name.is_empty() {
            v.push(String::new());
        } else {
            v.push(cx.resolve(name, label)?);
        }
    }
    Ok(v)
}

fn map_node(cx: &mut Ctx, node: &NodeProto, idx: usize) -> Res<()> {
    let label = node.label(idx);
    let dom = if node.domain.is_empty() || node.domain == "ai.onnx" {
        ""
    } else {
        node.domain.as_str()
    };
    if dom == "ai.onnx.ml" {
        return classical::map_ml_node(cx, node, &label);
    }
    if !dom.is_empty() {
        return Err(OnnxError::UnknownOp {
            op: node.op_type.clone(),
            node: label,
            domain: node.domain.clone(),
        });
    }
    let xs = ins(cx, node, &label)?;
    let need = |n: usize| -> Res<()> {
        if xs.len() < n || xs[..n].iter().any(|s| s.is_empty()) {
            Err(OnnxError::Malformed {
                what: format!("op {}", node.op_type),
                detail: format!("node '{label}' needs {n} inputs, has {}", xs.len()),
            })
        } else {
            Ok(())
        }
    };
    let out = node.outputs.first().cloned().unwrap_or_default();
    let out_sn = cx.san(&out);

    match node.op_type.as_str() {
        // ---- passthrough ----
        "Identity" => {
            need(1)?;
            let t = cx.info(&xs[0])?;
            cx.env.insert(out_sn.clone(), t);
            cx.alias.insert(out_sn, xs[0].clone());
        }
        "Dropout" => {
            need(1)?;
            let t = cx.info(&xs[0])?;
            cx.env.insert(out_sn.clone(), t.clone());
            cx.alias.insert(out_sn, xs[0].clone());
            if let Some(mask) = node.outputs.get(1) {
                if !mask.is_empty() {
                    let msn = cx.san(mask);
                    let n = numel(&t.shape);
                    cx.k_bools_named(&msn, &vec![true; n], &t.shape)?;
                }
            }
        }
        // ---- unary fp16 ----
        "Relu" => cx.u(&xs, &out_sn, "relu")?,
        "Sigmoid" => cx.u(&xs, &out_sn, "sigmoid")?,
        "Tanh" => cx.u(&xs, &out_sn, "tanh")?,
        "Sqrt" => cx.u(&xs, &out_sn, "sqrt")?,
        "Exp" => cx.u(&xs, &out_sn, "exp")?,
        "Abs" => cx.u(&xs, &out_sn, "abs")?,
        "Erf" => cx.u(&xs, &out_sn, "erf")?,
        "Floor" => cx.u(&xs, &out_sn, "floor")?,
        "Ceil" => cx.u(&xs, &out_sn, "ceil")?,
        "Round" => cx.u(&xs, &out_sn, "round")?,
        "Sign" => cx.u(&xs, &out_sn, "sign")?,
        "Sin" => cx.u(&xs, &out_sn, "sin")?,
        "Cos" => cx.u(&xs, &out_sn, "cos")?,
        "Tan" => cx.u(&xs, &out_sn, "tan")?,
        "Sinh" => cx.u(&xs, &out_sn, "sinh")?,
        "Cosh" => cx.u(&xs, &out_sn, "cosh")?,
        "Asin" => cx.u(&xs, &out_sn, "asin")?,
        "Acos" => cx.u(&xs, &out_sn, "acos")?,
        "Atan" => cx.u(&xs, &out_sn, "atan")?,
        "Atanh" => cx.u(&xs, &out_sn, "atanh")?,
        "Softplus" => cx.u(&xs, &out_sn, "softplus")?,
        "Softsign" => cx.u(&xs, &out_sn, "softsign")?,
        "Not" => cx.u(&xs, &out_sn, "logical_not")?,
        "Log" => {
            need(1)?;
            let eps = cx.k_f16(0.0);
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "log",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("epsilon".into(), bind(&eps).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "Neg" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let m1 = cx.k_f16(-1.0);
            cx.binary("mul", &xs[0], &m1, &out_sn)?;
            let _ = t;
        }
        "Reciprocal" => {
            need(1)?;
            let eps = cx.k_f16(0.0);
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "inverse",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("epsilon".into(), bind(&eps).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "LeakyRelu" => {
            need(1)?;
            let a = cx.k_f16(node.attr_f("alpha", 0.01));
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "leaky_relu",
                vec![("x".into(), bind(&xs[0]).1), ("alpha".into(), bind(&a).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "Elu" => {
            need(1)?;
            let a = cx.k_f16(node.attr_f("alpha", 1.0));
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "elu",
                vec![("x".into(), bind(&xs[0]).1), ("alpha".into(), bind(&a).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "Selu" => {
            need(1)?;
            // gamma * elu(alpha)
            let alpha = node.attr_f("alpha", 1.673_263_2);
            let gamma = node.attr_f("gamma", 1.050_701);
            let a = cx.k_f16(alpha);
            let t = cx.info(&xs[0])?.clone();
            let e_name = cx.b.fresh("selu_e");
            let e = cx.e1(
                "elu",
                vec![("x".into(), bind(&xs[0]).1), ("alpha".into(), bind(&a).1)],
                &e_name,
                t.dtype,
                &t.shape,
            );
            let g = cx.k_f16(gamma);
            cx.binary("mul", &e, &g, &out_sn)?;
        }
        "Mish" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let sp_name = cx.b.fresh("mish_sp");
            let sp = cx.e1(
                "softplus",
                vec![("x".into(), bind(&xs[0]).1)],
                &sp_name,
                t.dtype,
                &t.shape,
            );
            let th_name = cx.b.fresh("mish_t");
            let th = cx.e1(
                "tanh",
                vec![("x".into(), bind(&sp).1)],
                &th_name,
                t.dtype,
                &t.shape,
            );
            cx.binary("mul", &xs[0], &th, &out_sn)?;
        }
        "HardSigmoid" => {
            need(1)?;
            let a = cx.k_f16(node.attr_f("alpha", 0.2));
            let bb = cx.k_f16(node.attr_f("beta", 0.5));
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "sigmoid_hard",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("alpha".into(), bind(&a).1),
                    ("beta".into(), bind(&bb).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "ThresholdedRelu" => {
            need(1)?;
            let a = cx.k_f16(node.attr_f("alpha", 1.0));
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "thresholded_relu",
                vec![("x".into(), bind(&xs[0]).1), ("alpha".into(), bind(&a).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "PRelu" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "prelu",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("alpha".into(), bind(&xs[1]).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "Gelu" => {
            need(1)?;
            let approx = node
                .attr_str("approximate")
                .map(|s| s == "tanh")
                .unwrap_or(false);
            let m = cx.k_str(if approx {
                "TANH_APPROXIMATION"
            } else {
                "EXACT"
            });
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "gelu",
                vec![("x".into(), bind(&xs[0]).1), ("mode".into(), bind(&m).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "Clip" => map_clip(cx, node, &xs, &out_sn)?,
        "Celu" => {
            need(1)?;
            // celu = max(0,x) + min(0, alpha*(exp(x/alpha)-1)) = elu with alpha
            let a = cx.k_f16(node.attr_f("alpha", 1.0));
            let t = cx.info(&xs[0])?.clone();
            cx.e1(
                "elu",
                vec![("x".into(), bind(&xs[0]).1), ("alpha".into(), bind(&a).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        // ---- binary ----
        "Add" => {
            cx.binary("add", &xs[0], &xs[1], &out_sn)?;
        }
        "Sub" => {
            cx.binary("sub", &xs[0], &xs[1], &out_sn)?;
        }
        "Mul" => {
            cx.binary("mul", &xs[0], &xs[1], &out_sn)?;
        }
        "Div" => {
            cx.binary("real_div", &xs[0], &xs[1], &out_sn)?;
        }
        "Pow" => {
            cx.binary("pow", &xs[0], &xs[1], &out_sn)?;
        }
        "And" => {
            cx.bool_bin("logical_and", &xs, &out_sn)?;
        }
        "Or" => {
            cx.bool_bin("logical_or", &xs, &out_sn)?;
        }
        "Xor" => {
            cx.bool_bin("logical_xor", &xs, &out_sn)?;
        }
        "Equal" => {
            cx.compare("equal", &xs[0], &xs[1], &out_sn)?;
        }
        "Greater" => {
            cx.compare("greater", &xs[0], &xs[1], &out_sn)?;
        }
        "Less" => {
            cx.compare("less", &xs[0], &xs[1], &out_sn)?;
        }
        "GreaterOrEqual" => {
            cx.compare("greater_equal", &xs[0], &xs[1], &out_sn)?;
        }
        "LessOrEqual" => {
            cx.compare("less_equal", &xs[0], &xs[1], &out_sn)?;
        }
        // ---- variadic ----
        "Sum" | "Mean" | "Max" | "Min" => {
            need(1)?;
            let mut acc = xs[0].clone();
            for x in &xs[1..] {
                if x.is_empty() {
                    continue;
                }
                let tmp = cx.b.fresh("var");
                match node.op_type.as_str() {
                    "Min" => {
                        acc = cx.binary("minimum", &acc, x, &tmp)?;
                    }
                    "Max" => {
                        acc = cx.binary("maximum", &acc, x, &tmp)?;
                    }
                    _ => {
                        acc = cx.binary("add", &acc, x, &tmp)?;
                    }
                }
            }
            if node.op_type == "Mean" {
                let n = xs.iter().filter(|s| !s.is_empty()).count();
                let inv = cx.k_f16(1.0 / n as f32);
                let mn = cx.b.fresh("mean");
                acc = cx.binary("mul", &acc, &inv, &mn)?;
            }
            let t = cx.info(&acc)?.clone();
            cx.e1(
                "identity",
                vec![("x".into(), bind(&acc).1)],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        // ---- softmax family ----
        // Opset<13 Softmax/LogSoftmax coerce input to 2-D around `axis`
        // ([prod(dims[:axis]), prod(dims[axis:])]) and normalize over
        // axis 1; opset≥13 normalizes along `axis` directly.
        "Softmax" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let def = if cx.opset >= 13 { -1 } else { 1 };
            let ax = norm_axis(node.attr_i("axis", def), t.shape.len(), "Softmax")?;
            if cx.opset < 13 {
                let g0: i64 = t.shape[..ax as usize].iter().product();
                let g1: i64 = t.shape[ax as usize..].iter().product();
                let r1 = cx.b.fresh("sm_r1");
                let flat = reshape_named(cx, &xs[0], &[g0, g1], &r1)?;
                let a = cx.k_i32s(1);
                let sm = cx.b.fresh("sm_2d");
                let sm = cx.e1(
                    "softmax",
                    vec![("x".into(), bind(&flat).1), ("axis".into(), bind(&a).1)],
                    &sm,
                    t.dtype,
                    &[g0, g1],
                );
                reshape_named(cx, &sm, &t.shape, &out_sn)?;
            } else {
                let a = cx.k_i32s(ax as i32);
                cx.e1(
                    "softmax",
                    vec![("x".into(), bind(&xs[0]).1), ("axis".into(), bind(&a).1)],
                    &out_sn,
                    t.dtype,
                    &t.shape,
                );
            }
        }
        "LogSoftmax" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let def = if cx.opset >= 13 { -1 } else { 1 };
            let ax = norm_axis(node.attr_i("axis", def), t.shape.len(), "LogSoftmax")?;
            let (x2, s2, a2): (String, Vec<i64>, i64) = if cx.opset < 13 {
                let g0: i64 = t.shape[..ax as usize].iter().product();
                let g1: i64 = t.shape[ax as usize..].iter().product();
                let r1 = cx.b.fresh("lsm_r1");
                let flat = reshape_named(cx, &xs[0], &[g0, g1], &r1)?;
                (flat, vec![g0, g1], 1)
            } else {
                (xs[0].clone(), t.shape.clone(), ax)
            };
            let lse_shape = reduce_shape(&s2, &[a2], true);
            let a = cx.k_i32v(&[a2 as i32]);
            let kd = cx.k_bool(true);
            let lse_name = cx.b.fresh("lse");
            let lse = cx.e1(
                "reduce_log_sum_exp",
                vec![
                    ("x".into(), bind(&x2).1),
                    ("axes".into(), bind(&a).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &lse_name,
                t.dtype,
                &lse_shape,
            );
            if cx.opset < 13 {
                let sn = cx.b.fresh("lsm_sub");
                let s = cx.binary("sub", &x2, &lse, &sn)?;
                reshape_named(cx, &s, &t.shape, &out_sn)?;
            } else {
                cx.binary("sub", &xs[0], &lse, &out_sn)?;
            }
        }
        "Hardmax" => {
            need(1)?;
            // y = one_hot(argmax) — on axis
            let t = cx.info(&xs[0])?.clone();
            let def = if cx.opset >= 13 { -1 } else { 1 };
            let ax = norm_axis(node.attr_i("axis", def), t.shape.len(), "Hardmax")?;
            let axis_dim = t.shape[ax as usize];
            // coerce to 2-D like ONNX does (flatten around axis)
            let g0: i64 = t.shape[..ax as usize].iter().product();
            let g1: i64 = t.shape[ax as usize..].iter().product();
            let r = cx.b.fresh("hm_r");
            let r = reshape_named(cx, &xs[0], &[g0, g1], &r)?;
            let a1 = cx.k_i32s(1);
            let kd = cx.k_bool(false);
            let am_name = cx.b.fresh("hm_am");
            let am = cx.e1(
                "reduce_argmax",
                vec![
                    ("x".into(), bind(&r).1),
                    ("axis".into(), bind(&a1).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &am_name,
                DType::Int32,
                &[g0],
            );
            let d = cx.k_i32s(axis_dim as i32);
            let axc = cx.k_i32s(1);
            let on = cx.k_f16(1.0);
            let off = cx.k_f16(0.0);
            let oh_name = cx.b.fresh("hm_oh");
            let oh = cx.e1(
                "one_hot",
                vec![
                    ("indices".into(), bind(&am).1),
                    ("one_hot_vector_size".into(), bind(&d).1),
                    ("axis".into(), bind(&axc).1),
                    ("on_value".into(), bind(&on).1),
                    ("off_value".into(), bind(&off).1),
                ],
                &oh_name,
                t.dtype,
                &[g0, g1],
            );
            reshape_named(cx, &oh, &t.shape, &out_sn)?;
        }
        // ---- matmul family ----
        "MatMul" => map_matmul(cx, node, &xs, &out_sn)?,
        "Gemm" => map_gemm(cx, node, &xs, &out_sn)?,
        // ---- conv/pool ----
        "Conv" => map_conv(cx, node, &xs, &out_sn)?,
        "ConvTranspose" => map_conv_transpose(cx, node, &xs, &out_sn)?,
        "MaxPool" => map_pool(cx, node, &xs, &out_sn, false)?,
        "AveragePool" => map_pool(cx, node, &xs, &out_sn, true)?,
        "GlobalAveragePool" => map_global_pool(cx, node, &xs, &out_sn, true)?,
        "GlobalMaxPool" => map_global_pool(cx, node, &xs, &out_sn, false)?,
        "LpPool" => {
            return Err(OnnxError::Unsupported(format!(
                "LpPool (p={}) has no MIL pool equivalent",
                node.attr_i("p", 2)
            )))
        }
        // ---- reshape family ----
        "Concat" => {
            need(1)?;
            let t0 = cx.info(&xs[0])?.clone();
            let ax = norm_axis(node.attr_i("axis", 0), t0.shape.len(), "Concat")?;
            let mut out_shape = t0.shape.clone();
            out_shape[ax as usize] = 0;
            let refs: Vec<String> = xs.iter().filter(|s| !s.is_empty()).cloned().collect();
            for x in &refs {
                let ti = cx.info(x)?;
                if ti.shape.len() != t0.shape.len() {
                    return Err(OnnxError::BadShape(format!(
                        "Concat '{label}': rank mismatch {:?} vs {:?}",
                        ti.shape, t0.shape
                    )));
                }
                out_shape[ax as usize] += ti.shape[ax as usize];
            }
            let a = cx.k_i32s(ax as i32);
            let il = cx.k_bool(false);
            let r: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
            cx.e1(
                "concat",
                vec![
                    ("values".into(), bind_many(&r)),
                    ("axis".into(), bind(&a).1),
                    ("interleave".into(), bind(&il).1),
                ],
                &out_sn,
                t0.dtype,
                &out_shape,
            );
        }
        "Reshape" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let spec = cx.static_i64(&node.inputs[1], &label)?;
            let allow0 = node.attr_i("allowzero", 0) != 0;
            let mut shape = Vec::with_capacity(spec.len());
            let mut infer: Option<usize> = None;
            for (i, &d) in spec.iter().enumerate() {
                match d {
                    -1 => {
                        if infer.is_some() {
                            return Err(OnnxError::BadShape(format!(
                                "Reshape '{label}': two -1 dims"
                            )));
                        }
                        infer = Some(i);
                        shape.push(1);
                    }
                    0 if !allow0 => {
                        if i >= t.shape.len() {
                            return Err(OnnxError::BadShape(format!(
                                "Reshape '{label}': 0 dim {i} beyond input rank"
                            )));
                        }
                        shape.push(t.shape[i]);
                    }
                    d if d > 0 => shape.push(d),
                    d => return Err(OnnxError::BadShape(format!("Reshape '{label}': dim {d}"))),
                }
            }
            let total = numel(&t.shape) as i64;
            if let Some(i) = infer {
                let known: i64 = shape.iter().product();
                if known == 0 || total % known != 0 {
                    return Err(OnnxError::BadShape(format!(
                        "Reshape '{label}': cannot infer -1 dim ({total} elems, known {known})"
                    )));
                }
                shape[i] = total / known;
            }
            if numel(&shape) as i64 != total {
                return Err(OnnxError::BadShape(format!(
                    "Reshape '{label}': {:?} holds {} elems, input has {total}",
                    shape,
                    numel(&shape)
                )));
            }
            let s = cx.k_i32v(&as_i32s(&shape)?);
            cx.e1(
                "reshape",
                vec![("x".into(), bind(&xs[0]).1), ("shape".into(), bind(&s).1)],
                &out_sn,
                t.dtype,
                &shape,
            );
        }
        "Transpose" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let rank = t.shape.len();
            let perm = {
                let p = node.attr_ints("perm");
                if p.is_empty() {
                    (0..rank as i64).rev().collect()
                } else {
                    p
                }
            };
            if perm.len() != rank {
                return Err(OnnxError::BadShape(format!(
                    "Transpose '{label}': perm {:?} vs rank {rank}",
                    perm
                )));
            }
            let mut out_shape = vec![0i64; rank];
            for (i, &p) in perm.iter().enumerate() {
                let pp = norm_axis(p, rank, "Transpose")?;
                out_shape[i] = t.shape[pp as usize];
            }
            let p = cx.k_i32v(&as_i32s(
                &perm
                    .iter()
                    .map(|&v| if v < 0 { v + rank as i64 } else { v })
                    .collect::<Vec<_>>(),
            )?);
            cx.e1(
                "transpose",
                vec![("x".into(), bind(&xs[0]).1), ("perm".into(), bind(&p).1)],
                &out_sn,
                t.dtype,
                &out_shape,
            );
        }
        "Flatten" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let rank = t.shape.len() as i64;
            let ax = {
                let a = node.attr_i("axis", 1);
                if a < 0 {
                    a + rank
                } else {
                    a
                }
            };
            if ax < 0 || ax > rank {
                return Err(OnnxError::BadShape(format!(
                    "Flatten '{label}': axis {ax} vs rank {rank}"
                )));
            }
            let d0: i64 = t.shape[..ax as usize].iter().product();
            let d1: i64 = t.shape[ax as usize..].iter().product();
            let shape = [d0, d1];
            let s = cx.k_i32v(&[d0 as i32, d1 as i32]);
            cx.e1(
                "reshape",
                vec![("x".into(), bind(&xs[0]).1), ("shape".into(), bind(&s).1)],
                &out_sn,
                t.dtype,
                &shape,
            );
        }
        "Squeeze" | "Unsqueeze" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let rank = t.shape.len() as i64;
            let axes = axes_of(cx, node, &label, 1, "axes")?;
            if node.op_type == "Squeeze" {
                let out_shape: Vec<i64> = match &axes {
                    Some(ax) => {
                        let mut keep = Vec::new();
                        let normed: Vec<i64> = ax
                            .iter()
                            .map(|&a| norm_axis(a, rank as usize, "Squeeze"))
                            .collect::<Res<Vec<_>>>()?;
                        for (i, &d) in t.shape.iter().enumerate() {
                            if normed.contains(&(i as i64)) {
                                if d != 1 {
                                    return Err(OnnxError::BadShape(format!(
                                        "Squeeze '{label}': dim {i} has size {d}, not 1"
                                    )));
                                }
                            } else {
                                keep.push(d);
                            }
                        }
                        keep
                    }
                    None => t.shape.iter().copied().filter(|&d| d != 1).collect(),
                };
                let s = cx.k_i32v(&as_i32s(&out_shape)?);
                cx.e1(
                    "reshape",
                    vec![("x".into(), bind(&xs[0]).1), ("shape".into(), bind(&s).1)],
                    &out_sn,
                    t.dtype,
                    &out_shape,
                );
            } else {
                let ax = axes.ok_or_else(|| OnnxError::Malformed {
                    what: "Unsqueeze".into(),
                    detail: format!("node '{label}': missing axes"),
                })?;
                let mut normed: Vec<usize> = ax
                    .iter()
                    .map(|&a| {
                        let r2 = (rank + ax.len() as i64) as usize;
                        Ok(norm_axis(a, r2, "Unsqueeze")? as usize)
                    })
                    .collect::<Res<Vec<_>>>()?;
                normed.sort_unstable();
                let mut out_shape = t.shape.clone();
                for &a in &normed {
                    out_shape.insert(a, 1);
                }
                let s = cx.k_i32v(&as_i32s(&out_shape)?);
                cx.e1(
                    "reshape",
                    vec![("x".into(), bind(&xs[0]).1), ("shape".into(), bind(&s).1)],
                    &out_sn,
                    t.dtype,
                    &out_shape,
                );
            }
        }
        "Expand" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let spec = cx.static_i64(&node.inputs[1], &label)?;
            let mut target: Vec<i64> = spec
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    if d == -1 {
                        // -1 copies the input dim at that position
                        t.shape
                            .get(i + t.shape.len().saturating_sub(spec.len()))
                            .copied()
                            .ok_or_else(|| {
                                OnnxError::BadShape(format!("Expand '{label}': -1 at dim {i}"))
                            })
                    } else if d > 0 {
                        Ok(d)
                    } else {
                        Err(OnnxError::BadShape(format!("Expand '{label}': dim {d}")))
                    }
                })
                .collect::<Res<Vec<_>>>()?;
            let out_shape = bcast(&t.shape, &target)?;
            target = out_shape.clone();
            let n = numel(&target);
            let ones = cx.k_f16t(&target, &vec![1.0; n]);
            cx.binary("mul", &xs[0], &ones, &out_sn)?;
        }
        "Tile" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let reps = cx.static_i64(&node.inputs[1], &label)?;
            if reps.len() != t.shape.len() {
                return Err(OnnxError::BadShape(format!(
                    "Tile '{label}': reps {:?} vs shape {:?}",
                    reps, t.shape
                )));
            }
            let out_shape: Vec<i64> = t.shape.iter().zip(&reps).map(|(&d, &r)| d * r).collect();
            let r = cx.k_i32v(&as_i32s(&reps)?);
            cx.e1(
                "tile",
                vec![("x".into(), bind(&xs[0]).1), ("reps".into(), bind(&r).1)],
                &out_sn,
                t.dtype,
                &out_shape,
            );
        }
        "Shape" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let rank = t.shape.len() as i64;
            let mut start = node.attr_i("start", 0);
            let mut end = node.attr_i("end", rank);
            if start < 0 {
                start += rank;
            }
            if end < 0 {
                end += rank;
            }
            let dims: Vec<i64> = t.shape
                [(start.max(0).min(rank)) as usize..(end.max(start).min(rank)) as usize]
                .to_vec();
            cx.k_i32t_named(&out_sn, &[dims.len() as i64], &as_i32s(&dims)?);
        }
        "Size" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let n = numel(&t.shape) as i32;
            cx.k_i32t_named(&out_sn, &[], &[n]);
        }
        "ConstantOfShape" => {
            need(1)?;
            let spec = cx.static_i64(&node.inputs[0], &label)?;
            let shape: Vec<i64> = spec
                .iter()
                .map(|&d| {
                    if d >= 0 {
                        Ok(d)
                    } else {
                        Err(OnnxError::BadShape(format!(
                            "ConstantOfShape '{label}': dim {d}"
                        )))
                    }
                })
                .collect::<Res<Vec<_>>>()?;
            let n = numel(&shape);
            let (val, dt) = match node.attr_t("value") {
                Some(tp) => match tp.data_type {
                    elem::FLOAT | elem::FLOAT16 | elem::DOUBLE => {
                        let v = tp.as_f32()?.into_iter().next().unwrap_or(0.0);
                        (Value::f16s(&shape, &vec![v; n]), DType::Fp16)
                    }
                    elem::INT32 | elem::INT64 => {
                        let v = tp.as_i64()?.into_iter().next().unwrap_or(0) as i32;
                        (
                            Value::Imm(
                                ValueType::Tensor(TensorType {
                                    dtype: DType::Int32,
                                    shape: shape.clone(),
                                }),
                                Immediate::Ints(vec![v; n]),
                            ),
                            DType::Int32,
                        )
                    }
                    t => {
                        return Err(OnnxError::Unsupported(format!(
                            "ConstantOfShape '{label}': value dtype {t}"
                        )))
                    }
                },
                None => (Value::f16s(&shape, &vec![0.0; n]), DType::Fp16),
            };
            let vt = cx.tt(dt, &shape);
            cx.b.op(
                "const",
                vec![],
                vec![(&out_sn, vt)],
                vec![("val".into(), val)],
            );
            cx.env.insert(out_sn.clone(), TInfo { shape, dtype: dt });
            *cx.hist.entry("const".to_string()).or_insert(0) += 1;
        }
        "Constant" => map_constant(cx, node, &out_sn)?,
        "Gather" => map_gather(cx, node, &xs, &out_sn, &label)?,
        "GatherElements" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let it = cx.info(&xs[1])?.clone();
            let ax = norm_axis(node.attr_i("axis", 0), t.shape.len(), "GatherElements")?;
            let a = cx.k_i32s(ax as i32);
            let vi = cx.k_bool(false);
            cx.e1(
                "gather_along_axis",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("indices".into(), bind(&xs[1]).1),
                    ("axis".into(), bind(&a).1),
                    ("validate_indices".into(), bind(&vi).1),
                ],
                &out_sn,
                t.dtype,
                &it.shape,
            );
        }
        "GatherND" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let it = cx.info(&xs[1])?.clone();
            let bd = node.attr_i("batch_dims", 0);
            if bd != 0 {
                return Err(OnnxError::Unsupported(format!(
                    "GatherND '{label}': batch_dims={bd} unsupported"
                )));
            }
            let n_ix = *it.shape.last().unwrap_or(&0) as usize;
            let mut out_shape: Vec<i64> = it.shape[..it.shape.len() - 1].to_vec();
            out_shape.extend_from_slice(&t.shape[n_ix..]);
            let vi = cx.k_bool(false);
            cx.e1(
                "gather_nd",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("indices".into(), bind(&xs[1]).1),
                    ("validate_indices".into(), bind(&vi).1),
                ],
                &out_sn,
                t.dtype,
                &out_shape,
            );
        }
        "Slice" => map_slice(cx, node, &xs, &out_sn, &label)?,
        "Split" => map_split(cx, node, &xs, &label)?,
        "Cast" => map_cast(cx, node, &xs, &out_sn)?,
        "Where" => {
            need(3)?;
            let c = cx.info(&xs[0])?.clone();
            let t = cx.info(&xs[1])?.clone();
            let b = cx.info(&xs[2])?.clone();
            if c.dtype != DType::Bool {
                return Err(OnnxError::BadShape(format!(
                    "Where '{label}': cond must be bool, got {:?}",
                    c.dtype
                )));
            }
            let shape = bcast(&bcast(&c.shape, &t.shape)?, &b.shape)?;
            cx.e1(
                "select",
                vec![
                    ("cond".into(), bind(&xs[0]).1),
                    ("a".into(), bind(&xs[1]).1),
                    ("b".into(), bind(&xs[2]).1),
                ],
                &out_sn,
                t.dtype,
                &shape,
            );
        }
        "Pad" => map_pad(cx, node, &xs, &out_sn, &label)?,
        "BatchNormalization" => {
            need(5)?;
            let t = cx.info(&xs[0])?.clone();
            let eps = cx.k_f16(node.attr_f("epsilon", 1e-5));
            cx.e1(
                "batch_norm",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("mean".into(), bind(&xs[3]).1),
                    ("variance".into(), bind(&xs[4]).1),
                    ("epsilon".into(), bind(&eps).1),
                    ("gamma".into(), bind(&xs[1]).1),
                    ("beta".into(), bind(&xs[2]).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
            // extra outputs (running stats) — inference graphs don't use them
            for extra in node.outputs.iter().skip(1) {
                if !extra.is_empty() {
                    let esn = cx.san(extra);
                    cx.alias.insert(esn.clone(), xs[0].clone());
                    cx.env.insert(esn, t.clone());
                }
            }
        }
        "InstanceNormalization" => {
            need(3)?;
            let t = cx.info(&xs[0])?.clone();
            let eps = cx.k_f16(node.attr_f("epsilon", 1e-5));
            cx.e1(
                "instance_norm",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("epsilon".into(), bind(&eps).1),
                    ("gamma".into(), bind(&xs[1]).1),
                    ("beta".into(), bind(&xs[2]).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "LayerNormalization" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let rank = t.shape.len();
            let ax = norm_axis(node.attr_i("axis", -1), rank, "LayerNormalization")?;
            let axes: Vec<i32> = (ax..rank as i64).map(|a| a as i32).collect();
            let eps = node.attr_f("epsilon", 1e-5);
            let a = cx.k_i32v(&axes);
            let e = cx.k_f16(eps);
            let mut inputs = vec![
                ("x".into(), bind(&xs[0]).1),
                ("axes".into(), bind(&a).1),
                ("epsilon".into(), bind(&e).1),
            ];
            if xs.len() > 1 && !xs[1].is_empty() {
                inputs.push(("gamma".into(), bind(&xs[1]).1));
            }
            if xs.len() > 2 && !xs[2].is_empty() {
                inputs.push(("beta".into(), bind(&xs[2]).1));
            }
            cx.e1("layer_norm", inputs, &out_sn, t.dtype, &t.shape);
            // outputs 1,2 (mean, invstd) unsupported — alias to input
            for extra in node.outputs.iter().skip(1) {
                if !extra.is_empty() {
                    let esn = cx.san(extra);
                    cx.alias.insert(esn.clone(), xs[0].clone());
                    cx.env.insert(esn, t.clone());
                }
            }
        }
        "Upsample" | "Resize" => map_resize(cx, node, &xs, &out_sn, &label)?,
        "ArgMax" | "ArgMin" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let ax = norm_axis(node.attr_i("axis", 0), t.shape.len(), &node.op_type)?;
            let keep = node.attr_i("keepdims", 1) != 0;
            let out_shape = reduce_shape(&t.shape, &[ax], keep);
            let a = cx.k_i32s(ax as i32);
            let kd = cx.k_bool(keep);
            cx.e1(
                if node.op_type == "ArgMax" {
                    "reduce_argmax"
                } else {
                    "reduce_argmin"
                },
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("axis".into(), bind(&a).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &out_sn,
                DType::Int32,
                &out_shape,
            );
        }
        "CumSum" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let axv = cx.static_i64(&node.inputs[1], &label)?;
            let ax = norm_axis(*axv.first().unwrap_or(&0), t.shape.len(), "CumSum")?;
            let a = cx.k_i32s(ax as i32);
            let ex = cx.k_bool(node.attr_i("exclusive", 0) != 0);
            let rv = cx.k_bool(node.attr_i("reverse", 0) != 0);
            cx.e1(
                "cumsum",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("axis".into(), bind(&a).1),
                    ("exclusive".into(), bind(&ex).1),
                    ("reverse".into(), bind(&rv).1),
                ],
                &out_sn,
                t.dtype,
                &t.shape,
            );
        }
        "TopK" => {
            need(2)?;
            let t = cx.info(&xs[0])?.clone();
            let kv = cx.static_i64(&node.inputs[1], &label)?;
            let k = *kv.first().unwrap_or(&0);
            let ax = norm_axis(node.attr_i("axis", -1), t.shape.len(), "TopK")?;
            let largest = node.attr_i("largest", 1) != 0;
            let mut out_shape = t.shape.clone();
            out_shape[ax as usize] = k;
            let kc = cx.k_i32s(k as i32);
            let a = cx.k_i32s(ax as i32);
            let asc = cx.k_bool(!largest);
            let (vout, iout) = (
                out_sn.clone(),
                cx.san(node.outputs.get(1).map(|s| s.as_str()).unwrap_or("")),
            );
            let names = cx.emit(
                "topk",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("k".into(), bind(&kc).1),
                    ("axis".into(), bind(&a).1),
                    ("ascending".into(), bind(&asc).1),
                ],
                vec![
                    (&vout, t.dtype, &out_shape),
                    (&iout, DType::Int32, &out_shape),
                ],
            );
            let _ = names;
        }
        "OneHot" => {
            need(3)?;
            let it = cx.info(&xs[0])?.clone();
            let depth = *cx
                .static_i64(&node.inputs[1], &label)?
                .first()
                .unwrap_or(&0);
            let vals = cx.static_f32(&node.inputs[2], &label)?;
            let off = vals.first().copied().unwrap_or(0.0);
            let on = vals.get(1).copied().unwrap_or(1.0);
            let ax = node.attr_i("axis", -1);
            let rank = it.shape.len() + 1;
            let axn = norm_axis(ax, rank, "OneHot")?;
            let mut out_shape = it.shape.clone();
            out_shape.insert(axn as usize, depth);
            let d = cx.k_i32s(depth as i32);
            let a = cx.k_i32s(axn as i32);
            let onc = cx.k_f16(on);
            let offc = cx.k_f16(off);
            cx.e1(
                "one_hot",
                vec![
                    ("indices".into(), bind(&xs[0]).1),
                    ("one_hot_vector_size".into(), bind(&d).1),
                    ("axis".into(), bind(&a).1),
                    ("on_value".into(), bind(&onc).1),
                    ("off_value".into(), bind(&offc).1),
                ],
                &out_sn,
                DType::Fp16,
                &out_shape,
            );
        }
        "DepthToSpace" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let bs = node.attr_i("blocksize", 1);
            if t.shape.len() != 4 {
                return Err(OnnxError::Unsupported(format!(
                    "DepthToSpace '{label}': rank {} ≠ 4",
                    t.shape.len()
                )));
            }
            let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
            if c % (bs * bs) != 0 {
                return Err(OnnxError::BadShape(format!(
                    "DepthToSpace '{label}': C={c} not divisible by blocksize²"
                )));
            }
            let out_shape = [n, c / (bs * bs), h * bs, w * bs];
            let b = cx.k_i32s(bs as i32);
            cx.e1(
                "depth_to_space",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("block_size".into(), bind(&b).1),
                ],
                &out_sn,
                t.dtype,
                &out_shape,
            );
        }
        "SpaceToDepth" => {
            need(1)?;
            let t = cx.info(&xs[0])?.clone();
            let bs = node.attr_i("blocksize", 1);
            if t.shape.len() != 4 {
                return Err(OnnxError::Unsupported(format!(
                    "SpaceToDepth '{label}': rank {} ≠ 4",
                    t.shape.len()
                )));
            }
            let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
            if h % bs != 0 || w % bs != 0 {
                return Err(OnnxError::BadShape(format!(
                    "SpaceToDepth '{label}': spatial {h}x{w} not divisible by {bs}"
                )));
            }
            let out_shape = [n, c * bs * bs, h / bs, w / bs];
            let b = cx.k_i32s(bs as i32);
            cx.e1(
                "space_to_depth",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("block_size".into(), bind(&b).1),
                ],
                &out_sn,
                t.dtype,
                &out_shape,
            );
        }
        "ScatterElements" | "Scatter" => {
            need(3)?;
            let t = cx.info(&xs[0])?.clone();
            let ax = norm_axis(node.attr_i("axis", 0), t.shape.len(), "ScatterElements")?;
            let mode = match node.attr_str("reduction").as_deref() {
                Some("add") => "add",
                Some("mul") => "mul",
                Some("max") => "max",
                Some("min") => "min",
                _ => "update",
            };
            let a = cx.k_i32s(ax as i32);
            let m = cx.k_str(mode);
            let vi = cx.k_bool(false);
            let ushape = cx.info(&xs[2])?.shape.clone();
            cx.e1(
                "scatter_along_axis",
                vec![
                    ("data".into(), bind(&xs[0]).1),
                    ("indices".into(), bind(&xs[1]).1),
                    ("updates".into(), bind(&xs[2]).1),
                    ("axis".into(), bind(&a).1),
                    ("mode".into(), bind(&m).1),
                    ("validate_indices".into(), bind(&vi).1),
                ],
                &out_sn,
                t.dtype,
                &ushape_scatter(&t.shape, &ushape),
            );
        }
        "ScatterND" => {
            need(3)?;
            let t = cx.info(&xs[0])?.clone();
            let mode = match node.attr_str("reduction").as_deref() {
                Some("add") => "add",
                Some("mul") => "mul",
                _ => "update",
            };
            let m = cx.k_str(mode);
            let vi = cx.k_bool(false);
            let dshape = t.shape.clone();
            cx.e1(
                "scatter_nd",
                vec![
                    ("data".into(), bind(&xs[0]).1),
                    ("indices".into(), bind(&xs[1]).1),
                    ("updates".into(), bind(&xs[2]).1),
                    ("mode".into(), bind(&m).1),
                    ("validate_indices".into(), bind(&vi).1),
                ],
                &out_sn,
                t.dtype,
                &dshape,
            );
        }
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL1"
        | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" => {
            need(1)?;
            let ty = match node.op_type.as_str() {
                "ReduceSum" => "reduce_sum",
                "ReduceMean" => "reduce_mean",
                "ReduceMax" => "reduce_max",
                "ReduceMin" => "reduce_min",
                "ReduceProd" => "reduce_prod",
                "ReduceL1" => "reduce_l1_norm",
                "ReduceL2" => "reduce_l2_norm",
                "ReduceLogSum" => "reduce_log_sum",
                "ReduceLogSumExp" => "reduce_log_sum_exp",
                "ReduceSumSquare" => "reduce_sum_square",
                _ => unreachable!(),
            };
            let keep = node.attr_i("keepdims", 1) != 0;
            let axes = axes_of(cx, node, &label, 1, "axes")?;
            if axes.is_none() && node.attr_i("noop_with_empty_axes", 0) != 0 && xs.len() > 1 {
                let t = cx.info(&xs[0])?.clone();
                cx.alias.insert(out_sn.clone(), xs[0].clone());
                cx.env.insert(out_sn, t);
            } else {
                let ax = axes.unwrap_or_default();
                let dt = cx.info(&xs[0])?.dtype;
                cx.reduce(ty, &xs[0], &ax, keep, &out_sn, dt)?;
            }
        }
        "Einsum" => {
            need(1)?;
            let eq = node
                .attr_str("equation")
                .ok_or_else(|| OnnxError::Malformed {
                    what: "Einsum".into(),
                    detail: format!("node '{label}': missing equation"),
                })?;
            // shape inference for einsum is non-trivial — require value_info
            return Err(OnnxError::Unsupported(format!(
                "Einsum '{label}' ({eq}): use value_info-free graphs — unmapped"
            )));
        }
        other => {
            return Err(OnnxError::UnknownOp {
                op: other.to_string(),
                node: label,
                domain: node.domain.clone(),
            })
        }
    }
    Ok(())
}

fn ushape_scatter(data: &[i64], _updates: &[i64]) -> Vec<i64> {
    data.to_vec()
}

/// `axes` from attr (opset≤12 style) or input `idx` (opset≥13).
/// `None` = not present anywhere.
fn axes_of(
    cx: &mut Ctx,
    node: &NodeProto,
    label: &str,
    idx: usize,
    attr: &str,
) -> Res<Option<Vec<i64>>> {
    if let Some(a) = node.attrs.get(attr) {
        if !a.ints.is_empty() {
            return Ok(Some(a.ints.clone()));
        }
    }
    if let Some(name) = node.inputs.get(idx) {
        if !name.is_empty() {
            return Ok(Some(cx.static_i64(name, label)?));
        }
    }
    Ok(None)
}

impl<'a> Ctx<'a> {
    fn u(&mut self, xs: &[String], out: &str, ty: &str) -> Res<()> {
        self.unary(ty, &xs[0], out)?;
        Ok(())
    }
    fn bool_bin(&mut self, ty: &str, xs: &[String], out: &str) -> Res<()> {
        let (a, b) = (self.info(&xs[0])?, self.info(&xs[1])?);
        let shape = bcast(&a.shape, &b.shape)?;
        self.e1(
            ty,
            vec![("x".into(), bind(&xs[0]).1), ("y".into(), bind(&xs[1]).1)],
            out,
            DType::Bool,
            &shape,
        );
        Ok(())
    }
    fn k_bools_named(&mut self, name: &str, vs: &[bool], shape: &[i64]) -> Res<()> {
        let vt = self.tt(DType::Bool, shape);
        self.b.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::bools(vs))],
        );
        self.env.insert(
            name.to_string(),
            TInfo {
                shape: shape.to_vec(),
                dtype: DType::Bool,
            },
        );
        Ok(())
    }
    fn k_i32t_named(&mut self, name: &str, shape: &[i64], vs: &[i32]) {
        let vt = self.tt(DType::Int32, shape);
        self.b.op(
            "const",
            vec![],
            vec![(name, vt)],
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
        );
        self.env.insert(
            name.to_string(),
            TInfo {
                shape: shape.to_vec(),
                dtype: DType::Int32,
            },
        );
    }
}

/// reshape helper that also records env.
fn reshape_named(cx: &mut Ctx, x: &str, shape: &[i64], out: &str) -> Res<String> {
    let dt = cx.info(x)?.dtype;
    let s = cx.k_i32v(&as_i32s(shape)?);
    Ok(cx.e1(
        "reshape",
        vec![("x".into(), bind(x).1), ("shape".into(), bind(&s).1)],
        out,
        dt,
        shape,
    ))
}

// ---------- larger mappers ----------

fn map_clip(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    // opset<11: attrs min/max; ≥11: optional inputs 1,2 ("" = absent)
    let mut lo = node.attrs.get("min").and_then(|a| a.f);
    let mut hi = node.attrs.get("max").and_then(|a| a.f);
    if xs.len() > 1 && !xs[1].is_empty() {
        if let Some(tp) = cx.init_of(&node.inputs[1]) {
            lo = tp.as_f32()?.first().copied();
        }
    }
    if xs.len() > 2 && !xs[2].is_empty() {
        if let Some(tp) = cx.init_of(&node.inputs[2]) {
            hi = tp.as_f32()?.first().copied();
        }
    }
    if lo.is_none() && hi.is_none() {
        // no-op clip = identity
        cx.alias.insert(out.to_string(), xs[0].clone());
        cx.env.insert(out.to_string(), t);
        return Ok(());
    }
    let a = cx.k_f16(lo.unwrap_or(-65504.0));
    let b = cx.k_f16(hi.unwrap_or(65504.0));
    cx.e1(
        "clip",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("alpha".into(), bind(&a).1),
            ("beta".into(), bind(&b).1),
        ],
        out,
        t.dtype,
        &t.shape,
    );
    Ok(())
}

fn map_matmul(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let label = node.label(0);
    let mut a = cx.info(&xs[0])?.clone();
    let mut b = cx.info(&xs[1])?.clone();
    let mut x = xs[0].clone();
    let mut y = xs[1].clone();
    let mut squeeze_front = false;
    let mut squeeze_back = false;
    if a.shape.len() == 1 {
        let nm = cx.b.fresh("mm_a");
        x = reshape_named(cx, &x, &[1, a.shape[0]], &nm)?;
        a.shape = vec![1, a.shape[0]];
        squeeze_front = true;
    }
    if b.shape.len() == 1 {
        let nm = cx.b.fresh("mm_b");
        y = reshape_named(cx, &y, &[b.shape[0], 1], &nm)?;
        b.shape = vec![b.shape[0], 1];
        squeeze_back = true;
    }
    if a.shape.len() < 2 || b.shape.len() < 2 {
        return Err(OnnxError::BadShape(format!("MatMul '{label}': rank < 2")));
    }
    let m = a.shape[a.shape.len() - 2];
    let ka = a.shape[a.shape.len() - 1];
    let kb = b.shape[b.shape.len() - 2];
    let n = b.shape[b.shape.len() - 1];
    if ka != kb {
        return Err(OnnxError::BadShape(format!(
            "MatMul '{label}': inner dims {ka} vs {kb}"
        )));
    }
    let batch = bcast(&a.shape[..a.shape.len() - 2], &b.shape[..b.shape.len() - 2])?;
    let mut out_shape = batch;
    out_shape.push(m);
    out_shape.push(n);
    let tx = cx.k_bool(false);
    let ty = cx.k_bool(false);
    let mm_name = cx.b.fresh("mm");
    let mm = cx.e1(
        "matmul",
        vec![
            ("x".into(), bind(&x).1),
            ("y".into(), bind(&y).1),
            ("transpose_x".into(), bind(&tx).1),
            ("transpose_y".into(), bind(&ty).1),
        ],
        &mm_name,
        a.dtype,
        &out_shape,
    );
    if squeeze_front {
        out_shape.remove(out_shape.len() - 2);
    }
    if squeeze_back {
        out_shape.pop();
    }
    reshape_named(&mut *cx, &mm, &out_shape, out)?;
    Ok(())
}

fn map_gemm(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let label = node.label(0);
    let a = cx.info(&xs[0])?.clone();
    let b = cx.info(&xs[1])?.clone();
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OnnxError::Unsupported(format!(
            "Gemm '{label}': A/B must be 2-D"
        )));
    }
    let ta = node.attr_i("transA", 0) != 0;
    let tb = node.attr_i("transB", 0) != 0;
    let (m, k1) = if ta {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    let (k2, n) = if tb {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };
    if k1 != k2 {
        return Err(OnnxError::BadShape(format!(
            "Gemm '{label}': K {k1} vs {k2}"
        )));
    }
    let out_shape = [m, n];
    let tx = cx.k_bool(ta);
    let ty = cx.k_bool(tb);
    let mm_name = cx.b.fresh("gemm_mm");
    let mut acc = cx.e1(
        "matmul",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("y".into(), bind(&xs[1]).1),
            ("transpose_x".into(), bind(&tx).1),
            ("transpose_y".into(), bind(&ty).1),
        ],
        &mm_name,
        a.dtype,
        &out_shape,
    );
    let alpha = node.attr_f("alpha", 1.0);
    if alpha != 1.0 {
        let c = cx.k_f16(alpha);
        let nm = cx.b.fresh("gemm_a");
        acc = cx.binary("mul", &acc, &c, &nm)?;
    }
    if xs.len() > 2 && !xs[2].is_empty() {
        let beta = node.attr_f("beta", 1.0);
        let mut cval = xs[2].clone();
        if beta != 1.0 {
            let bc = cx.k_f16(beta);
            let nm = cx.b.fresh("gemm_bc");
            cval = cx.binary("mul", &cval, &bc, &nm)?;
        }
        let nm = cx.b.fresh("gemm_c");
        acc = cx.binary("add", &acc, &cval, &nm)?;
    }
    let t = cx.info(&acc)?.clone();
    cx.e1(
        "identity",
        vec![("x".into(), bind(&acc).1)],
        out,
        t.dtype,
        &t.shape,
    );
    Ok(())
}

/// conv output spatial size for NOTSET/VALID/custom pads.
fn conv_out_dim(i: i64, pad_b: i64, pad_e: i64, k: i64, dil: i64, s: i64) -> i64 {
    (i + pad_b + pad_e - dil * (k - 1) - 1).div_euclid(s) + 1
}

fn map_conv(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let label = node.label(0);
    let t = cx.info(&xs[0])?.clone();
    let w = cx.info(&xs[1])?.clone();
    let spatial = t.shape.len() - 2;
    if w.shape.len() != spatial + 2 {
        return Err(OnnxError::BadShape(format!(
            "Conv '{label}': weight rank {} vs input rank {}",
            w.shape.len(),
            t.shape.len()
        )));
    }
    let auto = node.attr_str("auto_pad").unwrap_or_else(|| "NOTSET".into());
    let strides = node.attr_ints("strides");
    let strides: Vec<i64> = if strides.is_empty() {
        vec![1; spatial]
    } else {
        strides
    };
    let dils = node.attr_ints("dilations");
    let dils: Vec<i64> = if dils.is_empty() {
        vec![1; spatial]
    } else {
        dils
    };
    let groups = node.attr_i("group", 1) as i32;
    let kshape: Vec<i64> = w.shape[2..].to_vec();
    let (pad_type, pads, out_spatial) = match auto.as_str() {
        "VALID" => (
            "valid".to_string(),
            vec![0i64; spatial * 2],
            (0..spatial)
                .map(|i| conv_out_dim(t.shape[2 + i], 0, 0, kshape[i], dils[i], strides[i]))
                .collect::<Vec<i64>>(),
        ),
        "SAME_UPPER" | "SAME_LOWER" => (
            if auto == "SAME_LOWER" {
                "same_lower"
            } else {
                "same"
            }
            .to_string(),
            vec![0i64; spatial * 2],
            (0..spatial)
                .map(|i| {
                    t.shape[2 + i].div_euclid(strides[i])
                        + i64::from(t.shape[2 + i] % strides[i] != 0)
                })
                .collect(),
        ),
        _ => {
            let p = node.attr_ints("pads");
            let pads = if p.is_empty() {
                vec![0i64; spatial * 2]
            } else {
                if p.len() != spatial * 2 {
                    return Err(OnnxError::BadShape(format!(
                        "Conv '{label}': pads has {} entries, expected {}",
                        p.len(),
                        spatial * 2
                    )));
                }
                p
            };
            (
                "custom".to_string(),
                pads.clone(),
                (0..spatial)
                    .map(|i| {
                        conv_out_dim(
                            t.shape[2 + i],
                            pads[i],
                            pads[i + spatial],
                            kshape[i],
                            dils[i],
                            strides[i],
                        )
                    })
                    .collect(),
            )
        }
    };
    let mut out_shape = vec![t.shape[0], w.shape[0]];
    out_shape.extend(out_spatial);
    for &d in &out_shape {
        if d <= 0 {
            return Err(OnnxError::BadShape(format!(
                "Conv '{label}': non-positive output dim in {out_shape:?}"
            )));
        }
    }
    let st = cx.k_i32v(&as_i32s(&strides)?);
    let pt = cx.k_str(&pad_type);
    let pd = cx.k_i32v(&as_i32s(&pads)?);
    let dl = cx.k_i32v(&as_i32s(&dils)?);
    let g = cx.k_i32s(groups);
    let mut inputs = vec![
        ("x".into(), bind(&xs[0]).1),
        ("weight".into(), bind(&xs[1]).1),
        ("strides".into(), bind(&st).1),
        ("pad_type".into(), bind(&pt).1),
        ("pad".into(), bind(&pd).1),
        ("dilations".into(), bind(&dl).1),
        ("groups".into(), bind(&g).1),
    ];
    if xs.len() > 2 && !xs[2].is_empty() {
        inputs.push(("bias".into(), bind(&xs[2]).1));
    }
    cx.e1("conv", inputs, out, t.dtype, &out_shape);
    Ok(())
}

fn map_conv_transpose(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let label = node.label(0);
    let t = cx.info(&xs[0])?.clone();
    let w = cx.info(&xs[1])?.clone();
    let spatial = t.shape.len() - 2;
    let auto = node.attr_str("auto_pad").unwrap_or_else(|| "NOTSET".into());
    let strides = {
        let s = node.attr_ints("strides");
        if s.is_empty() {
            vec![1; spatial]
        } else {
            s
        }
    };
    let dils = {
        let d = node.attr_ints("dilations");
        if d.is_empty() {
            vec![1; spatial]
        } else {
            d
        }
    };
    let groups = node.attr_i("group", 1) as i32;
    let kshape: Vec<i64> = w.shape[2..].to_vec();
    let opad = node.attr_ints("output_padding");
    let opad = if opad.is_empty() {
        vec![0; spatial]
    } else {
        opad
    };
    let p = node.attr_ints("pads");
    let pads = if p.is_empty() {
        vec![0i64; spatial * 2]
    } else {
        p
    };
    if matches!(auto.as_str(), "SAME_UPPER" | "SAME_LOWER") {
        return Err(OnnxError::Unsupported(format!(
            "ConvTranspose '{label}': auto_pad=SAME unsupported"
        )));
    }
    let mut out_shape = vec![t.shape[0], w.shape[1] * groups as i64];
    let mut full_out = Vec::with_capacity(spatial);
    for i in 0..spatial {
        let d = strides[i] * (t.shape[2 + i] - 1) + opad[i] + (kshape[i] - 1) * dils[i] + 1
            - pads[i]
            - pads[i + spatial];
        if d <= 0 {
            return Err(OnnxError::BadShape(format!(
                "ConvTranspose '{label}': non-positive output dim"
            )));
        }
        full_out.push(d);
    }
    out_shape.extend(full_out.iter().copied());
    let st = cx.k_i32v(&as_i32s(&strides)?);
    let pt = cx.k_str("custom");
    let pd = cx.k_i32v(&as_i32s(&pads)?);
    let dl = cx.k_i32v(&as_i32s(&dils)?);
    let g = cx.k_i32s(groups);
    let os = cx.k_i32v(&as_i32s(&out_shape)?);
    let mut inputs = vec![
        ("x".into(), bind(&xs[0]).1),
        ("weight".into(), bind(&xs[1]).1),
        ("strides".into(), bind(&st).1),
        ("pad_type".into(), bind(&pt).1),
        ("pad".into(), bind(&pd).1),
        ("dilations".into(), bind(&dl).1),
        ("groups".into(), bind(&g).1),
        ("output_shape".into(), bind(&os).1),
    ];
    if xs.len() > 2 && !xs[2].is_empty() {
        inputs.push(("bias".into(), bind(&xs[2]).1));
    }
    cx.e1("conv_transpose", inputs, out, t.dtype, &out_shape);
    Ok(())
}

fn pool_out_dim(i: i64, pad_b: i64, pad_e: i64, k: i64, s: i64, ceil: bool) -> i64 {
    let num = i + pad_b + pad_e - k;
    let mut out = if ceil {
        num.div_euclid(s) + i64::from(num.rem_euclid(s) != 0) + 1
    } else {
        num.div_euclid(s) + 1
    };
    if ceil && (out - 1) * s >= i + pad_b {
        out -= 1; // ONNX ceil_mode: window must start inside input+pad_begin
    }
    out
}

fn map_pool(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, avg: bool) -> Res<()> {
    let label = node.label(0);
    let t = cx.info(&xs[0])?.clone();
    let spatial = t.shape.len() - 2;
    let ks = node.attr_ints("kernel_shape");
    if ks.len() != spatial {
        return Err(OnnxError::BadShape(format!(
            "pool '{label}': kernel_shape {:?} vs spatial {spatial}",
            ks
        )));
    }
    let dils = node.attr_ints("dilations");
    if dils.iter().any(|&d| d != 1) {
        return Err(OnnxError::Unsupported(format!(
            "pool '{label}': dilations unsupported"
        )));
    }
    let strides = node.attr_ints("strides");
    let strides: Vec<i64> = if strides.is_empty() {
        vec![1; spatial]
    } else {
        strides
    };
    let auto = node.attr_str("auto_pad").unwrap_or_else(|| "NOTSET".into());
    let ceil = node.attr_i("ceil_mode", 0) != 0;
    let (pad_type, pads) = match auto.as_str() {
        "VALID" => ("valid", vec![0i64; spatial * 2]),
        "SAME_UPPER" => ("same", vec![0i64; spatial * 2]),
        "SAME_LOWER" => ("same_lower", vec![0i64; spatial * 2]),
        _ => {
            let p = node.attr_ints("pads");
            if p.is_empty() {
                ("custom", vec![0i64; spatial * 2])
            } else {
                ("custom", p)
            }
        }
    };
    let out_spatial: Vec<i64> = (0..spatial)
        .map(|i| {
            if pad_type == "same" || pad_type == "same_lower" {
                let s = strides[i];
                t.shape[2 + i].div_euclid(s) + i64::from(t.shape[2 + i] % s != 0)
            } else {
                pool_out_dim(
                    t.shape[2 + i],
                    pads[i],
                    pads[i + spatial],
                    ks[i],
                    strides[i],
                    ceil,
                )
            }
        })
        .collect();
    let mut out_shape = t.shape[..2].to_vec();
    out_shape.extend(out_spatial);
    let k = cx.k_i32v(&as_i32s(&ks)?);
    let st = cx.k_i32v(&as_i32s(&strides)?);
    let pt = cx.k_str(pad_type);
    let pd = cx.k_i32v(&as_i32s(&pads)?);
    let cm = cx.k_bool(ceil);
    let mut inputs = vec![
        ("x".into(), bind(&xs[0]).1),
        ("kernel_sizes".into(), bind(&k).1),
        ("strides".into(), bind(&st).1),
        ("pad_type".into(), bind(&pt).1),
        ("pad".into(), bind(&pd).1),
        ("ceil_mode".into(), bind(&cm).1),
    ];
    if avg {
        // ONNX count_include_pad=1 (default) ↔ MIL exclude=false
        let cip = node.attr_i("count_include_pad", 1) != 0;
        let ex = cx.k_bool(!cip);
        inputs.push(("exclude_padding_from_average".into(), bind(&ex).1));
    }
    cx.e1(
        if avg { "avg_pool" } else { "max_pool" },
        inputs,
        out,
        t.dtype,
        &out_shape,
    );
    Ok(())
}

fn map_global_pool(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, avg: bool) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    let spatial = t.shape.len() - 2;
    let ks: Vec<i32> = t.shape[2..].iter().map(|&d| d as i32).collect();
    let st = vec![1i32; spatial];
    let pt = cx.k_str("valid");
    let pd = cx.k_i32v(&vec![0; spatial * 2]);
    let cm = cx.k_bool(false);
    let k = cx.k_i32v(&ks);
    let stc = cx.k_i32v(&st);
    let mut out_shape = t.shape[..2].to_vec();
    out_shape.extend(std::iter::repeat(1).take(spatial));
    let mut inputs = vec![
        ("x".into(), bind(&xs[0]).1),
        ("kernel_sizes".into(), bind(&k).1),
        ("strides".into(), bind(&stc).1),
        ("pad_type".into(), bind(&pt).1),
        ("pad".into(), bind(&pd).1),
        ("ceil_mode".into(), bind(&cm).1),
    ];
    if avg {
        let ex = cx.k_bool(false);
        inputs.push(("exclude_padding_from_average".into(), bind(&ex).1));
    }
    let _ = node;
    cx.e1(
        if avg { "avg_pool" } else { "max_pool" },
        inputs,
        out,
        t.dtype,
        &out_shape,
    );
    Ok(())
}

fn map_gather(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, label: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    let it = cx.info(&xs[1])?.clone();
    let ax = norm_axis(node.attr_i("axis", 0), t.shape.len(), "Gather")?;
    // scalar indices → gather with [1], then squeeze the axis
    let (idx_name, idx_shape, squeeze) = if it.shape.is_empty() {
        let sn = cx.san(&node.inputs[1]);
        if let Some(tp) = cx.init.get(&sn) {
            let v = tp.as_i64()?;
            let c = cx.k_i32t(&[1], &[v[0] as i32]);
            (c, vec![1i64], true)
        } else {
            (xs[1].clone(), vec![1i64], true)
        }
    } else {
        (xs[1].clone(), it.shape.clone(), false)
    };
    let mut gshape: Vec<i64> = t.shape[..ax as usize].to_vec();
    gshape.extend_from_slice(&idx_shape);
    gshape.extend_from_slice(&t.shape[ax as usize + 1..]);
    let a = cx.k_i32s(ax as i32);
    let bd = cx.k_i32s(0);
    let vi = cx.k_bool(false);
    let g_name = cx.b.fresh("g");
    let g = cx.e1(
        "gather",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("indices".into(), bind(&idx_name).1),
            ("axis".into(), bind(&a).1),
            ("batch_dims".into(), bind(&bd).1),
            ("validate_indices".into(), bind(&vi).1),
        ],
        &g_name,
        t.dtype,
        &gshape,
    );
    if squeeze {
        let mut out_shape = t.shape[..ax as usize].to_vec();
        out_shape.extend_from_slice(&t.shape[ax as usize + 1..]);
        reshape_named(cx, &g, &out_shape, out)?;
    } else {
        let gi = cx.info(&g)?.clone();
        cx.e1(
            "identity",
            vec![("x".into(), bind(&g).1)],
            out,
            gi.dtype,
            &gi.shape,
        );
    }
    let _ = label;
    Ok(())
}

fn map_slice(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, label: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    let rank = t.shape.len() as i64;
    let starts = cx.static_i64(&node.inputs[1], label)?;
    let ends = cx.static_i64(&node.inputs[2], label)?;
    let axes = match node.inputs.get(3) {
        Some(n) if !n.is_empty() => cx.static_i64(n, label)?,
        _ => (0..starts.len() as i64).collect(),
    };
    let steps = match node.inputs.get(4) {
        Some(n) if !n.is_empty() => cx.static_i64(n, label)?,
        _ => vec![1i64; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(OnnxError::BadShape(format!(
            "Slice '{label}': starts/ends/axes/steps length mismatch"
        )));
    }
    // Normalize into per-rank begin/end/stride (+ optional reverse).
    let mut begin = vec![0i64; rank as usize];
    let mut end = t.shape.clone();
    let mut stride = vec![1i64; rank as usize];
    let mut reverse_axes: Vec<i64> = Vec::new();
    for i in 0..starts.len() {
        let ax = norm_axis(axes[i], rank as usize, "Slice")? as usize;
        let dim = t.shape[ax];
        let st = steps[i];
        if st == 0 {
            return Err(OnnxError::BadShape(format!("Slice '{label}': step 0")));
        }
        let (s_raw, e_raw) = (starts[i], ends[i]);
        if st > 0 {
            let s = (if s_raw < 0 { s_raw + dim } else { s_raw }).clamp(0, dim);
            let e = (if e_raw < 0 { e_raw + dim } else { e_raw }).clamp(0, dim);
            begin[ax] = s;
            end[ax] = e.max(s);
            stride[ax] = st;
        } else {
            // negative step → forward slice + reverse on that axis
            let s = if s_raw < 0 { s_raw + dim } else { s_raw }.clamp(-1, dim - 1);
            let e = if e_raw < 0 { e_raw + dim } else { e_raw }.clamp(-1, dim - 1);
            // elements: s, s+st, ... > e → forward range [e+1, s+1) stride -st
            begin[ax] = e + 1;
            end[ax] = s + 1;
            stride[ax] = -st;
            reverse_axes.push(ax as i64);
        }
    }
    let mut out_shape = vec![0i64; rank as usize];
    for a in 0..rank as usize {
        let len = (end[a] - begin[a]).max(0);
        out_shape[a] = len.div_euclid(stride[a]) + i64::from(len.rem_euclid(stride[a]) != 0);
    }
    let b = cx.k_i32v(&as_i32s(&begin)?);
    let e = cx.k_i32v(&as_i32s(&end)?);
    let st = cx.k_i32v(&as_i32s(&stride)?);
    let bm = cx.k_bools(&vec![false; rank as usize]);
    let em = cx.k_bools(&vec![false; rank as usize]);
    let sm = cx.k_bools(&vec![false; rank as usize]);
    let sl_name = cx.b.fresh("slice");
    let sliced = cx.e1(
        "slice_by_index",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("begin".into(), bind(&b).1),
            ("end".into(), bind(&e).1),
            ("stride".into(), bind(&st).1),
            ("begin_mask".into(), bind(&bm).1),
            ("end_mask".into(), bind(&em).1),
            ("squeeze_mask".into(), bind(&sm).1),
        ],
        &sl_name,
        t.dtype,
        &out_shape,
    );
    if reverse_axes.is_empty() {
        let si = cx.info(&sliced)?.clone();
        cx.e1(
            "identity",
            vec![("x".into(), bind(&sliced).1)],
            out,
            si.dtype,
            &si.shape,
        );
    } else {
        let ra = cx.k_i32v(&as_i32s(&reverse_axes)?);
        cx.e1(
            "reverse",
            vec![("x".into(), bind(&sliced).1), ("axes".into(), bind(&ra).1)],
            out,
            t.dtype,
            &out_shape,
        );
    }
    Ok(())
}

fn map_split(cx: &mut Ctx, node: &NodeProto, xs: &[String], label: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    let ax = norm_axis(node.attr_i("axis", 0), t.shape.len(), "Split")?;
    let dim = t.shape[ax as usize];
    let sizes: Vec<i64> = if let Some(name) = node.inputs.get(1) {
        if !name.is_empty() {
            cx.static_i64(name, label)?
        } else {
            vec![]
        }
    } else {
        node.attr_ints("split")
    };
    let n_out = node.outputs.len();
    let sizes: Vec<i64> = if sizes.is_empty() {
        if dim % n_out as i64 != 0 {
            return Err(OnnxError::BadShape(format!(
                "Split '{label}': dim {dim} not divisible into {n_out} parts"
            )));
        }
        vec![dim / n_out as i64; n_out]
    } else {
        if sizes.iter().sum::<i64>() != dim || sizes.len() != n_out {
            return Err(OnnxError::BadShape(format!(
                "Split '{label}': sizes {sizes:?} vs dim {dim}/{n_out} outputs"
            )));
        }
        sizes
    };
    let ss = cx.k_i32v(&as_i32s(&sizes)?);
    let ns = cx.k_i32s(n_out as i32);
    let a = cx.k_i32s(ax as i32);
    let mut outs: Vec<(String, DType, Vec<i64>)> = Vec::with_capacity(n_out);
    for (i, oname) in node.outputs.iter().enumerate() {
        let mut s = t.shape.clone();
        s[ax as usize] = sizes[i];
        outs.push((cx.san(oname), t.dtype, s));
    }
    let ov: Vec<(&str, DType, &[i64])> = outs
        .iter()
        .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
        .collect();
    cx.emit(
        "split",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("num_splits".into(), bind(&ns).1),
            ("split_sizes".into(), bind(&ss).1),
            ("axis".into(), bind(&a).1),
        ],
        ov,
    );
    Ok(())
}

fn map_cast(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str) -> Res<()> {
    let label = node.label(0);
    let t = cx.info(&xs[0])?.clone();
    let to = node.attr_i("to", 0) as i32;
    // Map ONNX target type into our lattice: floats → fp16 (fp32 kept
    // only when input was already fp32 — it never is), ints → int32.
    let (dt, mil_ty) = match to {
        elem::FLOAT | elem::FLOAT16 | elem::DOUBLE | elem::BFLOAT16 => (DType::Fp16, "fp16"),
        elem::INT32
        | elem::INT64
        | elem::INT8
        | elem::UINT8
        | elem::INT16
        | elem::UINT16
        | elem::UINT32
        | elem::UINT64 => (DType::Int32, "int32"),
        elem::BOOL => (DType::Bool, "bool"),
        t => return Err(OnnxError::Unsupported(format!("Cast '{label}': to={t}"))),
    };
    if dt == t.dtype {
        cx.alias.insert(out.to_string(), xs[0].clone());
        cx.env.insert(out.to_string(), t);
        return Ok(());
    }
    let d = cx.k_str(mil_ty);
    cx.e1(
        "cast",
        vec![("x".into(), bind(&xs[0]).1), ("dtype".into(), bind(&d).1)],
        out,
        dt,
        &t.shape,
    );
    Ok(())
}

fn map_constant(cx: &mut Ctx, node: &NodeProto, out: &str) -> Res<()> {
    let label = node.label(0);
    let (val, dt, shape) = if let Some(tp) = node.attr_t("value") {
        match tp.data_type {
            elem::FLOAT | elem::FLOAT16 | elem::DOUBLE => {
                let vs = tp.as_f32()?;
                (Value::f16s(&tp.dims, &vs), DType::Fp16, tp.dims.clone())
            }
            elem::INT32 | elem::INT64 => {
                let vs = tp.as_i64()?;
                let v32 = as_i32s(&vs)?;
                (
                    Value::Imm(
                        ValueType::Tensor(TensorType {
                            dtype: DType::Int32,
                            shape: tp.dims.clone(),
                        }),
                        Immediate::Ints(v32),
                    ),
                    DType::Int32,
                    tp.dims.clone(),
                )
            }
            t => {
                return Err(OnnxError::Unsupported(format!(
                    "Constant '{label}': dtype {t}"
                )))
            }
        }
    } else if let Some(f) = node.attrs.get("value_float").and_then(|a| a.f) {
        (Value::f16_scalar(f), DType::Fp16, vec![])
    } else if let Some(i) = node.attrs.get("value_int").and_then(|a| a.i) {
        (
            Value::Imm(
                ValueType::Tensor(TensorType {
                    dtype: DType::Int32,
                    shape: vec![],
                }),
                Immediate::Ints(vec![i as i32]),
            ),
            DType::Int32,
            vec![],
        )
    } else if let Some(a) = node.attrs.get("value_floats") {
        let vs = a.floats.clone();
        (
            Value::f16s(&[vs.len() as i64], &vs),
            DType::Fp16,
            vec![vs.len() as i64],
        )
    } else if let Some(a) = node.attrs.get("value_ints") {
        let vs = as_i32s(&a.ints)?;
        (
            Value::Imm(
                ValueType::Tensor(TensorType {
                    dtype: DType::Int32,
                    shape: vec![vs.len() as i64],
                }),
                Immediate::Ints(vs.clone()),
            ),
            DType::Int32,
            vec![vs.len() as i64],
        )
    } else {
        return Err(OnnxError::Unsupported(format!(
            "Constant '{label}': no supported value attribute"
        )));
    };
    let vt = cx.tt(dt, &shape);
    cx.b.op("const", vec![], vec![(out, vt)], vec![("val".into(), val)]);
    cx.env.insert(out.to_string(), TInfo { shape, dtype: dt });
    *cx.hist.entry("const".to_string()).or_insert(0) += 1;
    Ok(())
}

fn map_pad(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, label: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    let rank = t.shape.len();
    // pads: attr (opset<11) or input[1] initializer
    let pads = if xs.len() > 1 && !xs[1].is_empty() {
        cx.static_i64(&node.inputs[1], label)?
    } else {
        node.attr_ints("pads")
    };
    if pads.len() != rank * 2 {
        return Err(OnnxError::BadShape(format!(
            "Pad '{label}': pads {:?} vs rank {rank}",
            pads
        )));
    }
    let mode = node.attr_str("mode").unwrap_or_else(|| "constant".into());
    let mil_mode = match mode.as_str() {
        "constant" => "constant",
        "reflect" => "reflect",
        "edge" => "replicate",
        m => return Err(OnnxError::Unsupported(format!("Pad '{label}': mode '{m}'"))),
    };
    let cval = if xs.len() > 2 && !xs[2].is_empty() {
        cx.static_f32(&node.inputs[2], label)?
            .first()
            .copied()
            .unwrap_or(0.0)
    } else {
        node.attr_f("value", 0.0)
    };
    // ONNX layout [b0..b_{r-1}, e0..e_{r-1}] → MIL interleaved pairs over
    // the last N dims, where N covers every nonzero pad.
    let first_nz = (0..rank)
        .find(|&i| pads[i] != 0 || pads[i + rank] != 0)
        .unwrap_or(rank);
    let n = rank - first_nz;
    let mut mil_pads = Vec::with_capacity(n * 2);
    for i in first_nz..rank {
        mil_pads.push(pads[i]);
        mil_pads.push(pads[i + rank]);
    }
    let out_shape: Vec<i64> = (0..rank)
        .map(|i| t.shape[i] + pads[i] + pads[i + rank])
        .collect();
    if mil_pads.iter().any(|&p| p < 0) {
        return Err(OnnxError::Unsupported(format!(
            "Pad '{label}': negative pads (crop)"
        )));
    }
    let p = cx.k_i32v(&as_i32s(&mil_pads)?);
    let m = cx.k_str(mil_mode);
    let cv = cx.k_f16(cval);
    cx.e1(
        "pad",
        vec![
            ("x".into(), bind(&xs[0]).1),
            ("pad".into(), bind(&p).1),
            ("mode".into(), bind(&m).1),
            ("constant_val".into(), bind(&cv).1),
        ],
        out,
        t.dtype,
        &out_shape,
    );
    Ok(())
}

fn map_resize(cx: &mut Ctx, node: &NodeProto, xs: &[String], out: &str, label: &str) -> Res<()> {
    let t = cx.info(&xs[0])?.clone();
    if t.shape.len() != 4 {
        return Err(OnnxError::Unsupported(format!(
            "Resize '{label}': rank {} ≠ 4 (only 2-D spatial resize)",
            t.shape.len()
        )));
    }
    let mode = node.attr_str("mode").unwrap_or_else(|| "nearest".into());
    // scales: Upsample input[1] / Resize input[3] (or sizes at input[3])
    let (h, w) = (t.shape[2], t.shape[3]);
    // scales: Upsample input[1]; Resize input[2] (input[1] is roi).
    let scales: Option<Vec<f32>> = {
        let idx = if node.op_type == "Upsample" { 1 } else { 2 };
        match node.inputs.get(idx) {
            Some(n) if !n.is_empty() => Some(cx.static_f32(n, label)?),
            _ => None,
        }
    };
    let sizes: Option<Vec<i64>> = match node.inputs.get(3) {
        Some(n) if !n.is_empty() => Some(cx.static_i64(n, label)?),
        _ => None,
    };
    let (sh, sw) = if let Some(s) = &scales {
        if s.len() != 4 {
            return Err(OnnxError::BadShape(format!(
                "Resize '{label}': scales {s:?} not rank-4"
            )));
        }
        if s[0] != 1.0 || s[1] != 1.0 {
            return Err(OnnxError::Unsupported(format!(
                "Resize '{label}': batch/channel scaling {s:?}"
            )));
        }
        (s[2], s[3])
    } else if let Some(sz) = &sizes {
        if sz.len() != 4 {
            return Err(OnnxError::BadShape(format!(
                "Resize '{label}': sizes {sz:?} not rank-4"
            )));
        }
        (sz[2] as f32 / h as f32, sz[3] as f32 / w as f32)
    } else {
        return Err(OnnxError::Malformed {
            what: node.op_type.clone(),
            detail: format!("node '{label}': no scales/sizes"),
        });
    };
    let ih = sh.round() as i64;
    let iw = sw.round() as i64;
    if (sh - ih as f32).abs() > 1e-6 || (sw - iw as f32).abs() > 1e-6 || ih < 1 || iw < 1 {
        return Err(OnnxError::Unsupported(format!(
            "Resize '{label}': non-integer scales {sh}x{sw} — only integer upsample"
        )));
    }
    let out_shape = [t.shape[0], t.shape[1], h * ih, w * iw];
    let shc = cx.k_i32s(ih as i32);
    let swc = cx.k_i32s(iw as i32);
    match mode.as_str() {
        "nearest" => {
            cx.e1(
                "upsample_nearest_neighbor",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("scale_factor_height".into(), bind(&shc).1),
                    ("scale_factor_width".into(), bind(&swc).1),
                ],
                out,
                t.dtype,
                &out_shape,
            );
        }
        "linear" | "bilinear" => {
            let ac = matches!(
                node.attr_str("coordinate_transformation_mode").as_deref(),
                Some("align_corners")
            );
            let acc = cx.k_bool(ac);
            cx.e1(
                "upsample_bilinear",
                vec![
                    ("x".into(), bind(&xs[0]).1),
                    ("scale_factor_height".into(), bind(&shc).1),
                    ("scale_factor_width".into(), bind(&swc).1),
                    ("align_corners".into(), bind(&acc).1),
                ],
                out,
                t.dtype,
                &out_shape,
            );
        }
        m => {
            return Err(OnnxError::Unsupported(format!(
                "Resize '{label}': mode '{m}'"
            )))
        }
    }
    Ok(())
}

/// Access an initializer by ONNX name without materializing it.
impl<'a> Ctx<'a> {
    pub(crate) fn init_of(&self, onnx: &str) -> Option<&'a TensorProto> {
        self.names
            .get(onnx)
            .and_then(|sn| self.init.get(sn).copied())
            .or_else(|| {
                // fallback: scan by original name (unsanitized path)
                self.init
                    .iter()
                    .find(|(_, tp)| tp.name == onnx)
                    .map(|(_, tp)| *tp)
            })
    }
}
