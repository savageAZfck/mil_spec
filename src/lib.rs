//! `mil_spec` — a native Apple CoreML MIL program writer, in pure Rust.
//!
//! Emits `.mlpackage` packages (protobuf spec + v2 weight blobs + Manifest.json)
//! with zero Python and zero coremltools — the pieces a CoreML `mlprogram`
//! needs, written directly to Apple's public wire schema (mlmodel/format:
//! Model.proto + MIL.proto, BSD-3).
//!
//! # What it can write
//!
//! - Tensor ops: `const`, elementwise arithmetic, `conv`, `matmul`,
//!   `reshape`/`transpose`/`slice`/`concat`, `softmax`, `cast`, `reduce_*`
//! - Stateful graphs: `read_state`/`write_state` for KV-cache-style models
//! - Weight-only int8: `constexpr_blockwise_shift_scale` — the runtime
//!   dequantization pattern CoreML uses to keep int8 payloads resident on
//!   the Neural Engine
//! - `weight.bin` v2 blobs, both file-backed ([`BlobWriter`]) and in-memory
//!   ([`WeightBin`])
//! - `.mlpackage` directory layout with a valid `Manifest.json`
//!
//! # Example
//!
//! ```no_run
//! use mil_spec::*;
//!
//! let mut b = Block::new();
//! let y = b.mul("x", "x", &[1, 4, 1, 1], "y");
//! b.outputs = vec![y];
//!
//! let inputs = [Feature { name: "x".into(), shape: vec![1, 4, 1, 1],
//!                       dtype: DType::Fp16, is_state: false }];
//! let outputs = [Feature { name: "y".into(), shape: vec![1, 4, 1, 1],
//!                        dtype: DType::Fp16, is_state: false }];
//! let fn_inputs = [NVT { name: "x".into(),
//!     ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])) }];
//!
//! let spec = encode_model(&inputs, &outputs, &[], &b, &fn_inputs,
//!                         &ModelMeta::new(10, "CoreML9"));
//! write_mlpackage(std::path::Path::new("out.mlpackage"), &spec, None).unwrap();
//! // compile with: xcrun coremlc compile out.mlpackage .
//! ```

#![forbid(unsafe_code)]

pub mod ir;

use std::io::{Seek, SeekFrom, Write};

// ======== protobuf wire primitives ========

const WIRE_VARINT: u32 = 0;
const WIRE_LEN: u32 = 2;

fn tag(buf: &mut Vec<u8>, field: u32, wire: u32) {
    varint(buf, ((field << 3) | wire) as u64);
}

fn varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

fn f_varint(buf: &mut Vec<u8>, field: u32, v: u64) {
    tag(buf, field, WIRE_VARINT);
    varint(buf, v);
}

fn f_i32(buf: &mut Vec<u8>, field: u32, v: i32) {
    // proto int32 encodes negative values as 10-byte sign-extended varint
    f_varint(buf, field, v as i64 as u64);
}

fn f_i64(buf: &mut Vec<u8>, field: u32, v: i64) {
    f_varint(buf, field, v as u64);
}

fn f_bytes(buf: &mut Vec<u8>, field: u32, v: &[u8]) {
    tag(buf, field, WIRE_LEN);
    varint(buf, v.len() as u64);
    buf.extend_from_slice(v);
}

fn f_str(buf: &mut Vec<u8>, field: u32, v: &str) {
    f_bytes(buf, field, v.as_bytes());
}

fn f_msg(buf: &mut Vec<u8>, field: u32, v: &[u8]) {
    f_bytes(buf, field, v);
}

fn f_packed_i32(buf: &mut Vec<u8>, field: u32, vs: &[i32]) {
    let mut inner = Vec::with_capacity(vs.len() * 2);
    for &v in vs {
        varint(&mut inner, v as i64 as u64);
    }
    f_bytes(buf, field, &inner);
}

fn f_packed_i64(buf: &mut Vec<u8>, field: u32, vs: &[i64]) {
    let mut inner = Vec::with_capacity(vs.len() * 2);
    for &v in vs {
        varint(&mut inner, v as u64);
    }
    f_bytes(buf, field, &inner);
}

fn f_packed_f32(buf: &mut Vec<u8>, field: u32, vs: &[f32]) {
    let mut inner = Vec::with_capacity(vs.len() * 4);
    for &v in vs {
        inner.extend_from_slice(&v.to_le_bytes());
    }
    f_bytes(buf, field, &inner);
}

// map<K,string> / map<K,msg> entry: message { key=1, value=2 }
fn map_entry_str(buf: &mut Vec<u8>, field: u32, key: &str, val: &[u8]) {
    let mut e = Vec::with_capacity(key.len() + val.len() + 8);
    f_str(&mut e, 1, key);
    f_bytes(&mut e, 2, val);
    f_bytes(buf, field, &e);
}

// ======== MIL spec types ========

/// Element data type for tensors, blobs, and feature descriptions.
///
/// The numeric mappings are the MIL/ArrayFeatureType/BlobDataType enum
/// values from Apple's format — they are wire constants, not indices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    Bool,
    Fp16,
    Fp32,
    Int4,
    Int8,
    Int32,
    Int64,
    Str,
    Uint4,
}

impl DType {
    fn mil(self) -> i32 {
        match self {
            DType::Bool => 1,
            DType::Str => 2,
            DType::Fp16 => 10,
            DType::Fp32 => 11,
            DType::Int4 => 25,
            DType::Int8 => 21,
            DType::Int32 => 23,
            DType::Int64 => 24,
            DType::Uint4 => 35,
        }
    }
    // ArrayFeatureType::ArrayDataType enum
    fn array(self) -> i64 {
        match self {
            DType::Fp16 => 65552,
            DType::Fp32 => 65568,
            DType::Int32 => 131104,
            _ => 65568,
        }
    }
    /// `MILBlob::BlobDataType` enum value used in weight.bin metadata.
    pub fn blob(self) -> u32 {
        match self {
            DType::Fp16 => 1,
            DType::Fp32 => 2,
            DType::Int4 => 8,
            DType::Int8 => 4,
            DType::Int32 => 14,
            DType::Uint4 => 11,
            _ => 1,
        }
    }
    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            DType::Int8 => 1,
            DType::Fp16 => 2,
            DType::Fp32 | DType::Int32 => 4,
            _ => 4,
        }
    }
}

/// A tensor type: element dtype + static shape.
///
/// Shapes are fully static — CoreML's mlprogram path compiles against
/// concrete dimensions, and dynamic dims (`dim.flexible`) are not emitted
/// by this writer.
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<i64>,
}

impl TensorType {
    /// `tensor<fp16, shape>` convenience.
    pub fn f16(shape: &[i64]) -> Self {
        TensorType {
            dtype: DType::Fp16,
            shape: shape.to_vec(),
        }
    }
    /// `tensor<int32, [n]>` convenience.
    pub fn i32v(n: usize) -> Self {
        TensorType {
            dtype: DType::Int32,
            shape: vec![n as i64],
        }
    }
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        f_varint(&mut b, 1, self.dtype.mil() as u64);
        f_i64(&mut b, 2, self.shape.len() as i64); // rank
        for &d in &self.shape {
            // Dimension { constant=1 {size=1} }
            let mut cd = Vec::new();
            f_varint(&mut cd, 1, d as u64);
            let mut dim = Vec::new();
            f_msg(&mut dim, 1, &cd);
            f_msg(&mut b, 3, &dim);
        }
        b
    }
}

/// A value type as it appears in function inputs and op outputs.
///
/// `State` wraps a tensor type for stateful models (e.g. KV caches bound
/// to `read_state`/`write_state`).
pub enum ValueType {
    Tensor(TensorType),
    State(TensorType),
}

impl ValueType {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        match self {
            ValueType::Tensor(t) => f_msg(&mut b, 1, &t.encode()),
            ValueType::State(t) => {
                let mut st = Vec::new();
                // StateType.wrappedType is a ValueType (tensor)
                let mut inner = Vec::new();
                f_msg(&mut inner, 1, &t.encode());
                f_msg(&mut st, 1, &inner);
                f_msg(&mut b, 5, &st);
            }
        }
        b
    }
}

/// Named value type — an op output or function input declaration.
pub struct NVT {
    pub name: String,
    pub ty: ValueType,
}

impl NVT {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        f_str(&mut b, 1, &self.name);
        f_msg(&mut b, 2, &self.ty.encode());
        b
    }
}

/// Compile-time immediate value payload.
pub enum Immediate {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
    LongInts(Vec<i64>),
    Bools(Vec<bool>),
    Strings(Vec<String>),
}

/// A `Value` — how a const or attribute's payload is stored on the wire.
pub enum Value {
    /// Typed immediate (scalars and small tensors).
    Imm(ValueType, Immediate),
    /// Reference into `weight.bin`: dtype/shape, fileName, blob offset.
    Blob(ValueType, String, u64),
    /// Raw little-endian tensor bytes (e.g. fp16 immediates — MIL rejects
    /// f32 `floats` stored under an fp16 type).
    Bytes(ValueType, Vec<u8>),
    /// Bare scalar string.
    Str(String),
}

impl Value {
    /// `int32` vector const.
    pub fn i32s(vs: &[i32]) -> Self {
        Value::Imm(
            ValueType::Tensor(TensorType::i32v(vs.len())),
            Immediate::Ints(vs.to_vec()),
        )
    }
    /// `int32` single-element vector const.
    pub fn i32_1(v: i32) -> Self {
        Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Int32,
                shape: vec![1],
            }),
            Immediate::Ints(vec![v]),
        )
    }
    /// `bool` vector const.
    pub fn bools(vs: &[bool]) -> Self {
        Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Bool,
                shape: vec![vs.len() as i64],
            }),
            Immediate::Bools(vs.to_vec()),
        )
    }
    /// `fp16` tensor const; `vs` are f32 values quantized to fp16 storage.
    pub fn f16s(shape: &[i64], vs: &[f32]) -> Self {
        // fp16 immediates serialize as raw little-endian bytes (MIL reads them
        // from TensorValue.bytes; storing f32 `floats` under an fp16 type is
        // rejected as an element-count mismatch by the spec parser).
        let raw: Vec<u8> = vs
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        Value::Bytes(ValueType::Tensor(TensorType::f16(shape)), raw)
    }
    /// `fp16` scalar const.
    pub fn f16_scalar(v: f32) -> Self {
        Self::f16s(&[], &[v])
    }
    /// `fp32` scalar const.
    pub fn f32_scalar(v: f32) -> Self {
        Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Fp32,
                shape: vec![],
            }),
            Immediate::Floats(vec![v]),
        )
    }
    /// `str` scalar const (rank-0 string tensor).
    pub fn string(s: &str) -> Self {
        // scalar string: type tensorType<str, rank0> + immediateValue.strings
        Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Str,
                shape: vec![],
            }),
            Immediate::Strings(vec![s.to_string()]),
        )
    }

    /// The value's wire encoding. Passes and inspectors fingerprint consts
    /// by these bytes — two `Value`s that encode identically are the same
    /// value and may be merged.
    pub fn wire_bytes(&self) -> Vec<u8> {
        self.encode()
    }

    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        match self {
            Value::Imm(ty, imm) => {
                f_msg(&mut b, 2, &ty.encode());
                let mut tv = Vec::new();
                match imm {
                    Immediate::Floats(vs) => {
                        let mut r = Vec::new();
                        f_packed_f32(&mut r, 1, vs);
                        f_msg(&mut tv, 1, &r);
                    }
                    Immediate::Ints(vs) => {
                        let mut r = Vec::new();
                        f_packed_i32(&mut r, 1, vs);
                        f_msg(&mut tv, 2, &r);
                    }
                    Immediate::Bools(vs) => {
                        let mut r = Vec::new();
                        let mut inner = Vec::new();
                        for &v in vs {
                            varint(&mut inner, v as u64);
                        }
                        f_bytes(&mut r, 1, &inner);
                        f_msg(&mut tv, 3, &r);
                    }
                    Immediate::Strings(vs) => {
                        let mut r = Vec::new();
                        for s in vs {
                            f_str(&mut r, 1, s);
                        }
                        f_msg(&mut tv, 4, &r);
                    }
                    Immediate::LongInts(vs) => {
                        let mut r = Vec::new();
                        f_packed_i64(&mut r, 1, vs);
                        f_msg(&mut tv, 5, &r);
                    }
                }
                let mut iv = Vec::new();
                f_msg(&mut iv, 1, &tv);
                f_msg(&mut b, 3, &iv); // immediateValue = 3
            }
            Value::Blob(ty, file, off) => {
                f_msg(&mut b, 2, &ty.encode());
                let mut bv = Vec::new();
                f_str(&mut bv, 1, file);
                f_varint(&mut bv, 2, *off);
                f_msg(&mut b, 5, &bv); // blobFileValue = 5
            }
            Value::Bytes(ty, raw) => {
                f_msg(&mut b, 2, &ty.encode());
                let mut tv = Vec::new();
                let mut r = Vec::new();
                f_bytes(&mut r, 1, raw); // RepeatedBytes.values = 1
                f_msg(&mut tv, 7, &r); // TensorValue.bytes = 7
                let mut iv = Vec::new();
                f_msg(&mut iv, 1, &tv);
                f_msg(&mut b, 3, &iv);
            }
            Value::Str(s) => {
                // bare scalar string value
                f_msg(
                    &mut b,
                    2,
                    &ValueType::Tensor(TensorType {
                        dtype: DType::Str,
                        shape: vec![],
                    })
                    .encode(),
                );
                let mut tv = Vec::new();
                let mut r = Vec::new();
                f_str(&mut r, 1, s);
                f_msg(&mut tv, 4, &r);
                let mut iv = Vec::new();
                f_msg(&mut iv, 1, &tv);
                f_msg(&mut b, 3, &iv);
            }
        }
        b
    }
}

/// One binding inside an `Argument`: a name reference or an inline value.
pub enum Binding {
    Name(String),
    Val(Value),
}

/// An op input argument — an ordered list of name/value bindings.
pub struct Argument(pub Vec<Binding>);

impl Argument {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for binding in &self.0 {
            let mut bd = Vec::new();
            match binding {
                Binding::Name(n) => f_str(&mut bd, 1, n),
                Binding::Val(v) => f_msg(&mut bd, 2, &v.encode()),
            }
            f_msg(&mut b, 1, &bd);
        }
        b
    }
}

/// Bind a single named input: returns `(name, Argument)` pairs for `op`.
pub fn bind(name: &str) -> (String, Argument) {
    (
        name.to_string(),
        Argument(vec![Binding::Name(name.to_string())]),
    )
}

/// Bind multiple named inputs into one variadic `Argument` (e.g. `concat.values`).
pub fn bind_many(names: &[&str]) -> Argument {
    Argument(names.iter().map(|n| Binding::Name(n.to_string())).collect())
}

/// Bind an inline const `Value` as an `Argument`.
pub fn bind_const(v: Value) -> Argument {
    Argument(vec![Binding::Val(v)])
}

/// A single MIL operation.
pub struct Op {
    /// Op type string (`"mul"`, `"scaled_dot_product_attention"`, ...).
    pub ty: String,
    /// `(input_name, argument)` pairs.
    pub inputs: Vec<(String, Argument)>,
    /// Declared outputs.
    pub outputs: Vec<NVT>,
    /// `(attr_name, value)` pairs.
    pub attrs: Vec<(String, Value)>,
}

impl Op {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        f_str(&mut b, 1, &self.ty);
        for (k, a) in &self.inputs {
            map_entry_str(&mut b, 2, k, &a.encode());
        }
        for o in &self.outputs {
            f_msg(&mut b, 3, &o.encode());
        }
        for (k, v) in &self.attrs {
            map_entry_str(&mut b, 5, k, &v.encode());
        }
        b
    }
}

// ======== graph builder ========

/// A MIL block: an ordered op list plus the names of its output values.
///
/// The builder helpers emit ops with correct signatures and auto-generated
/// const inputs for scalars/vectors — call them and wire the returned value
/// names into downstream ops.
pub struct Block {
    pub ops: Vec<Op>,
    pub outputs: Vec<String>,
    counter: usize,
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl Block {
    pub fn new() -> Self {
        Block {
            ops: vec![],
            outputs: vec![],
            counter: 0,
        }
    }

    /// Fresh unique name with a hint prefix (for synthesized const names).
    pub fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    /// Emit an op; `outs` are (name, type) for each produced value.
    pub fn op(
        &mut self,
        ty: &str,
        inputs: Vec<(String, Argument)>,
        outs: Vec<(&str, ValueType)>,
        attrs: Vec<(String, Value)>,
    ) -> Vec<String> {
        let out_names: Vec<String> = outs.iter().map(|(n, _)| n.to_string()).collect();
        let op = Op {
            ty: ty.to_string(),
            inputs,
            outputs: outs
                .into_iter()
                .map(|(n, t)| NVT {
                    name: n.to_string(),
                    ty: t,
                })
                .collect(),
            attrs,
        };
        self.ops.push(op);
        out_names
    }

    /// Convenience single-output op; returns the output name.
    pub fn o1(
        &mut self,
        ty: &str,
        inputs: Vec<(String, Argument)>,
        out_name: &str,
        out_ty: ValueType,
    ) -> String {
        self.op(ty, inputs, vec![(out_name, out_ty)], vec![])[0].clone()
    }

    fn tt(&self, dt: DType, shape: &[i64]) -> ValueType {
        ValueType::Tensor(TensorType {
            dtype: dt,
            shape: shape.to_vec(),
        })
    }

    // --- typed helpers (names chosen by caller) ---

    /// `const` with an `int32` vector value.
    pub fn konst_i32(&mut self, name: &str, vs: &[i32]) -> String {
        let vt = self.tt(DType::Int32, &[vs.len() as i64]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::i32s(vs))],
        )[0]
        .clone()
    }

    /// `const` with an `int32` scalar (rank-0) value.
    pub fn konst_scalar_i32(&mut self, name: &str, v: i32) -> String {
        let vt = self.tt(DType::Int32, &[]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![(
                "val".into(),
                Value::Imm(
                    ValueType::Tensor(TensorType {
                        dtype: DType::Int32,
                        shape: vec![],
                    }),
                    Immediate::Ints(vec![v]),
                ),
            )],
        )[0]
        .clone()
    }

    /// `const` with an `fp16` scalar value.
    pub fn konst_f16(&mut self, name: &str, v: f32) -> String {
        let vt = self.tt(DType::Fp16, &[]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::f16_scalar(v))],
        )[0]
        .clone()
    }

    /// `const` with an `fp32` scalar value.
    pub fn konst_f32(&mut self, name: &str, v: f32) -> String {
        let vt = self.tt(DType::Fp32, &[]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::f32_scalar(v))],
        )[0]
        .clone()
    }

    /// `const` with a `bool` scalar value.
    pub fn konst_bool(&mut self, name: &str, v: bool) -> String {
        let vt = self.tt(DType::Bool, &[]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![(
                "val".into(),
                Value::Imm(
                    ValueType::Tensor(TensorType {
                        dtype: DType::Bool,
                        shape: vec![],
                    }),
                    Immediate::Bools(vec![v]),
                ),
            )],
        )[0]
        .clone()
    }

    /// `const` referencing a blob in `weight.bin` at `offset`.
    ///
    /// `file` is the BlobFileValue path — `"@model_path/weights/weight.bin"`
    /// for the package's own weight blob.
    pub fn konst_blob(
        &mut self,
        name: &str,
        file: &str,
        offset: u64,
        dtype: DType,
        shape: &[i64],
    ) -> String {
        let vt = self.tt(dtype, shape);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![(
                "val".into(),
                Value::Blob(
                    ValueType::Tensor(TensorType {
                        dtype,
                        shape: shape.to_vec(),
                    }),
                    file.to_string(),
                    offset,
                ),
            )],
        )[0]
        .clone()
    }

    /// Int8 weight with one fp16 scale per output channel (axis 0):
    /// `name = constexpr_blockwise_shift_scale(data=int8, scale=fp16)`.
    /// The ANE keeps the int8 payload and dequantizes on the fly.
    pub fn konst_q8(
        &mut self,
        name: &str,
        file: &str,
        data_off: u64,
        scale_off: u64,
        shape: &[i64],
    ) -> String {
        let mut scale_shape = vec![shape[0]];
        scale_shape.extend(std::iter::repeat(1).take(shape.len() - 1));
        let q = self.konst_blob(&format!("{name}_q8"), file, data_off, DType::Int8, shape);
        let s = self.konst_blob(
            &format!("{name}_scale"),
            file,
            scale_off,
            DType::Fp16,
            &scale_shape,
        );
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "constexpr_blockwise_shift_scale",
            vec![("data".into(), bind(&q).1), ("scale".into(), bind(&s).1)],
            name,
            vt,
        )
    }

    /// 4-bit palettized weight with a 16-entry fp16 LUT per output channel:
    /// `name = constexpr_lut_to_dense(indices=uint4, lut=fp16)`.
    /// ANE-supported sub-byte route — `constexpr_blockwise_shift_scale` on
    /// int4 fails plan-build -14. Indices are packed two nibbles per byte;
    /// the LUT blob holds `shape[0] * 16` fp16 values — for channel c,
    /// entry i is the dequantized weight for index i.
    pub fn konst_q4(
        &mut self,
        name: &str,
        file: &str,
        data_off: u64,
        lut_off: u64,
        shape: &[i64],
    ) -> String {
        let mut lut_shape: Vec<i64> = vec![shape[0]];
        lut_shape.extend(std::iter::repeat(1).take(shape.len() - 1));
        lut_shape.push(16);
        lut_shape.push(1);
        let q = self.konst_blob(&format!("{name}_q4"), file, data_off, DType::Uint4, shape);
        let s = self.konst_blob(
            &format!("{name}_lut"),
            file,
            lut_off,
            DType::Fp16,
            &lut_shape,
        );
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "constexpr_lut_to_dense",
            vec![("indices".into(), bind(&q).1), ("lut".into(), bind(&s).1)],
            name,
            vt,
        )
    }

    /// `mul` elementwise (fp16 output).
    pub fn mul(&mut self, a: &str, b: &str, shape: &[i64], name: &str) -> String {
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "mul",
            vec![("x".into(), bind(a).1), ("y".into(), bind(b).1)],
            name,
            vt,
        )
    }

    /// `add` elementwise (fp16 output).
    pub fn add(&mut self, a: &str, b: &str, shape: &[i64], name: &str) -> String {
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "add",
            vec![("x".into(), bind(a).1), ("y".into(), bind(b).1)],
            name,
            vt,
        )
    }

    /// `sub` elementwise (fp16 output).
    pub fn sub(&mut self, a: &str, b: &str, shape: &[i64], name: &str) -> String {
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "sub",
            vec![("x".into(), bind(a).1), ("y".into(), bind(b).1)],
            name,
            vt,
        )
    }

    /// 1x1 `conv` with unit strides, valid padding, one group — the layout
    /// CoreML uses to run linear layers on the ANE.
    pub fn conv1x1(
        &mut self,
        x: &str,
        w: &str,
        bias: Option<&str>,
        cout: i64,
        name: &str,
    ) -> String {
        let strides = self.fresh("stride");
        let strides = self.konst_i32(&strides, &[1, 1]);
        let pad_type = self.fresh("padtype");
        let pad_type = {
            let vt = self.tt(DType::Str, &[]);
            self.op(
                "const",
                vec![],
                vec![(&pad_type, vt)],
                vec![("val".into(), Value::Str("valid".into()))],
            )[0]
            .clone()
        };
        let pads = self.fresh("pad");
        let pads = self.konst_i32(&pads, &[0, 0, 0, 0]);
        let dil = self.fresh("dil");
        let dil = self.konst_i32(&dil, &[1, 1]);
        let grp = self.fresh("grp");
        let grp = self.konst_scalar_i32(&grp, 1);
        let mut inputs = vec![
            ("x".into(), bind(x).1),
            ("weight".into(), bind(w).1),
            ("strides".into(), bind(&strides).1),
            ("pad_type".into(), bind(&pad_type).1),
            ("pad".into(), bind(&pads).1),
            ("dilations".into(), bind(&dil).1),
            ("groups".into(), bind(&grp).1),
        ];
        if let Some(bi) = bias {
            inputs.push(("bias".into(), bind(bi).1));
        }
        let vt = self.tt(DType::Fp16, &[1, cout, 1, 1]);
        self.o1("conv", inputs, name, vt)
    }

    /// `conv1x1` over a sequence: x is `(1, cin, s, 1)` → out `(1, cout, s, 1)`.
    pub fn conv1x1_s(
        &mut self,
        x: &str,
        w: &str,
        bias: Option<&str>,
        cout: i64,
        s: i64,
        name: &str,
    ) -> String {
        let strides = self.fresh("stride");
        let strides = self.konst_i32(&strides, &[1, 1]);
        let pad_type = self.fresh("padtype");
        let pad_type = {
            let vt = self.tt(DType::Str, &[]);
            self.op(
                "const",
                vec![],
                vec![(&pad_type, vt)],
                vec![("val".into(), Value::Str("valid".into()))],
            )[0]
            .clone()
        };
        let pads = self.fresh("pad");
        let pads = self.konst_i32(&pads, &[0, 0, 0, 0]);
        let dil = self.fresh("dil");
        let dil = self.konst_i32(&dil, &[1, 1]);
        let grp = self.fresh("grp");
        let grp = self.konst_scalar_i32(&grp, 1);
        let mut inputs = vec![
            ("x".into(), bind(x).1),
            ("weight".into(), bind(w).1),
            ("strides".into(), bind(&strides).1),
            ("pad_type".into(), bind(&pad_type).1),
            ("pad".into(), bind(&pads).1),
            ("dilations".into(), bind(&dil).1),
            ("groups".into(), bind(&grp).1),
        ];
        if let Some(bi) = bias {
            inputs.push(("bias".into(), bind(bi).1));
        }
        let vt = self.tt(DType::Fp16, &[1, cout, s, 1]);
        self.o1("conv", inputs, name, vt)
    }

    /// `reshape` to `shape` (emits the int32 shape const).
    pub fn reshape(&mut self, x: &str, shape: &[i64], name: &str) -> String {
        let s = self.fresh("shape");
        let s = self.konst_i32(&s, &shape.iter().map(|d| *d as i32).collect::<Vec<_>>());
        let vt = self.tt(DType::Fp16, shape);
        self.o1(
            "reshape",
            vec![("x".into(), bind(x).1), ("shape".into(), bind(&s).1)],
            name,
            vt,
        )
    }

    /// `transpose` by `perm`.
    pub fn transpose(&mut self, x: &str, perm: &[i32], out_shape: &[i64], name: &str) -> String {
        let p = self.fresh("perm");
        let p = self.konst_i32(&p, perm);
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "transpose",
            vec![("x".into(), bind(x).1), ("perm".into(), bind(&p).1)],
            name,
            vt,
        )
    }

    /// `expand_dims` on `axes`.
    pub fn expand_dims(&mut self, x: &str, axes: &[i32], out_shape: &[i64], name: &str) -> String {
        let a = self.fresh("axes");
        let a = self.konst_i32(&a, axes);
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "expand_dims",
            vec![("x".into(), bind(x).1), ("axes".into(), bind(&a).1)],
            name,
            vt,
        )
    }

    /// `squeeze` on `axes`.
    pub fn squeeze(&mut self, x: &str, axes: &[i32], out_shape: &[i64], name: &str) -> String {
        let a = self.fresh("axes");
        let a = self.konst_i32(&a, axes);
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "squeeze",
            vec![("x".into(), bind(x).1), ("axes".into(), bind(&a).1)],
            name,
            vt,
        )
    }

    /// `slice_by_index` with unit stride and open masks.
    pub fn slice(
        &mut self,
        x: &str,
        begin: &[i32],
        end: &[i32],
        out_shape: &[i64],
        name: &str,
    ) -> String {
        let b0 = self.fresh("begin");
        let b0 = self.konst_i32(&b0, begin);
        let e0 = self.fresh("end");
        let e0 = self.konst_i32(&e0, end);
        let st = self.fresh("stride");
        let st = self.konst_i32(&st, &vec![1; begin.len()]);
        let bm = self.fresh("bmask");
        let bm = self.konst_i32_bm(&bm, begin.len());
        let em = self.fresh("emask");
        let em = self.konst_i32_bm(&em, end.len());
        let sm = self.fresh("smask");
        let sm = self.konst_i32_sm(&sm, end.len());
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "slice_by_index",
            vec![
                ("x".into(), bind(x).1),
                ("begin".into(), bind(&b0).1),
                ("end".into(), bind(&e0).1),
                ("stride".into(), bind(&st).1),
                ("begin_mask".into(), bind(&bm).1),
                ("end_mask".into(), bind(&em).1),
                ("squeeze_mask".into(), bind(&sm).1),
            ],
            name,
            vt,
        )
    }

    fn konst_i32_bm(&mut self, name: &str, n: usize) -> String {
        let vs: Vec<bool> = vec![false; n];
        let vt = self.tt(DType::Bool, &[n as i64]);
        self.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::bools(&vs))],
        )[0]
        .clone()
    }
    fn konst_i32_sm(&mut self, name: &str, n: usize) -> String {
        self.konst_i32_bm(name, n)
    }

    /// `concat` along `axis`.
    pub fn concat(&mut self, xs: &[String], axis: i32, out_shape: &[i64], name: &str) -> String {
        let a = self.fresh("axis");
        let a = self.konst_scalar_i32(&a, axis);
        let il = self.fresh("ilv");
        let il = self.konst_bool(&il, false);
        let refs: Vec<&str> = xs.iter().map(|s| s.as_str()).collect();
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "concat",
            vec![
                ("values".into(), bind_many(&refs)),
                ("axis".into(), bind(&a).1),
                ("interleave".into(), bind(&il).1),
            ],
            name,
            vt,
        )
    }

    /// `matmul` with optional transpose of the second operand.
    pub fn matmul(&mut self, x: &str, y: &str, ty: bool, out_shape: &[i64], name: &str) -> String {
        let tx = self.fresh("tx");
        let tx = self.konst_bool(&tx, false);
        let tyy = self.fresh("ty");
        let tyy = self.konst_bool(&tyy, ty);
        let vt = self.tt(DType::Fp16, out_shape);
        self.o1(
            "matmul",
            vec![
                ("x".into(), bind(x).1),
                ("y".into(), bind(y).1),
                ("transpose_x".into(), bind(&tx).1),
                ("transpose_y".into(), bind(&tyy).1),
            ],
            name,
            vt,
        )
    }

    /// `softmax` on `axis`; `fp32` selects the output dtype.
    pub fn softmax(
        &mut self,
        x: &str,
        axis: i32,
        out_shape: &[i64],
        name: &str,
        fp32: bool,
    ) -> String {
        let a = self.fresh("axis");
        let a = self.konst_scalar_i32(&a, axis);
        let dt = if fp32 { DType::Fp32 } else { DType::Fp16 };
        let vt = self.tt(dt, out_shape);
        self.o1(
            "softmax",
            vec![("x".into(), bind(x).1), ("axis".into(), bind(&a).1)],
            name,
            vt,
        )
    }

    /// `cast` to the named dtype (`"fp32"`, `"fp16"`, `"int32"`).
    pub fn cast(
        &mut self,
        x: &str,
        to: &str,
        out_shape: &[i64],
        name: &str,
        out_f32: bool,
    ) -> String {
        let d = self.fresh("dtype");
        let d = {
            let vt = self.tt(DType::Str, &[]);
            self.op(
                "const",
                vec![],
                vec![(&d, vt)],
                vec![("val".into(), Value::Str(to.into()))],
            )[0]
            .clone()
        };
        let dt = if out_f32 { DType::Fp32 } else { DType::Fp16 };
        let vt = self.tt(dt, out_shape);
        self.o1(
            "cast",
            vec![("x".into(), bind(x).1), ("dtype".into(), bind(&d).1)],
            name,
            vt,
        )
    }

    /// RMSNorm (fp16-safe):
    /// `m = max(|x|) (floored); xs = x/m;`
    /// `out = xs * rsqrt(mean(xs^2, axis=-3 keepdims) + eps/m^2) * w`
    ///     `= x * rsqrt(mean(x^2) + eps) * w`
    ///
    /// `w` is a bound const name of shape `(d,1,1)`; `x` is `(1,d,1,1)` or
    /// `(n,d,1,1)`. The dynamic max-abs prescale keeps `xs^2 <= 1` — plain
    /// `mean(x^2)` overflows fp16 on large activations, and a fixed
    /// downscale underflows `x^2` to zero on small ones (embedding-scale
    /// inputs), which is worse: `rsqrt(0)` gives inf. `eps/m^2` is formed
    /// as `(sqrt(eps)/m)^2` since eps itself is an fp16 denormal.
    pub fn rms_norm(
        &mut self,
        x: &str,
        w: &str,
        _d: i64,
        eps: f32,
        shape4: &[i64],
        pfx: &str,
    ) -> String {
        const M_FLOOR: f32 = 3.90625e-3; // 2^-8
        let absx = {
            let vt = self.tt(DType::Fp16, shape4);
            self.o1(
                "abs",
                vec![("x".into(), bind(x).1)],
                &format!("{pfx}_abs"),
                vt,
            )
        };
        // mean/max over channel axis (dim=1 for 4D conv layout; dim=-3 general)
        let axes = self.fresh("axes");
        let axes = self.konst_i32(&axes, &[1]);
        let kd = self.fresh("kd");
        let kd = self.konst_bool(&kd, true);
        let mut mshape = shape4.to_vec();
        mshape[1] = 1;
        let m = {
            let vt = self.tt(DType::Fp16, &mshape);
            self.o1(
                "reduce_max",
                vec![
                    ("x".into(), bind(&absx).1),
                    ("axes".into(), bind(&axes).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &format!("{pfx}_m"),
                vt,
            )
        };
        let fl = self.konst_f16(&format!("{pfx}_fl"), M_FLOOR);
        let mc = {
            let vt = self.tt(DType::Fp16, &mshape);
            self.o1(
                "maximum",
                vec![("x".into(), bind(&m).1), ("y".into(), bind(&fl).1)],
                &format!("{pfx}_mc"),
                vt,
            )
        };
        let xs = {
            let vt = self.tt(DType::Fp16, shape4);
            self.o1(
                "real_div",
                vec![("x".into(), bind(x).1), ("y".into(), bind(&mc).1)],
                &format!("{pfx}_xs"),
                vt,
            )
        };
        let sq = self.mul(&xs, &xs, shape4, &format!("{pfx}_sq"));
        let mean = {
            let vt = self.tt(DType::Fp16, &mshape);
            self.o1(
                "reduce_mean",
                vec![
                    ("x".into(), bind(&sq).1),
                    ("axes".into(), bind(&axes).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &format!("{pfx}_mean"),
                vt,
            )
        };
        let esq = self.konst_f16(&format!("{pfx}_esq"), eps.sqrt());
        let t = {
            let vt = self.tt(DType::Fp16, &mshape);
            self.o1(
                "real_div",
                vec![("x".into(), bind(&esq).1), ("y".into(), bind(&mc).1)],
                &format!("{pfx}_t"),
                vt,
            )
        };
        let e2 = self.mul(&t, &t, &mshape, &format!("{pfx}_e2"));
        let vp = self.add(&mean, &e2, &mshape, &format!("{pfx}_var"));
        let eps_c = self.konst_f16(&format!("{pfx}_rse"), 0.0);
        let r = {
            let vt = self.tt(DType::Fp16, &mshape);
            self.o1(
                "rsqrt",
                vec![
                    ("x".into(), bind(&vp).1),
                    ("epsilon".into(), bind(&eps_c).1),
                ],
                &format!("{pfx}_rsq"),
                vt,
            )
        };
        let xn = self.mul(&xs, &r, shape4, &format!("{pfx}_xn"));
        self.mul(&xn, w, shape4, &format!("{pfx}_out"))
    }

    /// `read_state` — binds a state input declared in the model description.
    pub fn read_state(&mut self, state: &str, shape: &[i64], name: &str) -> String {
        let vt = self.tt(DType::Fp16, shape);
        self.op(
            "read_state",
            vec![("input".into(), bind(state).1)],
            vec![(name, vt)],
            vec![("name".into(), Value::Str(name.to_string()))],
        )[0]
        .clone()
    }

    /// `write_state` — appends `value` into a state (newest-first KV layout).
    pub fn write_state(&mut self, state: &str, value: &str) {
        self.op(
            "write_state",
            vec![
                ("input".into(), bind(state).1),
                ("data".into(), bind(value).1),
            ],
            vec![],
            vec![("name".into(), Value::Str(format!("{state}_write_state")))],
        );
    }
}

// ======== Model/package serialization ========

fn feature_desc_multi(name: &str, shape: &[i64], dt: DType) -> Vec<u8> {
    // FeatureDescription { name=1, type=3 { multiArrayType=5 { shape=1, dataType=2 } } }
    let mut aft = Vec::new();
    for &d in shape {
        f_i64(&mut aft, 1, d);
    }
    f_varint(&mut aft, 2, dt.array() as u64);
    let mut ft = Vec::new();
    f_msg(&mut ft, 5, &aft);
    let mut fd = Vec::new();
    f_str(&mut fd, 1, name);
    f_msg(&mut fd, 3, &ft);
    fd
}

fn feature_desc_state(name: &str, shape: &[i64], dt: DType) -> Vec<u8> {
    // FeatureDescription { name=1, type=3 { stateType=8 { arrayType=1 } } }
    let mut aft = Vec::new();
    for &d in shape {
        f_i64(&mut aft, 1, d);
    }
    f_varint(&mut aft, 2, dt.array() as u64);
    let mut sft = Vec::new();
    f_msg(&mut sft, 1, &aft);
    let mut ft = Vec::new();
    f_msg(&mut ft, 8, &sft);
    let mut fd = Vec::new();
    f_str(&mut fd, 1, name);
    f_msg(&mut fd, 3, &ft);
    fd
}

/// A model feature: input, output, or state declaration in the
/// `ModelDescription`.
pub struct Feature {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DType,
    /// `true` → emitted as a `stateType` feature (paired with `read_state`/
    /// `write_state`); `false` → `multiArrayType` input/output.
    pub is_state: bool,
}

/// Model-level metadata written into `ModelDescription` and the Function's
/// opset specialization.
///
/// `spec_version` and `opset` must match an installed CoreML tooling pair —
/// e.g. `(8, "CoreML5")`, `(9, "CoreML7")`, `(10, "CoreML9")`. The Function
/// block is specialized under the same opset name.
pub struct ModelMeta {
    /// `Model.specificationVersion` (protobuf field 1, int32).
    pub spec_version: i32,
    /// Opset specialization name (`"CoreML5"`, `"CoreML9"`, ...).
    pub opset: String,
    /// `metadata.creator` — what tool produced this model.
    pub creator: String,
    /// `metadata.shortDescription` — human-readable model description.
    pub description: String,
}

impl ModelMeta {
    /// Meta with crate-default creator/description.
    pub fn new(spec_version: i32, opset: &str) -> Self {
        ModelMeta {
            spec_version,
            opset: opset.to_string(),
            creator: "mil-spec".into(),
            description: "CoreML mlprogram written by mil-spec".into(),
        }
    }
    /// Builder: set `metadata.creator`.
    pub fn creator(mut self, c: &str) -> Self {
        self.creator = c.to_string();
        self
    }
    /// Builder: set `metadata.shortDescription`.
    pub fn description(mut self, d: &str) -> Self {
        self.description = d.to_string();
        self
    }
}

/// Serialize a `Block` + feature descriptions into a `model.mlmodel` spec
/// (the protobuf payload inside a `.mlpackage`).
///
/// `states` must mirror the state features the block reads/writes via
/// `read_state`/`write_state`. `fn_inputs` are the program's input
/// declarations (usually the same names/shapes as `inputs`).
pub fn encode_model(
    inputs: &[Feature],
    outputs: &[Feature],
    states: &[Feature],
    block: &Block,
    fn_inputs: &[NVT],
    meta: &ModelMeta,
) -> Vec<u8> {
    // Block
    let mut blk = Vec::new();
    // block.inputs: not needed (no block-local names)
    for out in &block.outputs {
        f_str(&mut blk, 2, out);
    }
    for op in &block.ops {
        f_msg(&mut blk, 3, &op.encode());
    }

    // Function { inputs=1, opset=2, block_specializations=3 map<string,Block> }
    let mut fnc = Vec::new();
    for nvt in fn_inputs {
        f_msg(&mut fnc, 1, &nvt.encode());
    }
    f_str(&mut fnc, 2, &meta.opset);
    map_entry_str(&mut fnc, 3, &meta.opset, &blk);

    // Program { version=1, functions=2 map }
    let mut prog = Vec::new();
    f_i64(&mut prog, 1, 1);
    map_entry_str(&mut prog, 2, "main", &fnc);

    // ModelDescription { input=1, output=10, state=13, metadata=100 }
    let mut desc = Vec::new();
    for f in inputs {
        f_msg(
            &mut desc,
            1,
            &feature_desc_multi(&f.name, &f.shape, f.dtype),
        );
    }
    for f in outputs {
        f_msg(
            &mut desc,
            10,
            &feature_desc_multi(&f.name, &f.shape, f.dtype),
        );
    }
    for f in states {
        f_msg(
            &mut desc,
            13,
            &feature_desc_state(&f.name, &f.shape, f.dtype),
        );
    }
    // metadata
    let mut meta_msg = Vec::new();
    f_str(&mut meta_msg, 1, &meta.description);
    f_str(&mut meta_msg, 2, "1.0");
    f_str(&mut meta_msg, 3, &meta.creator);
    f_msg(&mut desc, 100, &meta_msg);

    // Model { specVersion=1, description=2, mlProgram=502 }
    let mut mdl = Vec::new();
    f_i32(&mut mdl, 1, meta.spec_version);
    f_msg(&mut mdl, 2, &desc);
    f_msg(&mut mdl, 502, &prog);
    mdl
}

// ======== weight.bin v2 blob writer ========

const BLOB_ALIGN: u64 = 64;
const BLOB_SENTINEL: u32 = 0xDEADBEEF;

/// File-backed `weight.bin` writer (v2 format): 64-byte storage header,
/// then 64-byte-aligned `blob_metadata` records + tensor data.
///
/// Use [`BlobWriter::append`] and keep the returned metadata offset — it is
/// the `BlobFileValue.offset` referenced by [`Value::Blob`] /
/// [`Block::konst_blob`].
pub struct BlobWriter {
    file: std::fs::File,
    count: u32,
}

impl BlobWriter {
    /// Create a `weight.bin` and write its 64-byte storage header
    /// (`count` is patched in on `finish`).
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::create(path)?;
        // storage_header: count u32, version u32 (=2), reserved 7*u64 = 0 (64B total)
        let mut hdr = Vec::with_capacity(64);
        hdr.extend_from_slice(&0u32.to_le_bytes()); // count patched at close
        hdr.extend_from_slice(&2u32.to_le_bytes());
        hdr.resize(64, 0);
        file.write_all(&hdr)?;
        Ok(BlobWriter { file, count: 0 })
    }

    fn aligned(&mut self) -> std::io::Result<u64> {
        let pos = self.file.stream_position()?;
        Ok(if pos % BLOB_ALIGN == 0 {
            pos
        } else {
            pos + (BLOB_ALIGN - pos % BLOB_ALIGN)
        })
    }

    /// Append raw bytes as one blob. Returns the BlobFileValue offset
    /// (the position of this blob's metadata record).
    pub fn append(&mut self, dtype: DType, data: &[u8]) -> std::io::Result<u64> {
        let meta_off = self.aligned()?;
        if meta_off > 0 {
            self.file.seek(SeekFrom::Start(meta_off))?;
        }
        // blob_metadata (64B): sentinel u32, mil_dtype u32, sizeInBytes u64,
        // offset u64 (data offset), padding_bits u64, reserved 4*u64
        let data_off = meta_off + BLOB_ALIGN;
        let mut meta = Vec::with_capacity(64);
        meta.extend_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta.extend_from_slice(&dtype.blob().to_le_bytes());
        meta.extend_from_slice(&(data.len() as u64).to_le_bytes());
        meta.extend_from_slice(&data_off.to_le_bytes());
        meta.extend_from_slice(&0u64.to_le_bytes());
        meta.resize(64, 0);
        self.file.write_all(&meta)?;
        self.file.write_all(data)?;
        self.count += 1;
        Ok(meta_off)
    }

    /// Patch the blob count into the header and flush.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.count.to_le_bytes())?;
        self.file.flush()
    }
}

/// In-memory `weight.bin` builder — the same v2 layout as [`BlobWriter`],
/// returned as a byte vector for embedding into a `.mlpackage` via
/// [`write_mlpackage`].
///
/// Offsets returned by [`WeightBin::put`] are BlobFileValue offsets, exactly
/// like `BlobWriter::append`.
pub struct WeightBin {
    data: Vec<u8>,
    cursor: u64,
    count: u32,
}

impl Default for WeightBin {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightBin {
    /// New empty blob store with the 64-byte storage header reserved.
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&0u32.to_le_bytes()); // count patched in finish
        data.extend_from_slice(&2u32.to_le_bytes());
        data.resize(64, 0);
        WeightBin {
            data,
            cursor: 64,
            count: 0,
        }
    }

    /// Append tensor bytes; returns the blob metadata offset for
    /// BlobFileValue references.
    ///
    /// Pads to 64-byte alignment *before* each record (matching the
    /// file-backed [`BlobWriter`]), so a finished `weight.bin` ends exactly
    /// after the last blob's data with no trailing pad.
    pub fn put(&mut self, _name: &str, dtype: DType, _shape: &[i64], bytes: &[u8]) -> u64 {
        let pad = (BLOB_ALIGN - self.cursor % BLOB_ALIGN) % BLOB_ALIGN;
        self.data.resize(self.data.len() + pad as usize, 0);
        self.cursor += pad;
        let meta_off = self.cursor;
        let data_off = meta_off + BLOB_ALIGN;
        let mut meta = Vec::with_capacity(64);
        meta.extend_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta.extend_from_slice(&dtype.blob().to_le_bytes());
        meta.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        meta.extend_from_slice(&data_off.to_le_bytes());
        meta.extend_from_slice(&0u64.to_le_bytes());
        meta.resize(64, 0);
        self.data.extend_from_slice(&meta);
        self.data.extend_from_slice(bytes);
        self.cursor = self.data.len() as u64;
        self.count += 1;
        meta_off
    }

    /// Patch the blob count into the header and return the finished bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.data[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.data
    }
}

// ======== .mlpackage writer ========

/// Write a `.mlpackage` directory: `Manifest.json` +
/// `Data/com.apple.CoreML/model.mlmodel` +
/// `Data/com.apple.CoreML/weights/weight.bin`.
///
/// `weights` is a ready-serialized weight.bin byte vector (`None` → the
/// weights directory is omitted). Compile the result with
/// `xcrun coremlc compile <dir>.mlpackage .` to produce a `.mlmodelc`.
pub fn write_mlpackage(
    dir: &std::path::Path,
    spec: &[u8],
    weights: Option<&[u8]>,
) -> std::io::Result<()> {
    let data_dir = dir.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data_dir)?;
    std::fs::write(data_dir.join("model.mlmodel"), spec)?;

    let model_id = uuid_str();
    let mut entries = String::new();
    entries.push_str(&format!(
        "    \"{}\": {{\"path\": \"com.apple.CoreML/model.mlmodel\", \"author\": \"com.apple.CoreML\", \"name\": \"model.mlmodel\", \"description\": \"CoreML Model Specification\"}}",
        model_id
    ));
    if let Some(w) = weights {
        let wdir = data_dir.join("weights");
        std::fs::create_dir_all(&wdir)?;
        std::fs::write(wdir.join("weight.bin"), w)?;
        entries.push_str(&format!(
            ",\n    \"{}\": {{\"path\": \"com.apple.CoreML/weights\", \"author\": \"com.apple.CoreML\", \"name\": \"weights\", \"description\": \"CoreML Model Weights\"}}",
            uuid_str()
        ));
    }
    let manifest = format!(
        "{{\n  \"fileFormatVersion\": \"1.0.0\",\n  \"itemInfoEntries\": {{\n{}\n  }},\n  \"rootModelIdentifier\": \"{}\"\n}}\n",
        entries, model_id
    );
    std::fs::write(dir.join("Manifest.json"), manifest)
}

/// Like [`write_mlpackage`] but takes `weight.bin` as a file path and
/// copies it into the package — for models whose weights are too large
/// to keep in memory. `None` omits the weights directory.
pub fn write_mlpackage_stream(
    dir: &std::path::Path,
    spec: &[u8],
    weight_src: Option<&std::path::Path>,
) -> std::io::Result<()> {
    let data_dir = dir.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data_dir)?;
    std::fs::write(data_dir.join("model.mlmodel"), spec)?;

    let model_id = uuid_str();
    let mut entries = String::new();
    entries.push_str(&format!(
        "    \"{}\": {{\"path\": \"com.apple.CoreML/model.mlmodel\", \"author\": \"com.apple.CoreML\", \"name\": \"model.mlmodel\", \"description\": \"CoreML Model Specification\"}}",
        model_id
    ));
    if let Some(wsrc) = weight_src {
        let wdir = data_dir.join("weights");
        std::fs::create_dir_all(&wdir)?;
        if wsrc != wdir.join("weight.bin") {
            std::fs::copy(wsrc, wdir.join("weight.bin"))?;
        }
        entries.push_str(&format!(
            ",\n    \"{}\": {{\"path\": \"com.apple.CoreML/weights\", \"author\": \"com.apple.CoreML\", \"name\": \"weights\", \"description\": \"CoreML Model Weights\"}}",
            uuid_str()
        ));
    }
    let manifest = format!(
        "{{\n  \"fileFormatVersion\": \"1.0.0\",\n  \"itemInfoEntries\": {{\n{}\n  }},\n  \"rootModelIdentifier\": \"{}\"\n}}\n",
        entries, model_id
    );
    std::fs::write(dir.join("Manifest.json"), manifest)
}

fn uuid_str() -> String {
    // RFC4122 v4 UUID, uppercased — matches Apple's package identifiers.
    let mut b = [0u8; 16];
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0) as u64;
    let mut s = seed ^ 0x9E3779B97F4A7C15;
    for x in b.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = (s & 0xff) as u8;
    }
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format!(
        "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_pkg(
        dir: &std::path::Path,
        blk: &Block,
        fn_inputs: &[NVT],
        outs: &[Feature],
        weights: Option<&[u8]>,
    ) {
        let inputs = [Feature {
            name: "x".into(),
            shape: vec![1, 4, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let spec = encode_model(
            &inputs,
            outs,
            &[],
            blk,
            fn_inputs,
            &ModelMeta::new(8, "CoreML5"),
        );
        write_mlpackage(dir, &spec, weights).unwrap();
        assert!(dir.join("Manifest.json").exists());
        assert!(dir.join("Data/com.apple.CoreML/model.mlmodel").exists());
    }

    #[test]
    fn probe_packages() {
        let root = std::env::temp_dir().join(format!("mil_probe_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // p1: y = add(x, x) — zero consts
        let mut b = Block::new();
        let y = b.add("x", "x", &[1, 4, 1, 1], "y");
        b.outputs = vec![y.clone()];
        let fin = vec![NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])),
        }];
        let outs = [Feature {
            name: "y".into(),
            shape: vec![1, 4, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        }];
        probe_pkg(&root.join("p1.mlpackage"), &b, &fin, &outs, None);

        // p2: + i32 vector const feeding reshape
        let mut b = Block::new();
        let s = b.konst_i32("shape_c", &[1, 4]);
        let y = {
            let vt = ValueType::Tensor(TensorType::f16(&[1, 4]));
            b.o1(
                "reshape",
                vec![("x".into(), bind("x").1), ("shape".into(), bind(&s).1)],
                "y",
                vt,
            )
        };
        b.outputs = vec![y];
        let outs2 = [Feature {
            name: "y".into(),
            shape: vec![1, 4],
            dtype: DType::Fp16,
            is_state: false,
        }];
        probe_pkg(&root.join("p2.mlpackage"), &b, &fin, &outs2, None);

        // p3: + f16 scalar const feeding mul
        let mut b = Block::new();
        let k = b.konst_f16("k", 2.0);
        let y = b.mul("x", &k, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        probe_pkg(&root.join("p3.mlpackage"), &b, &fin, &outs, None);

        // p4: + blob const feeding mul (4x4 fp16 weight)
        let wpath = root.join("w.bin");
        let mut w = BlobWriter::create(&wpath).unwrap();
        let wdata: Vec<u8> = (0..16)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();
        let off = w.append(DType::Fp16, &wdata).unwrap();
        w.finish().unwrap();
        let wbin = std::fs::read(&wpath).unwrap();
        let mut b = Block::new();
        let wb = b.konst_blob(
            "w",
            "@model_path/weights/weight.bin",
            off,
            DType::Fp16,
            &[4, 4, 1, 1],
        );
        let y = b.mul("x", &wb, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        probe_pkg(&root.join("p4.mlpackage"), &b, &fin, &outs, Some(&wbin));

        // p5: str const (cast dtype arg)
        let mut b = Block::new();
        let y = b.cast("x", "fp32", &[1, 4, 1, 1], "y", true);
        b.outputs = vec![y];
        let outs5 = [Feature {
            name: "y".into(),
            shape: vec![1, 4, 1, 1],
            dtype: DType::Fp32,
            is_state: false,
        }];
        probe_pkg(&root.join("p5.mlpackage"), &b, &fin, &outs5, None);

        // p6: i32 SCALAR const (rank 0) feeding softmax axis
        let mut b = Block::new();
        let ax = b.konst_scalar_i32("ax", 1);
        let y = {
            let vt = ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1]));
            b.o1(
                "softmax",
                vec![("x".into(), bind("x").1), ("axis".into(), bind(&ax).1)],
                "y",
                vt,
            )
        };
        b.outputs = vec![y];
        probe_pkg(&root.join("p6.mlpackage"), &b, &fin, &outs, None);

        // p7: fp16 VECTOR const [4] feeding mul (rank 1, fp16 storage test)
        let mut b = Block::new();
        let k = b.op(
            "const",
            vec![],
            vec![("k", ValueType::Tensor(TensorType::f16(&[4])))],
            vec![("val".into(), Value::f16s(&[4], &[1.0, 2.0, 3.0, 4.0]))],
        )[0]
        .clone();
        let y = b.mul("x", &k, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        probe_pkg(&root.join("p7.mlpackage"), &b, &fin, &outs, None);

        // p8: fp16 vector via BYTES storage (raw fp16 LE)
        let mut b = Block::new();
        let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let k = b.op(
            "const",
            vec![],
            vec![("k", ValueType::Tensor(TensorType::f16(&[4])))],
            vec![(
                "val".into(),
                Value::Bytes(ValueType::Tensor(TensorType::f16(&[4])), raw),
            )],
        )[0]
        .clone();
        let y = b.mul("x", &k, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        probe_pkg(&root.join("p8.mlpackage"), &b, &fin, &outs, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn weight_bin_layout() {
        // in-memory and file-backed writers must produce identical bytes
        let wdata: Vec<u8> = (0..16)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();

        let mut mem = WeightBin::new();
        let mem_off = mem.put("w", DType::Fp16, &[4, 4], &wdata);
        let mem_bytes = mem.finish();

        let root = std::env::temp_dir().join(format!("mil_wb_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let wpath = root.join("w.bin");
        let mut f = BlobWriter::create(&wpath).unwrap();
        let file_off = f.append(DType::Fp16, &wdata).unwrap();
        f.finish().unwrap();
        let file_bytes = std::fs::read(&wpath).unwrap();

        assert_eq!(mem_off, file_off);
        assert_eq!(mem_bytes, file_bytes);
        let _ = std::fs::remove_dir_all(&root);
    }
}
