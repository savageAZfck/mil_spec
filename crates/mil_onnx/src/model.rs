//! ONNX `ModelProto` reader — proto2 schema, decoded by hand over
//! [`crate::proto`]. No prost, no generated code.
//!
//! Field numbers follow `onnx/onnx.proto3` (the serialized form is the
//! same for proto2 and proto3). Only the fields a converter needs are
//! surfaced; everything else is decoded at the wire layer and ignored.

use crate::proto::{self, PMsg, WireVal};
use crate::OnnxError;
use std::collections::HashMap;

// ---------- ONNX enums (wire constants) ----------

/// `TensorProto.DataType`.
pub mod elem {
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

fn elem_name(t: i32) -> &'static str {
    match t {
        0 => "UNDEFINED",
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        16 => "BFLOAT16",
        _ => "unknown",
    }
}

// ---------- decoded model ----------

/// A tensor shape dimension.
#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    /// Concrete size.
    Value(i64),
    /// Named symbolic dim (`dim_param`) — not convertible.
    Param(String),
    /// Unspecified dim — not convertible.
    Unknown,
}

/// `ValueInfoProto`: name + optional tensor type.
#[derive(Clone, Debug)]
pub struct ValueInfo {
    /// Value name.
    pub name: String,
    /// `TensorProto.DataType` element type (`0` = undeclared).
    pub elem_type: i32,
    /// Shape if declared (`None` for non-tensor/unspecified types).
    pub shape: Option<Vec<Dim>>,
}

/// `TensorProto`.
#[derive(Clone, Debug, Default)]
pub struct TensorProto {
    /// Dimensions.
    pub dims: Vec<i64>,
    /// `TensorProto.DataType`.
    pub data_type: i32,
    /// Tensor name.
    pub name: String,
    /// `raw_data` payload (little-endian).
    pub raw: Option<Vec<u8>>,
    /// `float_data`.
    pub float_data: Vec<f32>,
    /// `int32_data` (also packed bool/uint8/int16 payloads).
    pub int32_data: Vec<i32>,
    /// `int64_data`.
    pub int64_data: Vec<i64>,
    /// `double_data`.
    pub double_data: Vec<f64>,
    /// `string_data`.
    pub string_data: Vec<Vec<u8>>,
    /// `uint64_data`.
    pub uint64_data: Vec<u64>,
    /// `data_location == EXTERNAL`.
    pub external: bool,
}

impl TensorProto {
    /// Element count implied by `dims`.
    pub fn count(&self) -> usize {
        self.dims.iter().map(|d| (*d).max(0) as usize).product()
    }

    fn no_data(&self) -> OnnxError {
        OnnxError::Malformed {
            what: format!("tensor '{}'", self.name),
            detail: format!(
                "no data payload for {} elems of type {}",
                self.count(),
                elem_name(self.data_type)
            ),
        }
    }

    /// Elements as f32 (FLOAT / FLOAT16 / DOUBLE / int types widened).
    pub fn as_f32(&self) -> Result<Vec<f32>, OnnxError> {
        match self.data_type {
            elem::FLOAT => {
                if let Some(raw) = &self.raw {
                    if raw.len() % 4 != 0 {
                        return Err(self.no_data());
                    }
                    Ok(raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect())
                } else {
                    Ok(self.float_data.clone())
                }
            }
            elem::FLOAT16 => {
                if let Some(raw) = &self.raw {
                    if raw.len() % 2 != 0 {
                        return Err(self.no_data());
                    }
                    Ok(raw
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect())
                } else {
                    // FLOAT16 also travels in int32_data (raw u16 bits).
                    Ok(self
                        .int32_data
                        .iter()
                        .map(|&v| half::f16::from_bits(v as u16).to_f32())
                        .collect())
                }
            }
            elem::DOUBLE => {
                if let Some(raw) = &self.raw {
                    if raw.len() % 8 != 0 {
                        return Err(self.no_data());
                    }
                    Ok(raw
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect())
                } else {
                    Ok(self.double_data.iter().map(|&v| v as f32).collect())
                }
            }
            elem::INT64 => Ok(self.as_i64()?.iter().map(|&v| v as f32).collect()),
            elem::INT32
            | elem::UINT8
            | elem::INT8
            | elem::UINT16
            | elem::INT16
            | elem::UINT32
            | elem::BOOL => Ok(self.as_i64()?.iter().map(|&v| v as f32).collect()),
            t => Err(OnnxError::Unsupported(format!(
                "tensor '{}': dtype {} cannot be read as f32",
                self.name,
                elem_name(t)
            ))),
        }
    }

    /// Elements as i64 (integer/bool tensors).
    pub fn as_i64(&self) -> Result<Vec<i64>, OnnxError> {
        let from_raw = |width: usize, signed: bool| -> Result<Vec<i64>, OnnxError> {
            let raw = self.raw.as_ref().unwrap();
            if raw.len() % width != 0 {
                return Err(self.no_data());
            }
            Ok(raw
                .chunks_exact(width)
                .map(|c| {
                    let mut b8 = [0u8; 8];
                    b8[..width].copy_from_slice(c);
                    let v = u64::from_le_bytes(b8);
                    if signed {
                        // sign-extend the low `width` bytes
                        let sh = 64 - (width as u32 * 8);
                        ((v << sh) as i64) >> sh
                    } else {
                        v as i64
                    }
                })
                .collect())
        };
        match self.data_type {
            elem::INT64 => {
                if self.raw.is_some() {
                    from_raw(8, true)
                } else {
                    Ok(self.int64_data.clone())
                }
            }
            elem::INT32
            | elem::UINT8
            | elem::INT8
            | elem::UINT16
            | elem::INT16
            | elem::UINT32
            | elem::BOOL => {
                if self.raw.is_some() {
                    let w = match self.data_type {
                        elem::INT32 | elem::UINT32 => 4,
                        elem::INT16 | elem::UINT16 => 2,
                        _ => 1,
                    };
                    let signed = matches!(self.data_type, elem::INT32 | elem::INT16 | elem::INT8);
                    from_raw(w, signed)
                } else {
                    Ok(self.int32_data.iter().map(|&v| v as i64).collect())
                }
            }
            elem::UINT64 => {
                if self.raw.is_some() {
                    from_raw(8, false)
                } else {
                    Ok(self.uint64_data.iter().map(|&v| v as i64).collect())
                }
            }
            t => Err(OnnxError::Unsupported(format!(
                "tensor '{}': dtype {} cannot be read as int",
                self.name,
                elem_name(t)
            ))),
        }
    }
}

/// `AttributeProto` — scalar/repeated fields as decoded.
#[derive(Clone, Debug, Default)]
pub struct AttributeProto {
    /// Attribute name.
    pub name: String,
    /// `f` scalar float.
    pub f: Option<f32>,
    /// `i` scalar int.
    pub i: Option<i64>,
    /// `s` scalar bytes/string.
    pub s: Option<Vec<u8>>,
    /// `t` embedded tensor.
    pub t: Option<TensorProto>,
    /// `floats` list.
    pub floats: Vec<f32>,
    /// `ints` list.
    pub ints: Vec<i64>,
    /// `strings` list.
    pub strings: Vec<Vec<u8>>,
}

/// `NodeProto`.
#[derive(Clone, Debug, Default)]
pub struct NodeProto {
    /// Input value names (`""` = absent optional input).
    pub inputs: Vec<String>,
    /// Output value names.
    pub outputs: Vec<String>,
    /// Node name (may be empty).
    pub name: String,
    /// Op type (`"Relu"`, `"Conv"`, ...).
    pub op_type: String,
    /// Domain (`""`/`"ai.onnx"`, `"ai.onnx.ml"`, ...).
    pub domain: String,
    /// Attributes by name.
    pub attrs: HashMap<String, AttributeProto>,
}

impl NodeProto {
    /// int attribute or default.
    pub fn attr_i(&self, name: &str, default: i64) -> i64 {
        self.attrs.get(name).and_then(|a| a.i).unwrap_or(default)
    }
    /// float attribute or default.
    pub fn attr_f(&self, name: &str, default: f32) -> f32 {
        self.attrs.get(name).and_then(|a| a.f).unwrap_or(default)
    }
    /// int-list attribute or empty.
    pub fn attr_ints(&self, name: &str) -> Vec<i64> {
        self.attrs
            .get(name)
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    }
    /// float-list attribute or empty.
    pub fn attr_floats(&self, name: &str) -> Vec<f32> {
        self.attrs
            .get(name)
            .map(|a| a.floats.clone())
            .unwrap_or_default()
    }
    /// string attribute (lossy) or None.
    pub fn attr_str(&self, name: &str) -> Option<String> {
        self.attrs
            .get(name)
            .and_then(|a| a.s.as_ref())
            .map(|s| String::from_utf8_lossy(s).into_owned())
    }
    /// tensor attribute.
    pub fn attr_t(&self, name: &str) -> Option<&TensorProto> {
        self.attrs.get(name).and_then(|a| a.t.as_ref())
    }
    /// The node name or a synthesized `op_type@index` label for errors.
    pub fn label(&self, idx: usize) -> String {
        if self.name.is_empty() {
            format!("{}#{}", self.op_type, idx)
        } else {
            self.name.clone()
        }
    }
}

/// `GraphProto`.
#[derive(Clone, Debug, Default)]
pub struct GraphProto {
    /// Nodes in topological order.
    pub nodes: Vec<NodeProto>,
    /// Graph name.
    pub name: String,
    /// Initializers (weights).
    pub initializers: Vec<TensorProto>,
    /// Declared graph inputs.
    pub inputs: Vec<ValueInfo>,
    /// Declared graph outputs.
    pub outputs: Vec<ValueInfo>,
    /// `value_info` (intermediate shapes, often present on exports).
    pub value_info: Vec<ValueInfo>,
}

/// `ModelProto`.
#[derive(Clone, Debug, Default)]
pub struct ModelProto {
    /// `ir_version`.
    pub ir_version: i64,
    /// Producer string.
    pub producer: String,
    /// `opset_import` — `(domain, version)` pairs.
    pub opsets: Vec<(String, i64)>,
    /// The graph.
    pub graph: GraphProto,
}

impl ModelProto {
    /// Opset version for `domain` (`""` or `"ai.onnx"` → the standard
    /// domain). Absent → 1 (most conservative defaults).
    pub fn opset(&self, domain: &str) -> i64 {
        let d = if domain == "ai.onnx" { "" } else { domain };
        self.opsets
            .iter()
            .find(|(dom, _)| dom == d || (d.is_empty() && dom == "ai.onnx"))
            .map(|(_, v)| *v)
            .unwrap_or(1)
    }

    /// Parse a serialized `ModelProto`.
    pub fn decode(bytes: &[u8]) -> Result<ModelProto, OnnxError> {
        let m = proto::decode(bytes).map_err(|e| OnnxError::Malformed {
            what: "ModelProto".into(),
            detail: format!("{e}"),
        })?;
        let graph = m.msg(7).ok_or_else(|| OnnxError::Malformed {
            what: "ModelProto".into(),
            detail: "missing graph (field 7)".into(),
        })?;
        let mut opsets = Vec::new();
        for om in m.msgs(8) {
            opsets.push((om.string(1).unwrap_or_default(), om.i64(2).unwrap_or(0)));
        }
        Ok(ModelProto {
            ir_version: m.i64(1).unwrap_or(0),
            producer: m.string(2).unwrap_or_default(),
            opsets,
            graph: GraphProto::decode(&graph)?,
        })
    }
}

impl GraphProto {
    fn decode(m: &PMsg) -> Result<GraphProto, OnnxError> {
        let mut g = GraphProto {
            name: m.string(2).unwrap_or_default(),
            ..Default::default()
        };
        for nm in m.msgs(1) {
            g.nodes.push(NodeProto::decode(&nm)?);
        }
        for tm in m.msgs(5) {
            g.initializers.push(TensorProto::decode(&tm)?);
        }
        for vm in m.msgs(11) {
            g.inputs.push(ValueInfo::decode(&vm));
        }
        for vm in m.msgs(12) {
            g.outputs.push(ValueInfo::decode(&vm));
        }
        for vm in m.msgs(13) {
            g.value_info.push(ValueInfo::decode(&vm));
        }
        Ok(g)
    }
}

impl NodeProto {
    fn decode(m: &PMsg) -> Result<NodeProto, OnnxError> {
        let mut n = NodeProto {
            name: m.string(3).unwrap_or_default(),
            op_type: m.string(4).unwrap_or_default(),
            domain: m.string(7).unwrap_or_default(),
            ..Default::default()
        };
        for v in m.all(1) {
            if let WireVal::Len(b) = v {
                n.inputs.push(String::from_utf8_lossy(b).into_owned());
            }
        }
        for v in m.all(2) {
            if let WireVal::Len(b) = v {
                n.outputs.push(String::from_utf8_lossy(b).into_owned());
            }
        }
        if n.op_type.is_empty() {
            return Err(OnnxError::Malformed {
                what: "NodeProto".into(),
                detail: format!("node '{}' has no op_type", n.name),
            });
        }
        for am in m.msgs(5) {
            let a = AttributeProto::decode(&am)?;
            n.attrs.insert(a.name.clone(), a);
        }
        Ok(n)
    }
}

impl AttributeProto {
    fn decode(m: &PMsg) -> Result<AttributeProto, OnnxError> {
        Ok(AttributeProto {
            name: m.string(1).unwrap_or_default(),
            f: m.f32(2),
            i: m.i64(3),
            s: m.bytes(4).map(|b| b.to_vec()),
            t: match m.msg(5) {
                Some(tm) => Some(TensorProto::decode(&tm)?),
                None => None,
            },
            floats: proto::f32s(m.all(7).cloned())?,
            ints: proto::varints(m.all(8).cloned())?,
            strings: m
                .all(9)
                .filter_map(|v| match v {
                    WireVal::Len(b) => Some(b.clone()),
                    _ => None,
                })
                .collect(),
        })
    }
}

impl TensorProto {
    fn decode(m: &PMsg) -> Result<TensorProto, OnnxError> {
        let mut t = TensorProto {
            data_type: m.varint(2).map(|v| v as i32).unwrap_or(0),
            name: m.string(8).unwrap_or_default(),
            raw: m.bytes(9).map(|b| b.to_vec()),
            ..Default::default()
        };
        for v in proto::varints(m.all(1).cloned())? {
            t.dims.push(v);
        }
        t.float_data = proto::f32s(m.all(4).cloned())?;
        t.int32_data = proto::varints(m.all(5).cloned())?
            .into_iter()
            .map(|v| v as i32)
            .collect();
        for v in m.all(6) {
            if let WireVal::Len(b) = v {
                t.string_data.push(b.clone());
            }
        }
        t.int64_data = proto::varints(m.all(7).cloned())?;
        t.double_data = proto::f64s(m.all(10).cloned())?;
        t.uint64_data = proto::varints(m.all(11).cloned())?
            .into_iter()
            .map(|v| v as u64)
            .collect();
        t.external = m.varint(14) == Some(1) || m.get(13).is_some();
        Ok(t)
    }
}

impl ValueInfo {
    fn decode(m: &PMsg) -> ValueInfo {
        // TypeProto.tensor_type = 1 { elem_type = 1, shape = 2 { dim = 1 } }
        let (elem_type, shape) = match m.msg(2).and_then(|t| t.msg(1)) {
            Some(tt) => {
                let et = tt.varint(1).map(|v| v as i32).unwrap_or(0);
                let sh = tt.msg(2).map(|s| {
                    s.msgs(1)
                        .iter()
                        .map(|d| {
                            if let Some(v) = d.i64(1) {
                                Dim::Value(v)
                            } else if let Some(p) = d.string(2) {
                                Dim::Param(p)
                            } else {
                                Dim::Unknown
                            }
                        })
                        .collect::<Vec<_>>()
                });
                (et, sh)
            }
            None => (0, None),
        };
        ValueInfo {
            name: m.string(1).unwrap_or_default(),
            elem_type,
            shape,
        }
    }
}
