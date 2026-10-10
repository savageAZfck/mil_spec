//! `mil_onnx` — ONNX → MIL frontend and sklearn-style classical-ML
//! conversion, in pure Rust. The `coremltools.convert(onnx_model)` and
//! `coremltools.converters.sklearn` replacements.
//!
//! # ONNX path
//!
//! [`ModelProto::decode`] reads the ONNX proto2 wire format with a
//! hand-rolled decoder ([`proto`]) — no prost, no generated code.
//! [`convert`] then lowers the graph onto [`mil_spec::Block`] ops with
//! fully static shape inference, streams large initializers through the
//! `weight.bin` v2 blob format, and returns a [`BuiltModel`] that writes
//! a compile-ready `.mlpackage`.
//!
//! All float tensors lower to **fp16** — the dtype the whole `mil_spec`
//! toolchain emits and the one the ANE wants. Integer tensors lower to
//! int32, bools to bool. Dynamic shapes (`dim_param` / unspecified dims)
//! are rejected: MIL program specialization needs concrete dims, the
//! same contract `encode_model` documents.
//!
//! # Classical ML path
//!
//! `coremltools` converts sklearn trees/GLMs by reading pickles — which
//! needs Python. [`classical`] instead defines a small JSON schema
//! (documented in [`classical`]'s module docs) for
//! tree ensembles and generalized linear models, and lowers them to
//! MIL *programs* — the same `.mlpackage` `mlprogram` path the ONNX
//! frontend emits, verified end-to-end through `coremlc`:
//!
//! - **GLM** → `linear` + post-eval (`sigmoid` / `softmax` / `probit`
//!   via `erf`), plus an `argmax`→`gather` label output for classifiers.
//! - **Tree ensemble** → a *violation-matrix* decomposition: every leaf
//!   is a conjunction of node predicates, so
//!   `off_l + le·Mle[l,:] + lt·Mlt[l,:] == 0` selects leaf `l` — two
//!   `matmul`s, an `equal`, and a value `matmul` produce the ensemble
//!   score. Exact for axis-aligned trees — no approximation.
//!
//! `ai.onnx.ml` nodes (`TreeEnsembleClassifier`/`Regressor`,
//! `LinearClassifier`/`Regressor`, `Normalizer`, `Scaler`,
//! `OneHotEncoder`, `LabelEncoder`/`Imputer` unmapped) inside ONNX
//! graphs route through the same lowering — the attributes *are* the
//! sklearn parameters, so skl2onnx exports work end-to-end.
//!
//! # Example
//!
//! ```no_run
//! let onnx_bytes = std::fs::read("model.onnx").unwrap();
//! let built = mil_onnx::convert_bytes(&onnx_bytes).unwrap();
//! built.write_mlpackage(std::path::Path::new("model.mlpackage")).unwrap();
//! // xcrun coremlc compile model.mlpackage .
//! ```

#![forbid(unsafe_code)]

pub mod classical;
pub mod map;
pub mod model;
pub mod proto;

pub use map::{ConvertOptions, INLINE_MAX_BYTES};
pub use model::{AttributeProto, Dim, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfo};

use mil_spec::{Block, Feature, ModelMeta, NVT};
use std::fmt;
use std::path::Path;

/// Conversion failure — every variant names *what* rejected and *why*.
#[derive(Debug)]
pub enum OnnxError {
    /// Byte stream isn't well-formed protobuf.
    Malformed {
        /// Message/context being decoded.
        what: String,
        /// Detail (offset, field, expected).
        detail: String,
    },
    /// Valid ONNX, but a feature this frontend doesn't map.
    Unsupported(String),
    /// A node uses an op with no mapping. Carries op_type + node name.
    UnknownOp {
        /// `op_type` string.
        op: String,
        /// Node name (or synthesized label).
        node: String,
        /// Domain (`""`, `"ai.onnx.ml"`, custom).
        domain: String,
    },
    /// A node references a value that is neither an input, an
    /// initializer, nor a previously-produced tensor.
    MissingInput {
        /// The referenced value name.
        name: String,
        /// The consuming node.
        node: String,
    },
    /// An input needed statically (shape/axes/starts…) is produced at
    /// runtime — the frontend is static-shape only.
    Dynamic(String),
    /// Declared shape/type is inconsistent.
    BadShape(String),
    /// Classical-ML JSON schema violation.
    Schema(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::Malformed { what, detail } => {
                write!(f, "malformed {what}: {detail}")
            }
            OnnxError::Unsupported(m) => write!(f, "unsupported: {m}"),
            OnnxError::UnknownOp { op, node, domain } => {
                let dom = if domain.is_empty() { "ai.onnx" } else { domain };
                write!(f, "no mapping for op {dom}::{op} (node '{node}')")
            }
            OnnxError::MissingInput { name, node } => {
                write!(f, "node '{node}' references unknown tensor '{name}'")
            }
            OnnxError::Dynamic(m) => write!(f, "dynamic value: {m}"),
            OnnxError::BadShape(m) => write!(f, "bad shape: {m}"),
            OnnxError::Schema(m) => write!(f, "classical-ml schema: {m}"),
        }
    }
}
impl std::error::Error for OnnxError {}

/// A converted graph — everything [`mil_spec::encode_model`] needs.
pub struct BuiltModel {
    /// The MIL block.
    pub block: Block,
    /// Model input features.
    pub inputs: Vec<Feature>,
    /// Model output features.
    pub outputs: Vec<Feature>,
    /// Function input declarations.
    pub fn_inputs: Vec<NVT>,
    /// Serialized `weight.bin` (None = no blob-backed weights).
    pub weight_bin: Option<Vec<u8>>,
    /// Ops that were emitted, by type — for coverage reporting.
    pub op_histogram: std::collections::BTreeMap<String, usize>,
}

impl BuiltModel {
    /// Encode + write a `.mlpackage` directory.
    pub fn write_mlpackage(&self, dir: &Path) -> std::io::Result<()> {
        let meta = ModelMeta::new(10, "CoreML9")
            .creator("mil_onnx")
            .description("converted by mil_onnx");
        let spec = mil_spec::encode_model(
            &self.inputs,
            &self.outputs,
            &[],
            &self.block,
            &self.fn_inputs,
            &meta,
        );
        mil_spec::write_mlpackage(dir, &spec, self.weight_bin.as_deref())
    }
}

/// Decode + convert a serialized ONNX `ModelProto`.
pub fn convert_bytes(bytes: &[u8]) -> Result<BuiltModel, OnnxError> {
    let m = ModelProto::decode(bytes)?;
    convert(&m)
}

/// Convert a decoded [`ModelProto`] into a [`BuiltModel`].
pub fn convert(model: &ModelProto) -> Result<BuiltModel, OnnxError> {
    map::convert_graph(model, &ConvertOptions::default())
}

/// [`convert`] with explicit options.
pub fn convert_with(model: &ModelProto, opts: &ConvertOptions) -> Result<BuiltModel, OnnxError> {
    map::convert_graph(model, opts)
}

/// Convert a classical-ML JSON description (schema documented in
/// [`classical`]) into a [`BuiltModel`].
pub fn convert_classical_json(bytes: &[u8]) -> Result<BuiltModel, OnnxError> {
    classical::convert_json(bytes)
}
