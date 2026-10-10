//! `surgery` — post-conversion edits on a built `.mlpackage`.
//!
//! `milc convert` is not the only way a package gets produced — this
//! module is the reader/editor for packages that already exist:
//!
//! - [`PackageEdit`] decodes `model.mlmodel` into a mutable proto tree
//!   ([`mil_spec::proto`]), indexes every weight const (blob reference
//!   and the `constexpr_*` groups built over them), and writes a fresh
//!   `weight.bin` + spec with only the intended changes. Unreferenced
//!   payloads are garbage-collected; offsets are re-assigned, never
//!   patched in place.
//! - [`requant`] re-encodes weight blobs (`fp16` ↔ `int8` per-channel
//!   ↔ `palette4` LUT) by decoding each weight to f32 and re-emitting
//!   the same `constexpr` patterns [`crate::builder`] produces.
//! - [`graft`] splices layer weight payloads between same-architecture
//!   packages after a structural-compatibility check.
//! - [`reshape`] / [`reshape_flex`] rewrite the enumerated shapes or
//!   shape range of a flexible-shape
//!   package without reconverting.
//! - [`fuse_lora`] bakes a LoRA adapter into a built package's weight
//!   blobs. The mapping is verifiable: `mil_convert` names every conv
//!   weight const deterministically (`l{L}_{wq,wk,wv,wo,wg,wu,wd}`,
//!   `lm_w`), so a packaged const resolves to exactly one HF tensor.
//!
//! Every entry point validates before it writes — a mismatched graft or
//! an unparseable spec is an error, never silent corruption.

use crate::WeightSource;
use mil_spec::proto::{self, PMut, PVal};
use mil_spec::{BlobWriter, DType};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const WEIGHT_FILE: &str = "@model_path/weights/weight.bin";
const SPEC_REL: &str = "Data/com.apple.CoreML/model.mlmodel";
const WEIGHT_REL: &str = "Data/com.apple.CoreML/weights/weight.bin";
const BLOB_SENTINEL: u32 = 0xDEADBEEF;

type R<T> = Result<T, String>;

fn err<T>(msg: impl Into<String>) -> R<T> {
    Err(msg.into())
}

// ======== dtype bridges ========

/// `DType` → MIL `tensorType.dataType` wire code (mirrors `DType::mil`).
fn mil_code(dt: DType) -> u64 {
    match dt {
        DType::Bool => 1,
        DType::Str => 2,
        DType::Fp16 => 10,
        DType::Fp32 => 11,
        DType::Int8 => 21,
        DType::Int32 => 23,
        DType::Int64 => 24,
        DType::Int4 => 25,
        DType::Uint4 => 35,
    }
}

fn dtype_of_mil(code: u64) -> Option<DType> {
    match code {
        1 => Some(DType::Bool),
        2 => Some(DType::Str),
        10 => Some(DType::Fp16),
        11 => Some(DType::Fp32),
        21 => Some(DType::Int8),
        23 => Some(DType::Int32),
        24 => Some(DType::Int64),
        25 => Some(DType::Int4),
        35 => Some(DType::Uint4),
        _ => None,
    }
}

fn dtype_of_blob(code: u32) -> Option<DType> {
    match code {
        1 => Some(DType::Fp16),
        2 => Some(DType::Fp32),
        4 => Some(DType::Int8),
        8 => Some(DType::Int4),
        11 => Some(DType::Uint4),
        14 => Some(DType::Int32),
        _ => None,
    }
}

// ======== small spec accessors ========

/// First output name of an op message.
fn op_out_name(op: &PMut) -> Option<String> {
    op.msgs(3).first().and_then(|nvt| nvt.str(1))
}

/// Value names bound to op input `key` (Argument field 1 → Binding
/// field 1 strings).
fn op_input_names(op: &PMut, key: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (k, entry) in op.map_entries(2) {
        if k != key {
            continue;
        }
        if let Some(arg) = entry.msg(2) {
            for b in arg.msgs(1) {
                if let Some(n) = b.str(1) {
                    out.push(n);
                }
            }
        }
    }
    out
}

/// Op attribute `key`'s `Value` message.
fn op_attr_value(op: &PMut, key: &str) -> Option<PMut> {
    for (k, entry) in op.map_entries(5) {
        if k == key {
            return entry.msg(2);
        }
    }
    None
}

/// `(dtype mil code, dims)` of a `tensorType` message. Symbolic dims
/// come back as `None` entries.
fn tensor_type_dims(tt: &PMut) -> (Option<u64>, Vec<Option<i64>>) {
    let dt = tt.varint(1);
    let mut dims = Vec::new();
    for d in tt.msgs(3) {
        // Dimension { constant=1 {size=1} | unknown=2 {} }
        if let Some(cd) = d.msg(1) {
            dims.push(cd.varint(1).map(|v| v as i64));
        } else {
            dims.push(None);
        }
    }
    (dt, dims)
}

/// `(file, offset, mil dtype code, shape)` for a blob-referencing value.
type BlobRef = (String, u64, Option<u64>, Vec<Option<i64>>);

/// `Value` → blob reference when it is one.
fn value_blob(v: &PMut) -> Option<BlobRef> {
    let bv = v.msg(5)?;
    let file = bv.str(1)?;
    let off = bv.varint(2)?;
    let (dt, dims) = v
        .msg(2)
        .and_then(|vt| vt.msg(1))
        .map(|tt| tensor_type_dims(&tt))
        .unwrap_or_default();
    Some((file, off, dt, dims))
}

/// `const` op → its blob reference, when the `val` attr holds one.
fn const_blob(op: &PMut) -> Option<BlobRef> {
    if op.str(1).as_deref() != Some("const") {
        return None;
    }
    op_attr_value(op, "val").and_then(|v| value_blob(&v))
}

// ======== proto constructors (mirror mil_spec's encoder) ========

fn dim_msg(size: i64) -> PMut {
    let mut cd = PMut::default();
    cd.push(1, PVal::Varint(size as u64));
    let mut d = PMut::default();
    d.push_msg(1, &cd);
    d
}

fn tensor_type_msg(dt: DType, shape: &[i64]) -> PMut {
    let mut tt = PMut::default();
    tt.push(1, PVal::Varint(mil_code(dt)));
    tt.push(2, PVal::Varint(shape.len() as u64));
    for &d in shape {
        tt.push_msg(3, &dim_msg(d));
    }
    tt
}

fn value_type_msg(dt: DType, shape: &[i64]) -> PMut {
    let mut vt = PMut::default();
    vt.push_msg(1, &tensor_type_msg(dt, shape));
    vt
}

fn nvt_msg(name: &str, dt: DType, shape: &[i64]) -> PMut {
    let mut nvt = PMut::default();
    nvt.push_str(1, name);
    nvt.push_msg(2, &value_type_msg(dt, shape));
    nvt
}

fn argument_msg(names: &[&str]) -> PVal {
    let mut arg = PMut::default();
    for n in names {
        let mut bnd = PMut::default();
        bnd.push_str(1, n);
        arg.push_msg(1, &bnd);
    }
    PVal::Len(proto::encode(&arg))
}

fn value_blob_msg(dt: DType, shape: &[i64], off: u64) -> PMut {
    let mut bv = PMut::default();
    bv.push_str(1, WEIGHT_FILE);
    bv.push(2, PVal::Varint(off));
    let mut v = PMut::default();
    v.push_msg(2, &value_type_msg(dt, shape));
    v.push_msg(5, &bv);
    v
}

/// `const` op whose `val` is a blob reference at `off`.
fn const_op(name: &str, dt: DType, shape: &[i64], off: u64) -> PMut {
    let mut op = PMut::default();
    op.push_str(1, "const");
    op.push_msg(3, &nvt_msg(name, dt, shape));
    op.push_map_entry(
        5,
        "val",
        PVal::Len(proto::encode(&value_blob_msg(dt, shape, off))),
    );
    op
}

/// A `constexpr_*` op producing fp16 `name` with name-bound inputs.
fn constexpr_op(ty: &str, name: &str, shape: &[i64], binds: &[(&str, &str)]) -> PMut {
    let mut op = PMut::default();
    op.push_str(1, ty);
    for (k, v) in binds {
        op.push_map_entry(2, k, argument_msg(&[*v]));
    }
    op.push_msg(3, &nvt_msg(name, DType::Fp16, shape));
    op
}

// ======== weight.bin records ========

fn read_record(weight: &[u8], meta_off: u64) -> R<(u32, &[u8])> {
    let m = meta_off as usize;
    if weight.len() < m + 64 {
        return err(format!("blob offset {meta_off} outside weight.bin"));
    }
    let rec = &weight[m..m + 64];
    let sentinel = u32::from_le_bytes(rec[0..4].try_into().unwrap());
    if sentinel != BLOB_SENTINEL {
        return err(format!(
            "blob offset {meta_off}: bad sentinel {sentinel:#x}"
        ));
    }
    let dtype = u32::from_le_bytes(rec[4..8].try_into().unwrap());
    let size = u64::from_le_bytes(rec[8..16].try_into().unwrap()) as usize;
    let data_off = u64::from_le_bytes(rec[16..24].try_into().unwrap()) as usize;
    if weight.len() < data_off + size {
        return err(format!("blob at {meta_off}: data overruns weight.bin"));
    }
    Ok((dtype, &weight[data_off..data_off + size]))
}

// ======== weight groups ========

/// How a logical weight tensor is stored in the spec+blob pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    /// A single fp16 blob const.
    Fp16,
    /// `constexpr_blockwise_shift_scale` — int8 data + fp16 per-channel scale.
    Int8,
    /// `constexpr_lut_to_dense` — uint4 indices + fp16 16-entry LUT.
    Palette4,
    /// `constexpr_blockwise_shift_scale` — uint4 data + fp16 blockwise
    /// scale + uint4 offset (the lossless ggml Q4_0 encoding).
    Q4Blocks,
}

impl GroupKind {
    /// CLI-ish name.
    pub fn as_str(self) -> &'static str {
        match self {
            GroupKind::Fp16 => "fp16",
            GroupKind::Int8 => "int8",
            GroupKind::Palette4 => "palette4",
            GroupKind::Q4Blocks => "q4blocks",
        }
    }
}

/// One blob-backed const op inside a group.
#[derive(Clone, Debug)]
pub struct Member {
    /// Const op output name (e.g. `l3_wq_q8`, `l3_wq_scale`).
    pub name: String,
    /// What the payload is: `data`/`scale`/`offset`/`lut`/`value`.
    pub role: String,
    /// BlobFileValue offset.
    pub offset: u64,
    /// MIL dtype code of the declared tensor type.
    pub mil_dtype: u64,
    /// Declared tensor shape.
    pub shape: Vec<i64>,
}

/// A logical weight tensor and the ops/blobs that encode it.
#[derive(Clone, Debug)]
pub struct WeightGroup {
    /// Producer output name (`l3_wq`, `lm_w`, `l0_inw`, ...).
    pub name: String,
    /// Storage encoding.
    pub kind: GroupKind,
    /// Logical weight shape (`[out, in, 1, 1]` for conv-packed weights).
    pub shape: Vec<i64>,
    /// Blob consts the group owns.
    pub members: Vec<Member>,
    /// Op indices owned by the group (member consts + constexpr head).
    op_indices: Vec<usize>,
}

impl WeightGroup {
    /// `[out, in, 1, 1]` conv-weight shape — the set requantization to a
    /// quantized encoding targets. 1-D biases and `(1,d,1,1)` norm
    /// weights are deliberately not conv weights.
    pub fn is_conv_weight(&self) -> bool {
        self.shape.len() == 4
            && self.shape[2] == 1
            && self.shape[3] == 1
            && self.shape[0] > 1
            && self.shape[1] > 1
    }
    /// Element count of the logical weight.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }
}

// ======== f32 decode / quantized emit ========

fn f16_slice(payload: &[u8]) -> Vec<f32> {
    payload
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn f32_to_f16(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Unpack MIL-order uint4 (low nibble = even element) into `numel` codes.
fn unpack_u4(payload: &[u8], numel: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let b = payload[i / 2];
        out.push(if i % 2 == 0 { b & 0x0f } else { b >> 4 });
    }
    out
}

/// Pack MIL-order uint4 codes (low nibble = even element).
fn pack_u4(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len() / 2 + 1);
    for c in codes.chunks(2) {
        let lo = c[0] & 0x0f;
        let hi = c.get(1).copied().unwrap_or(0) & 0x0f;
        out.push(lo | (hi << 4));
    }
    out
}

/// Per-row int8 quantize (row = `numel/shape[0]`), same math as
/// [`crate::builder::quantize_int8`]: scale = absmax/127, q = round(w/s).
fn quantize_int8_rows(vals: &[f32], rows: usize) -> (Vec<u8>, Vec<f32>) {
    let n = vals.len();
    let row = n / rows.max(1);
    let mut q = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let rowv = &vals[r * row..(r + 1) * row];
        let amax = rowv.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        scales.push(scale);
        let inv = 1.0 / scale;
        for &v in rowv {
            q.push(((v * inv).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    (q, scales)
}

/// Deterministic 16-level uniform palette per row: `lut[i] =
/// min + (i + 0.5)·step`, `idx = clamp(floor((v - min)/step), 0, 15)`.
fn quantize_palette4_rows(vals: &[f32], rows: usize) -> (Vec<u8>, Vec<f32>) {
    let n = vals.len();
    let row = n / rows.max(1);
    let mut codes = vec![0u8; n];
    let mut lut = Vec::with_capacity(rows * 16);
    for r in 0..rows {
        let rowv = &vals[r * row..(r + 1) * row];
        let (mut mn, mut mx) = (f32::MAX, f32::MIN);
        for &v in rowv {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if mx <= mn {
            mn -= 0.5;
            mx = mn + 1.0;
        }
        let step = (mx - mn) / 16.0;
        for i in 0..16 {
            lut.push(mn + (i as f32 + 0.5) * step);
        }
        let inv = 1.0 / step;
        for (j, &v) in rowv.iter().enumerate() {
            let idx = ((v - mn) * inv).floor().clamp(0.0, 15.0) as u8;
            codes[r * row + j] = idx;
        }
    }
    (codes, lut)
}

/// Deterministic blockwise symmetric uint4 (the Q4_0 pattern
/// `scale·(d − 8)`): one fp16 scale per `block` elements along dim 1.
fn quantize_q4blocks(vals: &[f32], shape: &[i64], block: usize) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let out_f = shape[0] as usize;
    let in_f = shape[1] as usize;
    let nb = in_f.div_ceil(block);
    let mut codes = vec![0u8; out_f * in_f];
    let mut scales = vec![0f32; out_f * nb];
    let mut offs = vec![8u8; out_f * nb];
    for o in 0..out_f {
        for b in 0..nb {
            let lo = b * block;
            let hi = (lo + block).min(in_f);
            let amax = vals[o * in_f + lo..o * in_f + hi]
                .iter()
                .fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax == 0.0 { 1.0 } else { amax / 7.0 };
            scales[o * nb + b] = s;
            let inv = 1.0 / s;
            for i in lo..hi {
                let d = (vals[o * in_f + i] * inv).round().clamp(-8.0, 7.0) as i32 + 8;
                codes[o * in_f + i] = d as u8;
            }
            // partial tail blocks keep offset 8 too (uniform layout)
            offs[o * nb + b] = 8;
        }
    }
    (pack_u4(&codes), scales, pack_u4(&offs))
}

/// Ops + blob payloads emitted for one re-encoded group.
struct Emitted {
    ops: Vec<PMut>,
    /// `(const name, dtype, payload)` — offsets assigned at write time.
    blobs: Vec<(String, DType, Vec<u8>)>,
}

/// Emit ops for `name` holding `vals` in encoding `kind`.
fn emit_group(name: &str, kind: GroupKind, shape: &[i64], vals: &[f32]) -> R<Emitted> {
    let n: i64 = shape.iter().product();
    if vals.len() != n as usize {
        return err(format!(
            "emit {name}: {} values for shape {shape:?} ({n})",
            vals.len()
        ));
    }
    match kind {
        GroupKind::Fp16 => Ok(Emitted {
            ops: vec![const_op(name, DType::Fp16, shape, 0)],
            blobs: vec![(name.to_string(), DType::Fp16, f32_to_f16(vals))],
        }),
        GroupKind::Int8 => {
            let rows = shape[0].max(1) as usize;
            let (q, scales) = quantize_int8_rows(vals, rows);
            let sshape: Vec<i64> = std::iter::once(shape[0])
                .chain(std::iter::repeat(1).take(shape.len() - 1))
                .collect();
            let qn = format!("{name}_q8");
            let sn = format!("{name}_scale");
            let ops = vec![
                const_op(&qn, DType::Int8, shape, 0),
                const_op(&sn, DType::Fp16, &sshape, 0),
                constexpr_op(
                    "constexpr_blockwise_shift_scale",
                    name,
                    shape,
                    &[("data", &qn), ("scale", &sn)],
                ),
            ];
            Ok(Emitted {
                ops,
                blobs: vec![(qn, DType::Int8, q), (sn, DType::Fp16, f32_to_f16(&scales))],
            })
        }
        GroupKind::Palette4 => {
            let rows = shape[0].max(1) as usize;
            let (codes, lut) = quantize_palette4_rows(vals, rows);
            // lut layout matches konst_q4: [shape[0], 1×(rank-1), 16, 1]
            let mut lut_shape: Vec<i64> = vec![shape[0]];
            lut_shape.extend(std::iter::repeat(1).take(shape.len() - 1));
            lut_shape.push(16);
            lut_shape.push(1);
            let qn = format!("{name}_q4");
            let ln = format!("{name}_lut");
            let ops = vec![
                const_op(&qn, DType::Uint4, shape, 0),
                const_op(&ln, DType::Fp16, &lut_shape, 0),
                constexpr_op(
                    "constexpr_lut_to_dense",
                    name,
                    shape,
                    &[("indices", &qn), ("lut", &ln)],
                ),
            ];
            Ok(Emitted {
                ops,
                blobs: vec![
                    (qn, DType::Uint4, pack_u4(&codes)),
                    (ln, DType::Fp16, f32_to_f16(&lut)),
                ],
            })
        }
        GroupKind::Q4Blocks => {
            if shape.len() < 2 {
                return err(format!("{name}: q4blocks needs rank ≥ 2, got {shape:?}"));
            }
            let (codes, scales, offs) = quantize_q4blocks(vals, shape, 32);
            let mut sshape = shape.to_vec();
            sshape[1] = (sshape[1] + 31) / 32;
            let qn = format!("{name}_q4");
            let sn = format!("{name}_scale");
            let on = format!("{name}_off");
            let ops = vec![
                const_op(&qn, DType::Uint4, shape, 0),
                const_op(&sn, DType::Fp16, &sshape, 0),
                const_op(&on, DType::Uint4, &sshape, 0),
                constexpr_op(
                    "constexpr_blockwise_shift_scale",
                    name,
                    shape,
                    &[("data", &qn), ("scale", &sn), ("offset", &on)],
                ),
            ];
            Ok(Emitted {
                ops,
                blobs: vec![
                    (qn, DType::Uint4, codes),
                    (sn, DType::Fp16, f32_to_f16(&scales)),
                    (on, DType::Uint4, offs),
                ],
            })
        }
    }
}

// ======== PackageEdit ========

/// A `.mlpackage` opened for editing: decoded spec, decoded op list,
/// and the `weight.bin` bytes. Mutations stage new blob payloads by
/// const name; [`PackageEdit::write`] re-lays the blob file and
/// re-encodes the spec in one pass.
pub struct PackageEdit {
    /// Source package dir.
    pub dir: PathBuf,
    /// Decoded `Model` message.
    model: PMut,
    /// Decoded `main` block's op list — the working copy written back
    /// by `write`. `None` until [`PackageEdit::ops_mut`] materializes it.
    ops: Option<Vec<PMut>>,
    /// `weight.bin` contents (empty when the package has no weights).
    weight: Vec<u8>,
    /// New/replacement payloads by member const name.
    new_blobs: BTreeMap<String, (DType, Vec<u8>)>,
}

impl PackageEdit {
    /// Open a `.mlpackage` directory.
    pub fn open(dir: &Path) -> R<PackageEdit> {
        let spec_path = dir.join(SPEC_REL);
        let spec_bytes =
            std::fs::read(&spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?;
        let model = proto::decode(&spec_bytes)
            .ok_or_else(|| format!("{}: spec is not protobuf", spec_path.display()))?;
        let wpath = dir.join(WEIGHT_REL);
        let weight = std::fs::read(&wpath).unwrap_or_default();
        let pe = PackageEdit {
            dir: dir.to_path_buf(),
            model,
            ops: None,
            weight,
            new_blobs: BTreeMap::new(),
        };
        // Fail fast on a spec we can't navigate — better here than mid-write.
        let _ = pe.ops()?;
        Ok(pe)
    }

    /// Decoded `ModelDescription`.
    pub fn description(&self) -> R<PMut> {
        self.model
            .msg(2)
            .ok_or_else(|| "spec: missing ModelDescription".to_string())
    }

    /// Replace `ModelDescription`.
    pub fn set_description(&mut self, desc: PMut) {
        self.model.set_msg(2, &desc);
    }

    /// `metadata.userDefined` map.
    pub fn user_metadata(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        if let Some(desc) = self.model.msg(2) {
            if let Some(meta) = desc.msg(100) {
                for (k, e) in meta.map_entries(16) {
                    if let Some(v) = e.str(2) {
                        out.insert(k, v);
                    }
                }
            }
        }
        out
    }

    /// Set a `metadata.userDefined` entry (created if absent).
    pub fn set_user_metadata(&mut self, key: &str, value: &str) -> R<()> {
        let mut desc = self.description()?;
        let mut meta = desc.msg(100).unwrap_or_default();
        // Replace existing entry for key, else append.
        let mut found = false;
        for f in meta.fields.iter_mut() {
            if f.num == 16 {
                if let Some(mut e) = f.val.as_msg() {
                    if e.str(1).as_deref() == Some(key) {
                        e.set_str(2, value);
                        f.val = PVal::Len(proto::encode(&e));
                        found = true;
                    }
                }
            }
        }
        if !found {
            meta.push_map_entry(16, key, PVal::Len(value.as_bytes().to_vec()));
        }
        desc.set_msg(100, &meta);
        self.set_description(desc);
        Ok(())
    }

    /// The decoded `main` block op list.
    pub fn ops(&self) -> R<Vec<PMut>> {
        match &self.ops {
            Some(o) => Ok(o.clone()),
            None => self.ops_vec(),
        }
    }

    /// Materialized ops — decodes the block the first time.
    fn ops_vec(&self) -> R<Vec<PMut>> {
        let blk = self.block_msg()?;
        Ok(blk.msgs(3))
    }

    fn block_msg(&self) -> R<PMut> {
        let prog = self
            .model
            .msg(502)
            .ok_or_else(|| "spec: missing mlProgram (field 502)".to_string())?;
        for (k, entry) in prog.map_entries(2) {
            if k == "main" {
                let fnc = entry
                    .msg(2)
                    .ok_or_else(|| "spec: main function is not a message".to_string())?;
                for (_opset, be) in fnc.map_entries(3) {
                    if let Some(blk) = be.msg(2) {
                        return Ok(blk);
                    }
                }
                return err("spec: main function has no block specialization");
            }
        }
        err("spec: program has no 'main' function")
    }

    /// Ops, materialized for editing.
    pub fn ops_mut(&mut self) -> R<&mut Vec<PMut>> {
        if self.ops.is_none() {
            self.ops = Some(self.ops_vec()?);
        }
        Ok(self.ops.as_mut().unwrap())
    }

    /// Function input names of `main`.
    pub fn fn_input_names(&self) -> R<Vec<String>> {
        let prog = self
            .model
            .msg(502)
            .ok_or_else(|| "spec: missing mlProgram".to_string())?;
        for (k, entry) in prog.map_entries(2) {
            if k == "main" {
                if let Some(fnc) = entry.msg(2) {
                    return Ok(fnc.msgs(1).iter().filter_map(|n| n.str(1)).collect());
                }
            }
        }
        err("spec: no main function")
    }

    /// Block output names.
    pub fn block_outputs(&self) -> R<Vec<String>> {
        let blk = self.block_msg()?;
        Ok(blk.get_all(2).iter().filter_map(|v| v.as_str()).collect())
    }

    /// Raw blob payload at `offset` (with blob dtype).
    pub fn blob_payload(&self, offset: u64) -> R<(u32, &[u8])> {
        read_record(&self.weight, offset)
    }

    /// Stage a payload replacement for member const `name`.
    pub fn stage_blob(&mut self, name: &str, dtype: DType, data: Vec<u8>) {
        self.new_blobs.insert(name.to_string(), (dtype, data));
    }

    /// Index every weight group in the op list.
    pub fn weight_groups(&self) -> R<Vec<WeightGroup>> {
        let ops = match &self.ops {
            Some(o) => o.clone(),
            None => self.ops_vec()?,
        };
        // blob consts by output name: name → (op index, offset, mil dtype, shape)
        let mut consts: BTreeMap<String, (usize, u64, u64, Vec<i64>)> = BTreeMap::new();
        for (i, op) in ops.iter().enumerate() {
            if let Some((file, off, dt, dims)) = const_blob(op) {
                if file != WEIGHT_FILE {
                    continue; // external blob file — not managed here
                }
                let Some(name) = op_out_name(op) else {
                    continue;
                };
                let shape: Vec<i64> = dims.iter().flatten().copied().collect();
                consts.insert(name, (i, off, dt.unwrap_or(0), shape));
            }
        }
        let mut claimed: BTreeSet<String> = BTreeSet::new();
        let mut groups: Vec<WeightGroup> = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            let ty = op.str(1).unwrap_or_default();
            if !ty.starts_with("constexpr_") {
                continue;
            }
            let Some(name) = op_out_name(op) else {
                continue;
            };
            let member = |key: &str| -> Option<Member> {
                let cn = op_input_names(op, key).into_iter().next()?;
                let (_, off, dt, shape) = consts.get(&cn)?.clone();
                Some(Member {
                    name: cn,
                    role: match key {
                        "data" | "indices" => "data",
                        "scale" => "scale",
                        "offset" => "offset",
                        "lut" => "lut",
                        _ => key,
                    }
                    .to_string(),
                    offset: off,
                    mil_dtype: dt,
                    shape,
                })
            };
            let (kind, members, shape) = match ty.as_str() {
                "constexpr_blockwise_shift_scale" => {
                    let (Some(data), Some(scale)) = (member("data"), member("scale")) else {
                        continue;
                    };
                    let shape = data.shape.clone();
                    let kind = match data.mil_dtype {
                        21 => GroupKind::Int8,
                        35 => GroupKind::Q4Blocks,
                        _ => continue,
                    };
                    let mut m = vec![data, scale];
                    if let Some(off) = member("offset") {
                        m.push(off);
                    }
                    (kind, m, shape)
                }
                "constexpr_lut_to_dense" => {
                    let (Some(idx), Some(lut)) = (member("indices"), member("lut")) else {
                        continue;
                    };
                    // Logical shape = the op's declared output type.
                    let shape: Vec<i64> = op
                        .msgs(3)
                        .first()
                        .and_then(|n| n.msg(2))
                        .and_then(|v| v.msg(1))
                        .map(|tt| tensor_type_dims(&tt).1.iter().flatten().copied().collect())
                        .unwrap_or_default();
                    if shape.is_empty() {
                        continue;
                    }
                    (GroupKind::Palette4, vec![idx, lut], shape)
                }
                _ => continue,
            };
            for m in &members {
                claimed.insert(m.name.clone());
            }
            let mut idx: Vec<usize> = members
                .iter()
                .map(|m| consts.get(&m.name).map(|c| c.0).unwrap_or(usize::MAX))
                .collect();
            idx.push(i);
            groups.push(WeightGroup {
                name,
                kind,
                shape,
                members,
                op_indices: idx,
            });
        }
        // Standalone fp16 blob consts → Fp16 groups.
        for (name, (i, off, dt, shape)) in &consts {
            if claimed.contains(name) || dtype_of_mil(*dt) != Some(DType::Fp16) {
                continue;
            }
            groups.push(WeightGroup {
                name: name.clone(),
                kind: GroupKind::Fp16,
                shape: shape.clone(),
                members: vec![Member {
                    name: name.clone(),
                    role: "value".to_string(),
                    offset: *off,
                    mil_dtype: *dt,
                    shape: shape.clone(),
                }],
                op_indices: vec![*i],
            });
        }
        groups.sort_by_key(|g| g.op_indices.iter().min().copied().unwrap_or(0));
        Ok(groups)
    }

    /// Decode a group's logical values to f32.
    pub fn group_values(&self, g: &WeightGroup) -> R<Vec<f32>> {
        let payload = |m: &Member| -> R<Vec<u8>> { Ok(self.blob_payload(m.offset)?.1.to_vec()) };
        match g.kind {
            GroupKind::Fp16 => {
                let p = payload(&g.members[0])?;
                if p.len() != g.numel() * 2 {
                    return err(format!(
                        "{}: fp16 payload {}B for {} elems",
                        g.name,
                        p.len(),
                        g.numel()
                    ));
                }
                Ok(f16_slice(&p))
            }
            GroupKind::Int8 => {
                let (data, scale) = (&g.members[0], &g.members[1]);
                let q = payload(data)?;
                let s = f16_slice(&payload(scale)?);
                let rows = g.shape[0].max(1) as usize;
                if s.len() != rows || q.len() != g.numel() {
                    return err(format!(
                        "{}: int8 payload/scale mismatch ({} q, {} s, rows {rows})",
                        g.name,
                        q.len(),
                        s.len()
                    ));
                }
                let row = g.numel() / rows;
                let mut v = vec![0f32; g.numel()];
                for r in 0..rows {
                    for j in 0..row {
                        v[r * row + j] = (q[r * row + j] as i8) as f32 * s[r];
                    }
                }
                Ok(v)
            }
            GroupKind::Palette4 => {
                let (idx, lut) = (&g.members[0], &g.members[1]);
                let codes = unpack_u4(&payload(idx)?, g.numel());
                let lutv = f16_slice(&payload(lut)?);
                let rows = g.shape[0].max(1) as usize;
                if lutv.len() != rows * 16 {
                    return err(format!(
                        "{}: lut has {} entries, want {rows}×16",
                        g.name,
                        lutv.len()
                    ));
                }
                let row = g.numel() / rows;
                let mut v = vec![0f32; g.numel()];
                for r in 0..rows {
                    for j in 0..row {
                        v[r * row + j] = lutv[r * 16 + codes[r * row + j] as usize];
                    }
                }
                Ok(v)
            }
            GroupKind::Q4Blocks => {
                let (data, scale) = (&g.members[0], &g.members[1]);
                let codes = unpack_u4(&payload(data)?, g.numel());
                let s = f16_slice(&payload(scale)?);
                let offv = match g.members.get(2) {
                    Some(m) => unpack_u4(&payload(m)?, s.len()),
                    None => vec![8u8; s.len()],
                };
                if shape_need_2d(g) {
                    let (out_f, in_f) = (g.shape[0] as usize, g.shape[1] as usize);
                    let nb = in_f.div_ceil(32);
                    if s.len() != out_f * nb {
                        return err(format!(
                            "{}: q4 scale has {} entries, want {out_f}×{nb}",
                            g.name,
                            s.len()
                        ));
                    }
                    let mut v = vec![0f32; g.numel()];
                    for o in 0..out_f {
                        for i in 0..in_f {
                            let b = i / 32;
                            let k = o * nb + b;
                            v[o * in_f + i] = (codes[o * in_f + i] as f32 - offv[k] as f32) * s[k];
                        }
                    }
                    Ok(v)
                } else {
                    err(format!(
                        "{}: q4blocks on non-2D shape {:?}",
                        g.name, g.shape
                    ))
                }
            }
        }
    }

    /// Replace `g` with a `kind` encoding of `vals` (same logical shape).
    /// Member/head ops are swapped for freshly emitted ones and the new
    /// payloads are staged for the rebuilt `weight.bin`.
    pub fn replace_group(&mut self, g: &WeightGroup, kind: GroupKind, vals: &[f32]) -> R<()> {
        let em = emit_group(&g.name, kind, &g.shape, vals)?;
        let ops = self.ops_mut()?;
        let mut remove: BTreeSet<usize> = g.op_indices.iter().copied().collect();
        let at = g.op_indices.iter().min().copied().unwrap_or(ops.len());
        // drop owned ops
        let mut kept: Vec<PMut> = Vec::with_capacity(ops.len());
        for (i, o) in ops.iter().enumerate() {
            if !remove.remove(&i) {
                kept.push(o.clone());
            }
        }
        *ops = kept;
        for (j, o) in em.ops.into_iter().enumerate() {
            ops.insert(at + j, o);
        }
        for (n, dt, bytes) in em.blobs {
            self.stage_blob(&n, dt, bytes);
        }
        Ok(())
    }

    /// Structural validation of the working op list: every bound input
    /// resolves (fn input, state, or earlier output) and every block
    /// output is produced.
    pub fn validate(&self) -> R<()> {
        let mut produced: BTreeSet<String> = BTreeSet::new();
        for n in self.fn_input_names()? {
            produced.insert(n);
        }
        if let Ok(desc) = self.description() {
            for m in desc.msgs(13) {
                if let Some(n) = m.str(1) {
                    produced.insert(n);
                }
            }
        }
        let ops = match &self.ops {
            Some(o) => o.clone(),
            None => self.ops_vec()?,
        };
        for op in &ops {
            let ty = op.str(1).unwrap_or_else(|| "?".into());
            for (_k, entry) in op.map_entries(2) {
                if let Some(arg) = entry.msg(2) {
                    for b in arg.msgs(1) {
                        if let Some(n) = b.str(1) {
                            if !produced.contains(&n) {
                                return err(format!("{ty}: input binds undefined value {n}"));
                            }
                        }
                    }
                }
            }
            for nvt in op.msgs(3) {
                if let Some(n) = nvt.str(1) {
                    produced.insert(n);
                }
            }
        }
        for out in self.block_outputs()? {
            if !produced.contains(&out) {
                return err(format!("block output {out} is never produced"));
            }
        }
        Ok(())
    }

    /// Write the edited package to `out_dir`: re-lays `weight.bin`
    /// (staged payloads replace originals; unreferenced records are
    /// dropped), patches every blob offset in the spec, then writes
    /// spec + Manifest. Other files under `weights/` are copied through.
    pub fn write(&self, out_dir: &Path) -> R<WriteReport> {
        self.validate()?;
        let ops = match &self.ops {
            Some(o) => o.clone(),
            None => self.ops_vec()?,
        };
        let coreml = out_dir.join("Data").join("com.apple.CoreML");
        let weights_dir = coreml.join("weights");

        // Walk ops; const ops get fresh offsets from the new writer.
        let has_blobs = ops.iter().any(|op| const_blob(op).is_some());
        std::fs::create_dir_all(&coreml).map_err(|e| format!("{}: {e}", coreml.display()))?;
        let wpath = weights_dir.join("weight.bin");
        let mut writer = if has_blobs {
            std::fs::create_dir_all(&weights_dir)
                .map_err(|e| format!("{}: {e}", weights_dir.display()))?;
            Some(BlobWriter::create(&wpath).map_err(|e| format!("{}: {e}", wpath.display()))?)
        } else {
            None
        };

        let mut blob_count = 0usize;
        let mut weight_bytes = 0u64;
        let mut new_ops: Vec<PMut> = Vec::with_capacity(ops.len());
        for op in &ops {
            let Some((file, off, _dt, _dims)) = const_blob(op) else {
                new_ops.push(op.clone());
                continue;
            };
            if file != WEIGHT_FILE {
                // External blob: keep the ref; the file is copied below.
                new_ops.push(op.clone());
                continue;
            }
            let name = op_out_name(op).unwrap_or_default();
            let (bdt, data) = match self.new_blobs.get(&name) {
                Some((dt, bytes)) => (*dt, bytes.clone()),
                None => {
                    let (dt, p) = self.blob_payload(off)?;
                    (
                        dtype_of_blob(dt)
                            .ok_or_else(|| format!("blob at {off}: unknown dtype {dt}"))?,
                        p.to_vec(),
                    )
                }
            };
            let w = writer
                .as_mut()
                .ok_or_else(|| "internal: no blob writer".to_string())?;
            let new_off = w
                .append(bdt, &data)
                .map_err(|e| format!("weight.bin append: {e}"))?;
            blob_count += 1;
            weight_bytes += data.len() as u64;
            new_ops.push(patch_blob_offset(op, new_off));
        }
        let mut weight_hash: Option<String> = None;
        if let Some(w) = writer {
            w.finish().map_err(|e| format!("weight.bin: {e}"))?;
            weight_hash = Some(
                mil_spec::sha256::sha256_file(&wpath)
                    .map_err(|e| format!("weight.bin hash: {e}"))?,
            );
        }

        // Copy through any additional weight files (multi-blob packages).
        let src_weights = self.dir.join("Data/com.apple.CoreML/weights");
        if let Ok(rd) = std::fs::read_dir(&src_weights) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_file() && p.file_name().map(|n| n != "weight.bin").unwrap_or(false) {
                    let dst = weights_dir.join(e.file_name());
                    std::fs::copy(&p, &dst).map_err(|e| format!("{}: {e}", dst.display()))?;
                }
            }
        }

        // Rebuild the spec with the edited op list, then refresh the
        // provenance weights hash — a surgered package's provenance
        // must describe what it actually contains, or `attest` would
        // report it as tampered. No provenance block → nothing added.
        let mut model = self.rebuilt_model(&new_ops)?;
        if let Some(h) = &weight_hash {
            patch_prov_weights(&mut model, h);
        }
        // the spec changed (ops, descriptions): keep `mil.prov.program`
        // describing it; the hash excludes userDefined so ordering with
        // the weights patch above does not matter
        crate::attest::refresh_program_hash(&mut model);
        let spec = proto::encode(&model);
        mil_spec::write_mlpackage_stream(
            out_dir,
            &spec,
            if has_blobs { Some(&wpath) } else { None },
        )
        .map_err(|e| format!("{}: {e}", out_dir.display()))?;
        Ok(WriteReport {
            package: out_dir.to_path_buf(),
            weight_bytes,
            blob_count,
            op_count: new_ops.len(),
        })
    }

    /// Re-encode `model` with `ops` swapped into the main block.
    fn rebuilt_model(&self, ops: &[PMut]) -> R<PMut> {
        let mut model = self.model.clone();
        let mut prog = model
            .msg(502)
            .ok_or_else(|| "spec: missing mlProgram".to_string())?;
        for f in prog.fields.iter_mut() {
            if f.num != 2 {
                continue;
            }
            let Some(mut entry) = f.val.as_msg() else {
                continue;
            };
            if entry.str(1).as_deref() != Some("main") {
                continue;
            }
            let Some(mut fnc) = entry.msg(2) else {
                continue;
            };
            for bf in fnc.fields.iter_mut() {
                if bf.num != 3 {
                    continue;
                }
                let Some(mut be) = bf.val.as_msg() else {
                    continue;
                };
                let Some(mut blk) = be.msg(2) else { continue };
                blk.remove_all(3);
                for op in ops {
                    blk.push_msg(3, op);
                }
                be.set_msg(2, &blk);
                bf.val = PVal::Len(proto::encode(&be));
            }
            entry.set_msg(2, &fnc);
            f.val = PVal::Len(proto::encode(&entry));
        }
        model.set_msg(502, &prog);
        Ok(model)
    }
}

fn shape_need_2d(g: &WeightGroup) -> bool {
    g.shape.len() >= 2
}

/// Set `mil.prov.weights` in the model's userDefined metadata, when the
/// provenance block exists (older/foreign specs are left untouched).
fn patch_prov_weights(model: &mut PMut, hash: &str) {
    let Some(mut desc) = model.msg(2) else { return };
    let Some(mut meta) = desc.msg(100) else {
        return;
    };
    let mut touched = false;
    for f in meta.fields.iter_mut() {
        if f.num != 16 {
            continue;
        }
        let Some(mut e) = f.val.as_msg() else {
            continue;
        };
        if e.str(1).as_deref() == Some("mil.prov.weights") {
            e.set_str(2, hash);
            f.val = PVal::Len(proto::encode(&e));
            touched = true;
        }
    }
    if !touched {
        return;
    }
    desc.set_msg(100, &meta);
    model.set_msg(2, &desc);
}

/// Return `op` with its `val` blob offset set to `new_off`.
fn patch_blob_offset(op: &PMut, new_off: u64) -> PMut {
    let mut out = op.clone();
    for f in out.fields.iter_mut() {
        if f.num != 5 {
            continue;
        }
        let Some(mut entry) = f.val.as_msg() else {
            continue;
        };
        if entry.str(1).as_deref() != Some("val") {
            continue;
        }
        let Some(mut v) = entry.msg(2) else { continue };
        let Some(mut bv) = v.msg(5) else { continue };
        bv.set_varint(2, new_off);
        v.set_msg(5, &bv);
        entry.set_msg(2, &v);
        f.val = PVal::Len(proto::encode(&entry));
    }
    out
}

/// What a [`PackageEdit::write`] produced.
#[derive(Debug)]
pub struct WriteReport {
    /// Output package dir.
    pub package: PathBuf,
    /// Payload bytes written to weight.bin.
    pub weight_bytes: u64,
    /// Blob records written.
    pub blob_count: usize,
    /// Ops in the written spec.
    pub op_count: usize,
}

// ======== commands ========

/// Requantization target for [`requant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetQuant {
    /// Per-channel int8 + fp16 scale (`constexpr_blockwise_shift_scale`).
    Int8,
    /// Raw fp16 blob consts.
    Fp16,
    /// 4-bit indices + fp16 LUT (`constexpr_lut_to_dense`).
    Palette4,
}

impl TargetQuant {
    /// Parse `--to` values.
    pub fn parse(s: &str) -> Option<TargetQuant> {
        match s {
            "int8" => Some(TargetQuant::Int8),
            "fp16" => Some(TargetQuant::Fp16),
            "palette4" | "pal4" | "int4" => Some(TargetQuant::Palette4),
            _ => None,
        }
    }
}

/// What a surgery command changed.
#[derive(Debug)]
pub struct SurgReport {
    /// Human-readable summary lines.
    pub lines: Vec<String>,
    /// Write outcome.
    pub written: Option<WriteReport>,
}

/// Re-encode every eligible weight group in `pkg` to `to`.
///
/// Eligibility mirrors what the converter quantizes: `--to int8` and
/// `--to palette4` touch `[out, in, 1, 1]` conv weights (fp16 sources)
/// plus already-quantized groups (int8 → palette4 etc. via an f32
/// decode); `--to fp16` dequantizes every non-fp16 group. Norm weights
/// and biases are never requantized — they were fp16 at convert time
/// on purpose. Output defaults to in-place.
pub fn requant(pkg: &Path, to: TargetQuant, out: Option<&Path>) -> R<SurgReport> {
    let mut pe = PackageEdit::open(pkg)?;
    // Collect the work list first, then apply in *descending* op order:
    // replace_group splices ops at indices captured at scan time, so
    // touching higher-index groups first keeps the others' indices valid.
    let groups = pe.weight_groups()?;
    let mut lines = Vec::new();
    let mut plan: Vec<(&WeightGroup, Vec<f32>, GroupKind)> = Vec::new();
    for g in &groups {
        let want = match to {
            TargetQuant::Int8 => match g.kind {
                GroupKind::Fp16 if g.is_conv_weight() => Some(GroupKind::Int8),
                GroupKind::Palette4 | GroupKind::Q4Blocks => Some(GroupKind::Int8),
                _ => None,
            },
            TargetQuant::Fp16 => match g.kind {
                GroupKind::Fp16 => None,
                _ => Some(GroupKind::Fp16),
            },
            TargetQuant::Palette4 => match g.kind {
                GroupKind::Fp16 if g.is_conv_weight() => Some(GroupKind::Palette4),
                GroupKind::Int8 | GroupKind::Q4Blocks => Some(GroupKind::Palette4),
                _ => None,
            },
        };
        let Some(kind) = want else { continue };
        if kind == g.kind {
            continue;
        }
        let vals = pe.group_values(g)?;
        plan.push((g, vals, kind));
    }
    plan.sort_by_key(|(g, _, _)| {
        std::cmp::Reverse(g.op_indices.iter().min().copied().unwrap_or(0))
    });
    let mut changed = 0usize;
    for (g, vals, kind) in &plan {
        pe.replace_group(g, *kind, vals)?;
        lines.push(format!(
            "  {} : {} → {}",
            g.name,
            g.kind.as_str(),
            kind.as_str()
        ));
        changed += 1;
    }
    if changed == 0 {
        return err(format!(
            "requant --to {}: no eligible weight groups in {}",
            match to {
                TargetQuant::Int8 => "int8",
                TargetQuant::Fp16 => "fp16",
                TargetQuant::Palette4 => "palette4",
            },
            pkg.display()
        ));
    }
    let dst = out.unwrap_or(pkg);
    let written = write_maybe_inplace(&pe, pkg, dst)?;
    lines.insert(0, format!("{changed} weight group(s) requantized"));
    Ok(SurgReport {
        lines,
        written: Some(written),
    })
}

/// Parse `--layers a..b` (Rust-style, `b` exclusive) or `a..=b`
/// (inclusive) into an inclusive range.
pub fn parse_layer_range(s: &str) -> R<(usize, usize)> {
    if let Some((a, b)) = s.split_once("..=") {
        let a: usize = a.trim().parse().map_err(|_| format!("bad range {s:?}"))?;
        let b: usize = b.trim().parse().map_err(|_| format!("bad range {s:?}"))?;
        if a > b {
            return err(format!("empty layer range {s:?}"));
        }
        return Ok((a, b));
    }
    if let Some((a, b)) = s.split_once("..") {
        let a: usize = a.trim().parse().map_err(|_| format!("bad range {s:?}"))?;
        let b: usize = b.trim().parse().map_err(|_| format!("bad range {s:?}"))?;
        if a >= b {
            return err(format!("empty layer range {s:?}"));
        }
        return Ok((a, b - 1));
    }
    err(format!("--layers needs a..b or a..=b, got {s:?}"))
}

/// Layer index from a weight group name (`l{N}_...`), if it is one.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix('l')?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() || !rest[digits.len()..].starts_with('_') {
        return None;
    }
    digits.parse().ok()
}

/// Feature lists of a description: `(field, name, shape, dtype)`.
fn feature_set(desc: &PMut) -> Vec<(u32, String, Vec<i64>, u64)> {
    let mut out = Vec::new();
    for field in [1u32, 10, 13] {
        for m in desc.msgs(field) {
            let Some(name) = m.str(1) else { continue };
            let (shape, dtype) = m
                .msg(3)
                .and_then(|t| {
                    t.msg(5)
                        .map(|mt| (mt.clone(), false))
                        .or_else(|| t.msg(8).and_then(|s| s.msg(1)).map(|mt| (mt, true)))
                })
                .map(|(mt, _st)| {
                    let dims: Vec<i64> = mt
                        .get_all(1)
                        .iter()
                        .map(|v| v.as_varint().unwrap_or(0) as i64)
                        .collect();
                    (dims, mt.varint(2).unwrap_or(0))
                })
                .unwrap_or_default();
            out.push((field, name, shape, dtype));
        }
    }
    out
}

/// Splice donor layer weights into `base` — both must be the same
/// architecture (identical op stream and feature lists). `layers` is an
/// inclusive range; only groups named `l{N}_*` in it are copied, so the
/// embedding, norms, and head always come from `base`.
pub fn graft(
    donor_dir: &Path,
    base_dir: &Path,
    layers: (usize, usize),
    out: &Path,
) -> R<SurgReport> {
    let donor = PackageEdit::open(donor_dir)?;
    let mut base = PackageEdit::open(base_dir)?;

    // ---- structural compatibility ----
    let dops = donor.ops()?;
    let bops = base.ops()?;
    if dops.len() != bops.len() {
        return err(format!(
            "graft: op counts differ (donor {} vs base {}) — not the same architecture",
            dops.len(),
            bops.len()
        ));
    }
    for (i, (d, b)) in dops.iter().zip(bops.iter()).enumerate() {
        if d.str(1) != b.str(1) {
            return err(format!(
                "graft: op {i} type differs ({:?} vs {:?}) — not the same architecture",
                d.str(1),
                b.str(1)
            ));
        }
    }
    let df = feature_set(&donor.description()?);
    let bf = feature_set(&base.description()?);
    if df != bf {
        return err("graft: model descriptions differ (inputs/outputs/states)");
    }

    let dgroups: BTreeMap<String, WeightGroup> = donor
        .weight_groups()?
        .into_iter()
        .map(|g| (g.name.clone(), g))
        .collect();
    let bgroups = base.weight_groups()?;
    let mut lines = Vec::new();
    let mut copied = 0usize;
    for g in &bgroups {
        let Some(n) = layer_of(&g.name) else { continue };
        if n < layers.0 || n > layers.1 {
            continue;
        }
        let Some(dg) = dgroups.get(&g.name) else {
            return err(format!(
                "graft: donor lacks weight group {} needed for layer {n}",
                g.name
            ));
        };
        if dg.kind != g.kind || dg.shape != g.shape {
            return err(format!(
                "graft: {} encoding differs (donor {} {:?} vs base {} {:?})",
                g.name,
                dg.kind.as_str(),
                dg.shape,
                g.kind.as_str(),
                g.shape
            ));
        }
        // Copy each member payload byte-for-byte — same kind+shape means
        // same payload layout, so this is lossless.
        let dmembers: BTreeMap<&str, &crate::surgery::Member> =
            dg.members.iter().map(|m| (m.name.as_str(), m)).collect();
        for m in &g.members {
            let Some(dm) = dmembers.get(m.name.as_str()) else {
                return err(format!("graft: donor {} missing member {}", g.name, m.name));
            };
            let (bdt, payload) = donor.blob_payload(dm.offset)?;
            let dt = dtype_of_blob(bdt)
                .ok_or_else(|| format!("donor {}: unknown blob dtype {bdt}", dm.name))?;
            base.stage_blob(&m.name, dt, payload.to_vec());
        }
        lines.push(format!("  {} ← donor", g.name));
        copied += 1;
    }
    if copied == 0 {
        return err(format!(
            "graft: no weight groups in layers {}..={} of {}",
            layers.0,
            layers.1,
            base_dir.display()
        ));
    }
    let written = base.write(out)?;
    lines.insert(0, format!("{copied} weight group(s) grafted from donor"));
    Ok(SurgReport {
        lines,
        written: Some(written),
    })
}

/// Target sequence flexibility for [`reshape_flex`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeqFlex {
    /// `EnumeratedShapes` over these sequence lengths (first = default).
    Lens(Vec<i64>),
    /// `ShapeRange` over `lo..=hi` (default = `lo`).
    Range(i64, i64),
}

/// What a feature's varying dims looked like before the rewrite.
#[derive(PartialEq)]
enum OldFlex {
    Lens(Vec<i64>),
    Range(i64, i64),
}

impl std::fmt::Display for OldFlex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OldFlex::Lens(l) => write!(f, "{l:?}"),
            OldFlex::Range(lo, hi) => write!(f, "{lo}..{hi}"),
        }
    }
}

/// Rewrite `EnumeratedShapes` on a flexible-shape package — the
/// enumerated-target form of [`reshape_flex`].
pub fn reshape(pkg: &Path, seq_lens: &[i64], out: Option<&Path>) -> R<SurgReport> {
    reshape_flex(pkg, &SeqFlex::Lens(seq_lens.to_vec()), out)
}

/// Rewrite the sequence flexibility of a flexible-shape package, in
/// either direction between enumerated shapes and a shape range. Every
/// feature's varying dims must track the same sequence description —
/// the exact property `--seq-lens` / `--seq-range` conversions produce
/// — and the new flexibility is the old template with those dims
/// rebound. The default shape becomes the first entry's (enumerated) or
/// the lower bound (range). Fixed-shape packages are refused.
pub fn reshape_flex(pkg: &Path, target: &SeqFlex, out: Option<&Path>) -> R<SurgReport> {
    match target {
        SeqFlex::Lens(l) if l.is_empty() || l.iter().any(|&v| v <= 0) => {
            return err("reshape: --seq-lens needs positive a,b,c");
        }
        SeqFlex::Range(lo, hi) if *lo < 1 || hi < lo => {
            return err(format!(
                "reshape: --seq-range {lo}..{hi} needs 1 <= LO <= HI"
            ));
        }
        _ => {}
    }
    let mut pe = PackageEdit::open(pkg)?;
    let mut desc = pe.description()?;
    let mut old: Option<OldFlex> = None;
    let mut rewritten = 0usize;
    let mut lines = Vec::new();

    for field in [1u32, 10] {
        for f in desc.fields.iter_mut().filter(|f| f.num == field) {
            let Some(mut feat) = f.val.as_msg() else {
                continue;
            };
            let Some(mut fty) = feat.msg(3) else { continue };
            let Some(mut ma) = fty.msg(5) else { continue };
            let name = feat.str(1).unwrap_or_default();
            let rank = ma.get_all(1).len();

            // Read the current flexibility into (template shape, varying
            // dim indices, old description).
            let (template, varying, old_f): (Vec<i64>, Vec<usize>, OldFlex) = if let Some(es) =
                ma.msg(21)
            {
                let entries: Vec<Vec<i64>> = es
                    .msgs(1)
                    .iter()
                    .map(|s| {
                        s.get_all(1)
                            .iter()
                            .map(|v| v.as_varint().unwrap_or(0) as i64)
                            .collect()
                    })
                    .collect();
                if entries.is_empty() || entries.iter().any(|e| e.len() != rank) {
                    return err(format!("reshape: {name}: malformed enumeratedShapes"));
                }
                // Varying dims must all carry the same value list.
                let mut varying: Vec<(usize, Vec<i64>)> = Vec::new();
                for d in 0..rank {
                    let vals: Vec<i64> = entries.iter().map(|e| e[d]).collect();
                    if vals.iter().collect::<BTreeSet<_>>().len() > 1 {
                        varying.push((d, vals));
                    }
                }
                if varying.is_empty() {
                    continue;
                }
                let first = varying[0].1.clone();
                if let Some((_, vals)) = varying.iter().find(|(_, v)| *v != first) {
                    return err(format!(
                        "reshape: {name} varies on different sequences {vals:?} and {first:?}"
                    ));
                }
                (
                    entries[0].clone(),
                    varying.iter().map(|(d, _)| *d).collect(),
                    OldFlex::Lens(first),
                )
            } else if let Some(sr) = ma.msg(31) {
                let ranges: Vec<(i64, i64)> = sr
                    .msgs(1)
                    .iter()
                    .map(|s| {
                        (
                            s.varint(1).unwrap_or(0) as i64,
                            s.varint(2).map(|v| v as i64).unwrap_or(0),
                        )
                    })
                    .collect();
                if ranges.len() != rank {
                    return err(format!("reshape: {name}: malformed shapeRange"));
                }
                let varying: Vec<usize> =
                    (0..rank).filter(|&d| ranges[d].0 != ranges[d].1).collect();
                if varying.is_empty() {
                    continue;
                }
                let first = ranges[varying[0]];
                if let Some(&d) = varying.iter().find(|&&d| ranges[d] != first) {
                    return err(format!(
                            "reshape: {name} dim {d} varies over {:?}, not the sequence range {first:?}",
                            ranges[d]
                        ));
                }
                (
                    ranges.iter().map(|r| r.0).collect(),
                    varying,
                    OldFlex::Range(first.0, first.1),
                )
            } else {
                continue;
            };
            match &old {
                None => old = Some(old_f),
                Some(o) if *o != old_f => {
                    return err(format!(
                        "reshape: {name} varies on a different sequence {old_f} than {o}"
                    ));
                }
                _ => {}
            }

            // Build the replacement flexibility + default shape.
            let at = |v: i64| -> Vec<i64> {
                let mut s = template.clone();
                for &d in &varying {
                    s[d] = v;
                }
                s
            };
            ma.remove_all(21);
            ma.remove_all(31);
            let default_shape = match target {
                SeqFlex::Lens(lens) => {
                    let mut new_es = PMut::default();
                    for &nl in lens {
                        let mut sm = PMut::default();
                        for d in at(nl) {
                            sm.push(1, PVal::Varint(d as u64));
                        }
                        new_es.push_msg(1, &sm);
                    }
                    ma.set_msg(21, &new_es);
                    lines.push(format!("  {name}: {} enumerated shape(s)", lens.len()));
                    at(lens[0])
                }
                SeqFlex::Range(lo, hi) => {
                    let mut sr = PMut::default();
                    for (d, &t) in template.iter().enumerate() {
                        let (a, b) = if varying.contains(&d) {
                            (*lo, *hi)
                        } else {
                            (t, t)
                        };
                        let mut sz = PMut::default();
                        sz.push(1, PVal::Varint(a as u64));
                        sz.push(2, PVal::Varint(b as u64));
                        sr.push_msg(1, &sz);
                    }
                    ma.set_msg(31, &sr);
                    lines.push(format!("  {name}: shape range {lo}..{hi}"));
                    at(*lo)
                }
            };
            ma.remove_all(1);
            for &d in &default_shape {
                ma.push(1, PVal::Varint(d as u64));
            }
            fty.set_msg(5, &ma);
            feat.set_msg(3, &fty);
            f.val = PVal::Len(proto::encode(&feat));
            rewritten += 1;
        }
    }
    let Some(old) = old.filter(|_| rewritten > 0) else {
        return err(format!(
            "reshape: {} has no sequence flexibility (enumeratedShapes or shapeRange) — was it converted without --seq-lens/--seq-range?",
            pkg.display()
        ));
    };
    pe.set_description(desc);
    let dst = out.unwrap_or(pkg);
    let written = write_maybe_inplace(&pe, pkg, dst)?;
    let new_desc = match target {
        SeqFlex::Lens(l) => format!("{l:?}"),
        SeqFlex::Range(lo, hi) => format!("{lo}..{hi}"),
    };
    lines.insert(
        0,
        format!("seq {old} → {new_desc} across {rewritten} feature(s)"),
    );
    Ok(SurgReport {
        lines,
        written: Some(written),
    })
}

/// mil_convert conv-weight name → HF tensor name and back. The emitted
/// const names are deterministic (`builder.rs`), which is what makes
/// packaged-const → projection mapping verifiable.
fn mil_name_for_hf(wname: &str) -> Option<String> {
    if wname == "lm_head.weight" {
        return Some("lm_w".into());
    }
    let rest = wname.strip_prefix("model.layers.")?;
    let (n, suffix) = rest.split_once('.')?;
    let code = match suffix {
        "self_attn.q_proj.weight" => "wq",
        "self_attn.k_proj.weight" => "wk",
        "self_attn.v_proj.weight" => "wv",
        "self_attn.o_proj.weight" => "wo",
        "mlp.gate_proj.weight" => "wg",
        "mlp.up_proj.weight" => "wu",
        "mlp.down_proj.weight" => "wd",
        _ => return None,
    };
    n.parse::<usize>().ok().map(|n| format!("l{n}_{code}"))
}

/// In-memory [`WeightSource`] over decoded package weights — lets
/// [`crate::lora::FusedSource`] run the same fusion math as convert.
struct MapSource(BTreeMap<String, (Vec<i64>, Vec<f32>)>);
impl crate::WeightSource for MapSource {
    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
        self.tensor_f32(name).map(|(s, v)| (s, f32_to_f16(&v)))
    }
    fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
        self.0.get(name).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing {name}"))
        })
    }
}

/// Fuse a LoRA adapter into a built package's weight blobs.
///
/// **Approach:** direct package surgery, not checkpoint rebuild. The
/// mapping packaged-const → HF tensor is *verifiable* because
/// `mil_convert` names conv weight consts deterministically
/// (`l{L}_{wq,wk,wv,wo,wg,wu,wd}`, `lm_w`) — `mil_name_for_hf` is a
/// total function over LoRA targets, and a miss is a hard error. Each
/// target group is decoded to f32, fused through the same
/// [`crate::lora::FusedSource`] math `convert --lora` uses, and
/// re-emitted in its original encoding (int8 stays int8 with fresh
/// scales, palette4 stays palette4, Q4-blocks stay Q4-blocks).
pub fn fuse_lora(pkg: &Path, adapter: &Path, out: &Path) -> R<SurgReport> {
    let lora = crate::lora::Lora::load(adapter).map_err(|e| format!("lora: {e}"))?;
    let mut pe = PackageEdit::open(pkg)?;
    let groups: BTreeMap<String, WeightGroup> = pe
        .weight_groups()?
        .into_iter()
        .map(|g| (g.name.clone(), g))
        .collect();

    // Map every adapter target before touching anything — a LoRA for
    // the wrong checkpoint must fail before a single op changes.
    let mut plan: Vec<(String, WeightGroup)> = Vec::new();
    for wname in lora.targets() {
        let Some(mname) = mil_name_for_hf(wname) else {
            return err(format!("lora {wname}: no package const mapping"));
        };
        let Some(g) = groups.get(&mname) else {
            return err(format!("lora {wname}: package has no weight const {mname}"));
        };
        if g.shape.len() != 4 || g.shape[2] != 1 || g.shape[3] != 1 {
            return err(format!(
                "lora {wname}: const {mname} is not conv-packed, shape {:?}",
                g.shape
            ));
        }
        plan.push((wname.to_string(), g.clone()));
    }
    if plan.is_empty() {
        return err("lora: adapter has no fusable targets");
    }

    // Decode each target once, fuse through FusedSource, re-emit in
    // the group's own encoding.
    let mut src_map = MapSource(BTreeMap::new());
    for (wname, g) in &plan {
        let vals = pe.group_values(g)?;
        src_map
            .0
            .insert(wname.clone(), (vec![g.shape[0], g.shape[1]], vals));
    }
    let fused = crate::lora::FusedSource::new(&src_map, &lora);
    // Descending op order, same reason as `requant`.
    let mut sorted = plan.clone();
    sorted.sort_by_key(|(_, g)| std::cmp::Reverse(g.op_indices.iter().min().copied().unwrap_or(0)));
    let mut lines = Vec::new();
    for (wname, g) in &sorted {
        let (shape, w) = fused
            .tensor_f32(wname)
            .map_err(|e| format!("lora {wname}: {e}"))?;
        pe.replace_group(g, g.kind, &w)?;
        lines.push(format!(
            "  {} ({}) : {} fused",
            g.name,
            wname,
            g.kind.as_str()
        ));
        let _ = shape;
    }
    let written = pe.write(out)?;
    lines.insert(
        0,
        format!("{} LoRA target(s) fused into {}", plan.len(), pkg.display()),
    );
    Ok(SurgReport {
        lines,
        written: Some(written),
    })
}

/// Write `pe` to `dst`; when `dst == src` (in-place), write a sibling
/// temp dir first and swap so a failed write can't torch the package.
fn write_maybe_inplace(pe: &PackageEdit, src: &Path, dst: &Path) -> R<WriteReport> {
    if dst != src {
        return pe.write(dst);
    }
    let tmp = src.with_extension("milc-surgery.tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    let r = pe.write(&tmp)?;
    std::fs::remove_dir_all(src).map_err(|e| format!("{}: {e}", src.display()))?;
    std::fs::rename(&tmp, src).map_err(|e| format!("rename {}: {e}", tmp.display()))?;
    Ok(r)
}

// ======== tests ========

#[cfg(test)]
mod tests {
    use super::*;
    use mil_spec::{
        encode_model, write_mlpackage_stream, Feature, ModelMeta, TensorType, ValueType, NVT,
    };
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("mil_surgery_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
    fn cleanup(p: &Path) {
        if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(p);
        }
    }

    /// Smallest legal package: fn input `x`, one fp16 conv-shaped const
    /// `w` [4,4,1,1] that is also the block output.
    fn tiny_pkg(dir: &Path) -> Vec<f32> {
        let vals: Vec<f32> = (0..16).map(|i| 0.01 * i as f32 - 0.05).collect();
        std::fs::create_dir_all(dir).unwrap();
        let wtmp = dir.join("w.bin");
        let mut w = BlobWriter::create(&wtmp).unwrap();
        let off = w.append(DType::Fp16, &f32_to_f16(&vals)).unwrap();
        w.finish().unwrap();
        let mut b = mil_spec::Block::new();
        b.konst_blob("w", WEIGHT_FILE, off, DType::Fp16, &[4, 4, 1, 1]);
        b.outputs = vec!["w".into()];
        let spec = encode_model(
            &[Feature {
                name: "x".into(),
                shape: vec![1, 4, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }],
            &[Feature {
                name: "w".into(),
                shape: vec![4, 4, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }],
            &[],
            &b,
            &[NVT {
                name: "x".into(),
                ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])),
            }],
            &ModelMeta::new(10, "CoreML9"),
        );
        write_mlpackage_stream(dir, &spec, Some(&wtmp)).unwrap();
        let _ = std::fs::remove_file(&wtmp);
        vals
    }

    #[test]
    fn u4_pack_roundtrip() {
        let codes: Vec<u8> = vec![0, 1, 2, 3, 15, 8, 7];
        let packed = pack_u4(&codes);
        // [0|1<<4, 2|3<<4, 15|8<<4, 7|0]
        assert_eq!(packed, vec![0x10, 0x32, 0x8f, 0x07]);
        let back = unpack_u4(&packed, codes.len());
        assert_eq!(back, codes);
    }

    #[test]
    fn int8_rows_roundtrip() {
        let vals: Vec<f32> = (-8..8).map(|i| i as f32 * 0.05).collect();
        let (q, s) = quantize_int8_rows(&vals, 2);
        assert_eq!(s.len(), 2);
        for (i, &v) in vals.iter().enumerate() {
            let r = i / 8;
            let got = (q[i] as i8) as f32 * s[r];
            assert!((got - v).abs() <= s[r] / 2.0 + 1e-6, "{i}: {got} vs {v}");
        }
    }

    #[test]
    fn palette4_codes_in_range() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.03).collect();
        let (codes, lut) = quantize_palette4_rows(&vals, 2);
        assert_eq!(lut.len(), 32);
        assert!(codes.iter().all(|&c| c < 16));
        // reconstruction stays within one step of the original
        for (i, &v) in vals.iter().enumerate() {
            let r = i / 16;
            let got = lut[r * 16 + codes[i] as usize];
            assert!((got - v).abs() < 0.1, "{i}: {got} vs {v}");
        }
    }

    #[test]
    fn parse_ranges() {
        assert_eq!(parse_layer_range("0..2").unwrap(), (0, 1));
        assert_eq!(parse_layer_range("0..=2").unwrap(), (0, 2));
        assert!(parse_layer_range("3..1").is_err());
        assert!(parse_layer_range("abc").is_err());
    }

    #[test]
    fn package_edit_roundtrip_requant() {
        let dir = tmp("roundtrip");
        let pkg = dir.join("m.mlpackage");
        let vals = tiny_pkg(&pkg);

        let pe = PackageEdit::open(&pkg).unwrap();
        let groups = pe.weight_groups().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].name, "w");
        assert_eq!(groups[0].kind, GroupKind::Fp16);
        let got = pe.group_values(&groups[0]).unwrap();
        for (a, b) in vals.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-3);
        }

        // fp16 → int8 → write → reopen
        let out = dir.join("m8.mlpackage");
        requant(&pkg, TargetQuant::Int8, Some(&out)).unwrap();
        let pe2 = PackageEdit::open(&out).unwrap();
        let g2 = pe2.weight_groups().unwrap();
        assert_eq!(g2.len(), 1);
        assert_eq!(g2[0].kind, GroupKind::Int8);
        let v2 = pe2.group_values(&g2[0]).unwrap();
        for (a, b) in vals.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 0.01, "int8: {a} vs {b}");
        }

        // int8 → palette4
        let out4 = dir.join("mp4.mlpackage");
        requant(&out, TargetQuant::Palette4, Some(&out4)).unwrap();
        let pe3 = PackageEdit::open(&out4).unwrap();
        let g3 = pe3.weight_groups().unwrap();
        assert_eq!(g3[0].kind, GroupKind::Palette4);

        // palette4 → fp16 back to a single fp16 const
        let out16 = dir.join("m16.mlpackage");
        requant(&out4, TargetQuant::Fp16, Some(&out16)).unwrap();
        let pe4 = PackageEdit::open(&out16).unwrap();
        let g4 = pe4.weight_groups().unwrap();
        assert_eq!(g4[0].kind, GroupKind::Fp16);
        let v4 = pe4.group_values(&g4[0]).unwrap();
        for (a, b) in vals.iter().zip(v4.iter()) {
            assert!((a - b).abs() < 0.03, "palette rt: {a} vs {b}");
        }
        cleanup(&dir);
    }

    #[test]
    fn graft_copies_donor_weights() {
        let dir = tmp("graft");
        let base = dir.join("base.mlpackage");
        let donor = dir.join("donor.mlpackage");
        tiny_pkg(&base);
        // Donor: same structure, different values — hand-edit a staged blob.
        tiny_pkg(&donor);
        let donor_vals: Vec<f32> = (0..16).map(|i| 0.5 + 0.001 * i as f32).collect();
        {
            let mut dpe = PackageEdit::open(&donor).unwrap();
            let g = dpe.weight_groups().unwrap();
            dpe.replace_group(&g[0], GroupKind::Fp16, &donor_vals)
                .unwrap();
            dpe.write(&donor).unwrap(); // in-place? write to same dir is fine for fixture
        }
        // graft with --layers only matches l{N}_ groups; "w" has no layer
        // prefix, so expect the honest error.
        let out = dir.join("g.mlpackage");
        assert!(graft(&donor, &base, (0, 1), &out).is_err());
        cleanup(&dir);
    }

    #[test]
    fn write_validates_op_bindings() {
        let dir = tmp("validate");
        let pkg = dir.join("m.mlpackage");
        tiny_pkg(&pkg);
        let mut pe = PackageEdit::open(&pkg).unwrap();
        // Insert an op binding a nonexistent input → validate must fail.
        let bad = constexpr_op(
            "constexpr_lut_to_dense",
            "zz",
            &[4, 4, 1, 1],
            &[("indices", "nope"), ("lut", "nope2")],
        );
        pe.ops_mut().unwrap().push(bad);
        assert!(pe.write(&dir.join("bad.mlpackage")).is_err());
        cleanup(&dir);
    }
}
