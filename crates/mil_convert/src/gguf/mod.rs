//! `gguf` — zero-dependency GGUF reader (format used by llama.cpp).
//!
//! Parses v2/v3 headers, all metadata value types (including nested
//! arrays), `general.alignment`, tensor infos, and split files
//! (`name-00001-of-0000N.gguf`). Tensor bytes are read lazily and
//! dequantized through [`dequant`], which is bit-exact against ggml's
//! `to_float` at the pinned commit (see `tests/gguf_golden/`).
//!
//! Format reference: ggml/src/gguf.cpp @ llama.cpp commit
//! 10a60cf303566e10d6a7a2774c17d2085503d87b — same validation rules:
//! v1 rejected, endianness-mismatch rejected, tensor `ne[0]` must be a
//! multiple of the block size, offsets must be aligned and in-bounds.

pub mod config;
pub mod dequant;
pub mod names;
pub mod tables;
pub mod tokenizer;

use crate::{NativeQuant, WeightSource};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// `ggml_type` ids — ggml/include/ggml.h at the pinned commit.
///
/// Gaps are ids whose types were removed from ggml; they parse as
/// [`GgufType::Removed`] so a file naming one errors out with a clear
/// message instead of a bare "unknown type".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    /// Intermediate quantization result — no `to_float` in ggml.
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
    Q2_0,
    /// Type id retired from ggml (Q4_2, Q4_3, the `*_4_4`/`_4_8`/`_8_8`
    /// reblocked family, IQ4_NL repacks).
    Removed(u32),
    /// Id ≥ `GGML_TYPE_COUNT` at the pinned commit — a newer type this
    /// build predates, or a corrupt file.
    Unknown(u32),
}

impl GgufType {
    pub fn from_id(id: u32) -> GgufType {
        match id {
            0 => GgufType::F32,
            1 => GgufType::F16,
            2 => GgufType::Q4_0,
            3 => GgufType::Q4_1,
            6 => GgufType::Q5_0,
            7 => GgufType::Q5_1,
            8 => GgufType::Q8_0,
            9 => GgufType::Q8_1,
            10 => GgufType::Q2K,
            11 => GgufType::Q3K,
            12 => GgufType::Q4K,
            13 => GgufType::Q5K,
            14 => GgufType::Q6K,
            15 => GgufType::Q8K,
            16 => GgufType::Iq2Xxs,
            17 => GgufType::Iq2Xs,
            18 => GgufType::Iq3Xxs,
            19 => GgufType::Iq1S,
            20 => GgufType::Iq4Nl,
            21 => GgufType::Iq3S,
            22 => GgufType::Iq2S,
            23 => GgufType::Iq4Xs,
            24 => GgufType::I8,
            25 => GgufType::I16,
            26 => GgufType::I32,
            27 => GgufType::I64,
            28 => GgufType::F64,
            29 => GgufType::Iq1M,
            30 => GgufType::Bf16,
            34 => GgufType::Tq1_0,
            35 => GgufType::Tq2_0,
            39 => GgufType::Mxfp4,
            40 => GgufType::Nvfp4,
            41 => GgufType::Q1_0,
            42 => GgufType::Q2_0,
            // 4,5 = Q4_2/Q4_3; 31..=33 = Q4_0_4_4 family;
            // 36..=38 = IQ4_NL repacks — all removed from ggml.
            4 | 5 | 31 | 32 | 33 | 36 | 37 | 38 => GgufType::Removed(id),
            _ => GgufType::Unknown(id),
        }
    }

    /// ggml type name (`q4_0`, `iq2_xxs`, …).
    pub fn name(&self) -> String {
        let s = match self {
            GgufType::F32 => "f32",
            GgufType::F16 => "f16",
            GgufType::Q4_0 => "q4_0",
            GgufType::Q4_1 => "q4_1",
            GgufType::Q5_0 => "q5_0",
            GgufType::Q5_1 => "q5_1",
            GgufType::Q8_0 => "q8_0",
            GgufType::Q8_1 => "q8_1",
            GgufType::Q2K => "q2_K",
            GgufType::Q3K => "q3_K",
            GgufType::Q4K => "q4_K",
            GgufType::Q5K => "q5_K",
            GgufType::Q6K => "q6_K",
            GgufType::Q8K => "q8_K",
            GgufType::Iq2Xxs => "iq2_xxs",
            GgufType::Iq2Xs => "iq2_xs",
            GgufType::Iq3Xxs => "iq3_xxs",
            GgufType::Iq1S => "iq1_s",
            GgufType::Iq4Nl => "iq4_nl",
            GgufType::Iq3S => "iq3_s",
            GgufType::Iq2S => "iq2_s",
            GgufType::Iq4Xs => "iq4_xs",
            GgufType::I8 => "i8",
            GgufType::I16 => "i16",
            GgufType::I32 => "i32",
            GgufType::I64 => "i64",
            GgufType::F64 => "f64",
            GgufType::Iq1M => "iq1_m",
            GgufType::Bf16 => "bf16",
            GgufType::Tq1_0 => "tq1_0",
            GgufType::Tq2_0 => "tq2_0",
            GgufType::Mxfp4 => "mxfp4",
            GgufType::Nvfp4 => "nvfp4",
            GgufType::Q1_0 => "q1_0",
            GgufType::Q2_0 => "q2_0",
            GgufType::Removed(id) => return format!("removed({id})"),
            GgufType::Unknown(id) => return format!("unknown({id})"),
        };
        s.to_string()
    }
}

impl std::fmt::Display for GgufType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A GGUF metadata value — all 13 `gguf_type` variants.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Element type tag + values. `Arr` is never itself an element
    /// type — nested arrays are rejected like ggml does.
    Arr(ArrKind, Vec<Value>),
}

/// The element type of a [`Value::Arr`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrKind {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    U64,
    I64,
    F64,
}

impl Value {
    /// Scalar as i64 where lossless (ints, bool). None for floats/strings.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::U8(v) => Some(*v as i64),
            Value::I8(v) => Some(*v as i64),
            Value::U16(v) => Some(*v as i64),
            Value::I16(v) => Some(*v as i64),
            Value::U32(v) => Some(*v as i64),
            Value::I32(v) => Some(*v as i64),
            Value::Bool(v) => Some(*v as i64),
            Value::U64(v) => i64::try_from(*v).ok(),
            Value::I64(v) => Some(*v),
            _ => None,
        }
    }
    /// Scalar as u32.
    pub fn as_u32(&self) -> Option<u32> {
        self.as_i64().and_then(|v| u32::try_from(v).ok())
    }
    /// Scalar as f64 (ints and floats).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }
    /// String contents.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    /// Array elements.
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(_, v) => Some(v),
            _ => None,
        }
    }
}

/// One tensor's location in a shard's data section.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// GGUF tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Element count per dim, ggml order (`ne[0]` innermost).
    pub ne: Vec<i64>,
    /// Quantization/storage type.
    pub gguf_type: GgufType,
    /// Byte offset within the data section.
    pub offset: u64,
    /// Total stored bytes.
    pub nbytes: u64,
    /// Shard index this tensor lives in.
    pub shard: usize,
}

impl TensorInfo {
    /// Total element count.
    pub fn nelements(&self) -> i64 {
        self.ne.iter().product()
    }
    /// Shape in HF order (innermost dim last) — `ne` reversed.
    pub fn hf_shape(&self) -> Vec<i64> {
        self.ne.iter().rev().cloned().collect()
    }
}

#[derive(Debug)]
struct Shard {
    path: PathBuf,
    /// Absolute file offset of the tensor data section.
    data_base: u64,
}

/// An open GGUF file set — one file, or all shards of a split file
/// unified under a single tensor index.
#[derive(Debug)]
pub struct Gguf {
    shards: Vec<Shard>,
    /// GGUF name → tensor info.
    tensors: BTreeMap<String, TensorInfo>,
    /// Metadata key → value (merged from shard 0; split metadata
    /// excluded from the public view? — kept verbatim).
    meta: BTreeMap<String, Value>,
    /// Format version (2 or 3 — v1 is rejected at open).
    pub version: u32,
    /// `general.alignment` (default 32).
    pub alignment: u64,
    /// `general.architecture` (e.g. `qwen3`, `llama`), if present.
    pub arch: Option<String>,
    /// Q/K un-permute head counts for llama-arch files:
    /// `(n_head, n_head_kv)`, `None` when no un-permute applies.
    permute_heads: Option<(i64, i64)>,
}

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io: {e}"),
            GgufError::Format(m) => write!(f, "gguf: {m}"),
        }
    }
}
impl std::error::Error for GgufError {}
impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}
pub(crate) fn fmt_err(m: impl Into<String>) -> GgufError {
    GgufError::Format(m.into())
}

pub(crate) type Result<T> = std::result::Result<T, GgufError>;

const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_NAME: usize = 64;
const MAX_STRING_LEN: u64 = 1024 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 1024 * 1024 * 1024;

/// Bounds-checked cursor over the header bytes.
struct Rd<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.buf.len().saturating_sub(self.pos) {
            return Err(fmt_err("header truncated"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_STRING_LEN {
            return Err(fmt_err(format!("string length {n} exceeds maximum")));
        }
        let s = std::str::from_utf8(self.take(n as usize)?)
            .map_err(|_| fmt_err("string is not valid utf-8"))?;
        Ok(s.to_string())
    }
}

fn read_value(r: &mut Rd, ty: i32) -> Result<Value> {
    Ok(match ty {
        0 => Value::U8(r.u8()?),
        1 => Value::I8(r.i8()?),
        2 => Value::U16(r.u16()?),
        3 => Value::I16(r.i16()?),
        4 => Value::U32(r.u32()?),
        5 => Value::I32(r.i32()?),
        6 => Value::F32(r.f32()?),
        7 => Value::Bool(r.i8()? != 0),
        8 => Value::Str(r.string()?),
        10 => Value::U64(r.u64()?),
        11 => Value::I64(r.i64()?),
        12 => Value::F64(r.f64()?),
        9 => return Err(fmt_err("nested array metadata is not allowed")),
        other => return Err(fmt_err(format!("invalid metadata type {other}"))),
    })
}

fn read_kv(r: &mut Rd) -> Result<(String, Value)> {
    let key = r.string()?;
    if key.is_empty() {
        return Err(fmt_err("empty metadata key"));
    }
    let ty = r.i32()?;
    if ty == 9 {
        // array: element type tag + u64 count, then values
        let elem_ty = r.i32()?;
        let n = r.u64()?;
        if n > MAX_ARRAY_ELEMENTS {
            return Err(fmt_err(format!("array of {n} elements exceeds maximum")));
        }
        let kind = match elem_ty {
            0 => ArrKind::U8,
            1 => ArrKind::I8,
            2 => ArrKind::U16,
            3 => ArrKind::I16,
            4 => ArrKind::U32,
            5 => ArrKind::I32,
            6 => ArrKind::F32,
            7 => ArrKind::Bool,
            8 => ArrKind::Str,
            10 => ArrKind::U64,
            11 => ArrKind::I64,
            12 => ArrKind::F64,
            other => {
                return Err(fmt_err(format!(
                    "array of type {other} is not a valid metadata value"
                )))
            }
        };
        let mut vals = Vec::with_capacity(n.min(1 << 20) as usize);
        for _ in 0..n {
            vals.push(read_value(r, elem_ty)?);
        }
        Ok((key, Value::Arr(kind, vals)))
    } else {
        Ok((key, read_value(r, ty)?))
    }
}

/// Parse one shard's header; returns (metadata, tensor infos, data_base).
fn parse_shard(buf: &[u8], file_len: u64, shard_idx: usize) -> Result<ShardHeader> {
    let mut r = Rd { buf, pos: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(fmt_err("bad magic — not a GGUF file"));
    }
    let version = r.u32()?;
    if version == 0 {
        return Err(fmt_err("bad GGUF version 0"));
    }
    if version & 0x0000FFFF == 0 {
        return Err(fmt_err(format!(
            "GGUF version field {version:#x} — file is likely byte-swapped \
             (big-endian GGUF is not supported)"
        )));
    }
    if version == 1 {
        return Err(fmt_err("GGUF v1 is not supported (v2/v3 only)"));
    }
    if version > 3 {
        return Err(fmt_err(format!(
            "GGUF version {version} > 3 — this reader targets the pinned format"
        )));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > (file_len / 8).max(1) || n_kv > (file_len / 8).max(1) {
        return Err(fmt_err(format!(
            "implausible counts: {n_tensors} tensors / {n_kv} kv in {file_len} bytes"
        )));
    }

    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let (k, v) = read_kv(&mut r)?;
        if meta.insert(k.clone(), v).is_some() {
            return Err(fmt_err(format!("duplicate metadata key '{k}'")));
        }
    }

    let alignment = match meta.get("general.alignment") {
        None => 32u64,
        Some(Value::U32(a)) => *a as u64,
        Some(_) => return Err(fmt_err("general.alignment must be uint32")),
    };
    if alignment == 0 || alignment & (alignment - 1) != 0 {
        return Err(fmt_err(format!(
            "alignment {alignment} is not a power of 2"
        )));
    }

    let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(fmt_err(format!("tensor name too long: '{name}'")));
        }
        let n_dims = r.u32()? as usize;
        if n_dims > GGML_MAX_DIMS {
            return Err(fmt_err(format!(
                "tensor '{name}': {n_dims} dims > {GGML_MAX_DIMS}"
            )));
        }
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = r.i64()?;
            if d < 0 {
                return Err(fmt_err(format!("tensor '{name}': negative dim {d}")));
            }
            ne.push(d);
        }
        let ty = r.i32()?;
        if ty < 0 {
            return Err(fmt_err(format!("tensor '{name}': invalid type id {ty}")));
        }
        let gguf_type = GgufType::from_id(ty as u32);
        match gguf_type {
            GgufType::Unknown(id) => {
                return Err(fmt_err(format!(
                    "tensor '{name}': type id {id} >= GGML_TYPE_COUNT (43)"
                )))
            }
            GgufType::Removed(id) => {
                return Err(fmt_err(format!(
                    "tensor '{name}': type id {id} was removed from ggml \
                     (no to_float at the pinned commit)"
                )))
            }
            _ => {}
        }
        let n_elems: i64 = ne.iter().product();
        let qk = dequant::block_len(gguf_type) as i64;
        if qk == 0 {
            return Err(fmt_err(format!(
                "tensor '{name}': {gguf_type} has no layout definition"
            )));
        }
        // scalar types have block_len 1; quantized types need ne[0] divisibility
        if qk > 1 && (ne.is_empty() || ne[0] % qk != 0) {
            return Err(fmt_err(format!(
                "tensor '{name}': {gguf_type} row {} not a multiple of block {qk}",
                ne.first().copied().unwrap_or(0)
            )));
        }
        let nbytes = (n_elems as u64 / qk as u64) * dequant::block_size(gguf_type) as u64;
        let offset = r.u64()?;
        if offset % alignment != 0 {
            return Err(fmt_err(format!(
                "tensor '{name}': offset {offset} not aligned to {alignment}"
            )));
        }
        tensors.push(TensorInfo {
            name,
            ne,
            gguf_type,
            offset,
            nbytes,
            shard: shard_idx,
        });
    }

    // data section begins at the next aligned file offset
    let data_base = (r.pos as u64).div_ceil(alignment) * alignment;
    Ok(ShardHeader {
        version,
        alignment,
        meta,
        tensors,
        data_base,
    })
}

struct ShardHeader {
    version: u32,
    alignment: u64,
    meta: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    data_base: u64,
}

/// `foo-00002-of-00007.gguf` → ("foo", 2, 7).
fn split_name(path: &Path) -> Option<(String, u32, u32)> {
    let stem = path.file_stem()?.to_str()?;
    let (prefix, count_s) = stem.rsplit_once("-of-")?;
    let (base, no_s) = prefix.rsplit_once('-')?;
    let no: u32 = no_s.parse().ok()?;
    let count: u32 = count_s.parse().ok()?;
    if no == 0 || no > count || count == 0 {
        return None;
    }
    Some((base.to_string(), no, count))
}

impl Gguf {
    /// Open a GGUF file. If it is one shard of a split file
    /// (`split.count` > 1), all sibling shards must exist in the same
    /// directory and are merged into one tensor index.
    pub fn open(path: &Path) -> Result<Gguf> {
        Self::open_impl(path, true)
    }

    fn open_impl(path: &Path, follow_splits: bool) -> Result<Gguf> {
        let mut f = std::fs::File::open(path)?;
        let file_len = f.metadata()?.len();
        // The header must be read up front; bound it so a corrupt
        // count can't make us slurp a huge file into memory.
        let header_cap = file_len.min(512 * 1024 * 1024) as usize;
        let mut buf = Vec::new();
        let mut rd = std::io::Read::take(&mut f, header_cap as u64);
        rd.read_to_end(&mut buf)?;
        let hdr = parse_shard(&buf, file_len, 0)?;
        if hdr.data_base > file_len {
            return Err(fmt_err("data section offset beyond end of file"));
        }
        for t in &hdr.tensors {
            let end = hdr
                .data_base
                .checked_add(t.offset)
                .and_then(|v| v.checked_add(t.nbytes))
                .ok_or_else(|| fmt_err(format!("tensor '{}': offset overflow", t.name)))?;
            if end > file_len {
                return Err(fmt_err(format!(
                    "tensor '{}': data [{}, {}) beyond file end {}",
                    t.name,
                    hdr.data_base + t.offset,
                    end,
                    file_len
                )));
            }
        }

        let split_count = hdr
            .meta
            .get("split.count")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if !follow_splits || split_count <= 1 {
            return Self::from_header(path, hdr);
        }

        // ---- split file: discover siblings ----
        let (base, _no, count) = split_name(path).ok_or_else(|| {
            fmt_err(format!(
                "file declares split.count={split_count} but its name \
                 is not <base>-000NN-of-0000M.gguf: {}",
                path.display()
            ))
        })?;
        if split_count as u32 != count {
            return Err(fmt_err(format!(
                "split.count={split_count} but filename says {count} shards"
            )));
        }
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let mut all: Option<Gguf> = None;
        for i in 1..=count {
            let sibling = dir.join(format!("{base}-{i:05}-of-{count:05}.gguf"));
            if !sibling.exists() {
                return Err(fmt_err(format!(
                    "missing split shard {}",
                    sibling.display()
                )));
            }
            let g = Gguf::open_impl(&sibling, false)?;
            let this_no = g
                .meta
                .get("split.no")
                .and_then(Value::as_i64)
                .ok_or_else(|| fmt_err("split shard missing split.no"))?;
            if this_no + 1 != i as i64 {
                return Err(fmt_err(format!(
                    "{}: split.no={this_no}, expected {}",
                    sibling.display(),
                    i - 1
                )));
            }
            match &mut all {
                None => all = Some(g),
                Some(a) => a.merge(g)?,
            }
        }
        let all = all.unwrap();
        let expected = all.meta.get("split.tensors.count").and_then(Value::as_i64);
        if let Some(exp) = expected {
            if exp != all.tensors.len() as i64 {
                return Err(fmt_err(format!(
                    "split.tensors.count={exp} but {} tensors were indexed",
                    all.tensors.len()
                )));
            }
        }
        Ok(all)
    }

    fn from_header(path: &Path, hdr: ShardHeader) -> Result<Gguf> {
        let mut tensors = BTreeMap::new();
        for t in hdr.tensors {
            if tensors.insert(t.name.clone(), t).is_some() {
                return Err(fmt_err("duplicate tensor name"));
            }
        }
        let arch = hdr
            .meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .map(str::to_string);
        let permute_heads = Self::permute_heads(&arch, &hdr.meta);
        Ok(Gguf {
            shards: vec![Shard {
                path: path.to_path_buf(),
                data_base: hdr.data_base,
            }],
            tensors,
            meta: hdr.meta,
            version: hdr.version,
            alignment: hdr.alignment,
            permute_heads,
            arch,
        })
    }

    /// Merge another shard's tensors (split files share metadata).
    fn merge(&mut self, other: Gguf) -> Result<()> {
        if other.alignment != self.alignment {
            return Err(fmt_err("split shards disagree on alignment"));
        }
        let shard_off = self.shards.len();
        for (name, mut t) in other.tensors {
            t.shard += shard_off;
            if self.tensors.insert(name.clone(), t).is_some() {
                return Err(fmt_err(format!("tensor '{name}' in multiple shards")));
            }
        }
        self.shards.extend(other.shards);
        Ok(())
    }

    /// Whether the llama Q/K head-interleave un-permute applies.
    /// `conversion/llama.py` at the pinned commit permutes attn_q with
    /// n_head and attn_k with n_head_kv for arch `llama` — the same
    /// reshape/swapaxes is an involution, so the inverse is itself.
    fn permute_heads(arch: &Option<String>, meta: &BTreeMap<String, Value>) -> Option<(i64, i64)> {
        if arch.as_deref() != Some("llama") {
            return None;
        }
        let nh = meta.get("llama.attention.head_count")?.as_i64()?;
        let kvh = meta
            .get("llama.attention.head_count_kv")
            .and_then(Value::as_i64)
            .unwrap_or(nh);
        Some((nh, kvh))
    }

    /// All metadata.
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.meta
    }
    /// One metadata value.
    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }
    /// All tensor infos (GGUF names), sorted by name.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.values()
    }
    /// Tensor info by GGUF name.
    pub fn tensor(&self, ggml_name: &str) -> Option<&TensorInfo> {
        self.tensors.get(ggml_name)
    }
    /// Tensor info by HF canonical name (`model.layers.0.self_attn.q_proj.weight`).
    pub fn tensor_hf(&self, hf_name: &str) -> Option<&TensorInfo> {
        names::hf_to_ggml(hf_name).and_then(|g| self.tensors.get(g.as_str()))
    }
    /// Whether an HF-named tensor exists.
    pub fn has(&self, hf_name: &str) -> bool {
        self.tensor_hf(hf_name).is_some()
    }
    /// Shard paths making up this file set.
    pub fn paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
    }

    /// Raw stored bytes for a tensor (still quantized).
    pub fn tensor_bytes(&self, ggml_name: &str) -> Result<Vec<u8>> {
        let t = self
            .tensors
            .get(ggml_name)
            .ok_or_else(|| fmt_err(format!("tensor {ggml_name} not in file")))?;
        let sh = &self.shards[t.shard];
        let mut f = std::fs::File::open(&sh.path)?;
        f.seek(SeekFrom::Start(sh.data_base + t.offset))?;
        let mut buf = vec![0u8; t.nbytes as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Dequantized f32 contents, in HF order (`hf_shape()`).
    ///
    /// For llama-arch files the stored `attn_q`/`attn_k` rows carry the
    /// converter's RoPE head interleave — this applies the inverse
    /// (see [`Gguf::permute_heads`]).
    pub fn tensor_f32(&self, hf_name: &str) -> Result<(Vec<i64>, Vec<f32>)> {
        let ggml_name = names::hf_to_ggml(hf_name)
            .ok_or_else(|| fmt_err(format!("no ggml name for HF tensor {hf_name}")))?;
        let t = self
            .tensors
            .get(ggml_name.as_str())
            .ok_or_else(|| fmt_err(format!("tensor {ggml_name} not in file")))?;
        let raw = self.tensor_bytes(&ggml_name)?;
        let n = t.nelements() as usize;
        let mut dst = vec![0f32; n];
        dequant::dequant(t.gguf_type, &raw, &mut dst)
            .map_err(|e| fmt_err(format!("{ggml_name}: {}", e.0)))?;
        if let Some((nh, kvh)) = self.permute_heads {
            let heads = match names::permute_kind(&ggml_name) {
                Some(names::PermuteKind::Q) => Some(nh),
                Some(names::PermuteKind::K) => Some(kvh),
                None => None,
            };
            if let Some(h) = heads {
                let shape = t.hf_shape();
                let rows = shape.first().copied().unwrap_or(1) as usize;
                let cols = n / rows.max(1);
                if rows % (h as usize) == 0 && rows > 0 {
                    unpermute(&mut dst, h as usize, rows, cols);
                }
            }
        }
        Ok((t.hf_shape(), dst))
    }

    /// Dequantized then converted to f16 little-endian — same contract
    /// as `safetensors::Safetensors::tensor_f16`.
    pub fn tensor_f16(&self, hf_name: &str) -> Result<(Vec<i64>, Vec<u8>)> {
        let (shape, f32s) = self.tensor_f32(hf_name)?;
        let mut out = Vec::with_capacity(f32s.len() * 2);
        for v in f32s {
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        Ok((shape, out))
    }
}

/// The llama Q/K row interleave applied to a raw-quantized tensor.
/// Blocks live inside a row (ggml requires `ne[0] % block == 0`), so the
/// permute is a pure row reorder — the same index math as [`unpermute`]
/// at `row_bytes` granularity. `data` and `scales`/`offset` each get
/// reordered at their own row stride.
fn unpermute_rows(data: &mut [u8], row_bytes: usize, n_head: usize) {
    let rows = data.len() / row_bytes;
    if rows == 0 || rows % n_head != 0 {
        return;
    }
    let hd2 = rows / n_head / 2;
    let src = data.to_vec();
    for h in 0..n_head {
        for i in 0..2 {
            for j in 0..hd2 {
                let d = ((h * 2 + i) * hd2 + j) * row_bytes;
                let s = ((h * hd2 + j) * 2 + i) * row_bytes;
                data[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
            }
        }
    }
}

/// Lossless ggml → MIL-quantized transcode.
///
/// Q8_0 stores `(fp16 d, int8 q[32])` per 32 elements and dequantizes as
/// `y = q * d` — exactly `constexpr_blockwise_shift_scale` with a
/// per-32-block scale and no offset. Q4_0 stores `(fp16 d, uint4
/// nibble[32])` and dequantizes as `y = (n - 8) * d` — the same op with
/// a constant offset of 8. Both keep the source's original code points,
/// so the emitted tensor's dequantized value is bit-identical to ggml's
/// (an f16 product of exact f16 operands — a single rounding).
///
/// `rows`/`cols` are the HF `(out, in)` shape. Row-major raw bytes are
/// repacked into (codes | scales | offset) blobs; the llama Q/K row
/// interleave is un-applied at row granularity.
fn transcode_q8_0(raw: &[u8], rows: usize, cols: usize) -> NativeQuant {
    debug_assert_eq!(raw.len(), rows * (cols / 32) * 34);
    let nb = cols / 32;
    let mut data = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * nb * 2);
    for r in 0..rows {
        for b in 0..nb {
            let blk = &raw[(r * nb + b) * 34..(r * nb + b) * 34 + 34];
            scales.extend_from_slice(&blk[0..2]);
            data.extend_from_slice(&blk[2..34]);
        }
    }
    NativeQuant {
        data,
        data_dtype: mil_spec::DType::Int8,
        scales,
        offset: None,
        block: 32,
    }
}

/// Same transcode for Q4_0: `(fp16 d, packed nibble[16])` per block.
/// ggml nibble order is `(lo=elem j, hi=elem j+16)`; MIL packs
/// `(lo=even, hi=odd)` — the repack below moves each nibble into place.
/// The emitted offset is a constant 8 (uint4 `0x88` per byte pair).
fn transcode_q4_0(raw: &[u8], rows: usize, cols: usize) -> NativeQuant {
    debug_assert_eq!(raw.len(), rows * (cols / 32) * 18);
    let nb = cols / 32;
    let mut data = Vec::with_capacity(rows * cols / 2);
    let mut scales = Vec::with_capacity(rows * nb * 2);
    for r in 0..rows {
        for b in 0..nb {
            let blk = &raw[(r * nb + b) * 18..(r * nb + b) * 18 + 18];
            scales.extend_from_slice(&blk[0..2]);
            let nib = &blk[2..18];
            for m in 0..8usize {
                // MIL byte m covers elements (2m, 2m+1)
                let lo = nib[2 * m] & 0x0F;
                let hi = nib[2 * m + 1] & 0x0F;
                data.push(lo | (hi << 4));
            }
            for m in 0..8usize {
                let lo = nib[2 * m] >> 4;
                let hi = nib[2 * m + 1] >> 4;
                data.push(lo | (hi << 4));
            }
        }
    }
    // offset=8 per scale element, uint4-packed
    let off = vec![0x88u8; rows * nb / 2];
    NativeQuant {
        data,
        data_dtype: mil_spec::DType::Uint4,
        scales,
        offset: Some((off, mil_spec::DType::Uint4)),
        block: 32,
    }
}

impl Gguf {
    /// Raw-quantized transcode for Q8_0/Q4_0 2-D tensors — see
    /// [`WeightSource::native_qblocks`].
    pub fn native_qblocks_hf(&self, hf_name: &str) -> Result<Option<NativeQuant>> {
        let Some(ggml_name) = names::hf_to_ggml(hf_name) else {
            return Ok(None);
        };
        let Some(t) = self.tensors.get(ggml_name.as_str()) else {
            return Ok(None);
        };
        if t.gguf_type != GgufType::Q8_0 && t.gguf_type != GgufType::Q4_0 {
            return Ok(None);
        }
        let shape = t.hf_shape();
        if shape.len() != 2 {
            return Ok(None);
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        if cols % 32 != 0 {
            return Ok(None);
        }
        let raw = self.tensor_bytes(&ggml_name)?;
        let mut nq = match t.gguf_type {
            GgufType::Q8_0 => transcode_q8_0(&raw, rows, cols),
            _ => transcode_q4_0(&raw, rows, cols),
        };
        // The dequantized path un-permutes llama attn_q/attn_k rows —
        // apply the same reorder to the quantized rows.
        if let Some((nh, kvh)) = self.permute_heads {
            let heads = match names::permute_kind(&ggml_name) {
                Some(names::PermuteKind::Q) => Some(nh),
                Some(names::PermuteKind::K) => Some(kvh),
                None => None,
            };
            if let Some(h) = heads {
                // the interleave splits each head into two halves of
                // hd2 rows — only defined when rows == h * 2 * hd2
                if rows % (2 * h as usize) == 0 {
                    let data_rb = nq.data.len() / rows;
                    let scale_rb = nq.scales.len() / rows;
                    unpermute_rows(&mut nq.data, data_rb, h as usize);
                    unpermute_rows(&mut nq.scales, scale_rb, h as usize);
                    if let Some((off, _)) = &mut nq.offset {
                        let off_rb = off.len() / rows;
                        unpermute_rows(off, off_rb, h as usize);
                    }
                }
            }
        }
        Ok(Some(nq))
    }
}

impl WeightSource for Gguf {
    fn has(&self, name: &str) -> bool {
        Gguf::has(self, name)
    }
    fn native_qblocks(&self, name: &str) -> std::io::Result<Option<NativeQuant>> {
        self.native_qblocks_hf(name).map_err(|e| match e {
            GgufError::Io(e) => e,
            GgufError::Format(m) => std::io::Error::new(std::io::ErrorKind::InvalidData, m),
        })
    }
    fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
        Gguf::tensor_f16(self, name).map_err(|e| match e {
            GgufError::Io(e) => e,
            GgufError::Format(m) => {
                let kind = if m.contains("not in file") || m.contains("no ggml name") {
                    std::io::ErrorKind::NotFound
                } else {
                    std::io::ErrorKind::InvalidData
                };
                std::io::Error::new(kind, m)
            }
        })
    }
    fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
        Gguf::tensor_f32(self, name).map_err(|e| match e {
            GgufError::Io(e) => e,
            GgufError::Format(m) => {
                let kind = if m.contains("not in file") || m.contains("no ggml name") {
                    std::io::ErrorKind::NotFound
                } else {
                    std::io::ErrorKind::InvalidData
                };
                std::io::Error::new(kind, m)
            }
        })
    }
    fn shape(&self, name: &str) -> std::io::Result<Vec<i64>> {
        self.tensor_hf(name).map(|t| t.hf_shape()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {name}"),
            )
        })
    }
}

/// Inverse of the llama.cpp Q/K head interleave. Not an involution in
/// general (only when `rows/nh/2 == 2`).
///
/// Forward (converter): `w.reshape(nh, 2, rows/nh/2, cols).swapaxes(1, 2)`
/// — so `dst[h][i][j]` takes `src[h][j][i]` on a `(nh, hd2, 2)` view.
fn unpermute(w: &mut [f32], n_head: usize, rows: usize, cols: usize) {
    let hd2 = rows / n_head / 2;
    let mut src = vec![0f32; w.len()];
    src.copy_from_slice(w);
    for h in 0..n_head {
        for i in 0..2 {
            for j in 0..hd2 {
                let dst_base = ((h * 2 + i) * hd2 + j) * cols;
                let src_base = ((h * hd2 + j) * 2 + i) * cols;
                w[dst_base..dst_base + cols].copy_from_slice(&src[src_base..src_base + cols]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpermute_inverts_converter_permute() {
        // hd2 = 2 (where the permute happens to be an involution) and
        // hd2 = 32 (head_dim 64, the real SmolLM2/Llama case).
        for (nh, hd2, cols) in [(4usize, 2usize, 1usize), (3, 32, 5)] {
            let rows = nh * 2 * hd2;
            let orig: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            // converter forward: reshape(nh, 2, hd2, cols).swapaxes(1, 2)
            let mut fwd = vec![0f32; rows * cols];
            for h in 0..nh {
                for i in 0..2 {
                    for j in 0..hd2 {
                        let d = ((h * hd2 + j) * 2 + i) * cols;
                        let s = ((h * 2 + i) * hd2 + j) * cols;
                        fwd[d..d + cols].copy_from_slice(&orig[s..s + cols]);
                    }
                }
            }
            assert_ne!(fwd, orig);
            let mut w = fwd;
            unpermute(&mut w, nh, rows, cols);
            assert_eq!(w, orig, "nh={nh} hd2={hd2}");
        }
    }

    /// Dequantize a `NativeQuant` payload the way the emitted
    /// `constexpr_blockwise_shift_scale` defines it — `f16(scale *
    /// (code − offset))`, one scale per `block` elements along the row.
    fn dequant_native(nq: &NativeQuant, rows: usize, cols: usize) -> Vec<half::f16> {
        let nb = cols / nq.block as usize;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let s = half::f16::from_le_bytes([
                    nq.scales[2 * (r * nb + c / nq.block as usize)],
                    nq.scales[2 * (r * nb + c / nq.block as usize) + 1],
                ])
                .to_f32();
                let code = match nq.data_dtype {
                    mil_spec::DType::Int8 => nq.data[r * cols + c] as i8 as i32,
                    mil_spec::DType::Uint4 => {
                        let byte = nq.data[(r * cols + c) / 2];
                        let nib = if (r * cols + c) % 2 == 0 {
                            byte & 0x0F
                        } else {
                            byte >> 4
                        };
                        nib as i32
                    }
                    _ => unreachable!(),
                };
                let off = match &nq.offset {
                    Some((_, mil_spec::DType::Uint4)) => 8,
                    _ => 0,
                };
                out.push(half::f16::from_f32((code - off) as f32 * s));
            }
        }
        out
    }

    /// Build a Q8_0 raw payload: `rows` rows × `cols`, per-32-blocks
    /// `(fp16 d, int8 q[32])`.
    fn raw_q8_0(rows: usize, cols: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for r in 0..rows {
            for b in 0..cols / 32 {
                let d = half::f16::from_f32(0.01 * ((r * 7 + b * 3) % 40 + 1) as f32);
                raw.extend_from_slice(&d.to_le_bytes());
                for j in 0..32 {
                    raw.push((((r * 31 + b * 17 + j * 7) % 255) as i32 - 127) as i8 as u8);
                }
            }
        }
        raw
    }

    /// Build a Q4_0 raw payload: `(fp16 d, packed nibble[16])` per
    /// block — ggml order `(lo=elem j, hi=elem j+16)`.
    fn raw_q4_0(rows: usize, cols: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for r in 0..rows {
            for b in 0..cols / 32 {
                let d = half::f16::from_f32(0.002 * ((r * 11 + b * 5) % 60 + 1) as f32);
                raw.extend_from_slice(&d.to_le_bytes());
                for j in 0..16 {
                    let lo = ((r * 13 + b * 7 + j * 3) % 16) as u8;
                    let hi = ((r * 5 + b * 11 + j * 5) % 16) as u8;
                    raw.push(lo | (hi << 4));
                }
            }
        }
        raw
    }

    #[test]
    fn q8_0_transcode_is_bit_exact() {
        let (rows, cols) = (4usize, 96usize); // 3 blocks/row
        let raw = raw_q8_0(rows, cols);
        let nq = transcode_q8_0(&raw, rows, cols);
        // emitted semantic values vs ggml's own dequantizer
        let mut want = vec![0f32; rows * cols];
        dequant::dequant(GgufType::Q8_0, &raw, &mut want).unwrap();
        let got = dequant_native(&nq, rows, cols);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                half::f16::from_f32(*w).to_bits(),
                "elem {i}: emitted {g:?} != gguf {w}"
            );
        }
    }

    #[test]
    fn q4_0_transcode_is_bit_exact() {
        let (rows, cols) = (4usize, 96usize);
        let raw = raw_q4_0(rows, cols);
        let nq = transcode_q4_0(&raw, rows, cols);
        let mut want = vec![0f32; rows * cols];
        dequant::dequant(GgufType::Q4_0, &raw, &mut want).unwrap();
        let got = dequant_native(&nq, rows, cols);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                half::f16::from_f32(*w).to_bits(),
                "elem {i}: emitted {g:?} != gguf {w}"
            );
        }
    }

    #[test]
    fn unpermute_rows_matches_element_permute() {
        // Byte-level row reorder must equal the f32 `unpermute`.
        let (nh, hd2, cols) = (3usize, 32usize, 8usize);
        let rows = nh * 2 * hd2;
        let orig: Vec<f32> = (0..rows * cols).map(|i| (i % 97) as f32).collect();
        let mut bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
        unpermute_rows(&mut bytes, cols * 4, nh);
        let mut elems = orig.clone();
        unpermute(&mut elems, nh, rows, cols);
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, elems);
    }
}
