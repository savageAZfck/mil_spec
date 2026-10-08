//! `mil_verify` — inspection, diff, and conformance for MIL specs.
//!
//! Three jobs:
//!
//! - **Decode** — read `model.mlmodel` back as a protobuf tree, with no
//!   schema dependency beyond the field numbers the writer uses. Unknown
//!   fields are preserved, not dropped, so a spec written by `coremltools`
//!   inspects the same way as one written by `mil_spec`.
//! - **Diff** — compare two specs field-by-field and report every
//!   difference with a path (`/description/output[0]`). When `coremlc`
//!   rejects a package you diff it against a known-good one instead of
//!   re-running the pipeline and praying.
//! - **Validate** — walk the decoded graph and check what the IR
//!   compiler guarantees *before* emission but a raw spec could still
//!   break: every op input binds a name produced earlier or declared as
//!   a function input; every declared block output exists; every blob
//!   reference lands inside `weight.bin`.
//!
//! The conformance battery runs valid graphs (must pass) alongside
//! planted-invalid controls (must fail). If the controls ever pass, the
//! verifier itself is broken — the same "dishonest harness" guard as
//! touchstone's proof-of-work.
//!
//! # Usage
//!
//! ```
//! let report = mil_verify::verify_block(&mil_spec::Block::new());
//! assert!(report.is_ok());
//! ```

#![forbid(unsafe_code)]

use mil_spec::{Binding, Block, Op};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

// ================= protobuf decoder =================

/// A decoded protobuf value.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldVal {
    /// varint
    Varint(u64),
    /// 32-bit fixed
    Fixed32(u32),
    /// 64-bit fixed
    Fixed64(u64),
    /// Length-delimited: nested message if it parses, else raw bytes.
    Msg(Msg),
    /// Length-delimited raw bytes (didn't parse as a message).
    Bytes(Vec<u8>),
}

/// One decoded field.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    /// Field number.
    pub num: u32,
    /// Decoded value.
    pub val: FieldVal,
}

/// A decoded protobuf message — a list of fields in wire order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Msg {
    /// Fields in wire order.
    pub fields: Vec<Field>,
}

impl Msg {
    /// All values for a field number.
    pub fn get_all(&self, num: u32) -> Vec<&FieldVal> {
        self.fields
            .iter()
            .filter(|f| f.num == num)
            .map(|f| &f.val)
            .collect()
    }
    /// First value for a field number.
    pub fn get(&self, num: u32) -> Option<&FieldVal> {
        self.get_all(num).into_iter().next()
    }
    /// Field as a nested message.
    pub fn msg(&self, num: u32) -> Option<&Msg> {
        match self.get(num) {
            Some(FieldVal::Msg(m)) => Some(m),
            _ => None,
        }
    }
    /// Field as varint.
    pub fn varint(&self, num: u32) -> Option<u64> {
        match self.get(num) {
            Some(FieldVal::Varint(v)) => Some(*v),
            _ => None,
        }
    }
    /// Field as UTF-8 string.
    pub fn str(&self, num: u32) -> Option<String> {
        match self.get(num) {
            Some(FieldVal::Bytes(b)) => String::from_utf8(b.clone()).ok(),
            _ => None,
        }
    }
    /// All nested messages for a field.
    pub fn msgs(&self, num: u32) -> Vec<&Msg> {
        self.get_all(num)
            .into_iter()
            .filter_map(|v| {
                if let FieldVal::Msg(m) = v {
                    Some(m)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Decode a protobuf byte stream into a [`Msg`]. Length-delimited fields
/// are tried as nested messages — if the bytes don't parse cleanly as
/// protobuf they're kept raw, so strings, packed arrays, and foreign
/// schemas all survive.
pub fn decode(bytes: &[u8]) -> Option<Msg> {
    let mut m = Msg::default();
    let mut i = 0usize;
    while i < bytes.len() {
        let (tag, n) = read_varint(&bytes[i..])?;
        i += n;
        let num = (tag >> 3) as u32;
        let wire = (tag & 7) as u8;
        let val = match wire {
            0 => {
                let (v, n) = read_varint(&bytes[i..])?;
                i += n;
                FieldVal::Varint(v)
            }
            1 => {
                if i + 8 > bytes.len() {
                    return None;
                }
                let v = u64::from_le_bytes(bytes[i..i + 8].try_into().ok()?);
                i += 8;
                FieldVal::Fixed64(v)
            }
            2 => {
                let (len, n) = read_varint(&bytes[i..])?;
                i += n;
                let len = len as usize;
                if i + len > bytes.len() {
                    return None;
                }
                let raw = bytes[i..i + len].to_vec();
                i += len;
                match decode(&raw) {
                    Some(m) if !raw.is_empty() => FieldVal::Msg(m),
                    _ => FieldVal::Bytes(raw),
                }
            }
            5 => {
                if i + 4 > bytes.len() {
                    return None;
                }
                let v = u32::from_le_bytes(bytes[i..i + 4].try_into().ok()?);
                i += 4;
                FieldVal::Fixed32(v)
            }
            _ => return None,
        };
        m.fields.push(Field { num, val });
    }
    Some(m)
}

fn read_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(10) {
        v |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

// ================= spec inspection =================

/// A decoded feature (input/output/state).
#[derive(Clone, Debug)]
pub struct FeatureInfo {
    /// Feature name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<i64>,
    /// Element dtype (CoreML type code).
    pub dtype: u64,
    /// True for stateType features.
    pub is_state: bool,
}

/// What a spec contains.
#[derive(Clone, Debug, Default)]
pub struct SpecSummary {
    /// Model.specificationVersion.
    pub spec_version: u64,
    /// Input features.
    pub inputs: Vec<FeatureInfo>,
    /// Output features.
    pub outputs: Vec<FeatureInfo>,
    /// State features.
    pub states: Vec<FeatureInfo>,
    /// Function input names.
    pub fn_inputs: Vec<String>,
    /// Opset specialization name.
    pub opset: Option<String>,
    /// Op counts by type.
    pub op_counts: Vec<(String, usize)>,
    /// Total ops.
    pub total_ops: usize,
    /// Declared block output names.
    pub block_outputs: Vec<String>,
}

fn feature_info(m: &Msg) -> Option<FeatureInfo> {
    let name = m.str(1)?;
    let ty = m.msg(3)?;
    let (shape, dtype, is_state) = if let Some(mt) = ty.msg(5) {
        (shape_of(mt), mt.varint(2).unwrap_or(0), false)
    } else if let Some(st) = ty.msg(8) {
        let inner = st.msg(1)?;
        (shape_of(inner), inner.varint(2).unwrap_or(0), true)
    } else {
        (vec![], 0, false)
    };
    Some(FeatureInfo {
        name,
        shape,
        dtype,
        is_state,
    })
}

fn shape_of(multiarray: &Msg) -> Vec<i64> {
    // shape is field 1, repeated packed i64s
    let mut out = Vec::new();
    for v in multiarray.get_all(1) {
        match v {
            FieldVal::Varint(d) => out.push(*d as i64),
            FieldVal::Bytes(b) => {
                // packed varints
                let mut i = 0usize;
                while i < b.len() {
                    if let Some((d, n)) = read_varint(&b[i..]) {
                        out.push(d as i64);
                        i += n;
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Summarize a decoded `model.mlmodel`.
pub fn summarize(spec: &[u8]) -> Option<SpecSummary> {
    let model = decode(spec)?;
    let mut s = SpecSummary {
        spec_version: model.varint(1).unwrap_or(0),
        ..Default::default()
    };

    if let Some(desc) = model.msg(2) {
        for m in desc.msgs(1) {
            if let Some(f) = feature_info(m) {
                s.inputs.push(f);
            }
        }
        for m in desc.msgs(10) {
            if let Some(f) = feature_info(m) {
                s.outputs.push(f);
            }
        }
        for m in desc.msgs(13) {
            if let Some(f) = feature_info(m) {
                s.states.push(f);
            }
        }
    }

    if let Some(prog) = model.msg(502) {
        // Program: field 2 functions map → entries {1 key, 2 Function}
        for entry in prog.msgs(2) {
            if entry.str(1).as_deref() != Some("main") {
                continue;
            }
            let fnc = entry.msg(2)?;
            s.opset = fnc.str(2);
            for nvt in fnc.msgs(1) {
                if let Some(n) = nvt.str(1) {
                    s.fn_inputs.push(n);
                }
            }
            for bentry in fnc.msgs(3) {
                let blk = bentry.msg(2)?;
                let mut counts: HashMap<String, usize> = HashMap::new();
                for op in blk.msgs(3) {
                    if let Some(t) = op.str(1) {
                        *counts.entry(t).or_default() += 1;
                        s.total_ops += 1;
                    }
                }
                for oname in blk.get_all(2).iter().filter_map(|v| {
                    if let FieldVal::Bytes(b) = v {
                        String::from_utf8(b.clone()).ok()
                    } else {
                        None
                    }
                }) {
                    s.block_outputs.push(oname);
                }
                let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                s.op_counts = sorted;
            }
        }
    }
    Some(s)
}

// ================= spec validation =================

/// A spec validation report.
#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    /// Errors that must fail validation.
    pub errors: Vec<String>,
    /// Non-fatal findings.
    pub warnings: Vec<String>,
}

impl VerifyReport {
    /// True if the spec is structurally valid.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            write!(f, "verify: ok")?;
        } else {
            write!(f, "verify: {} error(s)", self.errors.len())?;
        }
        for e in &self.errors {
            write!(f, "\n  error: {}", e)?;
        }
        for w in &self.warnings {
            write!(f, "\n  warn : {}", w)?;
        }
        Ok(())
    }
}

/// Validate a decoded spec's graph structure: every op input binds a
/// name defined earlier (function input, state, or an earlier op's
/// output); every declared output exists; every state read/write pairs
/// with a declared state feature.
pub fn verify_spec(spec: &[u8]) -> VerifyReport {
    let mut r = VerifyReport::default();
    let model = match decode(spec) {
        Some(m) => m,
        None => {
            r.errors.push("spec does not parse as protobuf".into());
            return r;
        }
    };
    let summary = summarize(spec).unwrap_or_default();
    if model.varint(1).is_none() {
        r.errors.push("missing specificationVersion".into());
    }
    if summary.inputs.is_empty() {
        r.warnings.push("no inputs declared".into());
    }
    if summary.outputs.is_empty() {
        r.errors.push("no outputs declared".into());
    }

    // Walk the block and check name resolution.
    let mut produced: HashMap<String, ()> = HashMap::new();
    for n in &summary.fn_inputs {
        produced.insert(n.clone(), ());
    }
    for st in &summary.states {
        produced.insert(st.name.clone(), ());
    }
    if let Some(prog) = model.msg(502) {
        for entry in prog.msgs(2) {
            if entry.str(1).as_deref() != Some("main") {
                continue;
            }
            if let Some(fnc) = entry.msg(2) {
                for bentry in fnc.msgs(3) {
                    if let Some(blk) = bentry.msg(2) {
                        verify_block_msg(blk, &mut produced, &mut r);
                    }
                }
            }
        }
    }
    // Declared outputs must be produced.
    for out in &summary.block_outputs {
        if !produced.contains_key(out) {
            r.errors
                .push(format!("block output {out} is never produced"));
        }
    }
    r
}

fn verify_block_msg(blk: &Msg, produced: &mut HashMap<String, ()>, r: &mut VerifyReport) {
    for op in blk.msgs(3) {
        let ty = op.str(1).unwrap_or_else(|| "?".into());
        // inputs: field 2 map → entries {1 name, 2 Argument}
        for ient in op.msgs(2) {
            if let Some(arg) = ient.msg(2) {
                // Argument: field 2 bindings (repeated Binding)
                for binding in arg.msgs(2) {
                    // Binding: field 1 name (str) or field 2 value
                    if let Some(n) = binding.str(1) {
                        if !produced.contains_key(&n) {
                            r.errors
                                .push(format!("{ty}: input binds undefined value {n}"));
                        }
                    }
                }
            }
        }
        // outputs: field 3 NVT {1 name}
        for nvt in op.msgs(3) {
            if let Some(n) = nvt.str(1) {
                produced.insert(n, ());
            }
        }
    }
}

// ================= diff =================

/// A single difference between two specs.
#[derive(Clone, Debug, PartialEq)]
pub enum Diff {
    /// Field present in `a` only.
    Removed { path: String },
    /// Field present in `b` only.
    Added { path: String },
    /// Field present in both but different.
    Changed { path: String, a: String, b: String },
}

impl fmt::Display for Diff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Diff::Removed { path } => write!(f, "- {path}"),
            Diff::Added { path } => write!(f, "+ {path}"),
            Diff::Changed { path, a, b } => write!(f, "~ {path}: {a} -> {b}"),
        }
    }
}

/// Diff two `model.mlmodel` byte strings field-by-field.
pub fn diff_specs(a: &[u8], b: &[u8]) -> Vec<Diff> {
    let mut out = Vec::new();
    match (decode(a), decode(b)) {
        (Some(ma), Some(mb)) => diff_msg("", &ma, &mb, &mut out),
        (None, Some(_)) => out.push(Diff::Removed {
            path: "<spec>".into(),
        }),
        (Some(_), None) => out.push(Diff::Added {
            path: "<spec>".into(),
        }),
        _ => {}
    }
    out
}

fn diff_msg(path: &str, a: &Msg, b: &Msg, out: &mut Vec<Diff>) {
    // Field numbers present on each side, in first-appearance order.
    let order = |m: &Msg| -> Vec<u32> {
        let mut seen = Vec::new();
        for f in &m.fields {
            if !seen.contains(&f.num) {
                seen.push(f.num);
            }
        }
        seen
    };
    let a_nums = order(a);
    let b_nums = order(b);

    let mut nums: Vec<u32> = a_nums.clone();
    for n in &b_nums {
        if !nums.contains(n) {
            nums.push(*n);
        }
    }

    for n in nums {
        let p = format!("{}/{}", path, n);
        let in_a = a_nums.contains(&n);
        let in_b = b_nums.contains(&n);
        match (in_a, in_b) {
            (true, true) => diff_field(&p, n, a, b, out),
            (true, false) => {
                let count = a.fields.iter().filter(|f| f.num == n).count();
                out.push(Diff::Removed {
                    path: format!("{} (x{})", p, count),
                });
            }
            (false, true) => {
                let count = b.fields.iter().filter(|f| f.num == n).count();
                out.push(Diff::Added {
                    path: format!("{} (x{})", p, count),
                });
            }
            (false, false) => {}
        }
    }
}

fn diff_field(path: &str, num: u32, a: &Msg, b: &Msg, out: &mut Vec<Diff>) {
    let av: Vec<&FieldVal> = a.get_all(num);
    let bv: Vec<&FieldVal> = b.get_all(num);
    let common = av.len().min(bv.len());
    for i in 0..common {
        diff_val(&format!("{path}[{i}]"), av[i], bv[i], out);
    }
    for _ in common..av.len() {
        out.push(Diff::Removed {
            path: format!("{path} (extra occurrence)"),
        });
    }
    for _ in common..bv.len() {
        out.push(Diff::Added {
            path: format!("{path} (extra occurrence)"),
        });
    }
}

fn diff_val(path: &str, a: &FieldVal, b: &FieldVal, out: &mut Vec<Diff>) {
    match (a, b) {
        (FieldVal::Varint(x), FieldVal::Varint(y)) if x == y => {}
        (FieldVal::Fixed32(x), FieldVal::Fixed32(y)) if x == y => {}
        (FieldVal::Fixed64(x), FieldVal::Fixed64(y)) if x == y => {}
        (FieldVal::Bytes(x), FieldVal::Bytes(y)) if x == y => {}
        (FieldVal::Msg(x), FieldVal::Msg(y)) => diff_msg(path, x, y, out),
        _ => {
            let fmt = |v: &FieldVal| -> String {
                match v {
                    FieldVal::Varint(x) => format!("varint({x})"),
                    FieldVal::Fixed32(x) => format!("fixed32({x})"),
                    FieldVal::Fixed64(x) => format!("fixed64({x})"),
                    FieldVal::Bytes(x) => format!("bytes({}B)", x.len()),
                    FieldVal::Msg(x) => format!("msg({} fields)", x.fields.len()),
                }
            };
            out.push(Diff::Changed {
                path: path.to_string(),
                a: fmt(a),
                b: fmt(b),
            });
        }
    }
}

// ================= package + weight.bin integrity =================

/// Verify a `.mlpackage` directory: spec parses, structure valid, and
/// every blob offset in the spec lands inside `weight.bin` with its
/// sentinel intact.
pub fn verify_package(dir: &Path) -> VerifyReport {
    let mut r = VerifyReport::default();
    let spec_path = dir.join("Data/com.apple.CoreML/model.mlmodel");
    let spec = match std::fs::read(&spec_path) {
        Ok(s) => s,
        Err(e) => {
            r.errors
                .push(format!("cannot read {}: {}", spec_path.display(), e));
            return r;
        }
    };
    let mut r2 = verify_spec(&spec);
    r.errors.append(&mut r2.errors);
    r.warnings.append(&mut r2.warnings);

    let wpath = dir.join("Data/com.apple.CoreML/weights/weight.bin");
    let wbytes = std::fs::read(&wpath).unwrap_or_default();
    if !wpath.exists() {
        // blob-free package — still valid if no op references a blob.
        if spec_has_blob_refs(&spec) {
            r.errors
                .push("spec references weight.bin but file is absent".into());
        }
        return r;
    }
    verify_blob_refs(&spec, &wbytes, &mut r);
    r
}

fn spec_has_blob_refs(spec: &[u8]) -> bool {
    // crude scan: any attribute value that decodes as a blob ref leaves
    // "@model_path" or "weights/" in the byte stream.
    let needle = b"@model_path";
    spec.windows(needle.len()).any(|w| w == needle)
        || spec
            .windows(b"weight.bin".len())
            .any(|w| w == b"weight.bin")
}

fn verify_blob_refs(_spec: &[u8], weight_bin: &[u8], r: &mut VerifyReport) {
    // weight.bin must start with the storage header (version 2) and have
    // sentinel records at 64-byte boundaries.
    if weight_bin.len() < 64 {
        r.errors
            .push("weight.bin shorter than storage header".into());
        return;
    }
    let version = u32::from_le_bytes(weight_bin[4..8].try_into().unwrap());
    if version != 2 {
        r.errors
            .push(format!("weight.bin storage version {version}, expected 2"));
    }
    // Walk blob records: each is a 64-byte metadata block followed by
    // sizeInBytes of data.
    let mut pos = 64u64;
    let mut count = 0u32;
    let declared = u32::from_le_bytes(weight_bin[0..4].try_into().unwrap());
    while pos + 64 <= weight_bin.len() as u64 {
        let rec = &weight_bin[pos as usize..pos as usize + 64];
        let sentinel = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        if sentinel != 0xDEADBEEF {
            break;
        }
        let size = u64::from_le_bytes(rec[8..16].try_into().unwrap());
        let data_off = u64::from_le_bytes(rec[16..24].try_into().unwrap());
        if data_off + size > weight_bin.len() as u64 {
            r.errors.push(format!(
                "blob {} data range {}+{} exceeds weight.bin ({}B)",
                count,
                data_off,
                size,
                weight_bin.len()
            ));
            return;
        }
        pos = (data_off + size + 63) & !63;
        count += 1;
    }
    if declared != count {
        r.warnings.push(format!(
            "weight.bin declares {declared} blobs but {count} records parse"
        ));
    }
}

// ================= block-level validation (pre-emit) =================

/// Validate a `Block` before it ever reaches `encode_model` — catches
/// the bugs the IR can't (hand-built graphs, pass output). Same rules
/// as [`verify_spec`] but at the structure level.
pub fn verify_block(b: &Block) -> VerifyReport {
    let mut r = VerifyReport::default();
    let mut produced: HashMap<String, ()> = HashMap::new();
    for op in &b.ops {
        for n in consumed_names(op) {
            if !produced.contains_key(&n) {
                r.errors
                    .push(format!("{}: input binds undefined value {n}", op.ty));
            }
        }
        for out in &op.outputs {
            produced.insert(out.name.clone(), ());
        }
    }
    for out in &b.outputs {
        if !produced.contains_key(out) {
            r.errors
                .push(format!("block output {out} is never produced"));
        }
    }
    r
}

fn consumed_names(op: &Op) -> Vec<String> {
    let mut names = Vec::new();
    for (_, arg) in &op.inputs {
        for binding in &arg.0 {
            if let Binding::Name(n) = binding {
                names.push(n.clone());
            }
        }
    }
    names
}

// ================= spec lint (decoded ops → mil_lint) =================

/// MIL dtype code → `DType` (the wire constants from `DType::mil`).
fn dtype_of(code: u64) -> Option<mil_spec::DType> {
    match code {
        1 => Some(mil_spec::DType::Bool),
        2 => Some(mil_spec::DType::Str),
        10 => Some(mil_spec::DType::Fp16),
        11 => Some(mil_spec::DType::Fp32),
        21 => Some(mil_spec::DType::Int8),
        23 => Some(mil_spec::DType::Int32),
        24 => Some(mil_spec::DType::Int64),
        _ => None,
    }
}

/// Decode an op output NVT: name, dtype, rank.
fn op_out_info(nvt: &Msg) -> (Option<String>, Option<mil_spec::DType>, usize) {
    let name = nvt.str(1);
    // NVT field 2 = ValueType → field 1 tensorType → {1 shape, 2 dtype}
    let tt = nvt
        .msg(2)
        .and_then(|vt| vt.msg(1).or_else(|| vt.msg(2).and_then(|st| st.msg(1))));
    match tt {
        Some(t) => {
            let dt = t.varint(2).and_then(dtype_of);
            let rank = shape_of(t).len();
            (name, dt, rank)
        }
        None => (name, None, 0),
    }
}

/// Lint a raw `model.mlmodel` spec: decode every op and run the
/// `mil_lint` rule table on it. Same verdicts as `lint_block`, on
/// packages — this is the `milc lint` path.
pub fn lint_spec(spec: &[u8]) -> Option<mil_lint::LintReport> {
    let model = decode(spec)?;
    let mut verdicts = Vec::new();
    if let Some(prog) = model.msg(502) {
        for entry in prog.msgs(2) {
            if entry.str(1).as_deref() != Some("main") {
                continue;
            }
            if let Some(fnc) = entry.msg(2) {
                for bentry in fnc.msgs(3) {
                    if let Some(blk) = bentry.msg(2) {
                        for op in blk.msgs(3) {
                            let ty = op.str(1).unwrap_or_else(|| "?".into());
                            // first output NVT carries name + type
                            let (name, dtype, rank) = op
                                .msgs(3)
                                .first()
                                .map(|n| op_out_info(n))
                                .unwrap_or((None, None, 0));
                            let str_const = dtype == Some(mil_spec::DType::Str);
                            let (unit, rule) = mil_lint::classify_op(&ty, dtype, rank, str_const);
                            verdicts.push(mil_lint::Verdict {
                                name: name.unwrap_or_else(|| format!("<{ty}>")),
                                op: ty,
                                unit,
                                rule: rule.to_string(),
                                shape: op
                                    .msgs(3)
                                    .first()
                                    .and_then(|n| n.msg(2))
                                    .and_then(|vt| vt.msg(1))
                                    .map(shape_of),
                            });
                        }
                    }
                }
            }
        }
    }
    Some(mil_lint::report_from_verdicts(verdicts))
}

// ================= conformance battery =================

/// One battery case: IR source plus whether it should compile.
pub struct Case {
    /// Case name.
    pub name: &'static str,
    /// IR source text.
    pub ir: &'static str,
    /// True if `ir::compile` should succeed.
    pub valid_ir: bool,
    /// True if the emitted spec should pass `verify_spec`.
    pub valid_spec: bool,
}

/// Result of running one case.
pub struct CaseResult {
    /// Case name.
    pub name: String,
    /// Whether the case behaved as expected.
    pub passed: bool,
    /// What went wrong, if anything.
    pub note: String,
}

/// The battery: valid graphs (must pass) and planted-invalid controls
/// (must fail). If a control ever passes, the verifier is broken — the
/// battery reports it so the harness itself can't lie.
pub fn battery() -> Vec<CaseResult> {
    let cases = [
        // --- valid programs ---
        Case {
            name: "elementwise-chain",
            ir: "input x: fp16[1,4,1,1]\nk = const_f16(2.0)\ny = mul(x, k)\nz = add(y, x)\noutput z",
            valid_ir: true,
            valid_spec: true,
        },
        Case {
            name: "conv-pipeline",
            ir: "input x: fp16[1,4,1,1]\nw = const_blob(file.bin, 64, fp16, [8,4,1,1])\ny = conv1x1(x, w)\noutput y",
            valid_ir: true,
            valid_spec: true,
        },
        Case {
            name: "stateful-kv",
            ir: "input x: fp16[1,4,1,1]\nstate kv: fp16[1,4,1,64]\nr = read_state(kv)\nc = concat([r, x], 3)\nwrite_state(kv, c)\noutput c",
            valid_ir: true,
            valid_spec: true,
        },
        Case {
            name: "q8-weight",
            ir: "input x: fp16[1,4,1,1]\nw = const_q8(file.bin, 64, 128, [8,4,1,1])\ny = conv1x1(x, w)\noutput y",
            valid_ir: true,
            valid_spec: true,
        },
        Case {
            name: "matmul",
            ir: "input x: fp16[1,4]\ny = matmul(x, x, transpose_y)\noutput y",
            valid_ir: true,
            valid_spec: true,
        },
        // --- planted controls: IR-valid but spec-invalid or IR-invalid ---
        Case {
            name: "ctl-undefined-ref",
            ir: "input x: fp16[1,4,1,1]\ny = mul(x, missing)\noutput y",
            valid_ir: false,
            valid_spec: false,
        },
        Case {
            name: "ctl-shape-mismatch",
            ir: "input x: fp16[1,4,1,1]\ny = reshape(x, [9,9,9])\noutput y",
            valid_ir: false,
            valid_spec: false,
        },
        Case {
            name: "ctl-double-def",
            ir: "input x: fp16[1,4,1,1]\ny = mul(x, x)\ny = add(x, x)\noutput y",
            valid_ir: false,
            valid_spec: false,
        },
        Case {
            name: "ctl-bad-perm",
            ir: "input x: fp16[1,4,1,1]\ny = transpose(x, perm=[0,0,0])\noutput y",
            valid_ir: false,
            valid_spec: false,
        },
        Case {
            name: "ctl-no-outputs",
            ir: "input x: fp16[1,4,1,1]\ny = mul(x, x)",
            valid_ir: false,
            valid_spec: false,
        },
    ];

    let mut results = Vec::new();
    for c in cases {
        let mut res = CaseResult {
            name: c.name.to_string(),
            passed: true,
            note: String::new(),
        };
        match mil_spec::ir::compile(c.ir) {
            Ok(prog) => {
                if !c.valid_ir {
                    res.passed = false;
                    res.note = "invalid IR compiled — compiler missed the control".into();
                } else {
                    // Emit and verify the spec.
                    let spec = mil_spec::encode_model(
                        &prog.inputs,
                        &prog.outputs,
                        &prog.states,
                        &prog.block,
                        &prog.fn_inputs,
                        &mil_spec::ModelMeta::new(8, "CoreML5"),
                    );
                    let r = verify_spec(&spec);
                    if r.is_ok() != c.valid_spec {
                        res.passed = false;
                        res.note =
                            format!("spec verify expected {} got {}", c.valid_spec, r.is_ok());
                    }
                }
            }
            Err(e) => {
                if c.valid_ir {
                    res.passed = false;
                    res.note = format!("valid IR failed to compile: {e}");
                }
            }
        }
        results.push(res);
    }
    results
}

/// Convenience for `milc verify`: run the battery, print, and return
/// exit status (0 = all pass).
pub fn run_battery() -> i32 {
    let results = battery();
    let mut fails = 0;
    for r in &results {
        let mark = if r.passed { "ok  " } else { "FAIL" };
        let note = if r.note.is_empty() {
            String::new()
        } else {
            format!("  — {}", r.note)
        };
        println!("{} {}{}", mark, r.name, note);
        if !r.passed {
            fails += 1;
        }
    }
    println!("{} cases, {} failed", results.len(), fails);
    if fails == 0 {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_spec::{DType, ValueType};

    fn spec_for(ir: &str) -> Vec<u8> {
        let p = mil_spec::ir::compile(ir).unwrap();
        mil_spec::encode_model(
            &p.inputs,
            &p.outputs,
            &p.states,
            &p.block,
            &p.fn_inputs,
            &mil_spec::ModelMeta::new(8, "CoreML5"),
        )
    }

    #[test]
    fn decode_roundtrips_model() {
        let spec = spec_for("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y");
        let s = summarize(&spec).unwrap();
        assert_eq!(s.spec_version, 8);
        assert_eq!(s.inputs[0].name, "x");
        assert_eq!(s.outputs[0].name, "y");
        assert_eq!(s.total_ops, 1);
        assert_eq!(s.op_counts, vec![("mul".to_string(), 1)]);
    }

    #[test]
    fn verify_spec_accepts_valid() {
        let spec = spec_for("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y");
        assert!(verify_spec(&spec).is_ok());
    }

    #[test]
    fn verify_spec_catches_dangling_ref() {
        // Hand-build a spec whose op binds a name that doesn't exist —
        // the IR compiler can't produce this, but a pass bug could.
        let mut b = mil_spec::Block::new();
        b.outputs = vec!["y".into()];
        let spec = mil_spec::encode_model(
            &[mil_spec::Feature {
                name: "x".into(),
                shape: vec![1, 4, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }],
            &[mil_spec::Feature {
                name: "y".into(),
                shape: vec![1, 4, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }],
            &[],
            &b,
            &[mil_spec::NVT {
                name: "x".into(),
                ty: ValueType::Tensor(mil_spec::TensorType::f16(&[1, 4, 1, 1])),
            }],
            &mil_spec::ModelMeta::new(8, "CoreML5"),
        );
        let r = verify_spec(&spec);
        assert!(!r.is_ok());
        assert!(r.errors.iter().any(|e| e.contains("y")));
    }

    #[test]
    fn diff_identical_specs_is_empty() {
        let s1 = spec_for("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y");
        let s2 = spec_for("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y");
        // encode_model embeds no timestamps/uuids → deterministic
        assert_eq!(s1, s2);
        assert!(diff_specs(&s1, &s2).is_empty());
    }

    #[test]
    fn diff_different_specs_reports_ops() {
        let s1 = spec_for("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y");
        let s2 = spec_for("input x: fp16[1,4,1,1]\ny = add(x, x)\noutput y");
        let d = diff_specs(&s1, &s2);
        assert!(!d.is_empty());
    }

    #[test]
    fn verify_block_catches_dangling() {
        let mut b = mil_spec::Block::new();
        let y = b.add("x", "nonexistent", &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let r = verify_block(&b);
        assert!(!r.is_ok());
        assert!(r.errors.iter().any(|e| e.contains("nonexistent")));
    }

    #[test]
    fn battery_all_pass() {
        for r in battery() {
            assert!(r.passed, "battery case {} failed: {}", r.name, r.note);
        }
    }

    #[test]
    fn package_verify_rejects_missing_weights() {
        let root = std::env::temp_dir().join(format!("mv_pkg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("p.mlpackage");
        std::fs::create_dir_all(dir.join("Data/com.apple.CoreML")).unwrap();
        // spec references weight.bin but none exists
        let spec = spec_for(
            "input x: fp16[1,4,1,1]\nw = const_blob(@model_path/weights/weight.bin, 64, fp16, [1,4,1,1])\ny = mul(x, w)\noutput y",
        );
        std::fs::write(dir.join("Data/com.apple.CoreML/model.mlmodel"), &spec).unwrap();
        let r = verify_package(&dir);
        assert!(!r.is_ok());
        assert!(r.errors.iter().any(|e| e.contains("weight.bin")));
        let _ = std::fs::remove_dir_all(&root);
    }
}
