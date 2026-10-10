//! Minimal protobuf wire decoder — the substrate under `model.rs`.
//!
//! ONNX is proto2: fields may appear packed *or* unpacked, floats arrive
//! as fixed32 wire values, and nothing is ever a group. This decoder is
//! deliberately dumb — it returns `(field_number, raw value)` pairs and
//! leaves schema interpretation to [`crate::model`]. Wire types 3, 4, 6,
//! and 7 (groups / invalid) are rejected, so a malformed stream fails
//! cleanly instead of being misparsed.

use crate::OnnxError;

/// One decoded field value.
#[derive(Clone, Debug, PartialEq)]
pub enum WireVal {
    /// Wire type 0 — varint (int32/int64/bool/enum, sign-extended).
    Varint(u64),
    /// Wire type 1 — fixed64 (double, fixed64).
    Fixed64(u64),
    /// Wire type 2 — length-delimited (string, bytes, message, packed).
    Len(Vec<u8>),
    /// Wire type 5 — fixed32 (float, fixed32).
    Fixed32(u32),
}

/// One decoded field.
#[derive(Clone, Debug, PartialEq)]
pub struct PField {
    /// Field number.
    pub num: u32,
    /// Raw wire value.
    pub val: WireVal,
}

/// A decoded protobuf message — fields in wire order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PMsg {
    /// Fields in wire order.
    pub fields: Vec<PField>,
}

impl PMsg {
    /// All values for field `num`.
    pub fn all(&self, num: u32) -> impl Iterator<Item = &WireVal> {
        self.fields
            .iter()
            .filter(move |f| f.num == num)
            .map(|f| &f.val)
    }
    /// First value for field `num`.
    pub fn get(&self, num: u32) -> Option<&WireVal> {
        self.all(num).next()
    }
    /// First varint for field `num`.
    pub fn varint(&self, num: u32) -> Option<u64> {
        match self.get(num) {
            Some(WireVal::Varint(v)) => Some(*v),
            _ => None,
        }
    }
    /// First varint for `num` as `i64` (proto sign-extension undone).
    pub fn i64(&self, num: u32) -> Option<i64> {
        self.varint(num).map(|v| v as i64)
    }
    /// First length-delimited field `num` as UTF-8.
    pub fn string(&self, num: u32) -> Option<String> {
        match self.get(num) {
            Some(WireVal::Len(b)) => String::from_utf8(b.clone()).ok(),
            _ => None,
        }
    }
    /// First length-delimited field `num` as raw bytes.
    pub fn bytes(&self, num: u32) -> Option<&[u8]> {
        match self.get(num) {
            Some(WireVal::Len(b)) => Some(b),
            _ => None,
        }
    }
    /// Field `num` as a nested message (wire-2 bytes decoded again).
    pub fn msg(&self, num: u32) -> Option<PMsg> {
        match self.get(num) {
            Some(WireVal::Len(b)) => decode(b).ok(),
            _ => None,
        }
    }
    /// All nested messages for field `num`. Undecodable values are
    /// skipped (packed data / strings land here harmlessly).
    pub fn msgs(&self, num: u32) -> Vec<PMsg> {
        self.all(num)
            .filter_map(|v| match v {
                WireVal::Len(b) => decode(b).ok(),
                _ => None,
            })
            .collect()
    }
    /// First fixed32 field `num` as f32.
    pub fn f32(&self, num: u32) -> Option<f32> {
        match self.get(num) {
            Some(WireVal::Fixed32(v)) => Some(f32::from_bits(*v)),
            _ => None,
        }
    }
}

/// Decode a varint; returns `(value, bytes_consumed)`. Max 10 bytes —
/// a longer run is malformed.
fn varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for (i, &b) in buf.iter().enumerate().take(10) {
        v |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

/// Decode `buf` as a protobuf message. `what` names the message for
/// error reporting. Returns `Err` on truncation, overlong varints,
/// field number 0, or group/invalid wire types.
pub fn decode(buf: &[u8]) -> Result<PMsg, OnnxError> {
    let mut m = PMsg::default();
    let mut i = 0usize;
    while i < buf.len() {
        let (tag, n) = varint(&buf[i..]).ok_or_else(|| OnnxError::Malformed {
            what: "varint tag".into(),
            detail: format!("offset {i}"),
        })?;
        i += n;
        let num = (tag >> 3) as u32;
        let wire = (tag & 7) as u8;
        if num == 0 {
            return Err(OnnxError::Malformed {
                what: "field number".into(),
                detail: format!("field 0 at offset {}", i - n),
            });
        }
        let val = match wire {
            0 => {
                let (v, n) = varint(&buf[i..]).ok_or_else(|| OnnxError::Malformed {
                    what: "varint".into(),
                    detail: format!("field {num} at offset {i}"),
                })?;
                i += n;
                WireVal::Varint(v)
            }
            1 => {
                if i + 8 > buf.len() {
                    return Err(OnnxError::Malformed {
                        what: "fixed64".into(),
                        detail: format!("field {num} truncated"),
                    });
                }
                let v = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
                i += 8;
                WireVal::Fixed64(v)
            }
            2 => {
                let (len, n) = varint(&buf[i..]).ok_or_else(|| OnnxError::Malformed {
                    what: "length".into(),
                    detail: format!("field {num} at offset {i}"),
                })?;
                i += n;
                let len = len as usize;
                if i + len > buf.len() {
                    return Err(OnnxError::Malformed {
                        what: "bytes".into(),
                        detail: format!("field {num}: len {len} overruns buffer"),
                    });
                }
                let v = buf[i..i + len].to_vec();
                i += len;
                WireVal::Len(v)
            }
            5 => {
                if i + 4 > buf.len() {
                    return Err(OnnxError::Malformed {
                        what: "fixed32".into(),
                        detail: format!("field {num} truncated"),
                    });
                }
                let v = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap());
                i += 4;
                WireVal::Fixed32(v)
            }
            w => {
                return Err(OnnxError::Malformed {
                    what: "wire type".into(),
                    detail: format!("field {num}: unsupported wire type {w}"),
                })
            }
        };
        m.fields.push(PField { num, val });
    }
    Ok(m)
}

/// Read a packed-or-unpacked repeated varint field. Each `WireVal` may
/// be a lone varint (unpacked) or a `Len` blob of packed varints.
pub fn varints(vals: impl Iterator<Item = WireVal>) -> Result<Vec<i64>, OnnxError> {
    let mut out = Vec::new();
    for v in vals {
        match v {
            WireVal::Varint(x) => out.push(x as i64),
            WireVal::Len(b) => {
                let mut i = 0usize;
                while i < b.len() {
                    let (x, n) = varint(&b[i..]).ok_or_else(|| OnnxError::Malformed {
                        what: "packed varint".into(),
                        detail: "overlong".into(),
                    })?;
                    i += n;
                    out.push(x as i64);
                }
            }
            _ => {
                return Err(OnnxError::Malformed {
                    what: "packed varint".into(),
                    detail: "wrong wire type".into(),
                })
            }
        }
    }
    Ok(out)
}

/// Read a packed-or-unpacked repeated float field.
pub fn f32s(vals: impl Iterator<Item = WireVal>) -> Result<Vec<f32>, OnnxError> {
    let mut out = Vec::new();
    for v in vals {
        match v {
            WireVal::Fixed32(x) => out.push(f32::from_bits(x)),
            WireVal::Len(b) => {
                if b.len() % 4 != 0 {
                    return Err(OnnxError::Malformed {
                        what: "packed float".into(),
                        detail: format!("len {} not a multiple of 4", b.len()),
                    });
                }
                for c in b.chunks_exact(4) {
                    out.push(f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            _ => {
                return Err(OnnxError::Malformed {
                    what: "packed float".into(),
                    detail: "wrong wire type".into(),
                })
            }
        }
    }
    Ok(out)
}

/// Read a packed-or-unpacked repeated double field.
pub fn f64s(vals: impl Iterator<Item = WireVal>) -> Result<Vec<f64>, OnnxError> {
    let mut out = Vec::new();
    for v in vals {
        match v {
            WireVal::Fixed64(x) => out.push(f64::from_bits(x)),
            WireVal::Len(b) => {
                if b.len() % 8 != 0 {
                    return Err(OnnxError::Malformed {
                        what: "packed double".into(),
                        detail: format!("len {} not a multiple of 8", b.len()),
                    });
                }
                for c in b.chunks_exact(8) {
                    out.push(f64::from_le_bytes(c.try_into().unwrap()));
                }
            }
            _ => {
                return Err(OnnxError::Malformed {
                    what: "packed double".into(),
                    detail: "wrong wire type".into(),
                })
            }
        }
    }
    Ok(out)
}
