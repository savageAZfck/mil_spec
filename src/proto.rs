//! `proto` — a minimal lossless protobuf tree codec.
//!
//! The writer half of `mil_spec` emits `model.mlmodel` wire bytes
//! directly. This module is the reader/editor half: it decodes any
//! protobuf stream into a mutable field tree and re-encodes it,
//! preserving field order and unknown fields byte-for-byte.
//!
//! Length-delimited fields are kept as **raw bytes** — a nested message
//! is decoded on demand with [`PMut::msg`]/[`PMut::msgs`] and stored
//! back with [`PMut::set_msg`]. A spec that is decoded and re-encoded
//! without edits is byte-identical to the input, which makes it safe to
//! use for surgery (edit a few fields, keep the rest verbatim) and for
//! hashing (a spec stripped of one metadata entry re-encodes to the
//! exact pre-embedding bytes the writer produced).
//!
//! Supported wire types: varint (0), fixed64 (1), length-delimited (2),
//! fixed32 (5). Groups (3/4) are rejected — no CoreML schema uses them.

/// A decoded protobuf field value.
#[derive(Clone, Debug, PartialEq)]
pub enum PVal {
    /// varint (wire 0) — ints, bools, enums.
    Varint(u64),
    /// fixed64 (wire 1).
    Fixed64(u64),
    /// Length-delimited (wire 2) — raw payload. May be a nested message,
    /// a string, packed elements, or opaque bytes; decode with
    /// [`decode`] when it is a message.
    Len(Vec<u8>),
    /// fixed32 (wire 5).
    Fixed32(u32),
}

/// One field: number + value.
#[derive(Clone, Debug, PartialEq)]
pub struct PField {
    /// Field number.
    pub num: u32,
    /// Value.
    pub val: PVal,
}

/// A decoded protobuf message — fields in wire order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PMut {
    /// Fields in wire order.
    pub fields: Vec<PField>,
}

fn enc_varint(buf: &mut Vec<u8>, mut v: u64) {
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

fn dec_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(10) {
        v |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

impl PVal {
    /// The value as varint, if it is one.
    pub fn as_varint(&self) -> Option<u64> {
        match self {
            PVal::Varint(v) => Some(*v),
            _ => None,
        }
    }
    /// The value as UTF-8, if it is a length-delimited string.
    pub fn as_str(&self) -> Option<String> {
        match self {
            PVal::Len(b) => String::from_utf8(b.clone()).ok(),
            _ => None,
        }
    }
    /// The value decoded as a nested message, if it parses.
    pub fn as_msg(&self) -> Option<PMut> {
        match self {
            PVal::Len(b) => decode(b),
            _ => None,
        }
    }
    /// Length-delimited payload bytes.
    pub fn as_len(&self) -> Option<&[u8]> {
        match self {
            PVal::Len(b) => Some(b),
            _ => None,
        }
    }
}

impl PMut {
    /// All values for field `num`, in wire order.
    pub fn get_all(&self, num: u32) -> Vec<&PVal> {
        self.fields
            .iter()
            .filter(|f| f.num == num)
            .map(|f| &f.val)
            .collect()
    }
    /// First value for field `num`.
    pub fn get(&self, num: u32) -> Option<&PVal> {
        self.get_all(num).into_iter().next()
    }
    /// First value for `num` as varint.
    pub fn varint(&self, num: u32) -> Option<u64> {
        self.get(num).and_then(PVal::as_varint)
    }
    /// First value for `num` as UTF-8 string.
    pub fn str(&self, num: u32) -> Option<String> {
        self.get(num).and_then(PVal::as_str)
    }
    /// First value for `num` decoded as a nested message.
    pub fn msg(&self, num: u32) -> Option<PMut> {
        self.get(num).and_then(PVal::as_msg)
    }
    /// Every value for `num` decoded as nested messages.
    pub fn msgs(&self, num: u32) -> Vec<PMut> {
        self.get_all(num)
            .iter()
            .filter_map(|v| v.as_msg())
            .collect()
    }
    /// `map<string, V>` entries at `num` as `(key, value)` pairs.
    /// Each entry is a message `{ 1: key, 2: value }`.
    pub fn map_entries(&self, num: u32) -> Vec<(String, PMut)> {
        self.msgs(num)
            .iter()
            .filter_map(|e| e.str(1).map(|k| (k, e.clone())))
            .collect()
    }

    /// Append a field.
    pub fn push(&mut self, num: u32, val: PVal) {
        self.fields.push(PField { num, val });
    }
    /// Replace the first `num` field in place, or append when absent.
    pub fn set(&mut self, num: u32, val: PVal) {
        for f in &mut self.fields {
            if f.num == num {
                f.val = val;
                return;
            }
        }
        self.push(num, val);
    }
    /// Remove every field numbered `num`.
    pub fn remove_all(&mut self, num: u32) {
        self.fields.retain(|f| f.num != num);
    }
    /// `set` with a varint.
    pub fn set_varint(&mut self, num: u32, v: u64) {
        self.set(num, PVal::Varint(v));
    }
    /// `set` with a UTF-8 string.
    pub fn set_str(&mut self, num: u32, s: &str) {
        self.set(num, PVal::Len(s.as_bytes().to_vec()));
    }
    /// `set` with an encoded nested message.
    pub fn set_msg(&mut self, num: u32, m: &PMut) {
        self.set(num, PVal::Len(encode(m)));
    }
    /// `push` with a UTF-8 string.
    pub fn push_str(&mut self, num: u32, s: &str) {
        self.push(num, PVal::Len(s.as_bytes().to_vec()));
    }
    /// `push` with an encoded nested message.
    pub fn push_msg(&mut self, num: u32, m: &PMut) {
        self.push(num, PVal::Len(encode(m)));
    }
    /// `push` a `map<string,V>` entry `{1: key, 2: value}`.
    pub fn push_map_entry(&mut self, num: u32, key: &str, val: PVal) {
        let mut e = PMut::default();
        e.push_str(1, key);
        e.push(2, val);
        self.push_msg(num, &e);
    }
}

/// Decode a protobuf stream. `None` on malformed input — truncated
/// varints, group wire types, or a length that runs past the buffer.
pub fn decode(bytes: &[u8]) -> Option<PMut> {
    let mut m = PMut::default();
    let mut i = 0usize;
    while i < bytes.len() {
        let (tagv, n) = dec_varint(&bytes[i..])?;
        i += n;
        let num = (tagv >> 3) as u32;
        if num == 0 {
            return None;
        }
        let val = match (tagv & 7) as u8 {
            0 => {
                let (v, n) = dec_varint(&bytes[i..])?;
                i += n;
                PVal::Varint(v)
            }
            1 => {
                if i + 8 > bytes.len() {
                    return None;
                }
                let v = u64::from_le_bytes(bytes[i..i + 8].try_into().ok()?);
                i += 8;
                PVal::Fixed64(v)
            }
            2 => {
                let (len, n) = dec_varint(&bytes[i..])?;
                i += n;
                let len = len as usize;
                if i + len > bytes.len() {
                    return None;
                }
                let raw = bytes[i..i + len].to_vec();
                i += len;
                PVal::Len(raw)
            }
            5 => {
                if i + 4 > bytes.len() {
                    return None;
                }
                let v = u32::from_le_bytes(bytes[i..i + 4].try_into().ok()?);
                i += 4;
                PVal::Fixed32(v)
            }
            _ => return None,
        };
        m.fields.push(PField { num, val });
    }
    Some(m)
}

/// Encode a message back to wire bytes. Unedited `Len` payloads pass
/// through verbatim, so a decode→encode round-trip is byte-identical.
pub fn encode(m: &PMut) -> Vec<u8> {
    let mut out = Vec::new();
    for f in &m.fields {
        enc_varint(&mut out, ((f.num << 3) | wire_of(&f.val)) as u64);
        match &f.val {
            PVal::Varint(v) => enc_varint(&mut out, *v),
            PVal::Fixed64(v) => out.extend_from_slice(&v.to_le_bytes()),
            PVal::Len(b) => {
                enc_varint(&mut out, b.len() as u64);
                out.extend_from_slice(b);
            }
            PVal::Fixed32(v) => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
    out
}

fn wire_of(v: &PVal) -> u32 {
    match v {
        PVal::Varint(_) => 0,
        PVal::Fixed64(_) => 1,
        PVal::Len(_) => 2,
        PVal::Fixed32(_) => 5,
    }
}

/// Build a `Value`/message field value from raw nested-message bytes.
pub fn len(bytes: Vec<u8>) -> PVal {
    PVal::Len(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip_edges() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut b = Vec::new();
            enc_varint(&mut b, v);
            let (got, n) = dec_varint(&b).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, b.len());
        }
    }

    #[test]
    fn decode_encode_is_lossless() {
        // {1: varint 150, 2: "abc", 5: fixed32 0xdead}
        let raw = {
            let mut m = PMut::default();
            m.push(1, PVal::Varint(150));
            m.push_str(2, "abc");
            m.push(5, PVal::Fixed32(0xdead));
            encode(&m)
        };
        let m = decode(&raw).unwrap();
        assert_eq!(m.varint(1), Some(150));
        assert_eq!(m.str(2).as_deref(), Some("abc"));
        assert_eq!(encode(&m), raw);
    }

    #[test]
    fn nested_msg_edit() {
        let mut inner = PMut::default();
        inner.push_str(1, "main");
        let mut outer = PMut::default();
        outer.push_msg(2, &inner);
        outer.push(1, PVal::Varint(9));
        let raw = encode(&outer);

        let mut m = decode(&raw).unwrap();
        let mut f = m.msg(2).unwrap();
        f.set_str(1, "side");
        m.set_msg(2, &f);
        let raw2 = encode(&m);
        let m2 = decode(&raw2).unwrap();
        assert_eq!(m2.msg(2).unwrap().str(1).as_deref(), Some("side"));
        assert_eq!(m2.varint(1), Some(9));
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(&[0xff]).is_none()); // truncated varint
        assert!(decode(&[0x0b]).is_none()); // group wire type
        assert!(decode(&[0x0a, 0x05, 0x01]).is_none()); // len overruns
        assert!(decode(&[0x00]).is_none()); // field 0
    }
}
