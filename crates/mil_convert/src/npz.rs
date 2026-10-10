//! `npz` — zero-dependency reader for `.npz` archives (LoRA adapters
//! saved with `np.savez` / `np.savez_compressed`).
//!
//! An `.npz` is a ZIP archive of `.npy` members. We parse the central
//! directory, extract each member (stored or deflated — RFC 1951 raw
//! DEFLATE implemented below), then parse the `.npy` v1/v2 header to get
//! `(shape, f32 data)`.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

fn err(kind: io::ErrorKind, msg: impl Into<String>) -> io::Error {
    io::Error::new(kind, msg.into())
}
fn bad(msg: impl Into<String>) -> io::Error {
    err(io::ErrorKind::InvalidData, msg)
}

fn le16(b: &[u8], o: usize) -> u32 {
    u16::from_le_bytes([b[o], b[o + 1]]) as u32
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

// ---------------- NPY ----------------

/// One `.npy` member decoded to f32.
#[derive(Clone, Debug)]
pub struct Npy {
    /// Logical shape (row-major order).
    pub shape: Vec<i64>,
    /// Elements in row-major order.
    pub data: Vec<f32>,
}

/// Parse a `.npy` blob into shape + f32 elements.
fn parse_npy(buf: &[u8]) -> io::Result<Npy> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(bad("not an .npy blob (bad magic)"));
    }
    let (major, _) = (buf[6], buf[7]);
    let (hlen, off) = match major {
        1 => (le16(buf, 8) as usize, 10),
        2 | 3 => (le32(buf, 8) as usize, 12),
        v => return Err(bad(format!("unsupported .npy version {v}"))),
    };
    if buf.len() < off + hlen {
        return Err(bad(".npy header truncated"));
    }
    let hdr =
        std::str::from_utf8(&buf[off..off + hlen]).map_err(|_| bad(".npy header is not utf-8"))?;

    // descr: first quoted string after the key.
    let descr = {
        let k = hdr
            .find("'descr'")
            .or_else(|| hdr.find("\"descr\""))
            .ok_or_else(|| bad(".npy header missing 'descr'"))?;
        let rest = &hdr[k + 7..];
        let q0 = rest
            .find(['\'', '"'])
            .ok_or_else(|| bad(".npy descr not quoted"))?;
        let qc = rest.as_bytes()[q0];
        let q1 = rest[q0 + 1..]
            .find(qc as char)
            .ok_or_else(|| bad(".npy descr unterminated"))?;
        &rest[q0 + 1..q0 + 1 + q1]
    };
    let fortran = {
        let k = hdr
            .find("'fortran_order'")
            .or_else(|| hdr.find("\"fortran_order\""))
            .ok_or_else(|| bad(".npy header missing 'fortran_order'"))?;
        hdr[k..].contains("True")
    };
    let shape = {
        let k = hdr
            .find("'shape'")
            .or_else(|| hdr.find("\"shape\""))
            .ok_or_else(|| bad(".npy header missing 'shape'"))?;
        let rest = &hdr[k..];
        let p0 = rest.find('(').ok_or_else(|| bad(".npy shape missing ("))?;
        let p1 = rest.find(')').ok_or_else(|| bad(".npy shape missing )"))?;
        rest[p0 + 1..p1]
            .split(',')
            .filter_map(|t| {
                let t = t.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(
                        t.parse::<i64>()
                            .map_err(|_| bad(format!(".npy bad dim {t:?}"))),
                    )
                }
            })
            .collect::<io::Result<Vec<i64>>>()?
    };
    if fortran {
        return Err(bad(".npy fortran_order arrays unsupported"));
    }
    let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    let raw = &buf[off + hlen..];
    let data = match descr {
        "<f4" | "=f4" | "|f4" => {
            if raw.len() < n * 4 {
                return Err(bad(".npy data truncated"));
            }
            raw[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        "<f8" | "=f8" => {
            if raw.len() < n * 8 {
                return Err(bad(".npy data truncated"));
            }
            raw[..n * 8]
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        "<f2" | "=f2" => {
            if raw.len() < n * 2 {
                return Err(bad(".npy data truncated"));
            }
            raw[..n * 2]
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        d => return Err(bad(format!(".npy descr {d:?} unsupported"))),
    };
    Ok(Npy { shape, data })
}

// ---------------- DEFLATE (RFC 1951) ----------------

struct BitReader<'a> {
    d: &'a [u8],
    bit: usize,
}
impl<'a> BitReader<'a> {
    fn new(d: &'a [u8]) -> Self {
        BitReader { d, bit: 0 }
    }
    /// Read `n` bits LSB-first.
    fn bits(&mut self, n: u32) -> io::Result<u32> {
        let mut v = 0u32;
        for i in 0..n {
            let byte = self.bit / 8;
            if byte >= self.d.len() {
                return Err(bad("deflate: out of input"));
            }
            v |= (((self.d[byte] >> (self.bit % 8)) & 1) as u32) << i;
            self.bit += 1;
        }
        Ok(v)
    }
    /// Huffman codes are packed MSB-first: accumulate then match.
    fn huff(&mut self, h: &Huff) -> io::Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=15usize {
            code = (code << 1) | self.bits(1)? as i32;
            let count = h.counts[len] as i32;
            if code - first < count {
                return Ok(h.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
        }
        Err(bad("deflate: invalid huffman code"))
    }
    fn align(&mut self) {
        self.bit = (self.bit + 7) & !7;
    }
}

/// Canonical Huffman decoder built from code lengths (ZLIB-style).
struct Huff {
    counts: [u16; 16],
    symbols: Vec<u16>,
}
impl Huff {
    fn build(lengths: &[u8]) -> io::Result<Huff> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        // next_code / symbol order per RFC 1951 §3.2.2
        let mut offs = [0usize; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + counts[i - 1] as usize;
        }
        let mut symbols = vec![0u16; lengths.iter().filter(|&&l| l > 0).count()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                symbols[offs[l as usize]] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(Huff { counts, symbols })
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXT: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXT: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CLEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Raw DEFLATE decoder → decompressed bytes.
fn inflate(comp: &[u8], out_len: usize) -> io::Result<Vec<u8>> {
    let mut br = BitReader::new(comp);
    let mut out: Vec<u8> = Vec::with_capacity(out_len);
    loop {
        let bfinal = br.bits(1)?;
        let btype = br.bits(2)?;
        match btype {
            0 => {
                br.align();
                let p = br.bit / 8;
                if p + 4 > comp.len() {
                    return Err(bad("deflate: stored block truncated"));
                }
                let len = le16(comp, p) as usize;
                let nlen = le16(comp, p + 2) as usize;
                if len != (!nlen & 0xffff) {
                    return Err(bad("deflate: stored len/nlen mismatch"));
                }
                if p + 4 + len > comp.len() {
                    return Err(bad("deflate: stored data truncated"));
                }
                out.extend_from_slice(&comp[p + 4..p + 4 + len]);
                br.bit = (p + 4 + len) * 8;
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 {
                    // fixed tables
                    let mut ll = [0u8; 288];
                    for (i, l) in ll.iter_mut().enumerate() {
                        *l = match i {
                            0..=143 => 8,
                            144..=255 => 9,
                            256..=279 => 7,
                            _ => 8,
                        };
                    }
                    let dl = [5u8; 30];
                    (Huff::build(&ll)?, Huff::build(&dl)?)
                } else {
                    let hlit = br.bits(5)? as usize + 257;
                    let hdist = br.bits(5)? as usize + 1;
                    let hclen = br.bits(4)? as usize + 4;
                    if hlit > 286 || hdist > 30 {
                        return Err(bad("deflate: bad hlit/hdist"));
                    }
                    let mut cl = [0u8; 19];
                    for &i in CLEN_ORDER.iter().take(hclen) {
                        cl[i] = br.bits(3)? as u8;
                    }
                    let clh = Huff::build(&cl)?;
                    let total = hlit + hdist;
                    let mut lens = vec![0u8; total];
                    let mut i = 0;
                    while i < total {
                        let s = br.huff(&clh)?;
                        match s {
                            0..=15 => {
                                lens[i] = s as u8;
                                i += 1;
                            }
                            16 => {
                                if i == 0 {
                                    return Err(bad("deflate: rep with no prev"));
                                }
                                let rep = br.bits(2)? as usize + 3;
                                let prev = lens[i - 1];
                                for _ in 0..rep {
                                    if i >= total {
                                        return Err(bad("deflate: len overflow"));
                                    }
                                    lens[i] = prev;
                                    i += 1;
                                }
                            }
                            17 => {
                                let rep = br.bits(3)? as usize + 3;
                                i += rep;
                            }
                            18 => {
                                let rep = br.bits(7)? as usize + 11;
                                i += rep;
                            }
                            _ => return Err(bad("deflate: bad clen symbol")),
                        }
                        if i > total {
                            return Err(bad("deflate: clen overflow"));
                        }
                    }
                    (Huff::build(&lens[..hlit])?, Huff::build(&lens[hlit..])?)
                };
                loop {
                    let s = br.huff(&lit)?;
                    match s {
                        0..=255 => out.push(s as u8),
                        256 => break,
                        257..=285 => {
                            let li = (s - 257) as usize;
                            let len = LEN_BASE[li] as usize + br.bits(LEN_EXT[li])? as usize;
                            let ds = br.huff(&dist)? as usize;
                            if ds >= 30 {
                                return Err(bad("deflate: bad dist symbol"));
                            }
                            let d = DIST_BASE[ds] as usize + br.bits(DIST_EXT[ds])? as usize;
                            if d > out.len() {
                                return Err(bad("deflate: dist beyond output"));
                            }
                            for _ in 0..len {
                                let b = out[out.len() - d];
                                out.push(b);
                            }
                        }
                        _ => return Err(bad("deflate: bad literal symbol")),
                    }
                }
            }
            _ => return Err(bad("deflate: reserved btype")),
        }
        if bfinal == 1 {
            break;
        }
    }
    if out.len() != out_len {
        return Err(bad(format!(
            "deflate: expected {out_len} bytes, got {}",
            out.len()
        )));
    }
    Ok(out)
}

// ---------------- ZIP ----------------

/// A member of the archive: name + decompressed contents.
struct ZipMember {
    name: String,
    data: Vec<u8>,
}

/// Read every member of a ZIP archive (stored or deflated).
fn read_zip(buf: &[u8]) -> io::Result<Vec<ZipMember>> {
    // End of central directory record: scan back for PK\x05\x06.
    if buf.len() < 22 {
        return Err(bad("npz: too small for a zip"));
    }
    let mut eocd = None;
    for i in (0..=buf.len() - 22).rev() {
        if le32(buf, i) == 0x0605_4b50 {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.ok_or_else(|| bad("npz: no end-of-central-directory"))?;
    let count = le16(buf, eocd + 10) as usize;
    let mut cd = le32(buf, eocd + 16) as usize;

    let mut members = Vec::with_capacity(count);
    for _ in 0..count {
        if cd + 46 > buf.len() || le32(buf, cd) != 0x0201_4b50 {
            return Err(bad("npz: bad central directory entry"));
        }
        let method = le16(buf, cd + 10);
        let csize = le32(buf, cd + 20) as usize;
        let usize_ = le32(buf, cd + 24) as usize;
        let nlen = le16(buf, cd + 28) as usize;
        let elen = le16(buf, cd + 30) as usize;
        let clen = le16(buf, cd + 32) as usize;
        let lho = le32(buf, cd + 42) as usize;
        if cd + 46 + nlen > buf.len() {
            return Err(bad("npz: cd entry name truncated"));
        }
        let name = std::str::from_utf8(&buf[cd + 46..cd + 46 + nlen])
            .map_err(|_| bad("npz: non-utf8 member name"))?
            .to_string();

        // Local header → data offset (name/extra lengths can differ
        // from the CD copy, so read them here).
        if lho + 30 > buf.len() || le32(buf, lho) != 0x0403_4b50 {
            return Err(bad(format!("npz: {name}: bad local header")));
        }
        let lnlen = le16(buf, lho + 26) as usize;
        let lelen = le16(buf, lho + 28) as usize;
        let dstart = lho + 30 + lnlen + lelen;
        if dstart + csize > buf.len() {
            return Err(bad(format!("npz: {name}: data truncated")));
        }
        let comp = &buf[dstart..dstart + csize];
        let data = match method {
            0 => {
                if comp.len() != usize_ {
                    return Err(bad(format!("npz: {name}: stored size mismatch")));
                }
                comp.to_vec()
            }
            8 => inflate(comp, usize_).map_err(|e| bad(format!("npz: {name}: {e}")))?,
            m => return Err(bad(format!("npz: {name}: zip method {m} unsupported"))),
        };
        members.push(ZipMember { name, data });
        cd += 46 + nlen + elen + clen;
    }
    Ok(members)
}

// ---------------- public API ----------------

/// An opened `.npz` archive: every member decoded to `.npy` f32 data.
pub struct Npz {
    /// member name (without `.npy` suffix) → tensor
    map: BTreeMap<String, Npy>,
}

/// Open an `.npz` file and decode all members.
pub fn open(path: &Path) -> io::Result<Npz> {
    let buf = std::fs::read(path)?;
    let mut map = BTreeMap::new();
    for m in read_zip(&buf)? {
        let stem = m.name.strip_suffix(".npy").unwrap_or(&m.name).to_string();
        let npy = parse_npy(&m.data).map_err(|e| bad(format!("npz: {}: {e}", m.name)))?;
        map.insert(stem, npy);
    }
    if map.is_empty() {
        return Err(bad(format!("npz: no .npy members in {}", path.display())));
    }
    Ok(Npz { map })
}

impl Npz {
    /// All member names (without `.npy` suffix).
    pub fn names(&self) -> Vec<&str> {
        self.map.keys().map(|s| s.as_str()).collect()
    }
    /// Tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&Npy> {
        self.map.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a minimal .npy v1.0 f32 blob.
    fn npy_f32(shape: &[i64], data: &[f32]) -> Vec<u8> {
        let dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
                + if shape.len() == 1 { "," } else { "" }
        );
        let mut h = dict;
        // pad so total header is a multiple of 64 including magic+ver+len
        let pad = (64 - ((10 + h.len() + 1) % 64)) % 64;
        h.push_str(&" ".repeat(pad));
        h.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(h.as_bytes());
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Minimal single-member ZIP (stored method, no compression).
    fn zip_stored(name: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let n = name.as_bytes();
        // local header
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method stored
        out.extend_from_slice(&0u32.to_le_bytes()); // time/date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (unused)
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(n);
        out.extend_from_slice(data);
        // central directory
        let cd = out.len();
        out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // made by
        out.extend_from_slice(&20u16.to_le_bytes()); // need
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method
        out.extend_from_slice(&0u32.to_le_bytes()); // time/date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // extra+comment+disk+int_attr
        out.extend_from_slice(&0u32.to_le_bytes()); // ext attr
        out.extend_from_slice(&0u32.to_le_bytes()); // local offset
        out.extend_from_slice(n);
        let cdlen = out.len() - cd;
        // eocd
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // disk
        out.extend_from_slice(&1u16.to_le_bytes()); // entries on disk
        out.extend_from_slice(&1u16.to_le_bytes()); // total entries
        out.extend_from_slice(&(cdlen as u32).to_le_bytes());
        out.extend_from_slice(&(cd as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }

    #[test]
    fn npy_roundtrip() {
        let blob = npy_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = parse_npy(&blob).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn zip_stored_roundtrip() {
        let npy = npy_f32(&[4], &[7.0, 8.0, 9.0, 10.0]);
        let zip = zip_stored("w.npy", &npy);
        let members = read_zip(&zip).unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].name, "w.npy");
        let t = parse_npy(&members[0].data).unwrap();
        assert_eq!(t.data, vec![7.0, 8.0, 9.0, 10.0]);
    }

    /// Deflate stream produced by `zlib.compress(data)[2:-4]` over a
    /// known payload — captured once as a fixture (see test data below).
    const DEFLATE_STREAM: &[u8] = &[
        0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x28, 0xcf, 0x2f, 0xca, 0x49, 0x51, 0x04, 0x02, 0x00,
    ];
    const DEFLATE_PLAIN: &[u8] = b"hello world!!!!";

    #[test]
    fn inflate_fixed_huffman() {
        // zlib.compress(b"hello world!!!!") level default →
        // strip 2-byte zlib header + 4-byte adler32 → raw deflate.
        let out = inflate(DEFLATE_STREAM, DEFLATE_PLAIN.len()).unwrap();
        assert_eq!(out, DEFLATE_PLAIN);
    }
}
