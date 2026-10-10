//! Golden-vector records — a versioned, self-checksummed snapshot of
//! the reference forward's output for a fixed prompt, plus the check
//! that proves a compiled package still agrees with it.
//!
//! The workflow:
//!
//! 1. **Generate** — [`Golden::from_reference`] runs
//!    [`crate::reference::forward`] on a weight source (HF safetensors
//!    dir or GGUF) for a fixed token-id prompt and stores either the
//!    per-position top-k (`k` indices + values, ~`positions·k·8`
//!    bytes) or the full logit matrix. The record also pins the
//!    model's identity: config dims, a free-form model id, a source
//!    description, and the SHA-256 of the weight files the logits
//!    were computed from ([`hash_model_source`]).
//! 2. **Check** — [`Golden::check`] compares per-position logit
//!    columns extracted from a package prediction
//!    ([`columns_from_output`]) against the record under a
//!    [`Tolerances`] gate: top-1 agreement, top-k overlap, a bound on
//!    the value drift at shared top-k indices, and a NaN ban. A
//!    [`CheckReport`] lists every failure, so a dishonest "pass"
//!    can't hide inside a single bool.
//! 3. **Differential** — [`diff_columns`] produces a per-position
//!    report between two runs of the same package (the
//!    CPU-vs-default-units measurement the e2e asserts top-1
//!    agreement on).
//!
//! # Record format (v1, all little-endian)
//!
//! ```text
//! u8[4]   magic "MGLD"
//! u32     format_version          (= 1)
//! u64     created_unix            (seconds; 0 = unknown)
//! str     toolchain               (free text, e.g. "mil_verify 0.2.0")
//! str     model_id                (free text, e.g. "qwen3-0.6b-hf")
//! str     model_type              (config.json model_type)
//! i64×6   hidden, layers, heads, kv_heads, head_dim, vocab
//! str     source_desc             ("dir:qwen3-hf (2 files)", ...)
//! u8[32]  source_sha256           (hash_model_source)
//! u32     n_tokens; u32×n         input token ids
//! u8      payload tag             (1 = Full, 2 = TopK)
//!   Full: u32 n_pos, u32 vocab, f32[n_pos·vocab]
//!   TopK: u32 n_pos, u32 vocab, u32 k, (u32 idx, f32 val)[n_pos·k]
//! u8[32]  record_sha256           (over every preceding byte)
//! ```
//!
//! `str` = u32 byte length + UTF-8 bytes. The trailing digest makes a
//! truncated or bit-flipped record a read error, not a silent
//! acceptance; a reader that doesn't know a field still can't miss it
//! because the digest covers the whole byte range.
//!
//! The `golden` example (`cargo run -p mil_verify --example golden`)
//! is the CLI harness: `gen` / `check` / `diff` subcommands.

use mil_convert::config::ModelConfig;
use mil_convert::WeightSource;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::Path;

type Res<T> = io::Result<T>;

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ================= SHA-256 (self-contained, FIPS 180-4) =================

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Incremental SHA-256. `update` any number of times, then `finish`.
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Sha256 {
    /// Fresh hasher.
    pub fn new() -> Self {
        Sha256 {
            h: SHA_H0,
            buf: [0u8; 64],
            buf_len: 0,
            total: 0,
        }
    }

    /// Feed bytes.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
            if data.is_empty() {
                return;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            self.compress(block.try_into().unwrap());
            data = rest;
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.buf_len = data.len();
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (h, v) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *h = h.wrapping_add(v);
        }
    }

    /// Final digest.
    pub fn finish(mut self) -> [u8; 32] {
        let bit_len = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buf_len != 56 {
            self.update(&[0]);
        }
        // Length block — hand-pack to avoid update() mutating `total`.
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-256 of a byte slice.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finish()
}

/// SHA-256 of a file, streamed in 1 MiB reads.
pub fn sha256_file(path: &Path) -> Res<[u8; 32]> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finish())
}

/// SHA-256 identifying a weight source: a regular file (e.g. `.gguf`)
/// is hashed directly; a directory is hashed as `config.json` plus
/// every `*.safetensors`/`*.gguf` member in sorted name order, each
/// framed as `u32 name_len | name | u64 size | bytes` so equal file
/// contents under different layouts hash differently. Returns the
/// digest and a short descriptor for the record.
pub fn hash_model_source(path: &Path) -> Res<([u8; 32], String)> {
    if path.is_file() {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("weights");
        return Ok((sha256_file(path)?, format!("file:{name}")));
    }
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let cfg = path.join("config.json");
    if cfg.exists() {
        files.push(cfg);
    }
    let mut members: Vec<std::path::PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x == "safetensors" || x == "gguf")
                .unwrap_or(false)
        })
        .collect();
    members.sort();
    files.extend(members);
    if files.is_empty() {
        return Err(err(format!(
            "{}: no config.json/safetensors/gguf to hash",
            path.display()
        )));
    }
    let mut h = Sha256::new();
    for f in &files {
        let name = f
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .as_bytes();
        h.update(&(name.len() as u32).to_le_bytes());
        h.update(name);
        let size = f.metadata()?.len();
        h.update(&size.to_le_bytes());
        let mut fp = std::fs::File::open(f)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = fp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
    }
    let dir = path.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    Ok((h.finish(), format!("dir:{dir} ({} files)", files.len())))
}

/// Hex string for a digest — record dumps and logs.
pub fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

// ================= record =================

/// Record format magic.
pub const MAGIC: &[u8; 4] = b"MGLD";
/// Current record format version.
pub const FORMAT_VERSION: u32 = 1;
/// Default top-k stored per position (32 × 8 B ≈ 256 B/position).
pub const DEFAULT_TOP_K: usize = 32;

/// Per-position logits payload.
#[derive(Clone, Debug, PartialEq)]
pub enum Logits {
    /// `full[pos][vocab]` — exact values; largest and strictest.
    Full(Vec<Vec<f32>>),
    /// Per position the top-`k` `(vocab_index, value)` pairs, sorted by
    /// value descending — `idx[0]` is the golden argmax.
    TopK {
        /// Declared vocabulary size.
        vocab: u32,
        /// Entries per position.
        k: u32,
        /// `idx[pos][entry]` — vocab indices, best first.
        idx: Vec<Vec<u32>>,
        /// `val[pos][entry]` — the logit at the matching index.
        val: Vec<Vec<f32>>,
    },
}

/// A versioned golden record.
#[derive(Clone, Debug)]
pub struct Golden {
    /// Free-form model identity ("qwen3-0.6b-hf").
    pub model_id: String,
    /// `model_type` from the config ("qwen3", "llama", ...).
    pub model_type: String,
    /// Config dims that define the architecture.
    pub hidden_size: i64,
    /// Layer count.
    pub num_layers: u32,
    /// Query heads.
    pub num_heads: i64,
    /// KV heads.
    pub num_kv_heads: i64,
    /// Per-head dim.
    pub head_dim: i64,
    /// Vocabulary size.
    pub vocab_size: i64,
    /// What `source_sha256` covers (from [`hash_model_source`]).
    pub source_desc: String,
    /// SHA-256 of the weight source the logits came from.
    pub source_sha256: [u8; 32],
    /// Input token ids the logits were computed for.
    pub token_ids: Vec<u32>,
    /// Per-position logits (full or top-k).
    pub logits: Logits,
    /// Free text: toolchain that generated the record.
    pub toolchain: String,
    /// Seconds since the unix epoch at generation (0 = unknown).
    pub created_unix: u64,
}

/// Where a record's logits came from — the fields that pin model
/// identity rather than values.
#[derive(Clone, Debug)]
pub struct Provenance {
    /// Free-form model identity ("qwen3-0.6b-hf").
    pub model_id: String,
    /// What `source_sha256` covers (from [`hash_model_source`]).
    pub source_desc: String,
    /// SHA-256 of the weight source the logits came from.
    pub source_sha256: [u8; 32],
    /// Free text: toolchain that generated the record.
    pub toolchain: String,
}

impl Provenance {
    /// Convenience constructor.
    pub fn new(
        model_id: impl Into<String>,
        source_desc: impl Into<String>,
        source_sha256: [u8; 32],
        toolchain: impl Into<String>,
    ) -> Self {
        Provenance {
            model_id: model_id.into(),
            source_desc: source_desc.into(),
            source_sha256,
            toolchain: toolchain.into(),
        }
    }
}

/// Indices of the `k` largest values, best first.
fn top_k_idx(v: &[f32], k: usize) -> Vec<u32> {
    let mut ix: Vec<u32> = (0..v.len() as u32).collect();
    ix.sort_by(|&i, &j| {
        v[j as usize]
            .partial_cmp(&v[i as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ix.truncate(k.min(v.len()));
    ix
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Golden {
    fn skeleton(prov: &Provenance, cfg: &ModelConfig, token_ids: Vec<u32>) -> Golden {
        Golden {
            model_id: prov.model_id.clone(),
            model_type: cfg.model_type.clone(),
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_layers as u32,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
            source_desc: prov.source_desc.clone(),
            source_sha256: prov.source_sha256,
            token_ids,
            logits: Logits::Full(Vec::new()),
            toolchain: prov.toolchain.clone(),
            created_unix: now_unix(),
        }
    }

    /// Build a record around already-computed reference logits,
    /// keeping the top-`k` per position.
    pub fn from_logits_topk(
        prov: &Provenance,
        cfg: &ModelConfig,
        token_ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
        k: usize,
    ) -> Golden {
        let mut idx = Vec::with_capacity(logits.len());
        let mut val = Vec::with_capacity(logits.len());
        let vocab = logits.first().map(|l| l.len()).unwrap_or(0) as u32;
        for col in &logits {
            let ix = top_k_idx(col, k);
            val.push(ix.iter().map(|&i| col[i as usize]).collect());
            idx.push(ix);
        }
        let mut g = Self::skeleton(prov, cfg, token_ids);
        g.logits = Logits::TopK {
            vocab,
            k: k as u32,
            idx,
            val,
        };
        g
    }

    /// Build a record around already-computed reference logits,
    /// keeping the full matrix.
    pub fn from_logits_full(
        prov: &Provenance,
        cfg: &ModelConfig,
        token_ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
    ) -> Golden {
        let mut g = Self::skeleton(prov, cfg, token_ids);
        g.logits = Logits::Full(logits);
        g
    }

    /// Run [`crate::reference::forward`] and record the top-`k` logits.
    pub fn from_reference(
        prov: &Provenance,
        cfg: &ModelConfig,
        w: &dyn WeightSource,
        token_ids: Vec<u32>,
        k: usize,
    ) -> Res<Golden> {
        let logits = crate::reference::forward(cfg, w, &token_ids)?;
        Ok(Self::from_logits_topk(prov, cfg, token_ids, logits, k))
    }

    /// Positions covered by the record.
    pub fn positions(&self) -> usize {
        match &self.logits {
            Logits::Full(l) => l.len(),
            Logits::TopK { idx, .. } => idx.len(),
        }
    }

    /// Declared vocab size of the payload.
    pub fn vocab(&self) -> usize {
        match &self.logits {
            Logits::Full(l) => l.first().map(|c| c.len()).unwrap_or(0),
            Logits::TopK { vocab, .. } => *vocab as usize,
        }
    }

    /// Golden argmax at `pos`.
    pub fn top1(&self, pos: usize) -> Option<u32> {
        match &self.logits {
            Logits::Full(l) => l.get(pos).map(|c| top_k_idx(c, 1)[0]),
            Logits::TopK { idx, .. } => idx.get(pos).and_then(|v| v.first().copied()),
        }
    }

    /// One-line human summary.
    pub fn summary(&self) -> String {
        let payload = match &self.logits {
            Logits::Full(l) => format!("full {}×{}", l.len(), self.vocab()),
            Logits::TopK { k, .. } => format!("top{k}"),
        };
        format!(
            "{} ({}) {} pos={} src={} sha={}.. by {}",
            self.model_id,
            self.model_type,
            payload,
            self.positions(),
            self.source_desc,
            &hex(&self.source_sha256)[..16],
            self.toolchain,
        )
    }

    // ---------- serialization ----------

    /// Serialize to bytes (format v1).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        put_u32(&mut b, FORMAT_VERSION);
        put_u64(&mut b, self.created_unix);
        put_str(&mut b, &self.toolchain);
        put_str(&mut b, &self.model_id);
        put_str(&mut b, &self.model_type);
        for v in [
            self.hidden_size,
            self.num_layers as i64,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.vocab_size,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        put_str(&mut b, &self.source_desc);
        b.extend_from_slice(&self.source_sha256);
        put_u32(&mut b, self.token_ids.len() as u32);
        for &t in &self.token_ids {
            put_u32(&mut b, t);
        }
        match &self.logits {
            Logits::Full(cols) => {
                b.push(1);
                put_u32(&mut b, cols.len() as u32);
                put_u32(&mut b, cols.first().map(|c| c.len()).unwrap_or(0) as u32);
                for c in cols {
                    for &v in c {
                        b.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            Logits::TopK { vocab, k, idx, val } => {
                b.push(2);
                put_u32(&mut b, idx.len() as u32);
                put_u32(&mut b, *vocab);
                put_u32(&mut b, *k);
                for (ix, vs) in idx.iter().zip(val) {
                    debug_assert_eq!(ix.len(), vs.len());
                    for (&i, &v) in ix.iter().zip(vs) {
                        put_u32(&mut b, i);
                        b.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        let digest = sha256(&b);
        b.extend_from_slice(&digest);
        b
    }

    /// Serialize to a file.
    pub fn write(&self, path: &Path) -> Res<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.to_bytes())
    }

    /// Deserialize, verifying magic, version, and the trailing digest.
    pub fn from_bytes(b: &[u8]) -> Res<Golden> {
        if b.len() < 4 + 4 + 32 {
            return Err(err("record too short"));
        }
        if &b[..4] != MAGIC {
            return Err(err("bad magic — not a golden record"));
        }
        let (body, digest) = b.split_at(b.len() - 32);
        if sha256(body) != digest {
            return Err(err("record digest mismatch — corrupt golden file"));
        }
        let mut r = Reader { b: body, i: 4 };
        let version = r.u32()?;
        if version == 0 || version > FORMAT_VERSION {
            return Err(err(format!(
                "record version {version} > supported {FORMAT_VERSION}"
            )));
        }
        let created_unix = r.u64()?;
        let toolchain = r.str()?;
        let model_id = r.str()?;
        let model_type = r.str()?;
        let mut dims = [0i64; 6];
        for d in dims.iter_mut() {
            *d = r.i64()?;
        }
        let source_desc = r.str()?;
        let mut source_sha256 = [0u8; 32];
        source_sha256.copy_from_slice(r.bytes(32)?);
        let n_tok = r.u32()? as usize;
        let mut token_ids = Vec::with_capacity(n_tok);
        for _ in 0..n_tok {
            token_ids.push(r.u32()?);
        }
        let tag = r.u8()?;
        let logits = match tag {
            1 => {
                let n_pos = r.u32()? as usize;
                let vocab = r.u32()? as usize;
                let mut cols = Vec::with_capacity(n_pos);
                for _ in 0..n_pos {
                    let mut c = Vec::with_capacity(vocab);
                    for _ in 0..vocab {
                        c.push(r.f32()?);
                    }
                    cols.push(c);
                }
                Logits::Full(cols)
            }
            2 => {
                let n_pos = r.u32()? as usize;
                let vocab = r.u32()?;
                let k = r.u32()?;
                let mut idx = Vec::with_capacity(n_pos);
                let mut val = Vec::with_capacity(n_pos);
                for _ in 0..n_pos {
                    let mut ix = Vec::with_capacity(k as usize);
                    let mut vs = Vec::with_capacity(k as usize);
                    for _ in 0..k {
                        ix.push(r.u32()?);
                        vs.push(r.f32()?);
                    }
                    idx.push(ix);
                    val.push(vs);
                }
                Logits::TopK { vocab, k, idx, val }
            }
            t => return Err(err(format!("unknown payload tag {t}"))),
        };
        if r.i != body.len() {
            return Err(err(format!(
                "trailing bytes after payload ({} extra)",
                body.len() - r.i
            )));
        }
        Ok(Golden {
            model_id,
            model_type,
            hidden_size: dims[0],
            num_layers: dims[1] as u32,
            num_heads: dims[2],
            num_kv_heads: dims[3],
            head_dim: dims[4],
            vocab_size: dims[5],
            source_desc,
            source_sha256,
            token_ids,
            logits,
            toolchain,
            created_unix,
        })
    }

    /// Read a record file.
    pub fn read(path: &Path) -> Res<Golden> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    // ---------- checking ----------

    /// Compare `actual[pos][vocab]` columns against the record under
    /// `tol`. `actual` comes from [`columns_from_output`] (or any
    /// per-position logit slice on the same token prompt).
    pub fn check(&self, actual: &[Vec<f32>], tol: &Tolerances) -> CheckReport {
        let mut rep = CheckReport::default();
        if actual.len() != self.positions() {
            rep.failures.push(format!(
                "position count: record has {}, actual has {}",
                self.positions(),
                actual.len()
            ));
            return rep;
        }
        for (pos, col) in actual.iter().enumerate() {
            if col.len() != self.vocab() {
                rep.failures.push(format!(
                    "pos {pos}: vocab {} != record {}",
                    col.len(),
                    self.vocab()
                ));
                continue;
            }
            let mut pc = PositionCheck {
                pos,
                golden_top1: self.top1(pos).map(|v| v as usize).unwrap_or(0),
                cosine: f64::NAN,
                ..Default::default()
            };
            pc.nan = col.iter().filter(|v| v.is_nan()).count();
            let finite = pc.nan == 0;
            if finite {
                let k = match &self.logits {
                    Logits::Full(_) => tol.check_k.max(5),
                    Logits::TopK { k, .. } => *k as usize,
                };
                let actual_top = top_k_idx(col, k.max(1));
                pc.actual_top1 = actual_top[0] as usize;
                match &self.logits {
                    Logits::Full(gcol) => {
                        let gcol = &gcol[pos];
                        let golden_top = top_k_idx(gcol, k.max(1));
                        pc.golden_top1 = golden_top[0] as usize;
                        let gset: std::collections::HashSet<u32> =
                            golden_top.iter().copied().collect();
                        pc.overlap = actual_top.iter().filter(|i| gset.contains(i)).count();
                        pc.max_val_delta = col
                            .iter()
                            .zip(gcol)
                            .map(|(a, g)| (a - g).abs())
                            .fold(0f32, f32::max);
                        let dot: f64 = col
                            .iter()
                            .zip(gcol)
                            .map(|(a, g)| *a as f64 * *g as f64)
                            .sum();
                        let na: f64 = col
                            .iter()
                            .map(|v| *v as f64 * *v as f64)
                            .sum::<f64>()
                            .sqrt();
                        let nb: f64 = gcol
                            .iter()
                            .map(|v| *v as f64 * *v as f64)
                            .sum::<f64>()
                            .sqrt();
                        pc.cosine = dot / (na * nb + 1e-30);
                        pc.checked_vals = col.len();
                    }
                    Logits::TopK { idx, val, .. } => {
                        let aset: std::collections::HashSet<u32> =
                            actual_top.iter().copied().collect();
                        pc.overlap = idx[pos].iter().filter(|i| aset.contains(i)).count();
                        let mut worst = 0f32;
                        let mut checked = 0usize;
                        for (&i, &gv) in idx[pos].iter().zip(&val[pos]) {
                            if aset.contains(&i) {
                                worst = worst.max((col[i as usize] - gv).abs());
                                checked += 1;
                            }
                        }
                        pc.max_val_delta = worst;
                        pc.checked_vals = checked;
                    }
                }
            }
            // verdict for this position
            if tol.forbid_nan && pc.nan > 0 {
                rep.failures
                    .push(format!("pos {pos}: {} NaN logits", pc.nan));
            }
            if tol.top1 && finite && pc.actual_top1 != pc.golden_top1 {
                rep.failures.push(format!(
                    "pos {pos}: top-1 {} != golden {}",
                    pc.actual_top1, pc.golden_top1
                ));
            }
            if tol.min_overlap > 0 && finite && pc.overlap < tol.min_overlap {
                rep.failures.push(format!(
                    "pos {pos}: top-k overlap {} < {}",
                    pc.overlap, tol.min_overlap
                ));
            }
            if finite && pc.max_val_delta > tol.max_val_delta {
                rep.failures.push(format!(
                    "pos {pos}: logit drift {:.4} > {}",
                    pc.max_val_delta, tol.max_val_delta
                ));
            }
            rep.positions.push(pc);
        }
        rep
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}

/// Bounds-checked cursor over a record body.
struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Res<&'a [u8]> {
        if self.i + n > self.b.len() {
            return Err(err("record truncated"));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Res<i64> {
        Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Res<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Res<String> {
        let n = self.u32()? as usize;
        let b = self.bytes(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| err("non-UTF8 string field"))
    }
}

// ================= check reports =================

/// The gate a package's logits must pass against a golden record.
///
/// Defaults are measured, not guessed (`tests/golden_e2e.rs` output):
/// converted fp16 packages on CpuOnly drift from the f64 reference by
/// ≤0.35 absolute at shared top-k indices on logits of magnitude ~20
/// (qwen3-0.6B worst 0.3402, smollm2-135M worst 0.2218), and a single
/// corrupted fp16 weight byte produced 0.6335 — so `max_val_delta`
/// = 0.5 sits between the two populations with ~1.5× headroom each
/// way. `min_overlap` = 5 of the stored top-k: healthy runs measure
/// 31–32 of 32, corruption dropped it to 28 but the drift gate is
/// what actually fired.
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    /// Require position-wise argmax equality.
    pub top1: bool,
    /// Minimum shared indices between the golden top-k and the
    /// actual top-k (TopK records: k = stored k; Full records:
    /// k = `check_k`). 0 disables.
    pub min_overlap: usize,
    /// Top-k depth used for overlap on Full records.
    pub check_k: usize,
    /// Max allowed |actual − golden| at shared top-k indices (or the
    /// whole vector for Full records).
    pub max_val_delta: f32,
    /// Any NaN logit fails.
    pub forbid_nan: bool,
}

impl Default for Tolerances {
    fn default() -> Self {
        Tolerances {
            top1: true,
            min_overlap: 5,
            check_k: 32,
            max_val_delta: 0.5,
            forbid_nan: true,
        }
    }
}

/// Per-position outcome of [`Golden::check`].
#[derive(Clone, Debug, Default)]
pub struct PositionCheck {
    /// Position index.
    pub pos: usize,
    /// Golden argmax.
    pub golden_top1: usize,
    /// Actual argmax.
    pub actual_top1: usize,
    /// Shared indices within the compared top-k.
    pub overlap: usize,
    /// Indices whose values were compared.
    pub checked_vals: usize,
    /// Largest |actual − golden| among compared indices.
    pub max_val_delta: f32,
    /// NaN logits in the actual column.
    pub nan: usize,
    /// Full-record cosine similarity (NaN for TopK records).
    pub cosine: f64,
}

/// Whole-record outcome of [`Golden::check`].
#[derive(Clone, Debug, Default)]
pub struct CheckReport {
    /// One entry per compared position.
    pub positions: Vec<PositionCheck>,
    /// Human-readable failures; empty = pass.
    pub failures: Vec<String>,
}

impl CheckReport {
    /// True when no tolerance was violated.
    pub fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for p in &self.positions {
            write!(
                f,
                "pos {}: top1 {}vs{} overlap={} dmax={:.4} nan={}",
                p.pos, p.actual_top1, p.golden_top1, p.overlap, p.max_val_delta, p.nan
            )?;
            if p.cosine.is_finite() {
                write!(f, " cos={:.6}", p.cosine)?;
            }
            writeln!(f)?;
        }
        if self.is_ok() {
            write!(f, "golden check: PASS")?;
        } else {
            write!(f, "golden check: FAIL ({}):", self.failures.len())?;
            for e in &self.failures {
                write!(f, "\n  {e}")?;
            }
        }
        Ok(())
    }
}

// ================= run helpers (package-side) =================

/// An owned prediction input in the layout the converted decoder
/// packages declare — same contract as `tests/reference_e2e.rs`:
/// `x (1,d,S,1)` hidden states, `cos`/`sin (1,1,S,hd)` rotate-half
/// rope tables, `mask (1,1,S,max_kv)` additive causal mask, `pos`
/// scalar int32. Build with [`package_inputs`]; convert to
/// `mil_infer::Input` at the call site (mil_infer is a dev-dep).
#[derive(Clone, Debug)]
pub struct PreparedInput {
    /// Feature name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<i64>,
    /// Little-endian element bytes matching `dtype`.
    pub data: Vec<u8>,
    /// Element dtype.
    pub dtype: mil_spec::DType,
}

/// Marshal `ids` into the package input contract for a model
/// converted with `embed: false`. Embedding rows are read from the
/// same weight source the reference/golden used, so both sides see
/// identical input values (Q8_0 embed tables included).
pub fn package_inputs(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
    max_kv: usize,
) -> Res<Vec<PreparedInput>> {
    let d = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let s = ids.len();
    // x (1, d, s, 1): x[c*s + p] = emb[ids[p]][c]
    let (shape, emb) = w.tensor_f16("model.embed_tokens.weight")?;
    if shape.len() != 2 || shape[1] as usize != d {
        return Err(err(format!("embed shape {shape:?}")));
    }
    let mut x = vec![0u8; s * d * 2];
    for (p, &id) in ids.iter().enumerate() {
        let off = id as usize * d * 2;
        for c in 0..d {
            x[(c * s + p) * 2] = emb[off + c * 2];
            x[(c * s + p) * 2 + 1] = emb[off + c * 2 + 1];
        }
    }
    // cos/sin (1,1,s,hd): HF rotate-half, freq = theta^{-2(c%hd/2)/hd}
    let mut cos = Vec::with_capacity(s * hd * 2);
    let mut sin = Vec::with_capacity(s * hd * 2);
    for p in 0..s {
        for c in 0..hd {
            let f =
                (p as f64) * (cfg.rope_theta as f64).powf(-2.0 * (c % (hd / 2)) as f64 / hd as f64);
            cos.extend_from_slice(&half::f16::from_f64(f.cos()).to_le_bytes());
            sin.extend_from_slice(&half::f16::from_f64(f.sin()).to_le_bytes());
        }
    }
    // mask (1,1,s,max_kv): 0 where j <= p, -1e4 elsewhere
    let mut mask = Vec::with_capacity(s * max_kv * 2);
    for p in 0..s {
        for j in 0..max_kv {
            let v = if j <= p { 0.0f32 } else { -1e4 };
            mask.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    Ok(vec![
        PreparedInput {
            name: "x".into(),
            shape: vec![1, d as i64, s as i64, 1],
            data: x,
            dtype: mil_spec::DType::Fp16,
        },
        PreparedInput {
            name: "cos".into(),
            shape: vec![1, 1, s as i64, hd as i64],
            data: cos,
            dtype: mil_spec::DType::Fp16,
        },
        PreparedInput {
            name: "sin".into(),
            shape: vec![1, 1, s as i64, hd as i64],
            data: sin,
            dtype: mil_spec::DType::Fp16,
        },
        PreparedInput {
            name: "mask".into(),
            shape: vec![1, 1, s as i64, max_kv as i64],
            data: mask,
            dtype: mil_spec::DType::Fp16,
        },
        PreparedInput {
            name: "pos".into(),
            shape: vec![1],
            data: 0i32.to_le_bytes().to_vec(),
            dtype: mil_spec::DType::Int32,
        },
    ])
}

/// Extract per-position logit columns from a package output laid out
/// `(1, vocab, s, 1)` — column `p` is `flat[v*s + p]` for `v` in
/// `0..vocab`.
pub fn columns_from_output(flat: &[f32], s: usize) -> Vec<Vec<f32>> {
    if s == 0 || flat.len() % s != 0 {
        return Vec::new();
    }
    let vocab = flat.len() / s;
    (0..s)
        .map(|p| (0..vocab).map(|v| flat[v * s + p]).collect())
        .collect()
}

// ================= unit differential =================

/// Per-position difference between two logit dumps of the same
/// package — the CPU-vs-default-units report.
#[derive(Clone, Debug)]
pub struct UnitDiff {
    /// Position index.
    pub pos: usize,
    /// max |a − b|.
    pub max_abs: f32,
    /// mean |a − b|.
    pub mean_abs: f32,
    /// cosine(a, b).
    pub cosine: f64,
    /// argmax of `a`.
    pub top1_a: usize,
    /// argmax of `b`.
    pub top1_b: usize,
}

/// Diff two `[pos][vocab]` logit dumps position by position.
pub fn diff_columns(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<UnitDiff> {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(pos, (x, y))| {
            let mut max_abs = 0f32;
            let mut sum = 0f64;
            let mut dot = 0f64;
            let mut nx = 0f64;
            let mut ny = 0f64;
            for (&u, &v) in x.iter().zip(y) {
                let d = (u - v).abs();
                max_abs = max_abs.max(d);
                sum += d as f64;
                dot += u as f64 * v as f64;
                nx += u as f64 * u as f64;
                ny += v as f64 * v as f64;
            }
            UnitDiff {
                pos,
                max_abs,
                mean_abs: (sum / x.len().max(1) as f64) as f32,
                cosine: dot / (nx.sqrt() * ny.sqrt() + 1e-30),
                top1_a: top_k_idx(x, 1)[0] as usize,
                top1_b: top_k_idx(y, 1)[0] as usize,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_of(s: &str) -> String {
        let d = sha256(s.as_bytes());
        hex(&d)
    }

    #[test]
    fn sha256_vectors() {
        // FIPS 180-4 / RFC 4634 test vectors.
        assert_eq!(
            hex_of(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_of("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // 1M 'a' — exercises the streaming path across many blocks.
        let mut h = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(
            crate::golden::hex(&h.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        // Odd-sized updates crossing block boundaries.
        let mut h2 = Sha256::new();
        for i in 0..77 {
            h2.update(&[i as u8]);
        }
        let mut one = Sha256::new();
        one.update(&(0u8..77).collect::<Vec<u8>>());
        let mut two = Sha256::new();
        let data = (0u8..77).collect::<Vec<u8>>();
        two.update(&data[..5]);
        two.update(&data[5..70]);
        two.update(&data[70..]);
        let want = one.finish();
        assert_eq!(h2.finish(), want);
        assert_eq!(two.finish(), want);
    }

    fn tiny_cfg() -> ModelConfig {
        ModelConfig::from_json(
            br#"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":2,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
                "intermediate_size":16,"vocab_size":32,"rms_norm_eps":1e-6,
                "rope_theta":10000.0,"max_position_embeddings":64,
                "tie_word_embeddings":false}"#,
        )
        .unwrap()
    }

    fn sample_golden(k: usize) -> Golden {
        // 3 positions × 32 vocab; position p's logits are a shifted
        // ramp so argmax differs per position.
        let logits: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                (0..32)
                    .map(|v| (v as f32 + p as f32 * 3.0) * 0.1 - 1.0)
                    .collect()
            })
            .collect();
        Golden::from_logits_topk(
            &Provenance::new("tiny", "synthetic", [7u8; 32], "mil_verify test"),
            &tiny_cfg(),
            vec![1, 2, 3],
            logits,
            k,
        )
    }

    #[test]
    fn record_roundtrip_topk() {
        let g = sample_golden(8);
        let bytes = g.to_bytes();
        let back = Golden::from_bytes(&bytes).unwrap();
        assert_eq!(back.model_id, "tiny");
        assert_eq!(back.num_layers, 2);
        assert_eq!(back.token_ids, vec![1, 2, 3]);
        assert_eq!(back.positions(), 3);
        assert_eq!(back.top1(0), Some(31));
        assert_eq!(back.source_sha256, [7u8; 32]);
        // A byte flip anywhere in the body must trip the digest.
        let mut bad = bytes.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 0x40;
        assert!(Golden::from_bytes(&bad).is_err());
        // Truncation, bad magic, and future versions all fail.
        assert!(Golden::from_bytes(&bytes[..bytes.len() - 40]).is_err());
        let mut badmagic = bytes.clone();
        badmagic[0] = b'X';
        assert!(Golden::from_bytes(&badmagic).is_err());
        let mut future = bytes.clone();
        future[4] = 99; // version field — digest also breaks, either way it must fail
        assert!(Golden::from_bytes(&future).is_err());
    }

    #[test]
    fn record_roundtrip_full() {
        let logits: Vec<Vec<f32>> = (0..2)
            .map(|p| (0..16).map(|v| v as f32 * p as f32 - 3.0).collect())
            .collect();
        let g = Golden::from_logits_full(
            &Provenance::new("t", "synthetic", [0u8; 32], "test"),
            &tiny_cfg(),
            vec![9],
            logits.clone(),
        );
        let back = Golden::from_bytes(&g.to_bytes()).unwrap();
        match &back.logits {
            Logits::Full(cols) => assert_eq!(*cols, logits),
            _ => panic!("payload kind lost"),
        }
    }

    #[test]
    fn check_passes_on_identical_logits() {
        let k = 8;
        let g = sample_golden(k);
        // Reconstruct full columns consistent with the golden top-k:
        // position p ramps 0.1·(v + 3p) − 1.
        let actual: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                (0..32)
                    .map(|v| (v as f32 + p as f32 * 3.0) * 0.1 - 1.0)
                    .collect()
            })
            .collect();
        let rep = g.check(&actual, &Tolerances::default());
        assert!(rep.is_ok(), "{rep}");
        assert_eq!(rep.positions[0].overlap, k);
    }

    #[test]
    fn check_catches_top1_flip_and_drift() {
        let g = sample_golden(8);
        // Swap the two strongest logits at position 1 → top-1 flip.
        let mut actual: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                (0..32)
                    .map(|v| (v as f32 + p as f32 * 3.0) * 0.1 - 1.0)
                    .collect()
            })
            .collect();
        actual[1].swap(30, 31);
        let rep = g.check(&actual, &Tolerances::default());
        assert!(!rep.is_ok());
        assert!(rep.failures.iter().any(|f| f.contains("top-1")));
        // A large uniform offset drifts values without changing ranks
        // → caught by max_val_delta only.
        let mut drifted: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                (0..32)
                    .map(|v| (v as f32 + p as f32 * 3.0) * 0.1 - 0.4)
                    .collect()
            })
            .collect();
        drifted[0][0] += 1e-9; // keep ordering deterministic
        let rep = g.check(&drifted, &Tolerances::default());
        assert!(!rep.is_ok());
        assert!(rep.failures.iter().any(|f| f.contains("drift")));
        // NaN is its own failure class.
        let mut nan = actual.clone();
        nan[2][5] = f32::NAN;
        let rep = g.check(&nan, &Tolerances::default());
        assert!(!rep.is_ok());
        assert!(rep.failures.iter().any(|f| f.contains("NaN")));
    }

    #[test]
    fn diff_columns_reports() {
        let a = vec![vec![0.0f32, 1.0, 2.0, 3.0]];
        let mut bcol = a[0].clone();
        bcol[3] += 0.5;
        let d = diff_columns(&a, &[bcol]);
        assert_eq!(d.len(), 1);
        assert!((d[0].max_abs - 0.5).abs() < 1e-6);
        assert_eq!(d[0].top1_a, d[0].top1_b);
        assert!(d[0].cosine > 0.99);
    }
}
