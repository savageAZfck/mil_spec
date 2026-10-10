//! Signed-by-construction provenance.
//!
//! Every `mil_convert` package carries a deterministic provenance block
//! in the spec's `description.metadata.userDefined` map (CoreML's real
//! key/value metadata field — no fake schema). Keys are namespaced
//! `mil.prov.*`; values are plain strings so `milc inspect` can show
//! them verbatim.
//!
//! ```text
//! mil.prov.v          format version ("1")
//! mil.prov.tool       toolchain + version
//! mil.prov.time       unix seconds (non-deterministic on purpose —
//!                     it records when conversion happened)
//! mil.prov.config     sha256 of the canonical config JSON
//! mil.prov.options    sha256 of the canonical options string
//! mil.prov.weights    sha256 of weight.bin
//! mil.prov.src.count  number of source weight files
//! mil.prov.src.{i}    "name size sha256" per file, sorted by name
//! ```
//!
//! [`verify`] recomputes `mil.prov.weights` over the packaged
//! weight.bin (tamper detection) and — with `--source` — the per-file
//! source hashes. Missing provenance is reported, not an error.
//! Updatable-layer intent is `mil.updatable` (comma-separated layer
//! names) — a metadata marker, see module docs in `mil_convert`.

use mil_spec::proto::{self, PMut};
use mil_spec::sha256;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// `model.description.metadata.userDefined` map (model field 2 →
/// description field 100 → metadata field 16 map entries).
fn user_defined(model: &PMut) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Some(desc) = model.msg(2) {
        if let Some(meta) = desc.msg(100) {
            for e in meta.msgs(16) {
                if let (Some(k), Some(v)) = (e.str(1), e.str(2)) {
                    out.insert(k, v);
                }
            }
        }
    }
    out
}

/// One source-weight-file fingerprint, deterministically ordered.
#[derive(Debug)]
pub struct SourceFp {
    /// File name only (path-independent so `--source dir` can differ).
    pub name: String,
    /// Byte size.
    pub size: u64,
    /// sha256 hex.
    pub hash: String,
}

/// Deterministic metadata entries to embed at convert time.
///
/// `weight_hash` is the sha256 hex of the packaged `weight.bin` — the
/// caller streams it during the write so a multi-GB model isn't
/// double-read.
pub fn provenance_map(
    cfg: &crate::ModelConfig,
    opts: &crate::Options,
    source_files: &[PathBuf],
    weight_hash: &str,
) -> R<BTreeMap<String, String>> {
    let mut m = BTreeMap::new();
    m.insert("mil.prov.v".into(), "1".into());
    m.insert(
        "mil.prov.tool".into(),
        format!("mil_convert {}", env!("CARGO_PKG_VERSION")),
    );
    m.insert(
        "mil.prov.time".into(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string(),
    );
    // Canonical config string — fixed field order, no JSON whitespace
    // sensitivity.
    let cfg_canon = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        cfg.model_type,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings
    );
    m.insert(
        "mil.prov.config".into(),
        sha256::sha256_hex(cfg_canon.as_bytes()),
    );
    let opts_canon = format!(
        "{}|{}|{:?}|{}|{}|{}|{}|{:?}|{}|{:?}|{:?}",
        opts.seq,
        opts.max_kv,
        opts.quant,
        opts.lm_head,
        opts.embed,
        opts.spec_version,
        opts.opset,
        opts.lora,
        opts.quant_policy.is_some(),
        opts.plan.is_some(),
        opts.seq_lens
    );
    m.insert(
        "mil.prov.options".into(),
        sha256::sha256_hex(opts_canon.as_bytes()),
    );
    m.insert("mil.prov.weights".into(), weight_hash.to_string());
    let mut fps = Vec::new();
    for p in source_files {
        let hash = sha256::sha256_file(p).map_err(|e| format!("{}: {e}", p.display()))?;
        fps.push(SourceFp {
            name: p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
            size: fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            hash,
        });
    }
    fps.sort_by(|a, b| a.name.cmp(&b.name));
    m.insert("mil.prov.src.count".into(), fps.len().to_string());
    for (i, fp) in fps.iter().enumerate() {
        m.insert(
            format!("mil.prov.src.{i}"),
            format!("{} {} {}", fp.name, fp.size, fp.hash),
        );
    }
    if !opts.updatable.is_empty() {
        m.insert("mil.updatable".into(), opts.updatable.join(","));
    }
    Ok(m)
}

/// What [`verify`] found.
#[derive(Debug)]
pub struct AttestReport {
    /// Every check, one line each.
    pub lines: Vec<String>,
    /// Any hard failure (tampered weights / source mismatch).
    pub ok: bool,
    /// Provenance was present at all.
    pub has_provenance: bool,
}

type R<T> = Result<T, String>;

/// Read the spec's userDefined metadata from an .mlpackage.
pub fn read_user_defined(pkg: &Path) -> R<BTreeMap<String, String>> {
    let spec = pkg.join("Data/com.apple.CoreML/model.mlmodel");
    let bytes = fs::read(&spec).map_err(|e| format!("{}: {e}", spec.display()))?;
    let model = proto::decode(&bytes).ok_or("spec decode failed")?;
    Ok(user_defined(&model))
}

/// Verify a package's provenance. `source` is a directory holding the
/// original weight files; fingerprints match on file *name* so the dir
/// needn't be the original checkout.
pub fn verify(pkg: &Path, source: Option<&Path>) -> R<AttestReport> {
    let ud = read_user_defined(pkg)?;
    let mut lines = Vec::new();
    let mut ok = true;
    let has = ud.contains_key("mil.prov.v");
    if !has {
        lines.push("no provenance block (package may predate --prov or be foreign)".into());
        return Ok(AttestReport {
            lines,
            ok: true,
            has_provenance: false,
        });
    }
    for k in ud.keys() {
        if k.starts_with("mil.prov.") || k.starts_with("mil.updatable") {
            lines.push(format!("{k} = {}", ud[k]));
        }
    }
    // weight.bin must match its recorded hash.
    let wpath = pkg.join("Data/com.apple.CoreML/weights/weight.bin");
    let want_w = ud.get("mil.prov.weights").cloned();
    match (want_w, sha256::sha256_file(&wpath)) {
        (Some(want), Ok(got)) => {
            if got == want {
                lines.push(format!("weight.bin sha256 OK ({want})"));
            } else {
                lines.push(format!("TAMPERED weight.bin: spec {want} != actual {got}"));
                ok = false;
            }
        }
        (Some(_), Err(e)) => {
            lines.push(format!("weight.bin unreadable: {e}"));
            ok = false;
        }
        (None, _) => {
            lines.push("no recorded weights hash — cannot verify blob".into());
        }
    }
    if let Some(dir) = source {
        let n: usize = ud
            .get("mil.prov.src.count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut checked = 0usize;
        for i in 0..n {
            let Some(entry) = ud.get(&format!("mil.prov.src.{i}")) else {
                continue;
            };
            let mut it = entry.split(' ');
            let (name, size, hash) = match (it.next(), it.next(), it.next()) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => continue,
            };
            let p = dir.join(name);
            match sha256::sha256_file(&p).and_then(|h| fs::metadata(&p).map(|m| (h, m.len()))) {
                Ok((got, got_size)) => {
                    let size_ok = size.parse::<u64>().map(|s| s == got_size).unwrap_or(false);
                    if got == hash && size_ok {
                        lines.push(format!("src {name} sha256 OK"));
                        checked += 1;
                    } else {
                        lines.push(format!(
                            "MISMATCH src {name}: spec {hash} ({size}B) != actual {got} ({got_size}B)"
                        ));
                        ok = false;
                    }
                }
                Err(e) => {
                    lines.push(format!("src {name}: {e}"));
                    ok = false;
                }
            }
        }
        lines.push(format!("{checked}/{n} source file(s) verified"));
    }
    Ok(AttestReport {
        lines,
        ok,
        has_provenance: has,
    })
}
