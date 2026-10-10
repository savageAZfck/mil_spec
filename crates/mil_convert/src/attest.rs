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
//! mil.prov.weights    sha256 of weight.bin ("none" if there is no blob)
//! mil.prov.program    sha256 of the canonical spec (see below)
//! mil.prov.src.count  number of source weight files
//! mil.prov.src.{i}    "name size sha256" per file, sorted by name
//! ```
//!
//! # What is covered
//!
//! - **`weight.bin`** — hashed byte for byte (`mil.prov.weights`).
//! - **The spec** — `mil.prov.program` is the sha256 of the *canonical
//!   spec bytes*: `model.mlmodel` decoded with [`mil_spec::proto`],
//!   every `userDefined` map entry (`Model.description.metadata`
//!   field 16) removed, and re-encoded. So the whole op graph, inline
//!   constants, feature descriptions (shapes, flexibility), and
//!   `shortDescription`/`creator` are covered; the `userDefined` map —
//!   which holds the provenance itself, and `mil.updatable` — is not.
//!   This is what protects packages with no `weight.bin`, whose weights
//!   live inside the spec as immediates.
//! - **Sources** (with `--source`) — per-file fingerprints prove which
//!   input a package *claims* to come from. They do not by themselves
//!   prove the package was built from it; the program and weights
//!   hashes pin what the package contains.
//!
//! Not covered: `Manifest.json`, and any file in the package other than
//! `model.mlmodel` and `weights/weight.bin`.
//!
//! Packages that don't derive from a transformer config (ONNX via
//! [`provenance_map_files`]) carry the same keys **without**
//! `mil.prov.config`; [`verify`] never requires it. When such a package
//! has no `weight.bin`, `mil.prov.weights` is [`NO_WEIGHTS`] (`"none"`)
//! and the check asserts the blob stays absent.
//! `--source` accepts a directory or a single file (a file stands for
//! its parent directory; entries match on name).
//!
//! [`verify`] recomputes `mil.prov.weights` and `mil.prov.program`
//! (mismatch → `TAMPERED`, failure) and — with `--source` — the
//! per-file source hashes. Provenance written before `mil.prov.program`
//! existed is reported as `spec not covered` and is not a failure;
//! missing provenance is reported, not an error. Package surgery
//! refreshes both hashes. Updatable-layer intent is `mil.updatable`
//! (comma-separated layer names) — a metadata marker, see module docs in
//! `mil_convert`.

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

/// Key holding the canonical-spec hash.
pub const PROGRAM_KEY: &str = "mil.prov.program";

/// Canonical spec bytes: `spec` decoded, every `userDefined` entry
/// (`Model.description(2).metadata(100)` field 16) removed, re-encoded.
fn canonical_spec(model: &PMut) -> Vec<u8> {
    let mut m = model.clone();
    if let Some(mut desc) = m.msg(2) {
        if let Some(mut meta) = desc.msg(100) {
            meta.remove_all(16);
            desc.set_msg(100, &meta);
            m.set_msg(2, &desc);
        }
    }
    proto::encode(&m)
}

/// `mil.prov.program` value for encoded spec bytes (`model.mlmodel`).
pub fn program_hash(spec: &[u8]) -> R<String> {
    let model = proto::decode(spec).ok_or("spec decode failed")?;
    Ok(sha256::sha256_hex(&canonical_spec(&model)))
}

/// Encode a spec with `mil.prov.program` embedded. `encode` produces
/// the spec bytes for a given [`mil_spec::ModelMeta`]; it is called once
/// without the key (to hash) and once with it. Because the canonical
/// form drops the whole `userDefined` map, adding the key does not
/// change the hash.
pub fn encode_with_program_hash(
    meta: mil_spec::ModelMeta,
    encode: impl Fn(&mil_spec::ModelMeta) -> R<Vec<u8>>,
) -> R<Vec<u8>> {
    let h = program_hash(&encode(&meta)?)?;
    encode(&meta.user_meta(PROGRAM_KEY, &h))
}

/// Refresh an existing `mil.prov.program` entry of `model` to the hash
/// of its current canonical form. A spec without that key (no
/// provenance, or provenance that predates it) is left untouched.
pub(crate) fn refresh_program_hash(model: &mut PMut) {
    let h = sha256::sha256_hex(&canonical_spec(model));
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
        if e.str(1).as_deref() == Some(PROGRAM_KEY) {
            e.set_str(2, &h);
            f.val = proto::PVal::Len(proto::encode(&e));
            touched = true;
        }
    }
    if touched {
        desc.set_msg(100, &meta);
        model.set_msg(2, &desc);
    }
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

/// Marker recorded in `mil.prov.weights` when the package has no
/// `weight.bin` (every weight inlined as an immediate). [`verify`]
/// then requires the blob to stay absent: a `weight.bin` appearing
/// later is reported as tampering.
pub const NO_WEIGHTS: &str = "none";

fn base_provenance(tool: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("mil.prov.v".into(), "1".into());
    m.insert("mil.prov.tool".into(), tool.to_string());
    m.insert(
        "mil.prov.time".into(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string(),
    );
    m
}

fn add_sources(m: &mut BTreeMap<String, String>, source_files: &[PathBuf]) -> R<()> {
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
    Ok(())
}

/// Provenance for packages that don't come from a transformer config
/// (e.g. `mil_onnx`): the same `mil.prov.*` keys as [`provenance_map`]
/// minus `mil.prov.config`.
///
/// `tool` is the full tool string (`"mil_onnx 0.1.0"`),
/// `options_desc` any canonical description of the conversion options
/// (hashed into `mil.prov.options`), and `weight_hash` the sha256 hex
/// of the packaged `weight.bin` — or `None` when the package has no
/// blob, recorded as [`NO_WEIGHTS`].
pub fn provenance_map_files(
    tool: &str,
    options_desc: &str,
    source_files: &[PathBuf],
    weight_hash: Option<&str>,
) -> R<BTreeMap<String, String>> {
    let mut m = base_provenance(tool);
    m.insert(
        "mil.prov.options".into(),
        sha256::sha256_hex(options_desc.as_bytes()),
    );
    m.insert(
        "mil.prov.weights".into(),
        weight_hash.unwrap_or(NO_WEIGHTS).to_string(),
    );
    add_sources(&mut m, source_files)?;
    Ok(m)
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
    let mut m = base_provenance(&format!("mil_convert {}", env!("CARGO_PKG_VERSION")));
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
    let mut opts_canon = format!(
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
    // appended only when set so hashes of pre-existing option sets
    // stay stable
    if let Some((lo, hi)) = opts.seq_range {
        opts_canon.push_str(&format!("|seq_range={lo}..{hi}"));
    }
    m.insert(
        "mil.prov.options".into(),
        sha256::sha256_hex(opts_canon.as_bytes()),
    );
    m.insert("mil.prov.weights".into(), weight_hash.to_string());
    add_sources(&mut m, source_files)?;
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
        (Some(want), got) if want == NO_WEIGHTS => {
            if wpath.exists() {
                lines.push(format!(
                    "UNEXPECTED weight.bin: spec records no weight blob{}",
                    got.map(|h| format!(" (blob sha256 {h})"))
                        .unwrap_or_default()
                ));
                ok = false;
            } else {
                lines.push("no weight.bin, as recorded (all weights inlined)".into());
            }
        }
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
    // The spec must match its canonical hash (op graph, inline consts,
    // feature descriptions — everything but userDefined).
    match ud.get(PROGRAM_KEY) {
        Some(want) => {
            let spec = pkg.join("Data/com.apple.CoreML/model.mlmodel");
            match fs::read(&spec)
                .map_err(|e| e.to_string())
                .and_then(|b| program_hash(&b))
            {
                Ok(got) if got == *want => {
                    lines.push(format!("program (model.mlmodel) sha256 OK ({want})"));
                }
                Ok(got) => {
                    lines.push(format!("TAMPERED program: spec {want} != actual {got}"));
                    ok = false;
                }
                Err(e) => {
                    lines.push(format!("TAMPERED program: spec unreadable: {e}"));
                    ok = false;
                }
            }
        }
        None => {
            lines.push("spec not covered (provenance predates mil.prov.program)".into());
        }
    }
    if let Some(src) = source {
        // A single file stands for its parent directory: entries still
        // match on file name, so only that file's fingerprint can be
        // verified and any others are reported as missing.
        let dir = if src.is_file() {
            src.parent().unwrap_or(Path::new("."))
        } else {
            src
        };
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
