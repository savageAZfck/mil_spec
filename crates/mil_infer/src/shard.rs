//! `shard` — `.milshards` bundle loading and chained prediction.
//!
//! A `milc convert --shard N` bundle is a directory of small
//! `.mlpackage`s — one per N-layer group plus vocab-sliced head
//! packages — tied together by `manifest.json`. The ANE execution-plan
//! builder rejects monolithic programs above roughly 3,200 ops (error
//! -14), which is why the bundle exists: every member stays under the
//! limit, so the whole model runs on the Neural Engine one package at
//! a time. This is the layout Bad Apple's production converter
//! measured on real hardware.
//!
//! [`ShardedModel`] compiles every member package, loads them on the
//! requested [`ComputeUnits`], then threads the hidden state through
//! the layer shards — each keeping its own packed-KV [`State`] — and
//! concatenates the head shards' logits slices into the full
//! `(1, vocab, S, 1)` output.

use crate::{ComputeUnits, Input, Output, Prediction};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::InferError;
type Result<T> = std::result::Result<T, InferError>;

fn err<T>(stage: &'static str, msg: impl Into<String>) -> Result<T> {
    Err(ierr(stage, msg))
}

fn ierr(stage: &'static str, msg: impl Into<String>) -> InferError {
    InferError {
        stage,
        message: msg.into(),
    }
}

/// Parsed `manifest.json` from a `.milshards` bundle. Only the fields
/// the runner needs are typed; the raw JSON stays available.
#[derive(Clone, Debug)]
pub struct ShardManifest {
    /// Raw manifest JSON (model config, quant, provenance summary).
    pub raw: serde_json::Value,
    /// Ordered `(file, layer_lo, layer_hi)` layer shard members.
    pub layer_shards: Vec<(String, usize, usize)>,
    /// Ordered `(file, row_lo, row_hi)` head shard members.
    pub head_shards: Vec<(String, i64, i64)>,
    /// Hidden size (for sanity-checking feed shapes).
    pub hidden_size: i64,
    /// Vocab size — the concatenated head output's channel count.
    pub vocab_size: i64,
    /// Sequence length the bundle was built for.
    pub seq: i64,
    /// `embed: true` bundles take `ids` instead of `x` on shard 0.
    pub embed: bool,
}

impl ShardManifest {
    /// Read `manifest.json` from a bundle directory.
    pub fn load(dir: &Path) -> Result<ShardManifest> {
        let text = std::fs::read_to_string(dir.join("manifest.json"))
            .map_err(|e| ierr("manifest", format!("{}: {e}", dir.display())))?;
        Self::parse(&text)
    }

    /// Parse manifest JSON text.
    pub fn parse(text: &str) -> Result<ShardManifest> {
        let raw: serde_json::Value =
            serde_json::from_str(text).map_err(|e| ierr("manifest", format!("json: {e}")))?;
        if raw["kind"].as_str() != Some("milshards") {
            return err("manifest", "kind is not \"milshards\"");
        }
        let members = |key: &str| -> Result<Vec<(String, i64, i64)>> {
            raw[key]
                .as_array()
                .ok_or_else(|| ierr("manifest", format!("{key}: missing array")))?
                .iter()
                .map(|m| {
                    let file = m["file"]
                        .as_str()
                        .ok_or_else(|| ierr("manifest", format!("{key}: member w/o file")))?
                        .to_string();
                    let rng = m["layers"]
                        .as_array()
                        .or_else(|| m["rows"].as_array())
                        .ok_or_else(|| ierr("manifest", format!("{key}: member w/o range")))?;
                    Ok((
                        file,
                        rng[0].as_i64().unwrap_or(0),
                        rng[1].as_i64().unwrap_or(0),
                    ))
                })
                .collect()
        };
        let layer_shards = members("layer_shards")?
            .into_iter()
            .map(|(f, a, b)| (f, a as usize, b as usize))
            .collect();
        let head_shards = members("head_shards")?;
        let hidden_size = raw["hidden_size"].as_i64().unwrap_or(0);
        let vocab_size = raw["vocab_size"].as_i64().unwrap_or(0);
        let seq = raw["seq"].as_i64().unwrap_or(0);
        let embed = raw["embed"].as_bool().unwrap_or(false);
        Ok(ShardManifest {
            raw,
            layer_shards,
            head_shards,
            hidden_size,
            vocab_size,
            seq,
            embed,
        })
    }
}

/// A loaded `.milshards` bundle: layer models each with their own
/// KV state, then head models run on the final hidden state.
#[cfg(target_os = "macos")]
pub struct ShardedModel {
    /// The parsed manifest.
    pub manifest: ShardManifest,
    layers: Vec<(crate::Model, crate::State)>,
    heads: Vec<crate::Model>,
    /// Directories holding the compiled member models (kept so temp
    /// dirs created by [`ShardedModel::load`] live as long as the run).
    pub compiled_dirs: Vec<PathBuf>,
}

#[cfg(target_os = "macos")]
impl ShardedModel {
    /// Compile every member package of `dir` into `work_dir` and load
    /// them all on `units`. `work_dir` is created if missing.
    pub fn load(dir: &Path, units: ComputeUnits, work_dir: &Path) -> Result<ShardedModel> {
        let manifest = ShardManifest::load(dir)?;
        std::fs::create_dir_all(work_dir)
            .map_err(|e| ierr("compile", format!("{}: {e}", work_dir.display())))?;
        let mut compiled_dirs = Vec::new();
        let mut layers = Vec::new();
        for (file, a, b) in &manifest.layer_shards {
            let pkg = dir.join(file);
            let cdir = work_dir.join(format!("layer_{a:02}-{b:02}"));
            let c = mil_compile::compile(&pkg, &cdir)
                .map_err(|e| ierr("compile", format!("{file}: {e}")))?;
            let m = crate::Model::load(&c.path, units)
                .map_err(|e| ierr("load", format!("{file}: {e}")))?;
            let st = m
                .new_state()
                .map_err(|e| ierr("state", format!("{file}: {e}")))?;
            layers.push((m, st));
            compiled_dirs.push(cdir);
        }
        let mut heads = Vec::new();
        for (file, lo, hi) in &manifest.head_shards {
            let pkg = dir.join(file);
            let cdir = work_dir.join(format!("head_v{lo}-{hi}"));
            let c = mil_compile::compile(&pkg, &cdir)
                .map_err(|e| ierr("compile", format!("{file}: {e}")))?;
            let m = crate::Model::load(&c.path, units)
                .map_err(|e| ierr("load", format!("{file}: {e}")))?;
            heads.push(m);
            compiled_dirs.push(cdir);
        }
        Ok(ShardedModel {
            manifest,
            layers,
            heads,
            compiled_dirs,
        })
    }

    /// Chain the layer shards and (when present) the head shards.
    ///
    /// `inputs` are the layer-shard feeds: `x` `(1,d,S,1)` hidden
    /// states — or `ids` `(1,S)` int32 for an `embed` bundle — plus
    /// `cos`, `sin`, `mask`, `pos` (and `seq` on flexible bundles).
    /// Every layer shard gets the same feeds with `x` replaced by the
    /// previous shard's output; the head shards each get the final `x`
    /// and their logits slices are concatenated in row order.
    ///
    /// Returns one [`Output`]: `logits` `(1, vocab, S, 1)` with head
    /// shards, or the raw hidden state `(1, d, S, 1)` without.
    pub fn predict(&self, inputs: &[Input]) -> Result<Prediction> {
        let t0 = Instant::now();
        let mut xbuf: Vec<u8> = Vec::new();
        let mut xshape: Vec<i64> = Vec::new();
        let mut xcode: i64 = 65552;
        for (i, (m, st)) in self.layers.iter().enumerate() {
            let p = if i == 0 {
                m.predict_with_state(Some(st), inputs)?
            } else {
                let mut feeds: Vec<Input> = inputs
                    .iter()
                    .filter(|f| f.name != "x" && f.name != "ids")
                    .cloned()
                    .collect();
                feeds.insert(
                    0,
                    Input {
                        name: "x",
                        shape: &xshape,
                        data: &xbuf,
                        dtype: mil_spec::DType::Fp16,
                    },
                );
                m.predict_with_state(Some(st), &feeds)?
            };
            let o = &p.outputs[0];
            xbuf = o.data.clone();
            xshape = o.shape.clone();
            xcode = o.dtype_code;
        }
        // With no layer shards the caller's `x` feeds the heads directly.
        if self.layers.is_empty() {
            let x = inputs
                .iter()
                .find(|f| f.name == "x")
                .ok_or_else(|| ierr("predict", "no layer shards and no `x` input"))?;
            xbuf = x.data.to_vec();
            xshape = x.shape.to_vec();
        }
        let out = if self.heads.is_empty() {
            Output {
                name: "x".into(),
                shape: xshape,
                dtype_code: xcode,
                data: xbuf,
            }
        } else {
            let feed = [Input {
                name: "x",
                shape: &xshape,
                data: &xbuf,
                dtype: mil_spec::DType::Fp16,
            }];
            let mut logits = Vec::new();
            let mut code = 65552;
            for (i, h) in self.heads.iter().enumerate() {
                let p = h.predict(&feed)?;
                let o = &p.outputs[0];
                code = o.dtype_code;
                if i == 0 {
                    logits.reserve(o.data.len() * self.heads.len());
                }
                logits.extend_from_slice(&o.data);
            }
            let s = if xshape.len() >= 3 { xshape[2] } else { 1 };
            Output {
                name: "logits".into(),
                shape: vec![1, self.manifest.vocab_size, s, 1],
                dtype_code: code,
                data: logits,
            }
        };
        Ok(Prediction {
            outputs: vec![out],
            latency: t0.elapsed(),
        })
    }
}

/// Non-macOS stub.
#[cfg(not(target_os = "macos"))]
pub struct ShardedModel;
#[cfg(not(target_os = "macos"))]
impl ShardedModel {
    /// Always fails off-macOS.
    pub fn load(_dir: &Path, _units: ComputeUnits, _work: &Path) -> Result<ShardedModel> {
        err("load", "mil_infer requires macOS")
    }
    /// Always fails off-macOS.
    pub fn predict(&self, _inputs: &[Input]) -> Result<Prediction> {
        err("predict", "mil_infer requires macOS")
    }
}
