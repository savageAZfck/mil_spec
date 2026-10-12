//! `ane_decode` — real token decode loop over a shard directory,
//! driving `ane_pipeline::Pipeline` with the same semantics Bad Apple's
//! Swift bridge uses on the CoreML path:
//!
//! - `x` = fp16 embedding row `token` sliced out of `embedding.f16`
//! - `rope_cos/sin[i] = cos/sin(position · inv_freq[i])`,
//!   `inv_freq[i] = rope_base^(-i / rope_half)`
//! - `attn_mask` additive: `-10000` masked, `0` open — the causal row
//!   opens one key per step across every broadcast row
//! - `kv_write_mask` one-hot at `position`
//! - argmax over the concatenated head logits (heads ordered by
//!   `vocab_start`, so the flat index IS the token id)
//!
//! One `Pipeline::step` per token — prefill and decode share the path.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::ane_pipeline::Pipeline;
use crate::e5rt::{DeviceMask, E5Error};

const MASKED: f32 = -10_000.0;

/// The `conversion_manifest.json` fields decode needs.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f64,
    pub embedding: PathBuf,
    /// Layer shard MILs in order.
    pub layers: Vec<PathBuf>,
    /// Head shard MILs ordered by vocab_start — flat logit index
    /// equals token id.
    pub heads: Vec<PathBuf>,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self, E5Error> {
        let text = std::fs::read_to_string(dir.join("conversion_manifest.json"))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| E5Error::Parse(e.to_string()))?;
        let model = &v["model"];
        let num = |k: &str| -> Result<usize, E5Error> {
            model[k]
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| E5Error::Parse(format!("manifest missing model.{k}")))
        };
        let mil_of =
            |dir: &Path, compiled: &str| dir.join(compiled.trim_end_matches('/')).join("model.mil");
        let mut layers: Vec<PathBuf> = v["shards"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|s| s["compiled_path"].as_str())
            .map(|c| mil_of(dir, c))
            .collect();
        layers.sort();
        let mut head_specs: Vec<(u64, PathBuf)> = v["shared"]["lm_head_shards"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|h| {
                Some((
                    h["vocab_start"].as_u64()?,
                    mil_of(dir, h["compiled_path"].as_str()?),
                ))
            })
            .collect();
        head_specs.sort_by_key(|(start, _)| *start);
        let embedding = dir.join(
            v["shared"]["embedding"]["path"]
                .as_str()
                .unwrap_or("embedding.f16"),
        );
        for p in layers.iter().chain(head_specs.iter().map(|(_, p)| p)) {
            if !p.exists() {
                return Err(E5Error::Parse(format!(
                    "manifest references missing shard {}",
                    p.display()
                )));
            }
        }
        Ok(Manifest {
            hidden_size: num("hidden_size")?,
            vocab_size: num("vocab_size")?,
            seq_len: num("seq_len")?,
            rope_dim: num("rope_dim")?,
            rope_freq_base: model["rope_freq_base"].as_f64().unwrap_or(1e6),
            embedding,
            layers,
            heads: head_specs.into_iter().map(|(_, p)| p).collect(),
        })
    }
}

/// Greedy decoder over a compiled shard directory.
pub struct Decoder {
    pipe: Pipeline,
    manifest: Manifest,
    embedding: File,
    inv_freq: Vec<f64>,
    position: usize,
    /// Persistent additive mask — rows share the causal pattern; one
    /// column opens per step and stays open.
    attn: Vec<u16>,
    attn_rows: usize,
    /// One-hot KV write pointer.
    kv_mask: Vec<u16>,
}

impl Decoder {
    /// Compile the whole shard set and zero the cache horizon.
    pub fn open_dir(dir: &Path, mask: DeviceMask) -> Result<Self, E5Error> {
        let manifest = Manifest::load(dir)?;
        let pipe = Pipeline::open(&manifest.layers, &manifest.heads, mask)?;
        let embedding = File::open(&manifest.embedding)?;
        let half = manifest.rope_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| manifest.rope_freq_base.powf(-(i as f64) / half as f64))
            .collect();
        let attn_rows =
            pipe.shared_elems("attn_mask").unwrap_or(manifest.seq_len) / manifest.seq_len;
        let mut d = Decoder {
            pipe,
            manifest,
            embedding,
            inv_freq,
            position: 0,
            attn: Vec::new(),
            attn_rows,
            kv_mask: Vec::new(),
        };
        d.attn = vec![half::f16::from_f32(MASKED).to_bits(); attn_rows * d.manifest.seq_len];
        d.kv_mask = vec![0u16; d.manifest.seq_len];
        d.reset()?;
        Ok(d)
    }

    /// Fresh horizon — zero states, re-mask everything.
    pub fn reset(&mut self) -> Result<(), E5Error> {
        self.position = 0;
        self.pipe.reset_state()?;
        for v in &mut self.attn {
            *v = half::f16::from_f32(MASKED).to_bits();
        }
        for v in &mut self.kv_mask {
            *v = 0;
        }
        self.pipe.set_shared("attn_mask", &self.attn)?;
        self.pipe.set_shared("kv_write_mask", &self.kv_mask)?;
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Bytes per fp16 hidden row (`hidden_size * 2`) — mesh seam sizing.
    pub fn hidden_bytes(&self) -> usize {
        self.manifest.hidden_size * 2
    }

    /// Feed one token through all layers. With `project`, also reads the
    /// head logits and returns the argmax token.
    pub fn push(&mut self, token: u32, project: bool) -> Result<Option<u32>, E5Error> {
        if token as usize >= self.manifest.vocab_size {
            return Err(E5Error::Call("decoder: token out of vocab", -1));
        }
        let mut row = vec![0u8; self.manifest.hidden_size * 2];
        self.embedding.read_exact_at(
            &mut row,
            token as u64 * self.manifest.hidden_size as u64 * 2,
        )?;
        self.push_row(&row, project)
    }

    /// Feed a raw fp16 hidden row (`[hidden]` bytes) instead of an
    /// embedding lookup — the mesh-brain tail-rank seam.
    pub fn push_hidden(&mut self, fp16_row: &[u8], project: bool) -> Result<Option<u32>, E5Error> {
        if fp16_row.len() != self.manifest.hidden_size * 2 {
            return Err(E5Error::Call("decoder: hidden row size mismatch", -1));
        }
        self.push_row(fp16_row, project)
    }

    /// Rewind the KV frontier to `target` — masks reopen for overwrite;
    /// stale KV at re-masked positions is invisible to attention.
    pub fn rewind(&mut self, target: usize) -> Result<(), E5Error> {
        if target > self.position {
            return Err(E5Error::Call("decoder: rewind forward", -1));
        }
        let masked = half::f16::from_f32(MASKED).to_bits();
        for row in 0..self.attn_rows {
            for p in target..self.position {
                self.attn[row * self.manifest.seq_len + p] = masked;
            }
        }
        if self.position > 0 {
            self.kv_mask[self.position - 1] = 0;
        }
        if target > 0 {
            self.kv_mask[target - 1] = half::f16::from_f32(1.0).to_bits();
        }
        self.position = target;
        self.pipe.set_shared("attn_mask", &self.attn)?;
        self.pipe.set_shared("kv_write_mask", &self.kv_mask)?;
        Ok(())
    }

    fn push_row(&mut self, row: &[u8], project: bool) -> Result<Option<u32>, E5Error> {
        let p = self.position;
        if p >= self.manifest.seq_len {
            return Err(E5Error::Call("decoder: context horizon exhausted", -1));
        }
        // x = hidden row
        let (slot, n) = self.pipe.embedding_slot()?;
        if n as usize != self.manifest.hidden_size {
            return Err(E5Error::Call("decoder: hidden/embedding mismatch", -1));
        }
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), slot as *mut u8, row.len()) };

        // rope tables at position p
        let cos: Vec<u16> = self
            .inv_freq
            .iter()
            .map(|f| half::f16::from_f64((p as f64 * f).cos()).to_bits())
            .collect();
        let sin: Vec<u16> = self
            .inv_freq
            .iter()
            .map(|f| half::f16::from_f64((p as f64 * f).sin()).to_bits())
            .collect();
        self.pipe.set_shared("rope_cos", &cos)?;
        self.pipe.set_shared("rope_sin", &sin)?;

        // causal open at p in every broadcast row; one-hot write
        let zero = 0u16;
        for row in 0..self.attn_rows {
            self.attn[row * self.manifest.seq_len + p] = zero;
        }
        if p > 0 {
            self.kv_mask[p - 1] = 0;
        }
        self.kv_mask[p] = half::f16::from_f32(1.0).to_bits();
        self.pipe.set_shared("attn_mask", &self.attn)?;
        self.pipe.set_shared("kv_write_mask", &self.kv_mask)?;

        self.pipe.step()?;
        self.position += 1;

        if !project {
            return Ok(None);
        }
        let logits = self.pipe.logits();
        if logits.is_empty() {
            return Err(E5Error::Call("decoder: no head logits bound", -1));
        }
        let (best, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                half::f16::from_bits(**a)
                    .to_f32()
                    .total_cmp(&half::f16::from_bits(**b).to_f32())
            })
            .ok_or(E5Error::Call("decoder: empty logits", -1))?;
        Ok(Some(best as u32))
    }

    /// Prefill `prompt` then greedily extend `max_new` tokens. Returns
    /// the generated ids (not echoing the prompt).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>, E5Error> {
        if prompt.is_empty() {
            return Err(E5Error::Call("decoder: empty prompt", -1));
        }
        for &t in &prompt[..prompt.len() - 1] {
            self.push(t, false)?;
        }
        let mut next = self.push(prompt[prompt.len() - 1], true)?;
        let mut out = Vec::new();
        for _ in 0..max_new {
            let Some(t) = next else { break };
            out.push(t);
            next = self.push(t, true)?;
        }
        Ok(out)
    }
}
