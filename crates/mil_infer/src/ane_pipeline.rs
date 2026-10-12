//! `ane_pipeline` — chained multi-shard decode on the ANE via e5rt.
//!
//! A `milshards`-style directory of `.mlmodelc` shards becomes one
//! encoded stream: N layer ops chained through two ping-pong `hidden`
//! buffers (layer i writes buf[i%2], layer i+1 reads it — no host copy
//! between layers), plus vocab head ops reading the final hidden.
//! Same-named live inputs (rope cos/sin, masks) share one buffer across
//! every op; each op's `state<>` inouts get resident per-op buffers.
//!
//! One `execute_sync` per step runs the whole model — the host only
//! embeds the token, writes the masks, and reads back logits.
//!
//! Port layout discovered from the ops themselves — nothing about the
//! shapes or names is hardcoded beyond `x`/`hidden`/`logits`.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use crate::e5rt::{self, Api, DeviceMask, E5Error, PortKind};

/// A compiled shard directory: `layer_*.mlmodelc` bodies and
/// `head_*.mlmodelc` vocab slices, chained into one stream.
pub struct Pipeline {
    api: Api,
    stream: *mut c_void,
    /// Ops in encode order — kept alive for the stream's lifetime.
    #[allow(dead_code)]
    ops: Vec<*mut c_void>,
    /// The two alternating hidden buffers.
    hidden: [*mut c_void; 2],
    hidden_elems: u64,
    /// Live inputs shared across every body op: (name, buf, elems).
    shared_in: Vec<(String, *mut c_void, u64)>,
    /// Per-op state buffers: (op_idx, name, buf, elems).
    states: Vec<(usize, String, *mut c_void, u64)>,
    /// Head logits buffers: (name, buf, elems).
    logits: Vec<(String, *mut c_void, u64)>,
}

/// Discover `layer_*`/`head_*` shards in a converted directory —
/// the `.mlmodelc` dirs holding `model.mil` + `weights/`, in name order.
pub fn shard_paths(dir: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>), E5Error> {
    let mut layers = Vec::new();
    let mut heads = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if p.extension().and_then(|s| s.to_str()) != Some("mlmodelc") {
            continue;
        }
        let mil = p.join("model.mil");
        if !mil.exists() {
            continue;
        }
        if name.starts_with("head_") {
            heads.push(mil);
        } else if name.starts_with("layer_") {
            layers.push(mil);
        }
    }
    layers.sort();
    heads.sort();
    if layers.is_empty() {
        return Err(E5Error::Call("no layer_*.mlmodelc shards found", -1));
    }
    Ok((layers, heads))
}

impl Pipeline {
    /// Compile every shard and wire the chain.
    ///
    /// `body` = layer shards in order; `heads` = vocab-slice ops that
    /// read the final hidden. Each body op gets per-op state buffers;
    /// every body op's input named the same as another's (masks, rope)
    /// is bound to one shared buffer.
    pub fn open(body: &[PathBuf], heads: &[PathBuf], mask: DeviceMask) -> Result<Self, E5Error> {
        let api = Api::open()?;
        let stream = api.stream()?;
        let mut ops = Vec::new();
        let mut shared_in: Vec<(String, *mut c_void, u64)> = Vec::new();
        let mut states = Vec::new();
        let mut hidden: [*mut c_void; 2] = [std::ptr::null_mut(); 2];
        let mut hidden_elems = 0u64;

        for (i, mil) in body.iter().enumerate() {
            let (op, _fns) = e5rt::Program::make_op_on(&api, mil, mask)?;
            ops.push(op);

            // hidden chain: op i reads x = hidden[(i+1)%2],
            // writes hidden = hidden[i%2] — op i+1 then reads hidden[i%2].
            for name in api.port_names(op, PortKind::Out) {
                let port = api.port(op, PortKind::Out, &name)?;
                let elems = api.port_elems(port);
                if hidden[0].is_null() {
                    hidden = [api.buf(elems * 2)?, api.buf(elems * 2)?];
                    hidden_elems = elems;
                }
                api.bind(port, hidden[i % 2])?;
            }
            for name in api.port_names(op, PortKind::In) {
                let port = api.port(op, PortKind::In, &name)?;
                let elems = api.port_elems(port);
                if name == "x" {
                    api.bind(port, hidden[(i + 1) % 2])?;
                    continue;
                }
                // same-named live input → shared buffer across ops
                match shared_in.iter().find(|(n, _, _)| n == &name) {
                    Some((_, buf, _)) => api.bind(port, *buf)?,
                    None => {
                        let buf = api.buf(elems * 2)?;
                        shared_in.push((name.clone(), buf, elems));
                        api.bind(port, buf)?;
                    }
                }
            }
            for name in api.port_names(op, PortKind::InOut) {
                let port = api.port(op, PortKind::InOut, &name)?;
                let elems = api.port_elems(port);
                let buf = api.buf(elems * 2)?;
                states.push((i, name, buf, elems));
                api.bind(port, buf)?;
            }
        }

        // Heads: each reads the last body op's hidden =
        // hidden[(n-1)%2] (layer n-1 wrote buf[(n-1)%2]).
        let mut logits = Vec::new();
        for mil in heads {
            let (op, _fns) = e5rt::Program::make_op_on(&api, mil, mask)?;
            ops.push(op);
            for name in api.port_names(op, PortKind::In) {
                let port = api.port(op, PortKind::In, &name)?;
                api.bind(port, hidden[(body.len() - 1) % 2])?;
            }
            for name in api.port_names(op, PortKind::Out) {
                let port = api.port(op, PortKind::Out, &name)?;
                let elems = api.port_elems(port);
                let buf = api.buf(elems * 2)?;
                logits.push((name.clone(), buf, elems));
                api.bind(port, buf)?;
            }
        }

        for op in &ops {
            api.encode(stream, *op)?;
        }
        Ok(Pipeline {
            api,
            stream,
            ops,
            hidden,
            hidden_elems,
            shared_in,
            states,
            logits,
        })
    }

    /// Open a shard directory (layers + heads discovered by name).
    pub fn open_dir(dir: &Path, mask: DeviceMask) -> Result<Self, E5Error> {
        let (layers, heads) = shard_paths(dir)?;
        Self::open(&layers, &heads, mask)
    }

    /// fp16 ptr to the embedding write slot (the buffer layer 0's `x`
    /// reads — layer 0 binds `x` to `hidden[1]`). Write the embedded
    /// token here before `step`.
    pub fn embedding_slot(&self) -> Result<(*mut u16, u64), E5Error> {
        Ok((self.api.ptr(self.hidden[1])?, self.hidden_elems))
    }

    /// Element count of a shared live input, if bound.
    pub fn shared_elems(&self, name: &str) -> Option<usize> {
        self.shared_in
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, e)| *e as usize)
    }

    /// Write a shared live input (mask, rope) — broadcast to all ops.
    pub fn set_shared(&self, name: &str, data: &[u16]) -> Result<(), E5Error> {
        let (_, buf, elems) = self
            .shared_in
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or(E5Error::Call("set_shared: no such input", -1))?;
        if data.len() as u64 > *elems {
            return Err(E5Error::Call("set_shared: overflow", -1));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.api.ptr(*buf)?, data.len()) };
        Ok(())
    }

    /// Zero all resident state buffers.
    pub fn reset_state(&self) -> Result<(), E5Error> {
        for (_, _, buf, elems) in &self.states {
            unsafe { std::ptr::write_bytes(self.api.ptr(*buf)?, 0, *elems as usize) };
        }
        Ok(())
    }

    /// One full decode step on the engine — every op, one submission.
    pub fn step(&self) -> Result<(), E5Error> {
        self.api.exec(self.stream)
    }

    /// Concatenated logits across all head ops (fp16 bits).
    pub fn logits(&self) -> Vec<u16> {
        let mut out = Vec::new();
        for (_, buf, elems) in &self.logits {
            if let Ok(p) = self.api.ptr(*buf) {
                out.extend(unsafe { std::slice::from_raw_parts(p, *elems as usize) });
            }
        }
        out
    }

    /// The last body op's `hidden` output (fp16 bits) — diagnostics.
    pub fn last_hidden(&self, n_layers: usize) -> Option<Vec<u16>> {
        if self.hidden[0].is_null() || n_layers == 0 {
            return None;
        }
        let buf = self.hidden[(n_layers - 1) % 2];
        let p = self.api.ptr(buf).ok()?;
        Some(unsafe { std::slice::from_raw_parts(p, self.hidden_elems as usize).to_vec() })
    }

    /// A resident state buffer's contents (fp16 bits) — diagnostics.
    pub fn state_f16(&self, op_idx: usize, name: &str) -> Option<Vec<u16>> {
        let (_, _, buf, elems) = self
            .states
            .iter()
            .find(|(i, n, _, _)| *i == op_idx && n == name)?;
        let p = self.api.ptr(*buf).ok()?;
        Some(unsafe { std::slice::from_raw_parts(p, *elems as usize).to_vec() })
    }

    /// State buffer names per op (diagnostics).
    pub fn state_names(&self) -> Vec<(usize, String)> {
        self.states
            .iter()
            .map(|(i, n, _, _)| (*i, n.clone()))
            .collect()
    }
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}
