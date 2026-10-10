//! `lora` — MLX/PEFT-style LoRA adapter loading and weight fusion.
//!
//! An adapter is a directory holding `adapter_config.json` plus a
//! safetensors or `.npz` file of low-rank pairs, or a bare
//! `*.safetensors` / `*.npz` file. For every base weight the adapter
//! carries `<base>.lora_a` and `<base>.lora_b` and the fused weight is
//!
//! ```text
//! W_fused = W + scale · (B @ A)
//! ```
//!
//! written out with the base tensor's own row orientation:
//!
//! - **MLX** (`mlx_lm.tuner.lora`): `a` is `[in, r]`, `b` is `[r, out]`,
//!   base `W` is `[out, in]`, so `delta[o][i] = Σ_r b[r][o]·a[i][r]`.
//! - **PEFT** (`lora_A`/`lora_B` names or transposed shapes): `a` is
//!   `[r, in]`, `b` is `[out, r]`, `delta[o][i] = Σ_r b[o][r]·a[r][i]`.
//!
//! Orientation is resolved per-pair from the shapes; both spell the
//! same `scale · (B @ A)` against the `[out, in]` base.
//!
//! Fusion happens inside [`FusedSource`] — a [`WeightSource`] wrapper —
//! in f32, *before* `tensor_f16` hands bytes to the fp16/int8 emitters,
//! so quantization sees already-fused weights.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::npz;
use crate::safetensors::{self, Safetensors};
use crate::WeightSource;

fn err(kind: io::ErrorKind, msg: impl Into<String>) -> io::Error {
    io::Error::new(kind, msg.into())
}
fn bad(msg: impl Into<String>) -> io::Error {
    err(io::ErrorKind::InvalidData, msg)
}

/// Base-name suffixes the converter can actually fuse into — anything
/// else in an adapter is a clear error rather than a silent drop.
const SUPPORTED_TARGETS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "lm_head",
];

/// One `(a, b)` low-rank pair, f32, as stored in the adapter file.
#[derive(Clone, Debug)]
pub struct LoraPair {
    /// `lora_a` elements, row-major.
    pub a: Vec<f32>,
    /// `lora_a` shape `[rows, cols]`.
    pub a_shape: [i64; 2],
    /// `lora_b` elements, row-major.
    pub b: Vec<f32>,
    /// `lora_b` shape `[rows, cols]`.
    pub b_shape: [i64; 2],
    /// Adapter base name (e.g. `model.layers.0.self_attn.q_proj`) for
    /// error messages.
    pub base: String,
}

/// Which way the pair's matrices lie relative to `[out, in]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// MLX: `a` `[in, r]`, `b` `[r, out]` → delta = (bᵀ @ aᵀ).
    Mlx,
    /// PEFT: `a` `[r, in]`, `b` `[out, r]` → delta = b @ a.
    Peft,
}

impl LoraPair {
    /// Resolve orientation against a base `W` of `[out, in]`.
    fn layout(&self, out: i64, in_: i64) -> io::Result<(Layout, i64)> {
        let [a0, a1] = self.a_shape;
        let [b0, b1] = self.b_shape;
        if a0 == in_ && b1 == out && a1 == b0 && a1 > 0 {
            Ok((Layout::Mlx, a1))
        } else if a1 == in_ && b0 == out && a0 == b1 && a0 > 0 {
            Ok((Layout::Peft, a0))
        } else {
            Err(bad(format!(
                "lora {}: a{a0}x{a1} b{b0}x{b1} cannot fuse into W {out}x{in_}",
                self.base
            )))
        }
    }
}

/// A loaded adapter: scale + every fused-target pair keyed by base
/// weight name (`…q_proj.weight`).
pub struct Lora {
    /// Multiplier applied to `B @ A`.
    pub scale: f32,
    /// weight name → pair
    pairs: BTreeMap<String, LoraPair>,
    /// where the adapter came from (for errors/logging)
    path: PathBuf,
}

/// Uniform access over whichever container the adapter ships in.
enum AdapterFile {
    St(Safetensors),
    Npz(npz::Npz),
}
impl AdapterFile {
    fn names(&self) -> Vec<String> {
        match self {
            AdapterFile::St(s) => s.names().iter().map(|s| s.to_string()).collect(),
            AdapterFile::Npz(n) => n.names().iter().map(|s| s.to_string()).collect(),
        }
    }
    fn tensor(&self, name: &str) -> io::Result<(Vec<i64>, Vec<f32>)> {
        match self {
            AdapterFile::St(s) => s.tensor_f32(name),
            AdapterFile::Npz(n) => n
                .tensor(name)
                .map(|t| (t.shape.clone(), t.data.clone()))
                .ok_or_else(|| err(io::ErrorKind::NotFound, format!("npz: no member {name}"))),
        }
    }
}

/// Read the LoRA scale from `adapter_config.json` text.
///
/// - MLX: `lora_parameters.scale` (a direct multiplier).
/// - MLX alt: `lora_parameters.alpha / lora_parameters.rank`.
/// - PEFT: top-level `lora_alpha / r`.
fn scale_from_config(text: &str) -> io::Result<f32> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| bad(format!("adapter_config.json: {e}")))?;
    let f = |v: &serde_json::Value| v.as_f64().map(|x| x as f32);
    let lp = &v["lora_parameters"];
    if let Some(s) = f(&lp["scale"]) {
        return Ok(s);
    }
    let alpha = f(&lp["alpha"])
        .or_else(|| f(&v["lora_alpha"]))
        .or_else(|| f(&v["alpha"]));
    let rank = f(&lp["rank"]).or_else(|| f(&v["r"]));
    match (alpha, rank) {
        (Some(a), Some(r)) if r > 0.0 => Ok(a / r),
        _ => Err(bad(
            "adapter_config.json: no lora scale (need lora_parameters.scale or alpha+rank)",
        )),
    }
}

/// Adapter config next to `weights`: `dir/adapter_config.json`, or the
/// sibling of a bare file. `None` (→ scale 1.0) only when the caller
/// pointed straight at a weight file with no config beside it.
fn load_scale(weights: &Path) -> io::Result<f32> {
    let cfg = if weights.is_dir() {
        weights.join("adapter_config.json")
    } else {
        weights
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("adapter_config.json")
    };
    if cfg.exists() {
        scale_from_config(&std::fs::read_to_string(&cfg)?)
    } else if weights.is_dir() {
        Err(err(
            io::ErrorKind::NotFound,
            format!("lora: {} has no adapter_config.json", weights.display()),
        ))
    } else {
        Ok(1.0)
    }
}

impl Lora {
    /// Load an adapter directory (`adapter_config.json` +
    /// `adapters.safetensors` / `*.npz`) or a bare
    /// `*.safetensors` / `*.npz` file.
    pub fn load(path: &Path) -> io::Result<Lora> {
        let weights = if path.is_dir() {
            let st = path.join("adapters.safetensors");
            if st.exists() {
                st
            } else {
                // any *.safetensors or *.npz in the dir
                let mut cand: Vec<PathBuf> = std::fs::read_dir(path)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        matches!(
                            p.extension().and_then(|x| x.to_str()),
                            Some("safetensors") | Some("npz")
                        )
                    })
                    .collect();
                cand.sort();
                cand.into_iter().next().ok_or_else(|| {
                    err(
                        io::ErrorKind::NotFound,
                        format!(
                            "lora: no adapters.safetensors or *.npz under {}",
                            path.display()
                        ),
                    )
                })?
            }
        } else {
            path.to_path_buf()
        };
        let scale = load_scale(&weights)?;
        let file = match weights.extension().and_then(|x| x.to_str()) {
            Some("safetensors") => AdapterFile::St(safetensors::open(&weights)?),
            Some("npz") => AdapterFile::Npz(npz::open(&weights)?),
            _ => {
                return Err(bad(format!(
                    "lora: {} is not .safetensors or .npz",
                    weights.display()
                )))
            }
        };

        // Pair up every <base>.lora_a / <base>.lora_b (also accept the
        // PEFT lora_A / lora_B spellings).
        let mut a_names: BTreeMap<String, String> = BTreeMap::new();
        let mut b_names: BTreeMap<String, String> = BTreeMap::new();
        for n in file.names() {
            for sfx in [".lora_a", ".lora_A"] {
                if let Some(base) = n.strip_suffix(sfx) {
                    a_names.insert(base.to_string(), n.clone());
                }
            }
            for sfx in [".lora_b", ".lora_B"] {
                if let Some(base) = n.strip_suffix(sfx) {
                    b_names.insert(base.to_string(), n.clone());
                }
            }
        }
        let mut pairs = BTreeMap::new();
        for (base, an) in &a_names {
            let bn = b_names
                .get(base)
                .ok_or_else(|| bad(format!("lora {base}: lora_a present but lora_b missing")))?;
            if !SUPPORTED_TARGETS.iter().any(|s| base.ends_with(s)) {
                return Err(bad(format!(
                    "lora {base}: unsupported target (need one of {SUPPORTED_TARGETS:?})"
                )));
            }
            let (a_shape, a) = file.tensor(an)?;
            let (b_shape, b) = file.tensor(bn)?;
            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(bad(format!(
                    "lora {base}: lora_a/lora_b must be 2D, got {a_shape:?}/{b_shape:?}"
                )));
            }
            let weight = format!("{base}.weight");
            pairs.insert(
                weight,
                LoraPair {
                    a,
                    a_shape: [a_shape[0], a_shape[1]],
                    b,
                    b_shape: [b_shape[0], b_shape[1]],
                    base: base.clone(),
                },
            );
        }
        for base in b_names.keys() {
            if !a_names.contains_key(base) {
                return Err(bad(format!(
                    "lora {base}: lora_b present but lora_a missing"
                )));
            }
        }
        if pairs.is_empty() {
            return Err(bad(format!(
                "lora: {} contains no lora_a/lora_b pairs",
                weights.display()
            )));
        }
        Ok(Lora {
            scale,
            pairs,
            path: weights,
        })
    }

    /// Base weight names this adapter fuses into.
    pub fn targets(&self) -> Vec<&str> {
        self.pairs.keys().map(|s| s.as_str()).collect()
    }

    /// The pair for a base weight name, if this adapter touches it.
    pub fn pair(&self, weight_name: &str) -> Option<&LoraPair> {
        self.pairs.get(weight_name)
    }

    /// Pre-flight check against a checkpoint: every target weight must
    /// exist and the a/b geometry must fuse into its `[out, in]` shape.
    /// Run this *before* any graph emission so a bad adapter fails
    /// before `weight.bin` is opened.
    pub fn validate(&self, src: &dyn WeightSource) -> io::Result<()> {
        for (wname, pair) in &self.pairs {
            if !src.has(wname) {
                return Err(err(
                    io::ErrorKind::NotFound,
                    format!(
                        "lora {}: checkpoint has no weight {wname} (adapter {})",
                        pair.base,
                        self.path.display()
                    ),
                ));
            }
            let shape = src.shape(wname)?;
            if shape.len() != 2 {
                return Err(bad(format!(
                    "lora {}: weight {wname} must be 2D, got {shape:?}",
                    pair.base
                )));
            }
            pair.layout(shape[0], shape[1])?;
        }
        Ok(())
    }

    /// Fused `W + scale·(B @ A)` for `wname` in f32 — the single place
    /// the orientation math lives.
    fn fuse_f32(&self, src: &dyn WeightSource, wname: &str) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let pair = self.pairs.get(wname).expect("fuse on non-target");
        let (shape, mut w) = src.tensor_f32(wname)?;
        if shape.len() != 2 {
            return Err(bad(format!("lora {}: {wname} not 2D", pair.base)));
        }
        let (out, in_) = (shape[0] as usize, shape[1] as usize);
        let (layout, r) = pair.layout(shape[0], shape[1])?;
        let r = r as usize;
        let s = self.scale;
        match layout {
            // a [in, r], b [r, out]: delta[o][i] = Σ_k b[k][o]·a[i][k]
            Layout::Mlx => {
                let [_, _] = pair.a_shape;
                for i in 0..in_ {
                    for o in 0..out {
                        let mut acc = 0f32;
                        for k in 0..r {
                            acc += pair.b[k * out + o] * pair.a[i * r + k];
                        }
                        w[o * in_ + i] += s * acc;
                    }
                }
            }
            // a [r, in], b [out, r]: delta[o][i] = Σ_k b[o][k]·a[k][i]
            Layout::Peft => {
                for o in 0..out {
                    for i in 0..in_ {
                        let mut acc = 0f32;
                        for k in 0..r {
                            acc += pair.b[o * r + k] * pair.a[k * in_ + i];
                        }
                        w[o * in_ + i] += s * acc;
                    }
                }
            }
        }
        Ok((shape, w))
    }
}

/// A [`WeightSource`] that fuses the adapter's pairs on the way out.
/// Untouched names delegate straight through; fused names are computed
/// in f32 then emitted as f16 bytes — so `Quant::Int8` in the builder
/// quantizes the *fused* values.
pub struct FusedSource<'a> {
    /// the underlying checkpoint
    inner: &'a dyn WeightSource,
    /// the adapter to fuse
    lora: &'a Lora,
}

impl<'a> FusedSource<'a> {
    /// Wrap `inner`; call [`Lora::validate`] first to fail fast.
    pub fn new(inner: &'a dyn WeightSource, lora: &'a Lora) -> FusedSource<'a> {
        FusedSource { inner, lora }
    }
}

impl WeightSource for FusedSource<'_> {
    fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }
    fn shape(&self, name: &str) -> io::Result<Vec<i64>> {
        self.inner.shape(name)
    }
    fn tensor_f16(&self, name: &str) -> io::Result<(Vec<i64>, Vec<u8>)> {
        if self.lora.pair(name).is_none() {
            return self.inner.tensor_f16(name);
        }
        let (shape, w) = self.lora.fuse_f32(self.inner, name)?;
        let mut bytes = Vec::with_capacity(w.len() * 2);
        for v in w {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        Ok((shape, bytes))
    }
    fn tensor_f32(&self, name: &str) -> io::Result<(Vec<i64>, Vec<f32>)> {
        if self.lora.pair(name).is_none() {
            return self.inner.tensor_f32(name);
        }
        self.lora.fuse_f32(self.inner, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory WeightSource for fusion math tests.
    struct Stub {
        t: HashMap<String, (Vec<i64>, Vec<f32>)>,
    }
    impl Stub {
        fn new() -> Stub {
            Stub { t: HashMap::new() }
        }
        fn add(&mut self, name: &str, shape: Vec<i64>, data: Vec<f32>) {
            self.t.insert(name.to_string(), (shape, data));
        }
    }
    impl WeightSource for Stub {
        fn has(&self, name: &str) -> bool {
            self.t.contains_key(name)
        }
        fn tensor_f16(&self, name: &str) -> io::Result<(Vec<i64>, Vec<u8>)> {
            let (s, d) = self
                .t
                .get(name)
                .ok_or_else(|| err(io::ErrorKind::NotFound, format!("missing {name}")))?;
            Ok((
                s.clone(),
                d.iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect(),
            ))
        }
        fn tensor_f32(&self, name: &str) -> io::Result<(Vec<i64>, Vec<f32>)> {
            self.t
                .get(name)
                .map(|(s, d)| (s.clone(), d.clone()))
                .ok_or_else(|| err(io::ErrorKind::NotFound, format!("missing {name}")))
        }
        fn shape(&self, name: &str) -> io::Result<Vec<i64>> {
            self.t
                .get(name)
                .map(|(s, _)| s.clone())
                .ok_or_else(|| err(io::ErrorKind::NotFound, format!("missing {name}")))
        }
    }

    /// Assemble a Lora directly (bypasses file loading) for math tests.
    fn mk(
        scale: f32,
        base: &str,
        a_shape: [i64; 2],
        a: Vec<f32>,
        b_shape: [i64; 2],
        b: Vec<f32>,
    ) -> Lora {
        let mut pairs = BTreeMap::new();
        pairs.insert(
            format!("{base}.weight"),
            LoraPair {
                a,
                a_shape,
                b,
                b_shape,
                base: base.to_string(),
            },
        );
        Lora {
            scale,
            pairs,
            path: PathBuf::from("<test>"),
        }
    }

    #[test]
    fn fuse_mlx_layout() {
        // W [out=2, in=3]; a [in=3, r=2], b [r=2, out=2]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = vec![0.7, 0.8, 0.9, 1.0];
        let l = mk(
            10.0,
            "model.layers.0.self_attn.q_proj",
            [3, 2],
            a.clone(),
            [2, 2],
            b.clone(),
        );
        let mut stub = Stub::new();
        stub.add(
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 3],
            w.clone(),
        );
        l.validate(&stub).unwrap();
        let fs = FusedSource::new(&stub, &l);
        let (shape, fused) = fs
            .tensor_f32("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(shape, vec![2, 3]);
        // delta[o][i] = 10 * Σ_k b[k][o]*a[i][k]
        for o in 0..2 {
            for i in 0..3 {
                let expect = w[o * 3 + i] + 10.0 * (b[o] * a[i * 2] + b[2 + o] * a[i * 2 + 1]);
                assert!((fused[o * 3 + i] - expect).abs() < 1e-6, "o{o} i{i}");
            }
        }
        // f16 path carries the same values within fp16 rounding
        let (_, bytes) = fs
            .tensor_f16("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        for (i, c) in bytes.chunks_exact(2).enumerate() {
            let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            assert!((v - fused[i]).abs() < 1e-3 + fused[i].abs() * 1e-3);
        }
    }

    #[test]
    fn fuse_peft_layout() {
        // W [out=2, in=3]; a [r=2, in=3], b [out=2, r=2]
        let w = vec![1.0; 6];
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = vec![2.0, 0.0, 0.0, 3.0];
        let l = mk(1.0, "model.layers.0.mlp.down_proj", [2, 3], a, [2, 2], b);
        let mut stub = Stub::new();
        stub.add("model.layers.0.mlp.down_proj.weight", vec![2, 3], w);
        l.validate(&stub).unwrap();
        let fs = FusedSource::new(&stub, &l);
        let (_, fused) = fs
            .tensor_f32("model.layers.0.mlp.down_proj.weight")
            .unwrap();
        // row0 += 2*[1,0,0], row1 += 3*[0,1,0]
        assert_eq!(fused, vec![3.0, 1.0, 1.0, 1.0, 4.0, 1.0]);
    }

    #[test]
    fn validate_missing_target() {
        let l = mk(
            1.0,
            "model.layers.9.self_attn.v_proj",
            [2, 2],
            vec![0.0; 4],
            [2, 2],
            vec![0.0; 4],
        );
        let stub = Stub::new();
        let e = l.validate(&stub).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
        assert!(e.to_string().contains("v_proj.weight"), "{e}");
    }

    #[test]
    fn validate_geometry_mismatch() {
        // a/b inner dims don't reconcile with W [4,4]
        let l = mk(
            1.0,
            "model.layers.0.self_attn.k_proj",
            [4, 8],
            vec![0.0; 32],
            [3, 4],
            vec![0.0; 12],
        );
        let mut stub = Stub::new();
        stub.add(
            "model.layers.0.self_attn.k_proj.weight",
            vec![4, 4],
            vec![0.0; 16],
        );
        let e = l.validate(&stub).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("cannot fuse"), "{e}");
    }

    #[test]
    fn untouched_delegates() {
        let l = mk(
            1.0,
            "model.layers.0.self_attn.q_proj",
            [2, 2],
            vec![0.0; 4],
            [2, 2],
            vec![0.0; 4],
        );
        let mut stub = Stub::new();
        stub.add(
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![1.0; 4],
        );
        stub.add("model.norm.weight", vec![4], vec![2.0; 4]);
        l.validate(&stub).unwrap();
        let fs = FusedSource::new(&stub, &l);
        let (s, d) = fs.tensor_f32("model.norm.weight").unwrap();
        assert_eq!(s, vec![4]);
        assert_eq!(d, vec![2.0; 4]);
    }

    #[test]
    fn scale_parsing() {
        assert_eq!(
            scale_from_config(r#"{"lora_parameters":{"rank":8,"scale":10}}"#).unwrap(),
            10.0
        );
        assert_eq!(
            scale_from_config(r#"{"lora_parameters":{"rank":8,"alpha":16}}"#).unwrap(),
            2.0
        );
        assert_eq!(
            scale_from_config(r#"{"r":8,"lora_alpha":32}"#).unwrap(),
            4.0
        );
        assert!(scale_from_config(r#"{"lora_parameters":{"rank":8}}"#).is_err());
    }
}
