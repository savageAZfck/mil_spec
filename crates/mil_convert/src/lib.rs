//! `mil_convert` — safetensors → `.mlpackage` in pure Rust.
//!
//! This is the `coremltools.convert` replacement: load a HuggingFace
//! checkpoint, generate a fat single-graph decoder MIL program, and
//! emit a `.mlpackage` `coremlc` accepts — no Python anywhere in the
//! path.
//!
//! # Pipeline
//!
//! 1. `config.json` → [`ModelConfig`] (Qwen2/Qwen3/Llama-class decoders)
//! 2. `*.safetensors` shards → [`safetensors::Safetensors`] index
//! 3. [`builder::build`] emits the whole decoder as one [`mil_spec::Block`]
//!    — packed-KV state, `slice_update` cache writes, GQA attention,
//!    per-channel int8 or fp16 conv weights
//! 4. [`mil_passes::optimize`] folds the helper-emitted consts (~30%
//!    smaller spec)
//! 5. `encode_model` + `write_mlpackage_stream` → the package
//!
//! The result is a *decode-step* graph: one call advances the KV cache
//! by `seq` tokens and returns logits. That's the shape a drafter needs —
//! the ANE proposes tokens while the GPU verifies.
//!
//! # Usage
//!
//! ```no_run
//! use mil_convert::{convert, builder::Options};
//! let report = convert(
//!     std::path::Path::new("Qwen3-0.6B"),
//!     std::path::Path::new("drafter.mlpackage"),
//!     &Options::default(),
//! ).unwrap();
//! println!("{} ops, {:.1}% ANE", report.op_count, report.ane_pct);
//! ```

#![forbid(unsafe_code)]

pub mod builder;
pub mod config;
pub mod gguf;
pub mod lora;
pub mod npz;
pub mod safetensors;

pub use builder::{Options, Quant};
pub use config::ModelConfig;
pub use gguf::Gguf;
pub use lora::{Lora, LoraPair};

use mil_spec::{encode_model, write_mlpackage_stream, BlobWriter, ModelMeta};
use std::path::Path;

/// Any tensor store the builder can pull weights from — safetensors
/// shards or a GGUF file. Lookups are by HF canonical name
/// (`model.layers.0.self_attn.q_proj.weight`); `tensor_f16` returns the
/// tensor's logical shape and its contents as f16 little-endian bytes.
pub trait WeightSource {
    /// Whether `name` exists.
    fn has(&self, name: &str) -> bool;
    /// `(shape, f16 LE bytes)` for `name`.
    fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)>;
    /// `(shape, f32 elements)` for `name`. Default decodes the f16
    /// form; stores that keep f32 natively should override.
    fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
        let (shape, bytes) = self.tensor_f16(name)?;
        Ok((
            shape,
            bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        ))
    }
    /// Logical shape for `name` without reading tensor bytes.
    fn shape(&self, name: &str) -> std::io::Result<Vec<i64>> {
        Ok(self.tensor_f16(name)?.0)
    }
}

impl WeightSource for Vec<safetensors::Safetensors> {
    fn has(&self, name: &str) -> bool {
        safetensors::find(self, name).is_some()
    }
    fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
        let st = safetensors::find(self, name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {name}"),
            )
        })?;
        st.tensor_f16(name)
    }
    fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
        let st = safetensors::find(self, name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {name}"),
            )
        })?;
        st.tensor_f32(name)
    }
    fn shape(&self, name: &str) -> std::io::Result<Vec<i64>> {
        let st = safetensors::find(self, name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {name}"),
            )
        })?;
        Ok(st.info(name).map(|i| i.shape.clone()).unwrap_or_default())
    }
}

/// What a conversion produced.
#[derive(Clone, Debug)]
pub struct ConvertReport {
    /// Ops in the emitted spec (after optimization).
    pub op_count: usize,
    /// Bytes of weight.bin.
    pub weight_bytes: u64,
    /// Package directory written.
    pub package: std::path::PathBuf,
    /// `mil_lint` ANE fraction (0–100) if `mil_lint` is linked — always
    /// reported so conversion failures show placement, not just shape.
    pub ane_pct: f64,
    /// `tokenizer.json` written next to the package (GGUF sources
    /// only), when the embedded tokenizer is a supported kind.
    pub tokenizer_json: Option<std::path::PathBuf>,
    /// Why no tokenizer.json was written — the named error from
    /// [`gguf::tokenizer`], if any.
    pub tokenizer_error: Option<String>,
}

/// Conversion error.
#[derive(Debug)]
pub enum ConvertError {
    /// Filesystem problem.
    Io(std::io::Error),
    /// config.json missing/bad.
    Config(String),
    /// Architecture not supported by the builder.
    Unsupported(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::Io(e) => write!(f, "io: {e}"),
            ConvertError::Config(m) => write!(f, "config: {m}"),
            ConvertError::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}
impl std::error::Error for ConvertError {}
impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        ConvertError::Io(e)
    }
}

type Result<T> = std::result::Result<T, ConvertError>;

/// Convert an HF model directory into a `.mlpackage`.
///
/// `model_dir` needs `config.json` plus one or more `*.safetensors`.
/// The package's `weight.bin` is streamed, so a model larger than RAM
/// still converts.
pub fn convert(model_dir: &Path, out_pkg: &Path, opts: &Options) -> Result<ConvertReport> {
    // ---- config ----
    let cfg_bytes = std::fs::read(model_dir.join("config.json"))
        .map_err(|e| ConvertError::Config(format!("cannot read config.json: {e}")))?;
    let cfg = ModelConfig::from_json(&cfg_bytes).map_err(ConvertError::Config)?;
    if !cfg.supported() {
        return Err(ConvertError::Unsupported(format!(
            "model_type {} — the builder handles qwen2/qwen3/llama/mistral",
            cfg.model_type
        )));
    }
    if cfg.num_kv_heads == 0 || cfg.num_heads % cfg.num_kv_heads != 0 {
        return Err(ConvertError::Config(format!(
            "num_heads {} not divisible by num_kv_heads {}",
            cfg.num_heads, cfg.num_kv_heads
        )));
    }

    // ---- weight shards ----
    let shards = safetensors::open_dir(model_dir)?;
    convert_impl(&cfg, &shards, out_pkg, opts)
}

/// Convert a GGUF file (single or split) into a `.mlpackage`.
///
/// Weights dequantize through [`gguf::dequant`] — bit-exact with ggml's
/// `to_float` — then flow through the same builder path as safetensors.
pub fn convert_gguf(path: &Path, out_pkg: &Path, opts: &Options) -> Result<ConvertReport> {
    let g = Gguf::open(path).map_err(|e| ConvertError::Config(format!("{e}")))?;
    let cfg = gguf::config::model_config(&g).map_err(ConvertError::Config)?;
    if !cfg.supported() {
        return Err(ConvertError::Unsupported(format!(
            "arch {} — the builder handles qwen2/qwen3/llama/mistral",
            cfg.model_type
        )));
    }
    if cfg.num_kv_heads == 0 || cfg.num_heads % cfg.num_kv_heads != 0 {
        return Err(ConvertError::Config(format!(
            "num_heads {} not divisible by num_kv_heads {}",
            cfg.num_heads, cfg.num_kv_heads
        )));
    }
    let mut report = convert_impl(&cfg, &g, out_pkg, opts)?;

    // Export the embedded tokenizer next to the package. An
    // unsupported tokenizer kind must not fail the weight conversion —
    // it lands in the report instead.
    let dir = out_pkg.parent().unwrap_or(Path::new("."));
    match gguf::tokenizer::to_tokenizer_json(&g)
        .and_then(|j| {
            let p = dir.join("tokenizer.json");
            std::fs::write(&p, j)
                .map(|_| p)
                .map_err(gguf::GgufError::Io)
        })
        .and_then(|p| {
            gguf::tokenizer::to_tokenizer_config_json(&g).and_then(|c| {
                std::fs::write(dir.join("tokenizer_config.json"), c)
                    .map(|_| p)
                    .map_err(gguf::GgufError::Io)
            })
        }) {
        Ok(p) => report.tokenizer_json = Some(p),
        Err(e) => report.tokenizer_error = Some(format!("{e}")),
    }
    Ok(report)
}

/// Shared build path once config + weights exist.
fn convert_impl(
    cfg: &ModelConfig,
    src: &dyn WeightSource,
    out_pkg: &Path,
    opts: &Options,
) -> Result<ConvertReport> {
    // Optional LoRA bake-in. Validate every adapter target against the
    // checkpoint before `weight.bin` is opened, then wrap the source so
    // emitters stream fused values through the normal fp16/int8 path.
    let lora = match &opts.lora {
        Some(p) => {
            Some(lora::Lora::load(p).map_err(|e| ConvertError::Config(format!("lora: {e}")))?)
        }
        None => None,
    };
    let fused_src;
    let src: &dyn WeightSource = match &lora {
        Some(l) => {
            l.validate(src).map_err(ConvertError::Io)?;
            fused_src = lora::FusedSource::new(src, l);
            &fused_src
        }
        None => src,
    };
    for l in 0..cfg.num_layers {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let name = format!("model.layers.{l}.{suffix}");
            if !src.has(&name) {
                return Err(ConvertError::Config(format!(
                    "missing tensor {name} in weights"
                )));
            }
        }
        if cfg.qk_norm {
            for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                let name = format!("model.layers.{l}.{suffix}");
                if !src.has(&name) {
                    return Err(ConvertError::Config(format!(
                        "q/k-norm arch needs tensor {name} — not in weights"
                    )));
                }
            }
        }
        if cfg.model_type == "qwen2" {
            // Qwen2 attention projections carry biases — without them the
            // converted graph silently diverges from the checkpoint.
            for suffix in [
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
            ] {
                let name = format!("model.layers.{l}.{suffix}");
                if !src.has(&name) {
                    return Err(ConvertError::Config(format!(
                        "qwen2 needs bias tensor {name} — not in weights"
                    )));
                }
            }
        }
    }
    if !src.has("model.norm.weight") {
        return Err(ConvertError::Config("missing model.norm.weight".into()));
    }
    if opts.lm_head && !cfg.tie_word_embeddings && !src.has("lm_head.weight") {
        return Err(ConvertError::Config("missing lm_head.weight".into()));
    }

    // ---- build ----
    std::fs::create_dir_all(out_pkg.parent().unwrap_or(Path::new(".")))?;
    let tmp_w = out_pkg.with_extension("weight.bin.tmp");
    let mut writer = BlobWriter::create(&tmp_w)?;
    let mut em = builder::WeightEmitter {
        w: &mut writer,
        quant: opts.quant,
        file: "@model_path/weights/weight.bin".into(),
        src,
    };
    let mut built = builder::build(cfg, opts, &mut em)?;
    writer.finish()?;

    // ---- optimize + emit ----
    let _ = mil_passes::optimize(&mut built.block);
    let meta = ModelMeta::new(opts.spec_version, &opts.opset)
        .creator("mil_convert")
        .description(&format!(
            "{} converted by mil_convert ({} layers, d={}, seq={})",
            cfg.model_type, cfg.num_layers, cfg.hidden_size, opts.seq
        ));
    let spec = encode_model(
        &built.inputs,
        &built.outputs,
        &built.states,
        &built.block,
        &built.fn_inputs,
        &meta,
    );
    write_mlpackage_stream(out_pkg, &spec, Some(&tmp_w))?;
    let weight_bytes = std::fs::metadata(tmp_w.clone())
        .map(|m| m.len())
        .unwrap_or(0);
    let _ = std::fs::remove_file(&tmp_w);

    Ok(ConvertReport {
        op_count: built.block.ops.len(),
        weight_bytes,
        package: out_pkg.to_path_buf(),
        ane_pct: 0.0, // filled by callers that link mil_lint
        tokenizer_json: None,
        tokenizer_error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal `.safetensors` file: header + f16 data.
    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
        let mut header = String::from("{");
        let mut data = Vec::new();
        let mut entries = Vec::new();
        for (name, shape, vals) in tensors {
            let off = data.len() as u64;
            for v in vals {
                data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
            }
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{shape_str}],\"data_offsets\":[{off},{}]}}",
                data.len()
            ));
        }
        header.push_str(&entries.join(","));
        header.push('}');
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&data).unwrap();
    }

    /// A tiny qwen3 config + matching weights.
    fn tiny_qwen3(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = r#"{
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": false
        }"#;
        std::fs::write(dir.join("config.json"), cfg).unwrap();

        let mut tensors: Vec<(String, Vec<i64>, Vec<f32>)> = Vec::new();
        let mut push = |name: &str, shape: Vec<i64>, fill: f32| {
            let n: i64 = shape.iter().product();
            tensors.push((
                name.to_string(),
                shape,
                (0..n).map(|i| fill + (i as f32 % 7.0) * 0.01).collect(),
            ));
        };
        for l in 0..2 {
            push(
                &format!("model.layers.{l}.input_layernorm.weight"),
                vec![16],
                1.0,
            );
            push(
                &format!("model.layers.{l}.post_attention_layernorm.weight"),
                vec![16],
                1.0,
            );
            push(
                &format!("model.layers.{l}.self_attn.q_proj.weight"),
                vec![16, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.self_attn.k_proj.weight"),
                vec![8, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.self_attn.v_proj.weight"),
                vec![8, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.self_attn.o_proj.weight"),
                vec![16, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.self_attn.q_norm.weight"),
                vec![8],
                1.0,
            );
            push(
                &format!("model.layers.{l}.self_attn.k_norm.weight"),
                vec![8],
                1.0,
            );
            push(
                &format!("model.layers.{l}.mlp.gate_proj.weight"),
                vec![32, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.mlp.up_proj.weight"),
                vec![32, 16],
                0.02,
            );
            push(
                &format!("model.layers.{l}.mlp.down_proj.weight"),
                vec![16, 32],
                0.02,
            );
        }
        push("model.norm.weight", vec![16], 1.0);
        push("lm_head.weight", vec![64, 16], 0.02);
        let refs: Vec<(&str, Vec<i64>, Vec<f32>)> = tensors
            .iter()
            .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &refs);
    }

    #[test]
    fn parses_safetensors() {
        let dir = std::env::temp_dir().join(format!("st_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(
            &dir.join("t.safetensors"),
            &[("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])],
        );
        let st = safetensors::open(&dir.join("t.safetensors")).unwrap();
        let (shape, bytes) = st.tensor_f16("w").unwrap();
        assert_eq!(shape, vec![2, 2]);
        let vals: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_qwen3_config() {
        let cfg = ModelConfig::from_json(
            br#"{"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":28,
                "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
                "intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-6,
                "rope_theta":1000000,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 28);
        assert!(cfg.qk_norm);
        assert!(cfg.supported());
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn converts_tiny_qwen3() {
        let root = std::env::temp_dir().join(format!("mc_tiny_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let model = root.join("model");
        tiny_qwen3(&model);
        let pkg = root.join("tiny.mlpackage");
        let opts = Options {
            seq: 1,
            max_kv: 16,
            quant: Quant::Int8,
            lm_head: true,
            embed: false,
            spec_version: 10,
            opset: "CoreML9".into(),
            lora: None,
        };
        let r = convert(&model, &pkg, &opts).unwrap();
        assert!(r.op_count > 0);
        assert!(pkg.join("Data/com.apple.CoreML/model.mlmodel").exists());
        assert!(pkg
            .join("Data/com.apple.CoreML/weights/weight.bin")
            .exists());
        assert!(r.weight_bytes > 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn int8_quantization_preserves_scales() {
        // 2x4 matrix — check per-row absmax scaling
        let vals = [1.0f32, 2.0, -1.0, -2.0, 0.5, -0.5, 0.25, -0.25];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let (q, s) = super::builder::quantize_int8(&bytes, 2, 4);
        // row 0 absmax 2.0 → scale 2/127; row 1 absmax 0.5 → scale 0.5/127
        let s0 = half::f16::from_le_bytes([s[0], s[1]]).to_f32();
        let s1 = half::f16::from_le_bytes([s[2], s[3]]).to_f32();
        assert!((s0 - 2.0 / 127.0).abs() < 1e-3);
        assert!((s1 - 0.5 / 127.0).abs() < 1e-3);
        assert_eq!(q[1] as i8, 127); // 2.0/scale → 127
        assert_eq!(q[4] as i8, 127); // 0.5/scale → 127
    }
}
