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
pub mod safetensors;

pub use builder::{Options, Quant};
pub use config::ModelConfig;

use mil_spec::{encode_model, write_mlpackage_stream, BlobWriter, ModelMeta};
use std::path::Path;

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
            if safetensors::find(&shards, &name).is_none() {
                return Err(ConvertError::Config(format!(
                    "missing tensor {name} in shards"
                )));
            }
        }
        if cfg.qk_norm {
            for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                let name = format!("model.layers.{l}.{suffix}");
                if safetensors::find(&shards, &name).is_none() {
                    return Err(ConvertError::Config(format!(
                        "q/k-norm arch needs tensor {name} — not in shards"
                    )));
                }
            }
        }
    }
    if safetensors::find(&shards, "model.norm.weight").is_none() {
        return Err(ConvertError::Config("missing model.norm.weight".into()));
    }
    if opts.lm_head
        && !cfg.tie_word_embeddings
        && safetensors::find(&shards, "lm_head.weight").is_none()
    {
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
        shards: &shards,
    };
    let mut built = builder::build(&cfg, opts, &mut em)?;
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
