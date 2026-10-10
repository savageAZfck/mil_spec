//! GGUF metadata → [`ModelConfig`].
//!
//! Keys follow llama.cpp's `{arch}.<key>` convention (gguf-py
//! `constants.py` `KEY_*`). Anything the builder needs and the file
//! doesn't state is an error — never a guess.

use super::{Gguf, Value};
use crate::config::ModelConfig;

fn need<'a>(g: &'a Gguf, key: &str) -> Result<&'a Value, String> {
    g.meta(key)
        .ok_or_else(|| format!("GGUF missing required metadata key `{key}`"))
}

fn i64_of(g: &Gguf, key: &str) -> Result<i64, String> {
    need(g, key)?
        .as_i64()
        .ok_or_else(|| format!("GGUF metadata `{key}` is not an integer"))
}

/// Scalar-or-first-element for array-valued head counts (hybrid
/// architectures store one value per layer; the supported dense decoders
/// use a scalar — a mixed array can't be right).
fn head_count(g: &Gguf, key: &str) -> Result<i64, String> {
    match need(g, key)? {
        Value::Arr(_, v) => v
            .first()
            .and_then(Value::as_i64)
            .ok_or_else(|| format!("GGUF `{key}` is an empty array")),
        v => v
            .as_i64()
            .ok_or_else(|| format!("GGUF metadata `{key}` is not an integer")),
    }
}

/// Build a [`ModelConfig`] from a GGUF's metadata + tensor index.
///
/// `arch` is the `general.architecture` string. Supported model types:
/// qwen2, qwen3, llama (mistral GGUFs also report `llama`), mistral.
pub fn model_config(g: &Gguf) -> Result<ModelConfig, String> {
    let arch = g
        .meta("general.architecture")
        .and_then(Value::as_str)
        .ok_or_else(|| "GGUF missing `general.architecture`".to_string())?
        .to_string();
    let k = |suffix: &str| format!("{arch}.{suffix}");

    let num_layers = i64_of(g, &k("block_count"))? as usize;
    let hidden_size = i64_of(g, &k("embedding_length"))?;
    let intermediate_size = i64_of(g, &k("feed_forward_length"))?;
    let num_heads = head_count(g, &k("attention.head_count"))?;
    let num_kv_heads = g
        .meta(&k("attention.head_count_kv"))
        .and_then(|v| match v {
            Value::Arr(_, a) => a.first().and_then(Value::as_i64),
            v => v.as_i64(),
        })
        .unwrap_or(num_heads);
    let head_dim = g
        .meta(&k("attention.key_length"))
        .and_then(Value::as_i64)
        .unwrap_or(hidden_size / num_heads);
    let rms_norm_eps = g
        .meta(&k("attention.layer_norm_rms_epsilon"))
        .and_then(Value::as_f64)
        .unwrap_or(1e-5) as f32;
    // llama-model.cpp:1442 — `hparams.rope_freq_base_train = 10000.0f`
    // is the load-time default when rope.freq_base is absent.
    let rope_theta = g
        .meta(&k("rope.freq_base"))
        .and_then(Value::as_f64)
        .unwrap_or(10000.0) as f32;
    let max_position_embeddings = i64_of(g, &k("context_length"))?;

    // vocab: explicit key > tokenizer tokens > embedding rows
    let vocab_size = match g.meta(&k("vocab_size")).and_then(Value::as_i64) {
        Some(v) => v,
        None => match g.meta("tokenizer.ggml.tokens").and_then(Value::as_arr) {
            Some(t) => t.len() as i64,
            None => g
                .tensor("token_embd.weight")
                .map(|t| t.hf_shape()[0])
                .ok_or_else(|| "GGUF gives no vocabulary size".to_string())?,
        },
    };

    let tie_word_embeddings = g.tensor("output.weight").is_none();
    let qk_norm = g.tensors().any(|t| t.name.ends_with("attn_q_norm.weight"));

    Ok(ModelConfig {
        model_type: arch,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        qk_norm,
        max_position_embeddings,
        tie_word_embeddings,
    })
}
