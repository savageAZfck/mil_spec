//! `config` — HuggingFace `config.json` → [`ModelConfig`].
//!
//! Parses the subset of HF transformer configs that drives graph
//! generation. Unknown fields are ignored; missing fields get the
//! architecture's documented defaults. Qwen2/Qwen3-class today —
//! the same layout covers Llama-family models through the same fields.

/// A parsed model configuration.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// `model_type` string ("qwen3", "qwen2", "llama", ...).
    pub model_type: String,
    /// Hidden dimension.
    pub hidden_size: i64,
    /// Transformer layer count.
    pub num_layers: usize,
    /// Query attention heads.
    pub num_heads: i64,
    /// Key/value heads (GQA); defaults to `num_heads`.
    pub num_kv_heads: i64,
    /// Per-head dimension; defaults to `hidden_size / num_heads`.
    pub head_dim: i64,
    /// MLP intermediate dimension.
    pub intermediate_size: i64,
    /// Vocabulary size.
    pub vocab_size: i64,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether q/k per-head norms exist (Qwen3 yes, Qwen2/Llama no).
    pub qk_norm: bool,
    /// `max_position_embeddings` — used to bound `max_kv`.
    pub max_position_embeddings: i64,
    /// `tie_word_embeddings` — if true, `lm_head` mirrors `embed_tokens`.
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Parse from a `config.json` file's bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("config.json: {e}"))?;
        let get = |k: &str| v.get(k);
        let i64_of = |k: &str| get(k).and_then(|x| x.as_i64());
        let need = |k: &str| -> Result<i64, String> {
            i64_of(k).ok_or_else(|| format!("config.json missing `{k}`"))
        };
        let model_type = get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_string();
        let hidden_size = need("hidden_size")?;
        let num_layers = i64_of("num_hidden_layers")
            .ok_or_else(|| "config.json missing `num_hidden_layers`".to_string())?
            as usize;
        let num_heads = need("num_attention_heads")?;
        let num_kv_heads = i64_of("num_key_value_heads").unwrap_or(num_heads);
        let head_dim = i64_of("head_dim").unwrap_or(hidden_size / num_heads);
        let intermediate_size = need("intermediate_size")?;
        let vocab_size = need("vocab_size")?;
        let rms_norm_eps = get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32;
        let rope_theta = get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e6) as f32;
        let qk_norm = matches!(model_type.as_str(), "qwen3" | "qwen3_moe");
        let max_position_embeddings = i64_of("max_position_embeddings").unwrap_or(32768);
        let tie_word_embeddings = get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        Ok(ModelConfig {
            model_type,
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

    /// Supported architecture check — today only the Qwen/Llama decoder
    /// family (SwiGLU MLP, RMSNorm, RoPE, GQA) maps to the builder.
    pub fn supported(&self) -> bool {
        matches!(
            self.model_type.as_str(),
            "qwen3" | "qwen3_moe" | "qwen2" | "llama" | "mistral"
        )
    }
}
