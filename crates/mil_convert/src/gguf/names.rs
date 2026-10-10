//! GGUF ↔ HF tensor-name mapping for the dense decoder family
//! (qwen2 / qwen3 / llama / mistral).
//!
//! Verified against `gguf-py/gguf/constants.py` (`MODEL_TENSOR` names)
//! and `conversion/` at the pinned llama.cpp commit. `mil_convert`
//! only consumes the dense-decoder core; anything else (MoE experts,
//! encoder/vision tensors, per-layer embeddings) maps to `None` so a
//! `has()` lookup cleanly reports absence.

/// Split `blk.<n>.<rest>` → (`n`, `rest`).
fn split_blk(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (n, rest) = rest.split_once('.')?;
    Some((n.parse().ok()?, rest))
}

/// HF canonical name → ggml tensor name. `None` when the name has no
/// GGUF counterpart.
pub fn hf_to_ggml(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let suffix = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.o_proj.bias" => "attn_output.bias",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.gate_proj.bias" => "ffn_gate.bias",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.up_proj.bias" => "ffn_up.bias",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.down_proj.bias" => "ffn_down.bias",
        _ => return None,
    };
    Some(format!("blk.{l}.{suffix}"))
}

/// ggml tensor name → HF canonical name (used by `milc gguf` display).
pub fn ggml_to_hf(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let (l, rest) = split_blk(name)?;
    let suffix = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_output.bias" => "self_attn.o_proj.bias",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_gate.bias" => "mlp.gate_proj.bias",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_up.bias" => "mlp.up_proj.bias",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "ffn_down.bias" => "mlp.down_proj.bias",
        _ => return None,
    };
    Some(format!("model.layers.{l}.{suffix}"))
}

/// Is this ggml name subject to the llama Q/K head-interleave permute?
pub enum PermuteKind {
    /// `attn_q` — permuted with `head_count`.
    Q,
    /// `attn_k` — permuted with `head_count_kv`.
    K,
}

/// Classify a ggml tensor name for the llama un-permute (`attn_q` /
/// `attn_k` weights *and* biases — `conversion/llama.py` permutes both).
pub fn permute_kind(ggml_name: &str) -> Option<PermuteKind> {
    let (_, rest) = split_blk(ggml_name)?;
    match rest {
        "attn_q.weight" | "attn_q.bias" => Some(PermuteKind::Q),
        "attn_k.weight" | "attn_k.bias" => Some(PermuteKind::K),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for (hf, gg) in [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.27.post_attention_layernorm.weight",
                "blk.27.ffn_norm.weight",
            ),
            (
                "model.layers.3.self_attn.q_proj.weight",
                "blk.3.attn_q.weight",
            ),
            ("model.layers.3.self_attn.q_proj.bias", "blk.3.attn_q.bias"),
            (
                "model.layers.3.self_attn.k_proj.weight",
                "blk.3.attn_k.weight",
            ),
            (
                "model.layers.3.self_attn.v_proj.weight",
                "blk.3.attn_v.weight",
            ),
            (
                "model.layers.3.self_attn.o_proj.weight",
                "blk.3.attn_output.weight",
            ),
            (
                "model.layers.3.self_attn.q_norm.weight",
                "blk.3.attn_q_norm.weight",
            ),
            (
                "model.layers.3.mlp.gate_proj.weight",
                "blk.3.ffn_gate.weight",
            ),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            (
                "model.layers.3.mlp.down_proj.weight",
                "blk.3.ffn_down.weight",
            ),
        ] {
            assert_eq!(hf_to_ggml(hf).as_deref(), Some(gg), "hf→ggml {hf}");
            assert_eq!(ggml_to_hf(gg).as_deref(), Some(hf), "ggml→hf {gg}");
        }
    }

    #[test]
    fn unmapped_is_none() {
        assert!(hf_to_ggml("model.layers.0.self_attn.rotary_emb.inv_freq").is_none());
        assert!(ggml_to_hf("blk.0.ffn_gate_inp.weight").is_none());
    }
}
