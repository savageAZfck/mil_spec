//! Structural proof for `gguf::tokenizer` — `#[ignore]`d: needs the
//! reference files under `~/.cache/mil_gguf_test/models/`.
//!
//! For every supported tokenizer family the generated tokenizer.json is
//! compared field-by-field against the model's real HuggingFace file:
//! `model.vocab` identical, `merges` identical in content AND rank
//! order, `added_tokens` identical (id, content, special flag), and the
//! `normalizer`/`pre_tokenizer`/`post_processor`/`decoder` blocks
//! identical. Merge *syntax* is normalized — HF serializes them either
//! as `"l r"` strings or `["l","r"]` pairs; we emit strings and compare
//! the pairs.
//!
//! ```sh
//! cargo test -p mil_convert --test gguf_tokenizer -- --ignored --nocapture
//! ```

use mil_convert::gguf::tokenizer::to_tokenizer_json;
use mil_convert::gguf::Gguf;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn cache() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/mil_gguf_test/models")
}

fn hf_json(rel: &str) -> Value {
    let p = cache().join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .unwrap()
}

fn gen(gguf: &str) -> Value {
    let g = Gguf::open(&cache().join(gguf)).unwrap();
    serde_json::from_str(&to_tokenizer_json(&g).unwrap()).unwrap()
}

/// HF merges come in two spellings; both normalize to (left, right)
/// pairs so content and rank order are what get compared.
fn merge_pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .expect("merges not an array")
        .iter()
        .map(|m| match m {
            Value::String(s) => {
                let (l, r) = s.split_once(' ').unwrap_or_else(|| panic!("merge {s:?}"));
                (l.to_string(), r.to_string())
            }
            Value::Array(a) => (
                a[0].as_str().unwrap().to_string(),
                a[1].as_str().unwrap().to_string(),
            ),
            other => panic!("merge entry {other}"),
        })
        .collect()
}

fn added_map(v: &Value) -> BTreeMap<i64, (String, bool)> {
    v.as_array()
        .expect("added_tokens not an array")
        .iter()
        .map(|t| {
            (
                t["id"].as_i64().unwrap(),
                (
                    t["content"].as_str().unwrap().to_string(),
                    t["special"].as_bool().unwrap(),
                ),
            )
        })
        .collect()
}

fn compare(gen: &Value, hf: &Value, name: &str) {
    compare_ex(gen, hf, name, &[])
}

/// `expected_special_diffs` pins the *exact* added-token `special`
/// flags GGUF cannot carry: llama.cpp collapses HF's
/// `additional_special_tokens` (special:false) into CONTROL alongside
/// real specials, so on Qwen exports six type-3 tokens read back as
/// special:true. Pinning the set keeps the check strict — any other
/// token drifting still fails.
fn compare_ex(gen: &Value, hf: &Value, name: &str, expected_special_diffs: &[&str]) {
    let (gv, hv) = (&gen["model"]["vocab"], &hf["model"]["vocab"]);
    assert_eq!(gv, hv, "{name}: vocab differs");
    assert_eq!(
        merge_pairs(&gen["model"]["merges"]),
        merge_pairs(&hf["model"]["merges"]),
        "{name}: merges differ (content or rank order)"
    );
    let (ga, ha) = (
        added_map(&gen["added_tokens"]),
        added_map(&hf["added_tokens"]),
    );
    assert_eq!(
        ga.keys().collect::<Vec<_>>(),
        ha.keys().collect::<Vec<_>>(),
        "{name}: added_token ids differ"
    );
    let mut special_diffs: Vec<String> = Vec::new();
    for (id, (gc, gs)) in &ga {
        let (hc, hs) = &ha[id];
        assert_eq!(gc, hc, "{name}: added_token {id} content differs");
        if gs != hs {
            special_diffs.push(hc.clone());
        }
    }
    assert_eq!(
        special_diffs,
        expected_special_diffs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        "{name}: unexpected added_token special-flag diffs"
    );
    for field in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        assert_eq!(&gen[field], &hf[field], "{name}: {field} differs");
    }
}

/// Qwen's HF files mark these six additional_special_tokens
/// special:false; the GGUF stores them as CONTROL and the format
/// carries nothing finer.
const QWEN_SPECIAL_DIFFS: &[&str] = &[
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
];

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn qwen3_matches_hf() {
    compare_ex(
        &gen("qwen3-gguf-unsloth/Qwen3-0.6B-Q8_0.gguf"),
        &hf_json("qwen3-hf/tokenizer.json"),
        "qwen3",
        QWEN_SPECIAL_DIFFS,
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn qwen25_matches_hf() {
    compare_ex(
        &gen("qwen25-gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
        &hf_json("qwen25-hf/tokenizer.json"),
        "qwen25",
        QWEN_SPECIAL_DIFFS,
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn smollm2_matches_hf() {
    compare(
        &gen("smollm2-gguf/SmolLM2-135M-Instruct-Q8_0.gguf"),
        &hf_json("smollm2-hf/tokenizer.json"),
        "smollm2",
    );
}

#[test]
#[ignore = "needs ~/.cache/mil_gguf_test downloads"]
fn stories15m_spm_matches_llama() {
    compare(
        &gen("stories15m-gguf/stories15M.gguf"),
        &hf_json("llama-tok/tokenizer.json"),
        "stories15m",
    );
}
