//! `tokenizer` — export a GGUF's embedded tokenizer metadata as
//! HuggingFace `tokenizer.json` / `tokenizer_config.json`.
//!
//! Two ggml tokenizer families:
//!
//! - `tokenizer.ggml.model = "gpt2"` — byte-level BPE. Vocab and merge
//!   list copy over verbatim; `tokenizer.ggml.token_type` selects the
//!   `added_tokens` block; the `pre_tokenizer` is chosen from
//!   `tokenizer.ggml.pre` using the same string table llama.cpp uses
//!   (`src/llama-vocab.cpp` at the pinned commit). Unknown `pre` is a
//!   hard error, never a fallback.
//! - `tokenizer.ggml.model = "llama"` — SentencePiece. HF represents
//!   these as a BPE whose merges are *derived from the piece scores*
//!   (transformers' `SentencePieceExtractor`/`generate_merges`); we
//!   port that algorithm exactly. Normalizer/decoder follow the
//!   Metaspace shape HF emits for llama-family fast tokenizers.
//!
//! Anything else (`"bert"`, `"t5"`, `"rwkv"`, …) gets a named
//! `UnsupportedTokenizer` error.

use super::{fmt_err, Gguf, Result, Value};
use serde_json::{json, Map};
use std::collections::HashMap;

/// Named error surface for the two strict-failure cases the API
/// contract promises: unknown tokenizer model, unknown pre-tokenizer.
fn unsupported(kind: &str, what: &str) -> super::GgufError {
    fmt_err(format!("unsupported {kind}: {what:?}"))
}

fn meta_str<'a>(g: &'a Gguf, key: &str) -> Option<&'a str> {
    g.meta(key).and_then(Value::as_str)
}

fn meta_i64(g: &Gguf, key: &str) -> Option<i64> {
    g.meta(key).and_then(Value::as_i64)
}

fn str_arr(g: &Gguf, key: &str) -> Result<Vec<String>> {
    let v = g
        .meta(key)
        .ok_or_else(|| fmt_err(format!("missing metadata {key}")))?;
    let arr = v
        .as_arr()
        .ok_or_else(|| fmt_err(format!("metadata {key} is not an array")))?;
    arr.iter()
        .map(|e| {
            e.as_str()
                .map(str::to_string)
                .ok_or_else(|| fmt_err(format!("metadata {key} contains a non-string element")))
        })
        .collect()
}

fn int_arr(g: &Gguf, key: &str) -> Result<Vec<i64>> {
    let v = g
        .meta(key)
        .ok_or_else(|| fmt_err(format!("missing metadata {key}")))?;
    let arr = v
        .as_arr()
        .ok_or_else(|| fmt_err(format!("metadata {key} is not an array")))?;
    arr.iter()
        .map(|e| {
            e.as_i64()
                .ok_or_else(|| fmt_err(format!("metadata {key} contains a non-int element")))
        })
        .collect()
}

fn f64_arr(g: &Gguf, key: &str) -> Result<Vec<f64>> {
    let v = g
        .meta(key)
        .ok_or_else(|| fmt_err(format!("missing metadata {key}")))?;
    let arr = v
        .as_arr()
        .ok_or_else(|| fmt_err(format!("metadata {key} is not an array")))?;
    arr.iter()
        .map(|e| {
            e.as_f64()
                .ok_or_else(|| fmt_err(format!("metadata {key} contains a non-float element")))
        })
        .collect()
}

/// Token id metadata (`*_token_id`) → the token string, `None` when the
/// key is absent or the id is the ggml null sentinel (u32::MAX).
fn token_by_id(g: &Gguf, key: &str, tokens: &[String]) -> Result<Option<String>> {
    let Some(id) = meta_i64(g, key) else {
        return Ok(None);
    };
    if id < 0 || id == u32::MAX as i64 {
        return Ok(None);
    }
    tokens
        .get(id as usize)
        .cloned()
        .map(Some)
        .ok_or_else(|| fmt_err(format!("{key}={id} out of vocab range")))
}

/// ggml `llama_token_type` values (same numbering as the sentencepiece
/// proto and `tokenizer.ggml.token_type`).
const TYPE_UNKNOWN: i64 = 2;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

const TYPE_BYTE: i64 = 6;

/// Is `ty` a "vocab" type in the HF sense (NORMAL or BYTE)?
fn is_vocab_type(ty: i64) -> bool {
    ty == 1 || ty == TYPE_BYTE
}

/// Reconstruct the HF `model.vocab` boundary. llama.cpp writes
/// `tokenizer.ggml.tokens` as HF `get_vocab()` — mergeable vocab and
/// added tokens interleaved by id. When every non-normal token sits in
/// a contiguous tail (Qwen), the HF file keeps them out of `vocab`;
/// when specials are interleaved in the ids (SmolLM2, Llama) the HF
/// file bakes them into `vocab`. Returns the vocab prefix length.
fn vocab_len(tokens: &[String], types: &[i64]) -> usize {
    for i in 0..tokens.len() {
        let t = types.get(i).copied().unwrap_or(1);
        if is_vocab_type(t) {
            continue;
        }
        // first non-normal token — clean tail iff nothing normal follows
        let tail_clean =
            (i..tokens.len()).all(|j| !is_vocab_type(types.get(j).copied().unwrap_or(1)));
        if tail_clean {
            return i;
        }
    }
    tokens.len()
}

/// `added_tokens` entries: UNKNOWN/CONTROL/USER_DEFINED tokens, sorted
/// by id. UNUSED (5) tokens are HF-invisible padding slots — never
/// emitted. `special` mirrors the ggml→HF direction llama.cpp uses:
/// CONTROL and UNKNOWN came from HF's `all_special_ids`; USER_DEFINED
/// did not. (GGUF can't recover HF's `additional_special_tokens`
/// special:false flag — collapsed into CONTROL at conversion.)
fn added_tokens(tokens: &[String], types: &[i64]) -> serde_json::Value {
    let mut added: Vec<(i64, &String, i64)> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let t = types.get(*i).copied().unwrap_or(1);
            t == TYPE_UNKNOWN || t == TYPE_CONTROL || t == TYPE_USER_DEFINED
        })
        .map(|(i, s)| (i as i64, s, types.get(i).copied().unwrap_or(1)))
        .collect();
    added.sort_by_key(|(id, _, _)| *id);
    serde_json::Value::Array(
        added
            .into_iter()
            .map(|(id, content, ty)| {
                json!({
                    "id": id,
                    "content": content,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": ty != TYPE_USER_DEFINED,
                })
            })
            .collect(),
    )
}

/// ---- `tokenizer.ggml.pre` → HF `pre_tokenizer` ----
///
/// The string table and regexes mirror `llama-vocab.cpp`'s
/// `llm_tokenizer_bpe` constructor at the pinned commit. `byte_encode`
/// marks the pres that use GPT-2 byte encoding — those close the
/// sequence with a `ByteLevel` pretokenizer; the rest are raw-UTF-8.
struct Pre {
    /// HF `pre_tokenizer` structure.
    pretok: serde_json::Value,
    /// Whether GPT-2 byte encoding applies (shapes decoder choice).
    byte_encode: bool,
}

fn split(regex: &str) -> serde_json::Value {
    json!({
        "type": "Split",
        "pattern": { "Regex": regex },
        "behavior": "Isolated",
        "invert": false,
    })
}

fn byte_level(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> serde_json::Value {
    json!({
        "type": "ByteLevel",
        "add_prefix_space": add_prefix_space,
        "trim_offsets": trim_offsets,
        "use_regex": use_regex,
    })
}

/// A `Sequence` of regex Splits + a closing ByteLevel — the HF
/// spelling of llama.cpp's sequential regex list.
fn regex_seq(regexes: &[&str], byte_encode: bool) -> serde_json::Value {
    let mut items: Vec<serde_json::Value> = regexes.iter().map(|r| split(r)).collect();
    if byte_encode {
        items.push(byte_level(false, true, false));
    }
    json!({ "type": "Sequence", "pretokenizers": items })
}

// Regexes verbatim from llama-vocab.cpp at the pinned commit
// (comments there preserve the original tokenizer.json form where the
// two differ — for HF output we keep llama.cpp's adapted form, which
// fancy-regex also accepts).
const RE_LLAMA3: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_JAIS2: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}";
const RE_DBRX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+(?!\\S)|\\s+";
const RE_GPT2: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
const RE_QWEN2: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_QWEN35: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_PORO: &str = " ?[^(\\s|.,!?…。，、।۔،)]+";
const RE_CHATGLM4: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_TEKKEN: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_GPT4O: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_GRANITE_EMB: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_SEED_CODER: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_UFAKZEKA: &str = "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_BAILING: &str = "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
const RE_EXAONE_MOE: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
const RE_MINICPM5: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_TINY_AYA: &str = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_K2_HORIZON: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const RE_NEWLINE: &str = "[^\\n]+|[\\n]+";
const RE_HF_QWEN2: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The `pre` string table — mirrors llama-vocab.cpp's chain at the
/// pinned commit, complete. Anything not listed errors out.
fn pre_tokenizer(pre: &str) -> Result<Pre> {
    // Verified HF structures for the three families our reference
    // tokenizers actually use.
    let qwen2 = || Pre {
        pretok: json!({
            "type": "Sequence",
            "pretokenizers": [
                split(RE_HF_QWEN2),
                byte_level(false, false, false),
            ],
        }),
        byte_encode: true,
    };
    let digits_bytelevel = || Pre {
        pretok: json!({
            "type": "Sequence",
            "pretokenizers": [
                { "type": "Digits", "individual_digits": true },
                byte_level(false, true, true),
            ],
        }),
        byte_encode: true,
    };
    let gpt2_bytelevel = || Pre {
        // HF GPT-2 convention: the family regex lives inside
        // ByteLevel's own `use_regex` split — no explicit Split node.
        pretok: byte_level(false, true, true),
        byte_encode: true,
    };
    let seq = |regexes: &'static [&'static str]| Pre {
        pretok: regex_seq(regexes, true),
        byte_encode: true,
    };
    let raw = |regexes: &'static [&'static str]| Pre {
        // byte_encode=false pres: raw-UTF-8 BPE, no ByteLevel node.
        pretok: regex_seq(regexes, false),
        byte_encode: false,
    };
    Ok(match pre {
        // qwen2 family — verified against Qwen3/Qwen2.5 HF files
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" => qwen2(),
        // llama.cpp [\\p{N}, gpt2] pair == Digits + ByteLevel(use_regex)
        // — verified against SmolLM2's HF file.
        "smollm" | "starcoder" | "refact" | "command-r" | "codeshell" | "exaone"
        | "minerva-7b" | "mellum2" => digits_bytelevel(),
        // single-regex byte-level family — plain ByteLevel.
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
        | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "jina-v1-en"
        | "jina-v2-code" | "roberta-bpe" | "mpt" | "olmo" | "jais" | "exaone4"
        | "trillion" | "granite-docling" => gpt2_bytelevel(),
        "whitespace" => {
            return Ok(Pre {
                pretok: json!({ "type": "WhitespaceSplit" }),
                byte_encode: false,
            })
        }
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral"
        | "midm-2.0" | "lfm2" | "jina-v5-nano" => seq(&[RE_LLAMA3]),
        "jais-2" => seq(&[RE_JAIS2]),
        "dbrx" | "smaug-bpe" => seq(&[RE_DBRX]),
        "falcon" => seq(&[
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            RE_GPT2,
            "[0-9][0-9][0-9]",
        ]),
        "stablelm2" | "hunyuan" | "solar-open" | "grok-2" => seq(&[RE_QWEN2]),
        "qwen35" => seq(&[RE_QWEN35]),
        "poro-chat" | "bloom" | "gpt3-finnish" => seq(&[RE_PORO]),
        "viking" => seq(&[RE_PORO, "\\p{N}"]),
        "chatglm-bpe" | "glm4" | "glm5" => seq(&[RE_CHATGLM4]),
        "tekken" => seq(&[RE_TEKKEN]),
        "gpt-4o" | "llama4" | "kanana2" | "talkie" => seq(&[RE_GPT4O]),
        "granite-embed-multi-97m" => seq(&[RE_GRANITE_EMB]),
        "seed-coder" => seq(&[RE_SEED_CODER]),
        "ufakzeka" => seq(&[RE_UFAKZEKA]),
        "bailingmoe" | "bailingmoe2" | "llada-moe" => seq(&[RE_BAILING]),
        "exaone-moe" => seq(&[RE_EXAONE_MOE]),
        "minicpm5" => seq(&["\\p{N}{1,3}", RE_MINICPM5]),
        "tiny_aya" | "cohere2moe" => seq(&["\\d{1,3}(?=(?:\\d{3})*\\b)", RE_TINY_AYA]),
        "kimi-k2" => seq(&["\\p{Han}+"]),
        "k2-horizon" => seq(&[RE_K2_HORIZON]),
        "chameleon" => seq(&[
            "<sentinel:[0-9]+>",
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
            "([\\t\\n]|    |  )",
            "\\p{N}",
            "[\\p{P}!-/:-@\\[-`{-~]",
            RE_GPT2,
        ]),
        "deepseek-llm" => seq(&[
            "[\r\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            "\\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}+",
        ]),
        "deepseek-coder" => seq(&[
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}",
        ]),
        "deepseek-v3" | "hunyuan-dense" | "joyai-llm" | "hy_v4" => seq(&[
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        ]),
        "spark2_5" => seq(&[
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
            "\\p{N}",
        ]),
        "youtu" => seq(&[
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ]),
        "superbpe" => seq(&["\\p{N}+", "(?=(\\d{3})+(?!\\d))"]),
        "afmoe" => seq(&[
            "\\p{AFMoE_digits}",
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ]),
        "laguna" => seq(&[RE_NEWLINE, RE_QWEN2]),
        "gemma4" | "granite-embed-multi-311m" | "mmbert" | "sarvam-moe" => {
            raw(&[RE_NEWLINE])
        }
        // llama.cpp's `default:` arm — the catch-all BPE presplit.
        "default" => seq(&[
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            RE_GPT2,
            "\\p{N}+",
            "[0-9][0-9][0-9]",
        ]),
        other => return Err(unsupported("pre-tokenizer", other)),
    })
}

/// Whether a `pre` family uses Qwen's `""`-valued subword fields
/// (their HF files carry empty strings; everyone else's are null).
fn is_qwen2_family(pre: &str) -> bool {
    matches!(
        pre,
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez"
    )
}

/// `tokenizer.ggml.model = "gpt2"` — byte-level BPE export.
fn gpt2_json(g: &Gguf) -> Result<serde_json::Value> {
    let tokens = str_arr(g, "tokenizer.ggml.tokens")?;
    let merges = str_arr(g, "tokenizer.ggml.merges")?;
    let types = match g.meta("tokenizer.ggml.token_type") {
        Some(_) => int_arr(g, "tokenizer.ggml.token_type")?,
        None => vec![1; tokens.len()],
    };
    if types.len() != tokens.len() {
        return Err(fmt_err(format!(
            "token_type has {} entries for {} tokens",
            types.len(),
            tokens.len()
        )));
    }
    let pre_name = meta_str(g, "tokenizer.ggml.pre").unwrap_or("default");
    let pre = pre_tokenizer(pre_name)?;

    let vocab: Map<String, serde_json::Value> = tokens[..vocab_len(&tokens, &types)]
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), json!(i)))
        .collect();

    let (sub_prefix, eow_suffix): (serde_json::Value, serde_json::Value) =
        if is_qwen2_family(pre_name) {
            (json!(""), json!(""))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
    let (normalizer, post_processor): (serde_json::Value, serde_json::Value) =
        if is_qwen2_family(pre_name) {
            (json!({ "type": "NFC" }), byte_level(false, false, false))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
    let decoder = if is_qwen2_family(pre_name) {
        byte_level(false, false, false)
    } else if pre.byte_encode {
        byte_level(true, true, true)
    } else {
        json!({ "type": "Fuse" })
    };

    Ok(json!({
        "version": "1.0",
        "truncation": serde_json::Value::Null,
        "padding": serde_json::Value::Null,
        "added_tokens": added_tokens(&tokens, &types),
        "normalizer": normalizer,
        "pre_tokenizer": pre.pretok,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": {
            "type": "BPE",
            "dropout": serde_json::Value::Null,
            "unk_token": serde_json::Value::Null,
            "continuing_subword_prefix": sub_prefix,
            "end_of_word_suffix": eow_suffix,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        },
    }))
}

/// Merge generation matching HF's SentencePiece→BPE output (as found
/// in real llama-family tokenizer.json files):
///
/// - every vocab piece contributes all splits whose halves are also
///   vocab pieces, each tagged with the piece's score;
/// - candidates order by `(vocab[left], vocab[right])` — the per-piece
///   `local` sort in `generate_merges`;
/// - the final list sorts by score **descending**, stable — ties keep
///   the `(left_id, right_id)` order. This reproduces the reference
///   file exactly, including the all-`-1e9` piece group at the tail.
fn generate_merges(tokens: &[String], scores: &[f64]) -> Vec<String> {
    let mut id: HashMap<&str, usize> = HashMap::new();
    for (i, t) in tokens.iter().enumerate() {
        id.insert(t.as_str(), i);
    }
    struct Cand {
        l: String,
        r: String,
        il: usize,
        ir: usize,
        score: f64,
    }
    let mut merges: Vec<Cand> = Vec::new();
    for (piece_id, piece) in tokens.iter().enumerate() {
        let chars: Vec<char> = piece.chars().collect();
        for i in 1..chars.len() {
            let l: String = chars[..i].iter().collect();
            let r: String = chars[i..].iter().collect();
            if let (Some(&il), Some(&ir)) = (id.get(l.as_str()), id.get(r.as_str())) {
                merges.push(Cand {
                    l,
                    r,
                    il,
                    ir,
                    score: scores[piece_id],
                });
            }
        }
    }
    // Stable (il, ir) ordering first, then a stable score-descending
    // sort — ties keep the id order.
    merges.sort_by_key(|c| (c.il, c.ir));
    merges.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merges
        .into_iter()
        .map(|c| format!("{} {}", c.l, c.r))
        .collect()
}

/// `tokenizer.ggml.model = "llama"` — SentencePiece, exported in HF's
/// BPE-of-SPM form (byte fallback, Metaspace-style normalizer).
fn llama_json(g: &Gguf) -> Result<serde_json::Value> {
    let tokens = str_arr(g, "tokenizer.ggml.tokens")?;
    let scores = f64_arr(g, "tokenizer.ggml.scores")?;
    if scores.len() != tokens.len() {
        return Err(fmt_err(format!(
            "scores has {} entries for {} tokens",
            scores.len(),
            tokens.len()
        )));
    }
    let types = match g.meta("tokenizer.ggml.token_type") {
        Some(_) => int_arr(g, "tokenizer.ggml.token_type")?,
        None => vec![1; tokens.len()],
    };
    let merges = generate_merges(&tokens, &scores);
    let vocab: Map<String, serde_json::Value> = tokens[..vocab_len(&tokens, &types)]
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), json!(i)))
        .collect();
    let unk = token_by_id(g, "tokenizer.ggml.unknown_token_id", &tokens)?;

    // HF's llama TemplateProcessing wraps the sequence in <s>.
    let post_processor = match token_by_id(g, "tokenizer.ggml.bos_token_id", &tokens)? {
        Some(bos) => {
            let bos_id = meta_i64(g, "tokenizer.ggml.bos_token_id").unwrap_or(0);
            let mut st = Map::new();
            st.insert(
                bos.clone(),
                json!({ "id": bos, "ids": [bos_id], "tokens": [bos] }),
            );
            json!({
                "type": "TemplateProcessing",
                "single": [
                    { "SpecialToken": { "id": bos, "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } },
                ],
                "pair": [
                    { "SpecialToken": { "id": bos, "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } },
                    { "SpecialToken": { "id": bos, "type_id": 1 } },
                    { "Sequence": { "id": "B", "type_id": 1 } },
                ],
                "special_tokens": st,
            })
        }
        None => serde_json::Value::Null,
    };

    Ok(json!({
        "version": "1.0",
        "truncation": serde_json::Value::Null,
        "padding": serde_json::Value::Null,
        "added_tokens": added_tokens(&tokens, &types),
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                { "type": "Prepend", "prepend": "▁" },
                { "type": "Replace", "pattern": { "String": " " }, "content": "▁" },
            ],
        },
        "pre_tokenizer": serde_json::Value::Null,
        "post_processor": post_processor,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                { "type": "Replace", "pattern": { "String": "▁" }, "content": " " },
                { "type": "ByteFallback" },
                { "type": "Fuse" },
                { "type": "Strip", "content": " ", "start": 1, "stop": 0 },
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": serde_json::Value::Null,
            "unk_token": unk,
            "continuing_subword_prefix": serde_json::Value::Null,
            "end_of_word_suffix": serde_json::Value::Null,
            "fuse_unk": true,
            "byte_fallback": true,
            "vocab": vocab,
            "merges": merges,
        },
    }))
}

/// HF `tokenizer.json` for this GGUF's embedded tokenizer.
pub fn to_tokenizer_json(g: &Gguf) -> Result<String> {
    let model = meta_str(g, "tokenizer.ggml.model")
        .ok_or_else(|| fmt_err("missing tokenizer.ggml.model"))?;
    let v = match model {
        "gpt2" => gpt2_json(g)?,
        "llama" => llama_json(g)?,
        other => return Err(unsupported("tokenizer model", other)),
    };
    serde_json::to_string_pretty(&v).map_err(|e| fmt_err(format!("json: {e}")))
}

/// HF `tokenizer_config.json` — the special tokens and chat template
/// the GGUF carries.
pub fn to_tokenizer_config_json(g: &Gguf) -> Result<String> {
    let tokens = str_arr(g, "tokenizer.ggml.tokens").unwrap_or_default();
    let tok = |key: &str| -> serde_json::Value {
        match token_by_id(g, key, &tokens) {
            Ok(Some(s)) => json!(s),
            _ => serde_json::Value::Null,
        }
    };
    let mut cfg = Map::new();
    cfg.insert("bos_token".into(), tok("tokenizer.ggml.bos_token_id"));
    cfg.insert("eos_token".into(), tok("tokenizer.ggml.eos_token_id"));
    cfg.insert("pad_token".into(), tok("tokenizer.ggml.padding_token_id"));
    cfg.insert("unk_token".into(), tok("tokenizer.ggml.unknown_token_id"));
    for (k, key) in [
        ("add_bos_token", "tokenizer.ggml.add_bos_token"),
        ("add_eos_token", "tokenizer.ggml.add_eos_token"),
    ] {
        if let Some(Value::Bool(b)) = g.meta(key) {
            cfg.insert(k.into(), json!(b));
        }
    }
    if let Some(t) = meta_str(g, "tokenizer.chat_template") {
        cfg.insert("chat_template".into(), json!(t));
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(cfg))
        .map_err(|e| fmt_err(format!("json: {e}")))
}
