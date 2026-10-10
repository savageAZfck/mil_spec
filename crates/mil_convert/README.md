# mil_convert

safetensors / GGUF → `.mlpackage` in pure Rust — the `coremltools.convert`
replacement: load a HuggingFace checkpoint *or* a llama.cpp GGUF file,
generate a fat single-graph decoder MIL program, and emit a `.mlpackage`
that `coremlc` accepts. No Python anywhere in the path.

## Frontends

| source | entry point | notes |
|---|---|---|
| `*.safetensors` directory | `mil_convert::convert` | HF checkpoints, `model.safetensors.index.json` shards |
| `*.gguf` file | `mil_convert::convert_gguf` | GGUF v2/v3, split shards (`*-000NN-of-0000N`), embedded tokenizer export |

`convert_gguf` also writes `tokenizer.json` and `tokenizer_config.json`
next to the output package (rebuilt from `tokenizer.ggml.*` metadata).

## Supported ggml tensor types

Dequantizers are ported from ggml (llama.cpp pin `10a60cf3`), verified
**bit-exact** against the C implementation by the golden-vector suite in
`tests/gguf_golden/`.

| class | types |
|---|---|
| float/int | F32, F16, BF16, F64, I8, I16, I32, I64 |
| legacy quants | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| ternary/micro | TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 |

Q8_1 has no `to_float` in ggml (intermediate-only type) and errors with a
named message; retired ids (Q4_2/Q4_3, the `*_4_4`/`_4_8`/`_8_8` reblocked
family, IQ4_NL repacks) and ids past the pinned `GGML_TYPE_COUNT` get
distinct `removed`/`unknown` errors.

## ggml → HF name mapping

`token_embd`→`model.embed_tokens`, `output_norm`→`model.norm`,
`output`→`lm_head`, `blk.N.attn_norm`→`input_layernorm`,
`blk.N.ffn_norm`→`post_attention_layernorm`,
`blk.N.attn_{q,k,v,output}`→`self_attn.{q,k,v,o}_proj` (.weight and .bias),
`blk.N.attn_{q,k}_norm`→`self_attn.{q,k}_norm`,
`blk.N.ffn_{gate,up,down}`→`mlp.{gate,up,down}_proj`.

For `general.architecture = llama` (includes Mistral GGUFs), `attn_q`/`attn_k`
are un-permuted on load — the exact inverse of llama.cpp's
`reshape(n_head, 2, …).swapaxes(1, 2)` permutation. Qwen2/Qwen3 GGUFs are
not permuted upstream and pass through.

## Changelog

### 0.2.0

- **GGUF frontend**: parser (v2/v3, split shards, endianness/version
  validation), all dequantizers bit-exact vs ggml, ggml↔HF name mapping,
  llama Q/K un-permute, `convert_gguf`, `milc gguf` inspect +
  `--tokenizer` export.
- **Tokenizer export**: `gguf::tokenizer::to_tokenizer_json` — GPT-2
  byte-level BPE and llama SentencePiece→BPE conversion verified
  structurally identical to the upstream HF `tokenizer.json` files.
- **Qwen2 attention biases**: `q_proj`/`k_proj`/`v_proj` biases are now
  emitted into the conv ops (previously dropped silently); required for
  Qwen2/Qwen2.5 checkpoints. Fixed in the builder, the GGUF name map, and
  preflight.
- **`WeightSource` trait** replaces the concrete `shards` field on
  `WeightEmitter` — breaking change to that struct's public API.
