# mil_spec

> **Status: beta.** Op coverage is the subset a stateful LLM graph needs —
> the wire format itself is stable and verified by `coremlc`.

**Write Apple CoreML `.mlpackage` programs in pure Rust. No Python, no
coremltools, no protoc.**

`mil_spec` emits the protobuf `model.mlmodel` spec, `weight.bin` v2 weight
blobs, and the `Manifest.json` package layout by hand — Apple documents the
schema ([`mlmodel/format`](https://github.com/apple/coremltools), BSD-3) but
the only writer most people have ever used is coremltools. This is the
other one.

```rust
use mil_spec::*;

let mut b = Block::new();
let y = b.mul("x", "x", &[1, 4, 1, 1], "y");
b.outputs = vec![y];

let inputs  = [Feature { name: "x".into(), shape: vec![1, 4, 1, 1],
                       dtype: DType::Fp16, is_state: false }];
let outputs = [Feature { name: "y".into(), shape: vec![1, 4, 1, 1],
                       dtype: DType::Fp16, is_state: false }];
let fn_inputs = [NVT { name: "x".into(),
    ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])) }];

let spec = encode_model(&inputs, &outputs, &[], &b, &fn_inputs,
                        &ModelMeta::new(10, "CoreML9"));
write_mlpackage("model.mlpackage".as_ref(), &spec, None).unwrap();
```

Compile with `xcrun coremlc compile model.mlpackage .` and you have a real
`.mlmodelc`.

## What it writes

| Piece | API |
|---|---|
| Elementwise (`mul`/`add`/`sub`, `real_div`/`floor_div`/`modulo`, `pow`, `maximum`/`minimum`, `exp`/`exp2`/`log`, `sqrt`/`rsqrt`/`square`, `abs`/`floor`/`ceil`/`round`/`neg`/`sign`, `sin`/`cos`/`tan`/`asin`/`acos`/`atan`, `sinh`/`cosh`/`tanh`/`atanh`, `erf`, `inverse`, `clip`/`threshold`, `select`) | [`Block`] builder helpers |
| Activations (`sigmoid`, `relu`, `relu6`, `leaky_relu`, `elu`, `gelu`, `softplus`, `softsign`, `sigmoid_hard`, `clamped_relu`, `prelu`, `softmax`, `log_softmax`) | [`Block`] builder helpers |
| Reductions (`reduce_sum`/`mean`/`max`/`min`/`prod`, `reduce_l1_norm`/`l2_norm`/`sum_square`/`log_sum`/`log_sum_exp`, `reduce_argmax`/`argmin`, `cumsum`) | [`Block`] builder helpers |
| Indexing (`gather`, `gather_along_axis`, `gather_nd`, `scatter`, `scatter_along_axis`, `scatter_nd`, `topk`, `argsort`, `one_hot`) | [`Block`] builder helpers |
| Shape/layout (`reshape`, `transpose`, `expand_dims`, `squeeze`, `slice`, `slice_by_size`, `concat`, `split`, `tile`, `pad`, `broadcast_to`, `flatten2d`, `stack`, `reverse`, `sliding_windows`, `cast`) | [`Block`] builder helpers |
| Normalization (`rms_norm`, `layer_norm`, `batch_norm`, `instance_norm`, `group_norm`, `l2_norm`, `local_response_norm`) | [`Block`] builder helpers |
| Pooling & upsample (`avg_pool`, `max_pool`, `avg_pool_global`, `max_pool_global`, `upsample_nearest`, `upsample_bilinear`) | [`Block`] builder helpers |
| Space transforms (`depth_to_space`, `space_to_depth`, `pixel_shuffle`) | [`Block`] builder helpers |
| Conv & linear (`conv` N-D, `conv_transpose`, `conv1x1`, `linear`, `matmul`) | [`Block`] builder helpers |
| Raw ops with full control | [`Block::op`], [`bind`], [`bind_many`], [`bind_const`] |
| Stateful graphs (KV caches) | [`Feature::is_state`], [`Block::read_state`], [`Block::write_state`] |
| Weight-only int8 (`constexpr_blockwise_shift_scale`) | [`Block::konst_q8`] |
| `weight.bin` v2, file-backed | [`BlobWriter`] |
| `weight.bin` v2, in-memory | [`WeightBin`] |
| `.mlpackage` + `Manifest.json` | [`write_mlpackage`] |
| RMSNorm peephole (fp16-safe prescaling) | [`Block::rms_norm`] |
| Validated IR front-end (text → checked graph) | [`ir::compile`] |

`ModelMeta` carries `spec_version` + `opset` — e.g. `(8, "CoreML5")`,
`(10, "CoreML9")` — the Function's block specialization and the
specification version move together.

## The IR front-end

`mil_spec::ir` is a compiler: a line-oriented SSA text form that lowers
onto the typed `Block` helpers and **validates before it emits** — names
must resolve, shapes must check out (broadcast, reshape element counts,
matmul inner dims, permutations, concat ranks, state reads/writes), and
declared outputs must be defined.

```text
input  x: fp16[1,4,1,1]
k    = const_f16(2.0)
y    = mul(x, k)
z    = reshape(y, [4,1,1,1])
output z
```

```rust
let prog = mil_spec::ir::compile(src)?;          // parse + lower + validate
let spec = mil_spec::encode_model(&prog.inputs, &prog.outputs, &prog.states,
                                  &prog.block, &prog.fn_inputs, &meta);
mil_spec::write_mlpackage(dir, &spec, weights)?;
```

See `examples/compile_ir.rs` for a stateful KV-cache program.

## The workspace

`mil_spec` is the core of a full native-Rust CoreML pipeline — the pieces
`coremltools` monopolizes, split into focused crates:

| Crate | Job |
|---|---|
| **`mil_convert`** | safetensors *and* GGUF → `.mlpackage`. Streams HF checkpoints or llama.cpp files (all ggml quant types, split shards, Q/K un-permute, embedded tokenizer export to `tokenizer.json`) into fat single-graph decoders: packed-KV state (`slice_update` at runtime `pos`), GQA attention, per-channel int8 or fp16 conv weights. Optional **LoRA bake-in** (`--lora`): MLX/PEFT adapters from safetensors or `.npz`, fused as `W + scale·(B @ A)` in f32 before quantization. Config-driven — Qwen2/Qwen3/Llama-class today. |
| **`mil_passes`** | Optimizer over `Block`: const dedup (the helpers emit ~5 identical consts per conv — this is the big spec shrinker), constant folding, no-op elimination, dead code, fixpoint. Deterministic, reports every change. |
| **`mil_lint`** | The differentiator — static **ANE-placement analysis**. Predicts per-op execution unit (ANE/GPU/CPU) with a stated rule *before* you compile, flags CPU islands and fp32 regions, and estimates dispatch count — the number that decides whether a graph lives or dies on the ANE. coremltools can't do this at all. |
| **`mil_compile`** | `.mlpackage` → `.mlmodelc`. Two backends: in-process `MLModel compileModelAtURL:` via the Objective-C runtime (no `xcrun`, no subprocess), and an `xcrun coremlc` driver with structured errors and `.mlmodelc` discovery. |
| **`mil_verify`** | Reads specs back: generic protobuf decoder, structural diff (`milc diff` shows why `coremlc` rejected your graph), name-resolution validation, `weight.bin` integrity, and a conformance battery — valid graphs must pass, planted-invalid controls must fail, or the verifier itself is broken. |
| **`milc`** | The CLI over all of it: `convert`, `lint`, `inspect`, `diff`, `compile`, `verify`, `check`, plus package surgery (`fuse-lora`, `requant`, `graft`, `reshape`) and `attest`. |

```bash
cargo build --release -p milc
milc convert Qwen3-0.6B -o drafter.mlpackage --seq 1 --max-kv 2048
milc convert model.gguf -o drafter.mlpackage   # GGUF auto-detected by magic
milc convert Qwen3-0.6B -o tuned.mlpackage --lora adapters/dream  # LoRA bake-in (MLX/PEFT, safetensors or npz)
milc gguf model.gguf --tokenizer tok.json     # inspect + tokenizer export
milc lint drafter.mlpackage     # per-op ANE/GPU/CPU + dispatch estimate
milc compile drafter.mlpackage  # via CoreML.framework, in-process
milc verify                     # conformance battery
```

A converted decoder reports ~100% ANE placement and 1 estimated dispatch —
the fat-graph shape the 11-shard drafter needed.

## Package surgery

`mil_convert::surgery` is a generic editor over a built `.mlpackage`:
decode the spec, index every weight group (raw fp16 consts, int8 and
Q4-block `constexpr_blockwise_shift_scale`, palette4
`constexpr_lut_to_dense`), stage blob replacements, and rewrite spec +
`weight.bin` in one pass — with op-binding validation before a byte
lands on disk. The CLI commands are thin shells over it:

```bash
milc fuse-lora drafter.mlpackage --lora adapters/dream -o tuned.mlpackage
milc requant   drafter.mlpackage --to int8|fp16|palette4 [-o out.mlpackage]
milc graft     donor.mlpackage --layers 0..4 --onto base.mlpackage -o out.mlpackage
milc reshape   flex.mlpackage --seq-lens 2,8,32 [-o out.mlpackage]
```

- **fuse-lora** fuses adapter deltas into an existing package's weight
  blobs. The packaged-const → HF-tensor mapping is *verifiable* because
  `mil_convert` names conv weight consts deterministically
  (`l{L}_{wq,wk,wv,wo,wg,wu,wd}`, `lm_w`); each target is decoded to
  f32, fused through the same `W + scale·(B @ A)` path as
  `convert --lora`, and re-emitted in its original encoding (int8 stays
  int8 with fresh scales). An adapter for the wrong checkpoint is a
  hard error before anything is written.
- **requant** re-encodes conv weights in place. fp16 → int8/palette4,
  int8 → fp16/palette4, palette4 → fp16 — decoded through f32, emitted
  deterministically. Norm weights and biases stay fp16, matching the
  converter's own policy. Without `-o` the package is rewritten
  atomically in place (temp dir + rename).
- **graft** splices `l{N}_*` layer weight groups from a donor package
  into a same-architecture base — op stream and model description must
  match exactly or it refuses, and payloads are copied losslessly
  (same encoding, same shape).
- **reshape** rewrites `EnumeratedShapes` on a flexible-shape package
  (the `convert --seq-lens` kind): every varying dim must track the
  same old list, and the new set becomes the enumerated entries with
  `seq_lens[0]` as default.

## Provenance (`milc attest`)

Every `mil_convert` package now carries a provenance block in
`description.metadata.userDefined` — the real CoreML key/value map, no
fake fields. Keys are `mil.prov.*`: toolchain + version, unix
timestamp, SHA-256 of the canonical config and options, SHA-256 of
`weight.bin`, and a `name size sha256` fingerprint per source weight
file in deterministic order.

```bash
milc attest drafter.mlpackage                    # re-hashes weight.bin
milc attest drafter.mlpackage --source Qwen3-0.6B  # + source fingerprints
```

Tampered `weight.bin` or a mismatched source file exits nonzero;
packages without provenance (older conversions, foreign tools) are
reported, not failed. Package surgery refreshes `mil.prov.weights` so
a legitimately-edited package still attests — that's the provenance
describing what the package *contains*.

## Updatable layers — the honest answer

CoreML's real updatable-model machinery (`NeuralNetwork.updatable`,
the `set_updatable` flags coremltools flips) lives on the
**neural-network proto**. `mlProgram` — what this toolchain emits —
has no per-op or per-const updatable field for it to map onto, so
there is nothing real to set. Instead `convert --updatable a,b,c`
writes a documented `mil.updatable` marker in the userDefined
metadata naming the weight consts; `milc inspect` shows it and the
package compiles unchanged. It is intent metadata, not CoreML
on-device training support — anything claiming otherwise would be a
fake.

## Why a hand-rolled writer

- **No toolchain.** coremltools drags in Python, protobuf codegen, and a
  version matrix. `mil_spec` is `cargo add` and `std`.
- **Ops coremltools won't emit for you.** Stateful KV models,
  `constexpr_blockwise_shift_scale` weight-only int8, fused
  `scaled_dot_product_attention` under the CoreML9 opset — the things that
  put a whole model on the Neural Engine.
- **Deterministic.** The same graph produces the same bytes. When `coremlc`
  rejects a model you want to diff the spec, not re-run a conversion
  pipeline and hope.

This writer was extracted from a production converter that runs a 4B model
at ~100% runtime-op Neural Engine placement — the encoding is battle-tested
against the real `coremlc`, not just the schema.

## Scope

Emits `mlprogram` (not NeuralNetwork protos). Shapes are static. The op
helpers cover the transformer-layer surface; anything else goes through
`Block::op` with raw `Op`/`Argument`/`Value` — the whole wire layer is
public for exactly that reason.

## License

MIT OR Apache-2.0, your choice.
