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
| Graph ops (`const`, `mul`/`add`/`sub`, `conv` 1x1, `matmul`, `reshape`/`transpose`/`slice`/`concat`, `softmax`, `cast`, `reduce_mean`/`rsqrt`) | [`Block`] builder helpers |
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
| **`mil_convert`** | safetensors → `.mlpackage`. Streams HF checkpoints into fat single-graph decoders: packed-KV state (`slice_update` at runtime `pos`), GQA attention, per-channel int8 or fp16 conv weights. Config-driven — Qwen2/Qwen3/Llama-class today. |
| **`mil_passes`** | Optimizer over `Block`: const dedup (the helpers emit ~5 identical consts per conv — this is the big spec shrinker), constant folding, no-op elimination, dead code, fixpoint. Deterministic, reports every change. |
| **`mil_lint`** | The differentiator — static **ANE-placement analysis**. Predicts per-op execution unit (ANE/GPU/CPU) with a stated rule *before* you compile, flags CPU islands and fp32 regions, and estimates dispatch count — the number that decides whether a graph lives or dies on the ANE. coremltools can't do this at all. |
| **`mil_compile`** | `.mlpackage` → `.mlmodelc`. Two backends: in-process `MLModel compileModelAtURL:` via the Objective-C runtime (no `xcrun`, no subprocess), and an `xcrun coremlc` driver with structured errors and `.mlmodelc` discovery. |
| **`mil_verify`** | Reads specs back: generic protobuf decoder, structural diff (`milc diff` shows why `coremlc` rejected your graph), name-resolution validation, `weight.bin` integrity, and a conformance battery — valid graphs must pass, planted-invalid controls must fail, or the verifier itself is broken. |
| **`milc`** | The CLI over all of it: `convert`, `lint`, `inspect`, `diff`, `compile`, `verify`, `check`. |

```bash
cargo build --release -p milc
milc convert Qwen3-0.6B -o drafter.mlpackage --seq 1 --max-kv 2048
milc lint drafter.mlpackage     # per-op ANE/GPU/CPU + dispatch estimate
milc compile drafter.mlpackage  # via CoreML.framework, in-process
milc verify                     # conformance battery
```

A converted decoder reports ~100% ANE placement and 1 estimated dispatch —
the fat-graph shape the 11-shard drafter needed.

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
