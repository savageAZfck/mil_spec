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

`ModelMeta` carries `spec_version` + `opset` — e.g. `(8, "CoreML5")`,
`(10, "CoreML9")` — the Function's block specialization and the
specification version move together.

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
