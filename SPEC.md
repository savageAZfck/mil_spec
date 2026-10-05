# SPEC — mil_spec (pure-Rust CoreML program writer)

## What it emits

A `.mlpackage` bundle:

```
model.mlpackage/
  Manifest.json            # fileFormatVersion, itemInfoEntries, rootModelIdentifier
  Data/com.apple.CoreML/
    model.mlmodel          # protobuf spec (MIL program), hand-encoded
    weights/weight.bin     # v2 blob format, file-backed or in-memory
```

`xcrun coremlc compile model.mlpackage .` must accept the output — the
platform toolchain is the verifier of last resort.

## Protobuf encoding

`f_u32/f_u64/f_i32/f_i64/f_str/f_bytes/f_msg/f_packed_*` — hand-rolled
wire-format writers. Varints for ints, tag = `field << 3 | wire_type`,
length-delimited fields carry `u32 len + bytes`. Packed scalar fields
serialize as an inner length-delimited buffer. ZigZag is *not* used
where the schema expects sint — the encoder matches the field types in
Apple's `format/` proto definitions (BSD-3, public).

## Model structure

- `Block` — builder for MIL ops: `const`, elementwise `mul`/`add`/`sub`,
  `conv` (1x1), `matmul`, `reshape`/`transpose`/`slice`/`concat`,
  `softmax`, `cast`, `reduce_mean`/`rsqrt`. Raw ops via `Block::op` +
  `bind`/`bind_many`/`bind_const`.
- `Feature` — named tensor spec `{name, shape, dtype, is_state}`.
  `is_state` marks KV-cache surfaces; `Block::read_state`/`write_state`
  wire stateful graphs.
- `NVT` (name + `ValueType`) — function signature entries: `Tensor` or
  `State` variants of `TensorType { dtype, shape }`.
- `encode_model(inputs, outputs, ops?, block, fn_inputs, meta)` → spec
  bytes. `write_mlpackage(dir, spec, weights)` → bundle + manifest.
- `ModelMeta::new(spec_version, "CoreML9")` — declared converter
  identity recorded in the spec.

## Weights

- `BlobWriter` — file-backed v2 writer (offset/length per tensor).
- `WeightBin` — in-memory v2.
- `Block::konst_q8` — weight-only int8 via `constexpr_blockwise_shift_scale`.

## IR front-end

`ir::compile(text)` — text IR → validated graph → `Block`. The IR
checker enforces shape/dtype legality *before* bytes are emitted:
unknown ops, arity mismatches, and shape errors are `Err`, never
silently encoded.

## Determinism

Encoding is deterministic given the same graph — field order, string
bytes, and blob layout are fixed. Two builds of the same model produce
byte-identical packages.
