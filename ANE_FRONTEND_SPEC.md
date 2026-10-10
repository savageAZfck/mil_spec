# ANEC MIL Frontend — Empirical Spec

Reverse-engineered rules for what `ANECompiler.framework`'s MIL text frontend
(`NetworkMILPath` / `NetworkMILName` in the input dict) accepts, probed via
`ANECValidate`/`ANECCompile` on macOS (M4, h16g). Private API, unstable — all
calls fork-isolated in `crates/mil_infer/examples/anec_probe.rs`.

## Entry points (confirmed ABI)

```
u64 ANECValidate(input: CFDictionary, params: CFDictionary, cb: Block)
u64 ANECCompile (input: CFDictionary, params: CFDictionary, cb: Block)
u64 ANECCompileOnline / Offline / JIT  (same shape)
cb(status: u64, result: CFDictionary)   // real ObjC block required
```

- `input.InputNetworks[]` entries: `NetworkName`, `NetworkMILPath` (dir,
  trailing `/`), `NetworkMILName` (basename). Legacy pair
  `NetworkSourcePath`/`NetworkSourceFileName` also read (expects a
  `Manifest.json` Espresso net dir).
- `input.ExternConstants`, `input.FileConstants` = CFArray (required for
  `CompileOnline`).
- `input.OutputFilePath` + `OutputFileName` → joined by literal concat.
- `params.TargetArchitecture`: CFString or CFArray — table recovered from
  binary: `h11 h12 h13 h13g h14 h14g h14c h15 h15g h15c h16 h16g h16c h16s
  h17 h17a h17c h17d h17g h17s h18 h18g h19` (all create validators).
- `params.OptionsFilePath` → plist with per-target flag sections; the flag
  vocabulary is embedded verbatim in produced Mach-O (`--bss-limit=-1`
  DEFAULT unlimited, `--memcache-size=4194304`, `--fdram-allocator=ffreuse`,
  `--fl2-cache-mode=resident`, `--use-extended-macho-format`, ...).

## Result dictionary

`status=0` + `ModelMaxDramUsage`, `NetworkStatusList[{Name, LiveInputList[{
BatchStride,Batches,Channels,Depth,Height,Width,*Strides,Interleave,
TensorType}], LiveOutputList}]`. Errors: `ErrorList` of
`InvalidMILProgram` / `InvalidNetwork` / `InvalidNetworkSyntax` /
`InvalidCompilationParam` / `InvalidMachOFileName` /
`InvalidJITShapesFileName` / `CompilationFailure` (+ per-network
`NetworkErrorList` e.g. `InvalidUnit`).

## MIL text rules (empirical)

- `program(1.0)`/`func main<ios17>` and `program(1.3)`/`ios18`-`ios19` both
  parse. `name =` attrs are dialect-scoped: `string("n")` under 1.3,
  `tensor<string, []>("n")` under 1.0 — mixing is a parse error.
- `const()`: `val = tensor<T, shape>([...])` or
  `tensor<T, shape>(BLOBFILE(path = tensor<string, []>("@model_path/..."),
   offset = tensor<uint64, []>(N)))`. Bare-scalar `val = fp16(x)` parses
  under 1.3 (coremlc's own serialization) — but tensor form always works.
- `BLOBFILE` path resolves against the mil's **containing dir** —
  `weights/weight.bin` must physically exist or the program is rejected.
- Rank-0 outputs may be spelled `fp16 x` or `tensor<fp16, []> x`.
- Function outputs: fp16/fp32 tensors produced by compute ops. Returning
  int32/bool/string tensors, const outputs, or multi-output tuples
  containing non-float → `InvalidMILProgram`. Returning an input param is
  legal; dead compute ops are tolerated (semantic stage, not parse).
- `state<T>` params + `read_state`/`write_state` work (ios18+);
  `write_state(data=X, input=st)` accepts any tensor X of matching type.
- **State contract (verified):** per state, exactly one `read_state`;
  every `slice_update` on state-derived data must be followed by a
  `write_state` to that state — chained `slice_update`s (su on su output)
  are rejected. Multi-layer kv-cache ⇒ **per-layer states**, each with its
  own read→slice_update→write triple. `write_state` requires a prior
  `read_state` of that state.
- `slice_update`/`slice_by_index` on **state** requires the full arg list
  (`begin_mask`, `end_mask`, `squeeze_mask`, `stride`); the minimal form
  parses on regular tensors but not on state.
- `begin`/`end` must be `const` — dynamic/index-computed slices are
  rejected (`InvalidMILProgram`). Dynamic positions belong to the
  `NetworkJITShapesPath` mechanism (unprobed).
- Zero-extent slices (`begin == end`) → `InvalidMILProgram`.
- Slice bounds are checked at compile: `offset + size > dim` surfaces as
  rc=22 with a real message ("The summation of offset:N and size:M in the
  input view is out-bounding the input batch dimension:K").
- **Non-float tensor compute is rejected**: `add`/`mul`/`cast`/`gather` on
  int32 tensors → `InvalidMILProgram`. int32/bool/string may exist only as
  `const()` attribute args. This frontend accepts the ANE-eligible subset
  only — CoreML performs the device split before this stage.
- `gather` rejected in all spellings tried (dynamic or const indices).
- Apple's own coremlc-produced `model.mil` files (1.3, ios19) generally
  FAIL this frontend — the shipped `model.mil` is pre-lowering MIL;
  ANEC's accepted dialect is the post-split ANE subgraph.

## Op matrix (minimal-program probes, h16g)

PARSE/OK/SEMAN as returned by `ANECValidate` (OK = status 0 + plan).

| op | result | notes |
|---|---|---|
| add mul sub real_div abs minimum maximum pow | OK | elementwise fp16 |
| sqrt exp tanh sigmoid silu clip(alpha,beta) gelu(mode) | OK | neuron family; gelu needs `mode` |
| softmax(axis) | OK | |
| reduce_mean(axes,keep_dims) | OK | reduction |
| rsqrt(epsilon,x) | OK | `epsilon` REQUIRED — bare `rsqrt(x)` → PARSE |
| conv(dilations,groups,pad,pad_type,strides,weight,x) | OK | full arg list required |
| matmul(transpose_x,transpose_y,x,y) | OK | transpose flags required |
| transpose(perm) reshape(shape) concat(axis,interleave,values) | OK | layout |
| tile(reps) slice_by_index(begin,end,...) slice_update(begin,end,update,x) | OK | |
| read_state write_state state<T> | OK | ios18+ |
| cast(dtype,x) fp16→fp32 | OK | int↔fp → PARSE |
| const BLOBFILE | OK | weights file must exist |
| gather | PARSE | all spellings rejected |
| scale_activation | PARSE | not an op |
| int32/bool compute ops | PARSE | non-float tensor math |
| multi-output incl. non-float | PARSE | outputs must be fp tensors |
| slice_update on state (full args + write_state) | OK | the rsw triple |
| chained slice_update / missing write_state | PARSE | |
| dynamic begin/end | PARSE | const indices only |

## Milestone

An 8-layer transformer shard (attention + MLP + per-layer kv-cache states,
750 ops, ~57MB weights) transformed to these rules validates `status=0`
and compiles via `ANECCompile` to a 56,950,784-byte Mach-O —
`ModelMaxDramUsage = 57,547,456` (all weights resident). No CoreML in the
path: `mil text → dict → ANECCompile → macho`.

## ANE layer vocabulary (validator exports)

AffineTransform ArgMinMax BatchToSpace Broadcast ChannelToSpace Concat
Conv CropResize CrossCorrelation CrossProduct Dropout DynamicGOC
DynamicSlice ElementWise Flatten GOC Gather GlobalArgMinMax InputView
InstanceNorm LRN LayerNorm Linear MatrixMult MinMaxNorm NMS Neuron Pad
PixelShuffle PixelUnshuffle Pool Random Reduction Resample Reshape
ResizeAs Resize RingBufferWriter SDPA ScaledElementWise Shape Softmax
Sort SpaceToBatch SpaceToChannel Tile TopK Transpose Unflatten

Espresso-net input bundle (per `NetworkSourcePath`): `Manifest.json`,
`weight.bin`, `init.json`, `validation.json`, `param.bin`, `mac_cfg.bin`,
`anehlo.bin`, `additional_weights.bin`. Outputs: `status.plist`, `ane.bin`,
`info.bin`, `section_symbols.json`, `perf_profile_hint.json`,
`refine.json`, `kernels_{before,after}_splitting.json`,
`_quantization_fusion_log.json`, `_max_shape.plist`,
`{online,offline}_options.plist`, `*.macho` (magic `0xcefaefbe`, segments
`__FVMLIB`, graph symbols like `main`, `z@output`, embedded flag line).

## Toolchain interface

The rules above are executable, not just documented:

- `mil_spec::ane_lower::lower_for_ane(text, pos)` — lowers a CoreML-emitted
  stateful MIL file to this dialect: bakes int32 index math
  (`concat`/`add`/`pos` chains) to consts, stripe-partitions each shared
  `state<tensor<T,[D,..]>>` along dim 0 into per-stripe states, rewrites
  the update chain into per-stripe `read_state → slice_update →
  write_state` triples, rebases `slice_by_index` readers to local
  indices, forces `ios18`, and prunes dead params/consts. Verified to
  produce a byte-equivalent compile result on the 8-layer shard.
- `mil_infer::anec::Anec` — direct `ANECompiler.framework` wrapper:
  `Anec::open()`, `.validate(mil, arch)`, `.compile(mil, dir, arch)` →
  typed `Verdict` (status, errors, `ModelMaxDramUsage`, artifact path).
- `milc ane lower|validate|compile <model.mil> [-o out] [--arch h16g]
  [--pos N]` — CLI for the full path:

      milc ane lower model.mil -o model.ane.mil   # → this dialect
      milc ane compile model.ane.mil -o out/      # → *.macho, no CoreML
