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

## Fuzzer findings (M4 / this build, `milc ane fuzz`)

Differential probing with the fuzzer's generator refined the matrix on
this machine's ANECompiler build:

- **`const` + `BLOBFILE` → `InvalidMILProgram` in every spelling**
  (`@model_path/...`, absolute path, relative `weights/weight.bin`,
  `tensor<string, []>`-wrapped path) — including `assets/chain/c64k1.mil`,
  a corpus file that produced byte-identical HWX. Embedded weight blobs
  do not survive this frontend build; weights need the
  `FileConstants`/`ExternConstants` dict lane (schema unprobed). Inline
  fp16 list consts are accepted fine.
- **`gelu` rejected in all spellings** (`mode` first/last, bare `string`
  or `tensor<string, []>` const) despite the matrix listing it OK.
- **`matmul` requires exactly one transpose flag** — `tx == ty` (both
  set or both clear) → `InvalidMILProgram`.
- **`clip` bounds must be fp16 consts** — int32 alpha/beta → reject.
- Function name is unchecked: `func frob<ios18>` validates.
- `const(` missing its closing paren is tolerated.
- Op arg order is free; dead consts are tolerated; `[1,4,1,1]`-scale
  shapes pass; bare `int32`/`bool`/`string`/`fp16` and `tensor<T, []>`
  scalar const spellings are interchangeable.
- 500-case run (`--compile`, seed 31337): 336 accepts, 160 correct
  rejects, 4 parser-laxity accepts, **0 crashes, 0 unexpected rejects**.

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

## Execution (e5rt)

`mil_infer::e5rt` executes compiled programs on the ANE via
`Espresso.framework`'s `e5rt_*` C API — `dlopen`+`dlsym` only, no
entitlement. `e5rt_e5_compiler_compile` sends the MIL through `aned`'s
full MIL→MLIR→LLIR→ANECIR→HWX pipeline (signs + caches per-PID); each
eval is bind→encode→`execute_sync` on the hardware lane.

Ports self-describe: `get_{input,output,inout}_names` +
`io_port_retain_tensor_desc` give names and element counts — `state<>`
params surface as *inout* ports, resident and written in-place across
calls. `Program::compile_auto` discovers and binds everything.

Verified on the 8-layer stateful transformer: ~1.1 ms/step on h16g,
all 8 `kv_*` states write back exactly their [2,3,5,64] update regions,
output fully populated. `milc ane run <model.mil> [--steps N]` drives it.

      milc ane run model.ane.mil --steps 3   # compile → bind → execute

## Perf-stats ABI (runtime contract — reverse engineered)

The daemon-side counter lane, mapped via `dyld_info -exports`, runtime
method introspection (`class_copyMethodList` + `method_getTypeEncoding`),
and staged call probes on the h16g M4:

- **Load side**: `kANEFPerformanceStatsMask` (no `Key` suffix on the
  option string) already rides `loadModel:options:` at `0x7f` — that
  mask is what makes `aned`'s `_ANEMachoPatcher` deploy the program at
  all. `kANEFModelLoadPerformanceStats` (also no suffix) asks the model
  to publish the procedure→stats-size map.
- **The map**: `kANEFModelProcedureNameToStatsSizeMapKey` lives inside
  `modelAttributes → ANEFModelDescription` — procedure name → required
  stats buffer bytes. **Absent on both our emitted binaries and
  ANEC-compiled ones** — the map is produced only for programs built
  with stats instrumentation, and `ProgramCreate stats buffer size:0`
  (ANEServices log) confirms none of our binaries carry it.
- **`_ANEPerformanceStats`** (framework export): the decoded-stats
  object. `statsWithRequestPerformanceBuffer:statsBufferSize:` has
  encoding `@32@0:8^^v16^I24` — `*buf` must point at a **valid
  IOSurfaceRef** (not malloc, not nil — both fault), `*size` at its
  byte extent. Factory wraps it and returns the stats object.
- **`_ANEPerformanceStatsIOSurface`**: sibling variant with
  `initWithIOSurface:statType:` (`@32@0:8@16q24`) — `statType=0x7f`
  instantiates cleanly; `stats` property unwraps to the decode object.
- **Request side**: `initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`
  accepts the stats object in the `perfStats:` slot; request builds,
  `perfStats`/`perfStatsArray` queryable.
- **Counter vocabulary** (exports): `hwExecutionTime`,
  `performanceCounters`, `perfCounterData`, `pStatsRawData`, plus
  per-counter keys `kANEPerfStatsCounter*` — `NEComputeCycles`,
  `Fp16Cycles`/`Int8Cycles`, `L2*StallCycles`, `DMARead*Bytes`,
  `DPEEnergy`, event types (`StartExecution`, `TaskSwitch`,
  `ContextSwitchOut`, `FetchDone`…) — a full per-task trace format.
- **The wall**: `-[request validate]` and
  `evaluateWithModel:options:request:qos:error:` both raise ObjC
  exceptions when a stats buffer is attached to a program whose
  `stats buffer size` is 0 — i.e. instrumentation is a **compile-time
  artifact**, not a runtime flag.
- **Dead ends verified**: `ANEC_DEBUG_MASK` alters output bytes but not
  stats descriptors; `EnableProfiling`/`EnableSummaryPerformanceStats`/
  `ForcePerfTracerRegisters`/`perfStatsMask` in params or input dict →
  byte-identical binaries (unknown keys ignored silently); e5rt
  `set_enable_profiling` plumbed (`E5_PROFILE=1`) but our test MIL
  segments to BNNS; `initWithRestrictedAccessAllowed:` client is
  refused by aned (`loadModel failed: unknown`) — entitlement-gated.
- **Open**: the compile-time switch that bakes `PerfTracer1Config`/
  `PerfTracer2Config` register programs + `perf_param` runtime params
  into `__kern_0` descriptors (ANECompiler strings show the machinery:
  `RunPerfTracerInitialization()`, "Insert instrumentation to enable
  performance recording"). Once identified, our emitter can produce
  stats-capable HWX and the whole lane lights up.

Working today:

- `EspressoProgram::bench(proc, steps)` — submit→complete latency over
  the XPC round trip. On self-emitted HWX (c64k1, 1 op): **min 44.8µs /
  mean 71.5µs / p99 140µs**, ~0.46 GB/s effective I/O.
  `ane_perf_probe --bench N <hwx>`.
- **Session telemetry** (`sessionHintWithModel:hint:options:report:error:`
  — enc `B56@0:8@16@24@32@40^@48`, report is a plain object arg the
  daemon *mutates*, must be NSMutableDictionary): wrap evals in
  `ANEHintClientSessionStart`/`Stop`/`Info` hints; options select
  report keys — `ANEClientSessionStatus`, `ANEClientTotalPages`,
  `ANEClientResidentPages` — and the daemon fills real numbers
  (verified: status 2, 1 total page, 4 resident pages around a live
  eval). Entitlement-free memory-residency telemetry.
- Export-name → string-value gotcha: `kANEFHint*` globals resolve via
  `dlsym`+deref — e.g. `kANEFHintSessionStart` →
  `"ANEHintClientSessionStart"`. Passing symbol names as strings fails
  silently (`ok=false`, nil error). `kANEFModelLoadPerformanceStats`
  has *no* export — it's an option-string constant only.

## Stateful emission decode (hwx backend)

Verified grammar (ANECCompile + ANECValidate, ios18 only — ios19 rejects
state params):

- `write_state(data=X, input=st)` accepts **only** state-derived X:
  the `read_state` result or a `slice_update`/`slice_by_index` chain on
  it. Writing an arbitrary computed tensor → `InvalidMILProgram`.
- `slice_update` on state requires the FULL arg list
  (`begin`,`begin_mask`,`end`,`end_mask`,`squeeze_mask`,`stride`,
  `update`,`x`); minimal form is rejected. `update` may be a const blob
  or runtime input.

Mach-O structure for `state<>` params (diffed vs stateless):

- One `__FVMLIB.__data` section per state, sized exactly to the dense
  tensor bytes (no 64B row alignment — states pack stride-2, e.g.
  [2,3,5,64] desc `ar1;0;2 → n=3 → c=5 → h=64` dense row-major).
- State sections come FIRST in bind order; io symbols shift (cur is
  symtab section 1 when it's the first state).
- `__text` gains a state-copy record: instruction id `0x9365` (4-D
  states) / `0x9361` (3-D), family `0x93xx` vs compute `0x9541`.
  Field map so far (passthrough corpus [4,16,32]/[8,4,32]/[4,16,64]):
  words 14-16 and 18-20 = dims reversed (innermost at w14); leading-1
  dims are dropped entirely ([1,4,16,32] ≡ [4,16,32] byte-identical);
  record grows by ~4 words when a dim crosses an encoding threshold
  (word 12 = 0x50001 long-form vs 0x40001).
- Espresso `modelAtURL` loads stateful artifacts directly: state ports
  surface as inouts (hidden from input_descs/output_descs), program
  runs — verified: `st_compile`-compiled rsw triple loads + evals.
