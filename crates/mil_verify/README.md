# mil_verify

Inspection, deterministic protobuf diff, conformance battery, and
golden-vector verification for MIL specs — the crate that answers
"is this package actually correct?" rather than "did it compile?".

- `verify_spec` / `verify_block` / `verify_package` — structural
  validation (name resolution, declared outputs, `weight.bin` blob
  bounds) on decoded or pre-emit graphs.
- `diff_specs` / `summarize` — field-by-field spec diff and summary,
  no schema dependency beyond wire field numbers.
- `lint_spec` — run the `mil_lint` dispatch table over a decoded spec.
- `battery` — valid graphs that must pass plus planted-invalid
  controls that must fail; if a control ever passes, the *verifier*
  is broken.
- `reference` — independent pure-Rust f64 transformer forward used as
  the correctness anchor.
- `golden` — versioned golden-vector records and the check that a
  compiled package still agrees with them.

## Golden-vector workflow

The golden path freezes the reference forward's logits for a fixed
prompt into a record, then proves a compiled package reproduces them.

### 1. Generate

```rust
let (sha, desc) = mil_verify::golden::hash_model_source(model_dir)?;
let prov = mil_verify::golden::Provenance::new("qwen3-0.6b", desc, sha, "mil_verify 0.2");
let g = mil_verify::golden::Golden::from_reference(
    &prov, &cfg, &weights, token_ids, /*top-k*/ 32,
)?;
g.write(path)?;
```

The record (`*.mgv`, format v1) stores: model identity + config dims,
SHA-256 of the weight source, the input token ids, per-position
top-`k` `(index, value)` pairs (~`positions·k·8` bytes — 1.6 KB for a
5-token prompt at `k = 32`), the generating toolchain, and a trailing
SHA-256 over the whole record. A `--full` variant keeps the entire
logit matrix when exact vectors are worth the size.

### 2. Check

```rust
let g = Golden::read(path)?;
let cols = golden::columns_from_output(&prediction_values, seq); // (1,vocab,s,1) → per-position
let report = g.check(&cols, &golden::Tolerances::default());
assert!(report.is_ok());
```

The default gate is measured, not guessed (`tests/golden_e2e.rs`
output): fp16 packages on `CpuOnly` drift ≤0.35 absolute at shared
top-k indices from the f64 reference on logits of magnitude ~20
(qwen3-0.6B worst 0.3402, smollm2-135M worst 0.2218), and a planted
one-byte `weight.bin` corruption produced 0.6335 — so
`max_val_delta = 0.5` sits between healthy and corrupt with ~1.5×
headroom each way. Also gated: per-position top-1 equality, top-k
overlap ≥ 5, and zero NaNs. The corruption control must fail — if it
passed, the check would be decorative.

### 3. CPU vs default units

`golden::diff_columns` reports per-position `max|Δ|`, `mean|Δ|`,
cosine, and both argmaxes between two runs of the same package. The
e2e asserts top-1 agreement between `ComputeUnits::CpuOnly` and
`ComputeUnits::All` — the ANE adds its own fp16 rounding on top of
the same graph, but greedy decoding must agree.

### CLI harness

The `golden` example is the runnable form (milc wiring is a separate
component):

```sh
cargo run -p mil_verify --release --example golden -- \
    gen ~/.cache/mil_gguf_test/models/qwen3-hf 791,6346,310,8625,374 qwen3.mgv
cargo run -p mil_verify --release --example golden -- \
    check out.mlmodelc qwen3.mgv ~/.cache/mil_gguf_test/models/qwen3-hf --units cpu
cargo run -p mil_verify --release --example golden -- \
    diff out.mlmodelc ~/.cache/mil_gguf_test/models/qwen3-hf 791,6346,310,8625,374
```

`check` exits nonzero on any tolerance violation and prints the
per-position report either way.

### Tests

| file | needs | gate |
|---|---|---|
| `src/golden.rs` unit tests | nothing | record roundtrip, digest tamper, check verdicts, sha256 vectors |
| `tests/reference_selfcheck.rs` | nothing (in-memory weights) | reference forward vs closed-form and independent f64 forwards on a 2-layer tiny model |
| `tests/golden_e2e.rs` (`--ignored`) | cached models + `coremlc` | full flow on qwen3-0.6B + smollm2-135M, CPU-vs-default diff, corruption control |
| `tests/reference_e2e.rs` (`--ignored`) | cached models + `coremlc` | reference vs package, GGUF isolation, anchors |

```sh
cargo test -p mil_verify                                        # fast suite
cargo test -p mil_verify --test golden_e2e --release -- --ignored --nocapture
```
