# Threat model — mil_spec

mil_spec is a writer, not a runner — the threat model is about what
malformed output could do downstream, and who catches it.

**Encoder emits malformed protobuf.** Apple's loader parses the spec
bytes with real protobuf — a wire-format bug produces a parse failure in
`coremlc` or CoreML, not silent misbehavior. The platform is the
verifier; this crate never runs what it writes.

**Invalid graph semantics reach the platform.** The IR front-end
(`ir::compile`) type-checks before encoding — unknown ops, wrong arity,
and shape/dtype mismatches are `Err`, not emitted bytes. Bypassing the
IR and hand-building a `Block` is the escape hatch: op-level builders
trust caller-supplied shapes, so invalid graphs built by hand fail at
`coremlc` or runtime, not in mil_spec.

**Weight blob corruption.** `weight.bin` offsets and lengths are
emitted per-tensor; a mismatch produces a load failure or wrong
numerics in CoreML, not a vulnerability in this crate — there is no
parser here for an attacker to feed.

**Adversarial input.** There is none in the usual sense: the crate is a
library given graphs by its caller. A hostile *graph* produces hostile
*models* — the sandbox is CoreML's own execution environment, which is
the same trust boundary every .mlpackage consumer already relies on.

**Unicode/path handling in output.** Package contents are written to a
caller-chosen directory; mil_spec writes fixed relative names and does
not traverse or follow caller paths beyond the root given.

## What this crate guarantees

- Deterministic, byte-stable output for a given graph.
- IR-checked graphs cannot encode invalid programs — validation refuses
  before bytes exist.
- The emitted protobuf, weight blob, and manifest formats match Apple's
  documented schemas; `coremlc` acceptance is the conformance bar.

## What it does not guarantee

- Hand-built `Block` graphs bypass IR validation — wrong shapes there
  are caught by the platform, not here.
- Semantic correctness of the model: mil_spec writes the graph you
  describe; whether that graph computes what you intended is the
  caller's problem.
- Op coverage is the subset a stateful LLM graph needs (see README) —
  ops outside the emitters must be written via `Block::op` raw binding.
