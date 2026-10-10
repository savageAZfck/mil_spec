# GGUF golden vectors

Bit-exact oracle for `src/gguf/dequant.rs`. Each `<type>.q` is
deterministic f32 input quantized by `ggml_quantize_chunk`, and
`<type>.f32` is the same bytes run through the type traits' `to_float`
— both produced by ggml itself, so the Rust port proves equality with
the reference implementation, not with a re-derivation.

Pinned reference: **llama.cpp commit
`10a60cf303566e10d6a7a2774c17d2085503d87b`** (ggml-org/llama.cpp,
shallow clone in `~/.cache/mil_gguf_test/llama.cpp`).

27 vectors — every quant type with a `to_float` at the pin:

    f16 bf16 q1_0 q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 mxfp4 nvfp4
    tq1_0 tq2_0 q2_K q3_K q4_K q5_K q6_K
    iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs

(f32 needs no quantizer — identity — and q8_1/q8_K have no `to_float`.)

## Regenerate

```sh
# 1. pinned sources
git clone --depth 1 https://github.com/ggml-org/llama.cpp \
    ~/.cache/mil_gguf_test/llama.cpp
cd ~/.cache/mil_gguf_test/llama.cpp && git fetch --depth 1 origin \
    10a60cf303566e10d6a7a2774c17d2085503d87b && git checkout FETCH_HEAD

# 2. compile just the two TUs the oracle needs (+ stubs for backend
#    symbols the quantize path never calls — stubs.c is in this dir).
#    -ffp-contract=off keeps FMA out so scalar f32 order is preserved.
cd ~/.cache/mil_gguf_test/ggml_objs
cc -O2 -ffp-contract=off -c \
    ../llama.cpp/ggml/src/ggml.c \
    ../llama.cpp/ggml/src/ggml-quants.c \
    stubs.c -I../llama.cpp/ggml/include -I../llama.cpp/ggml/src
# (ggml.c/ggml-quants.c each compile to ggml.o / ggml-quants.o;
#  stubs.c → stubs.o)

# 3. build the oracle + the table dumper
cc -O2 -ffp-contract=off \
    /path/to/mil_spec/crates/mil_convert/tests/gguf_golden/gen.c \
    ggml.o ggml-quants.o stubs.o -lm -o gen
cc -O2 -ffp-contract=off \
    /path/to/mil_spec/crates/mil_convert/tests/gguf_golden/dump_tables.c \
    ggml.o ggml-quants.o stubs.o -lm -o dump_tables

# 4. regenerate vectors + tables
./gen /path/to/mil_spec/crates/mil_convert/tests/gguf_golden/vectors
./dump_tables > /path/to/mil_spec/crates/mil_convert/src/gguf/tables.rs
# (re-add the MIT header comment at the top of tables.rs)
```

Input generation: LCG-seeded per type, 4 rows × 512 elements — normal
range, all-zero, large-magnitude (×40+3), small (×1e-3). Imatrix for
types that require one: `1.0 + 0.5·|rand|`, separately seeded.
`stubs.c` satisfies `ggml_backend_*`/`ggml_critical_section_*` symbols
that `ggml-quants.c` references but the scalar path never calls.
