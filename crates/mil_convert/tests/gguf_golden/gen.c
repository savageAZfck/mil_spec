// Golden dequantize oracle: deterministic f32 input → ggml_quantize_chunk →
// trait to_float → expected f32. Emits <type>.q (quantized bytes) and
// <type>.f32 (reference dequantized floats) into vectors/ so the Rust port
// can prove bit-exact equivalence.
//
// Build (see README.md): links ggml.c + ggml-quants.c compiled from the
// pinned llama.cpp commit with -ffp-contract=off.
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NROWS 4
#define NPER 512

static float frand(unsigned *s) {
    // deterministic LCG → f32 in [-1, 1]
    *s = *s * 1664525u + 1013904223u;
    return ((float)(*s >> 9) / (float)(1u << 22)) * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    const char *outdir = argc > 1 ? argv[1] : "vectors";

    enum ggml_type types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
        GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };
    const int ntypes = (int)(sizeof(types) / sizeof(types[0]));

    for (int ti = 0; ti < ntypes; ti++) {
        enum ggml_type t = types[ti];
        const struct ggml_type_traits *tr = ggml_get_type_traits(t);
        if (!tr->to_float) {
            fprintf(stderr, "skip %s: no to_float\n", tr->type_name);
            continue;
        }
        if (NPER % tr->blck_size) {
            fprintf(stderr, "skip %s: row %d not divisible by blck %lld\n",
                    tr->type_name, NPER, (long long)tr->blck_size);
            continue;
        }

        // deterministic input rows:
        //   row 0: mixed normal-range values
        //   row 1: all zeros
        //   row 2: large magnitude
        //   row 3: small magnitude
        float *src = malloc(NROWS * NPER * sizeof(float));
        unsigned seed = 0x9e3779b9u + (unsigned)t;
        for (int r = 0; r < NROWS; r++) {
            for (int i = 0; i < NPER; i++) {
                float v = frand(&seed);
                if (r == 1) v = 0.0f;
                if (r == 2) v = v * 40.0f + 3.0f;
                if (r == 3) v = v * 1e-3f;
                src[r * NPER + i] = v;
            }
        }

        float *imatrix = NULL;
        if (ggml_quantize_requires_imatrix(t)) {
            imatrix = malloc(NROWS * NPER * sizeof(float));
            unsigned iseed = 0xdeadbeefu + (unsigned)t;
            for (int i = 0; i < NROWS * NPER; i++) {
                imatrix[i] = 1.0f + 0.5f * fabsf(frand(&iseed));
            }
        }

        size_t row_bytes = ggml_row_size(t, NPER);
        void *dst = calloc(NROWS, row_bytes);
        size_t written = ggml_quantize_chunk(t, src, dst, 0, NROWS, NPER, imatrix);
        if (written != NROWS * row_bytes) {
            fprintf(stderr, "%s: quantize wrote %zu expected %zu\n",
                    tr->type_name, written, NROWS * row_bytes);
            return 2;
        }

        float *deq = malloc(NROWS * NPER * sizeof(float));
        tr->to_float(dst, deq, (int64_t)NROWS * NPER);

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.q", outdir, tr->type_name);
        FILE *fq = fopen(path, "wb");
        fwrite(dst, 1, NROWS * row_bytes, fq);
        fclose(fq);
        snprintf(path, sizeof(path), "%s/%s.f32", outdir, tr->type_name);
        FILE *ff = fopen(path, "wb");
        fwrite(deq, sizeof(float), NROWS * NPER, ff);
        fclose(ff);

        printf("%-8s row_bytes=%zu ok\n", tr->type_name, row_bytes);
        free(imatrix);
        free(dst);
        free(deq);
        free(src);
    }
    return 0;
}
