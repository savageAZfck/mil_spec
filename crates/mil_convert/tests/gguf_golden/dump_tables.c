#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include <stdio.h>
#define DUMP(name, T, t, n, w, fmt) do { \
    printf("pub const %s: [%s; %d] = [", name, T, n); \
    for (int i = 0; i < n; i++) printf("%s" fmt, i % w ? (i ? "," : "") : (i ? ",\n    " : "\n    "), t[i]); \
    printf("\n];\n\n"); } while (0)
int main(void) {
    DUMP("KMASK_IQ2XS", "u8", kmask_iq2xs, 8, 16, "%d");
    DUMP("KSIGNS_IQ2XS", "u8", ksigns_iq2xs, 128, 16, "%d");
    DUMP("KVALUES_IQ4NL", "i8", kvalues_iq4nl, 16, 16, "%d");
    DUMP("KVALUES_FP4", "i8", kvalues_fp4, 16, 16, "%d");
    DUMP("IQ2XXS_GRID", "u64", iq2xxs_grid, 256, 2, "0x%016llx");
    DUMP("IQ2XS_GRID", "u64", iq2xs_grid, 512, 2, "0x%016llx");
    DUMP("IQ2S_GRID", "u64", iq2s_grid, 1024, 2, "0x%016llx");
    DUMP("IQ3XXS_GRID", "u32", iq3xxs_grid, 256, 4, "0x%08x");
    DUMP("IQ3S_GRID", "u32", iq3s_grid, 512, 4, "0x%08x");
    DUMP("IQ1S_GRID", "u64", iq1s_grid, 2048, 2, "0x%016llx");
    return 0;
}
