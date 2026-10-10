// Link stubs for ggml internals the quantize/dequantize paths never call.
#include "ggml-backend.h"
void ggml_critical_section_start(void) {}
void ggml_critical_section_end(void) {}
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(struct ggml_backend_buffer * b) { (void)b; return GGML_BACKEND_BUFFER_USAGE_ANY; }
void ggml_backend_tensor_set(struct ggml_tensor * t, const void * d, size_t o, size_t s) { (void)t;(void)d;(void)o;(void)s; }
void ggml_backend_tensor_memset(struct ggml_tensor * t, uint8_t v, size_t o, size_t s) { (void)t;(void)v;(void)o;(void)s; }
