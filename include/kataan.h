/*
 * kataan.h — C ABI for the Kataan JavaScript engine.
 *
 * Build the library with one of:
 *   cargo rustc --lib --release --features ffi --crate-type staticlib
 *   cargo rustc --lib --release --features ffi --crate-type cdylib
 *
 * Conventions (shared with the sibling purecrypto C ABI):
 *   - Fallible functions return a KtStatus (0 = success, negative = error).
 *   - Variable-length output uses the in/out length convention: pass a buffer
 *     and a *len holding its capacity; on return *len is the actual (or, on
 *     KT_BUFFER_TOO_SMALL, the required) length. Call with *len == 0 to query.
 *   - Opaque handles are created and freed by the library.
 *
 * This header tracks the current surface (version + status codes + one-shot
 * eval). Persistent context/value handles are added as the engine grows; see
 * ROADMAP.md.
 */
#ifndef KATAAN_H
#define KATAAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. The numeric values are part of the ABI. */
typedef enum {
    KT_OK = 0,
    KT_NULL_POINTER = -1,
    KT_BUFFER_TOO_SMALL = -2,
    KT_INVALID_INPUT = -3,
    KT_INTERNAL = -100
} KtStatus;

/*
 * Returns the engine version as a static NUL-terminated string. The pointer is
 * valid for the lifetime of the program and must not be freed.
 */
const char *kt_version(void);

/*
 * Copies the engine version into `buf` (capacity *len bytes), writing the
 * number of bytes used (excluding any NUL) back into *len. Returns:
 *   KT_OK              on success,
 *   KT_BUFFER_TOO_SMALL if *len was too small (*len holds the required size),
 *   KT_NULL_POINTER     if `len` (or `buf` when needed) is NULL.
 * Call with *len == 0 to query the required length.
 */
int kt_version_copy(char *buf, size_t *len);

/*
 * Evaluates a JavaScript source string (`source`, `source_len` bytes of UTF-8;
 * a NUL terminator is not required) and writes its result, rendered as a
 * string, into `out` (capacity *out_len bytes) following the in/out length
 * convention. Call with *out_len == 0 to query the required length.
 *
 * Returns:
 *   KT_OK               on success (`out` holds the completion value),
 *   KT_INVALID_INPUT    on a parse error or an uncaught throw (`out` holds the
 *                       error message), or non-UTF-8 input,
 *   KT_BUFFER_TOO_SMALL if `out` was too small (*out_len holds the required size),
 *   KT_NULL_POINTER     if `out_len` (or `source`/`out` when needed) is NULL,
 *   KT_INTERNAL         if the engine panicked (caught at the boundary).
 *
 * Each call uses a fresh engine with the standard globals (no console); it does
 * not persist state between calls.
 */
int kt_eval(const char *source, size_t source_len, char *out, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* KATAAN_H */
