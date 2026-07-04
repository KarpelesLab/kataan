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
#include <stdint.h>
#include <stdbool.h>

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

/*
 * Evaluate `source` with a pre-installed global ArrayBuffer named `buffer`, built
 * as an engine-OWNED copy of the caller's `data` region (`data_len` bytes). After
 * the run, the buffer's (possibly script-mutated) bytes are written back into
 * `data` in place, so the caller observes JS writes made through a view over
 * `buffer` (e.g. `new Uint8Array(buffer)[0] = 9`). The completion value's string
 * is written into `out` per the in/out length convention.
 *
 * Returns KT_OK on success, KT_INVALID_INPUT on a parse error / uncaught throw
 * (`out` holds the message), or the same buffer/pointer/internal codes as kt_eval.
 * If `data_len > 0`, `data` must point to `data_len` readable/writable bytes.
 */
int kt_eval_with_buffer(const char *source, size_t source_len, unsigned char *data,
                        size_t data_len, char *out, size_t *out_len);

/*
 * Like kt_eval_with_buffer, but the global ArrayBuffer `buffer` wraps the caller's
 * `data` region ZERO-COPY: JS writes through a view over `buffer` hit `data` in
 * place and are visible to the caller immediately after the call returns (no copy
 * back). The engine does NOT free the region; `data`/`data_len` must remain a
 * valid, uniquely-owned mutable region for the entire duration of the call.
 *
 * Returns the same codes as kt_eval_with_buffer.
 */
int kt_eval_with_external_buffer(const char *source, size_t source_len, unsigned char *data,
                                 size_t data_len, char *out, size_t *out_len);

/*
 * Compile `source` (UTF-8 JavaScript, `source_len` bytes) to a portable `.ktbc`
 * bytecode artifact, written into `out` (capacity *out_len bytes) per the in/out
 * length convention. Call with *out_len == 0 to query the required size.
 *
 * Returns:
 *   KT_OK               on success (`out` holds the bytecode artifact),
 *   KT_INVALID_INPUT    on a parse/compile error (`out` holds the message) or
 *                       non-UTF-8 input,
 *   KT_BUFFER_TOO_SMALL if `out` was too small (*out_len holds the required size),
 *   KT_NULL_POINTER / KT_INTERNAL as for kt_eval.
 *
 * The artifact pairs with kt_load_bytecode: compile once, run many times.
 */
int kt_compile(const char *source, size_t source_len, char *out, size_t *out_len);

/*
 * Verify and run a `.ktbc` bytecode artifact (`bytecode`, `bytecode_len` bytes),
 * writing its completion value, rendered as a string, into `out` per the in/out
 * length convention. The artifact is verified (untrusted-load safe) before it
 * runs. Returns KT_OK on success, KT_INVALID_INPUT for a corrupt/unverifiable
 * artifact or an uncaught throw (`out` holds the message), or the same
 * buffer/pointer/internal codes as kt_eval.
 */
int kt_load_bytecode(const char *bytecode, size_t bytecode_len, char *out, size_t *out_len);

/*
 * Run `source` and write a D' snapshot of its completion value's object graph
 * into `out` (portable bytes) per the in/out length convention. The completion
 * must be a heap object (object/array/string/...); a primitive completion returns
 * KT_INVALID_INPUT with the message in `out`. The bytes pair with kt_restore and
 * may be persisted or moved across processes on a matching host.
 */
int kt_snapshot(const char *source, size_t source_len, char *out, size_t *out_len);

/*
 * Restore a snapshot written by kt_snapshot into a fresh runtime and write its
 * first root value, rendered as a string, into `out` per the in/out length
 * convention. Returns KT_OK on success, KT_INVALID_INPUT for a malformed snapshot,
 * or the same buffer/pointer/internal codes as kt_eval. (Data graphs restore
 * standalone; a snapshot containing closures must be reloaded by a runtime holding
 * the same code.)
 */
int kt_restore(const char *snapshot, size_t snapshot_len, char *out, size_t *out_len);

/*
 * --- Value layer -----------------------------------------------------------
 *
 * KtValue is an ABI-stable JavaScript value passed by copy (the engine's NaN-box
 * encoding). Construct and inspect it only through these functions; the numeric
 * layout is opaque. These are pure (they touch no engine state). The context and
 * function-registration bridge builds on this layer.
 */
typedef struct { uint64_t bits; } KtValue;

KtValue kt_value_undefined(void);
KtValue kt_value_null(void);
KtValue kt_value_number(double n);
KtValue kt_value_boolean(bool b);

bool   kt_value_is_number(KtValue v);
double kt_value_as_number(KtValue v);
bool   kt_value_is_boolean(KtValue v);
bool   kt_value_as_boolean(KtValue v);
bool   kt_value_is_undefined(KtValue v);
bool   kt_value_is_null(KtValue v);
bool   kt_value_is_object(KtValue v);

#ifdef __cplusplus
}
#endif

#endif /* KATAAN_H */
