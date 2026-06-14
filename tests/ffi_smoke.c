/*
 * ffi_smoke.c — minimal C-ABI smoke test for the Kataan engine.
 *
 * Built and run by CI (see .github/workflows/ci.yml). Exercises the C surface:
 * the version accessor, the in/out length-convention copy, and kt_eval.
 *
 *   cargo rustc --lib --release --features ffi --crate-type staticlib
 *   cc tests/ffi_smoke.c -I include target/release/libkataan.a \
 *      -lpthread -ldl -lm -o ffi_smoke && ./ffi_smoke
 */
#include "kataan.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    /* kt_version() returns a non-empty static C string. */
    const char *v = kt_version();
    if (v == NULL || v[0] == '\0') {
        fprintf(stderr, "FAIL: kt_version returned empty\n");
        return 1;
    }
    printf("kataan version: %s\n", v);

    /* Length query: *len == 0 must report the required size. */
    size_t len = 0;
    int rc = kt_version_copy(NULL, &len);
    if (rc != KT_BUFFER_TOO_SMALL) {
        fprintf(stderr, "FAIL: length query returned %d, want %d\n", rc, KT_BUFFER_TOO_SMALL);
        return 1;
    }
    if (len != strlen(v)) {
        fprintf(stderr, "FAIL: queried length %zu != strlen %zu\n", len, strlen(v));
        return 1;
    }

    /* Copy into an adequately sized buffer. */
    char buf[64];
    size_t cap = sizeof(buf);
    rc = kt_version_copy(buf, &cap);
    if (rc != KT_OK) {
        fprintf(stderr, "FAIL: copy returned %d\n", rc);
        return 1;
    }
    if (cap != strlen(v) || strncmp(buf, v, cap) != 0) {
        fprintf(stderr, "FAIL: copied version mismatch\n");
        return 1;
    }

    /* NULL len pointer must be rejected. */
    if (kt_version_copy(buf, NULL) != KT_NULL_POINTER) {
        fprintf(stderr, "FAIL: NULL len not rejected\n");
        return 1;
    }

    /* kt_eval runs JavaScript and returns the result as a string. */
    const char *src = "const sq = x => x * x; sq(8) + [1,2,3].length";
    char out[128];
    size_t out_len = sizeof(out);
    rc = kt_eval(src, strlen(src), out, &out_len);
    if (rc != KT_OK) {
        fprintf(stderr, "FAIL: kt_eval returned %d\n", rc);
        return 1;
    }
    out[out_len] = '\0';
    if (strcmp(out, "67") != 0) { /* 64 + 3 */
        fprintf(stderr, "FAIL: kt_eval result '%s', want '67'\n", out);
        return 1;
    }
    printf("kt_eval(\"sq(8) + [1,2,3].length\") = %s\n", out);

    /* An uncaught throw is reported as KT_INVALID_INPUT with the message. */
    const char *bad = "null.oops";
    out_len = sizeof(out);
    rc = kt_eval(bad, strlen(bad), out, &out_len);
    if (rc != KT_INVALID_INPUT) {
        fprintf(stderr, "FAIL: throwing script returned %d, want %d\n", rc, KT_INVALID_INPUT);
        return 1;
    }

    /* kt_compile produces a portable bytecode artifact... */
    const char *prog = "function dbl(n) { return n + n; } dbl(33)";
    unsigned char art[512];
    size_t art_len = sizeof(art);
    rc = kt_compile(prog, strlen(prog), (char *)art, &art_len);
    if (rc != KT_OK) {
        fprintf(stderr, "FAIL: kt_compile returned %d\n", rc);
        return 1;
    }
    /* ...which kt_load_bytecode verifies and runs back to the same result. */
    out_len = sizeof(out);
    rc = kt_load_bytecode((const char *)art, art_len, out, &out_len);
    if (rc != KT_OK) {
        fprintf(stderr, "FAIL: kt_load_bytecode returned %d\n", rc);
        return 1;
    }
    out[out_len] = '\0';
    if (strcmp(out, "66") != 0) {
        fprintf(stderr, "FAIL: kt_load_bytecode result '%s', want '66'\n", out);
        return 1;
    }
    printf("kt_compile + kt_load_bytecode(\"dbl(33)\") = %s (%zu-byte artifact)\n", out, art_len);

    /* A corrupt artifact is refused (verifier / decoder), not run. */
    art[art_len - 1] ^= 0xff;
    out_len = sizeof(out);
    rc = kt_load_bytecode((const char *)art, art_len, out, &out_len);
    if (rc != KT_INVALID_INPUT) {
        fprintf(stderr, "FAIL: corrupt artifact returned %d, want %d\n", rc, KT_INVALID_INPUT);
        return 1;
    }

    /* kt_eval_with_buffer: a global ArrayBuffer from owned bytes; JS reads the
     * seeded bytes and mutates one, written back into our region (round-trip). */
    unsigned char region[4] = {1, 2, 3, 4};
    const char *bsrc =
        "const v = new Uint8Array(buffer); const s = v[0]+v[1]+v[2]+v[3]; v[0] = 200; s";
    out_len = sizeof(out);
    rc = kt_eval_with_buffer(bsrc, strlen(bsrc), region, sizeof(region), out, &out_len);
    out[out_len] = '\0';
    if (rc != KT_OK || strcmp(out, "10") != 0 || region[0] != 200) {
        fprintf(stderr, "FAIL: kt_eval_with_buffer rc=%d out='%s' region[0]=%d\n",
                rc, out, region[0]);
        return 1;
    }
    printf("kt_eval_with_buffer: sum=%s, region[0] now %d\n", out, region[0]);

    /* kt_eval_with_external_buffer: zero-copy over our region; the JS write hits
     * our bytes in place (no copy back), observed right after the call. */
    unsigned char ext[8] = {9, 0, 0, 0, 0, 0, 0, 0};
    const char *esrc = "const v = new Uint8Array(buffer); const seen = v[0]; v[5] = 77; seen";
    out_len = sizeof(out);
    rc = kt_eval_with_external_buffer(esrc, strlen(esrc), ext, sizeof(ext), out, &out_len);
    out[out_len] = '\0';
    if (rc != KT_OK || strcmp(out, "9") != 0 || ext[5] != 77) {
        fprintf(stderr, "FAIL: kt_eval_with_external_buffer rc=%d out='%s' ext[5]=%d\n",
                rc, out, ext[5]);
        return 1;
    }
    printf("kt_eval_with_external_buffer: seen=%s, ext[5] now %d (zero-copy)\n", out, ext[5]);

    printf("ffi_smoke: all checks passed\n");
    return 0;
}
