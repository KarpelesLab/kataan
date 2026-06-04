/*
 * ffi_smoke.c — minimal C-ABI smoke test for the Kataan engine.
 *
 * Built and run by CI (see .github/workflows/ci.yml). Exercises the Phase-A
 * C surface: the version accessor and the in/out length-convention copy.
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

    printf("ffi_smoke: all checks passed\n");
    return 0;
}
