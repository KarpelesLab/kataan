#!/bin/sh
# Cargo runner: run "$@" under a hard address-space limit (children inherit it
# and cannot raise it). KATAAN_TEST_MAX_MEMORY_GB lowers or raises the default.
ulimit -v $(( ${KATAAN_TEST_MAX_MEMORY_GB:-32} * 1024 * 1024 )) || exit 1
exec "$@"
