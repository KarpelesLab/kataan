#!/bin/bash
# Compare kataan against node on the larger mixed-workload programs in bench/js2
# (ray tracer, interpreter, solver, simulation, ...). Same options as run.sh:
#
#   bench/run2.sh [N] [filter]    best-of-N wall clock per engine (default 3)
#
# Every program prints a deterministic checksum; the whole output (not just the
# last line) must match node's or the row is marked WRONG.
exec env BENCH_DIR="${BENCH_DIR:-bench/js2}" OUT_LINES="${OUT_LINES:-100}" "$(dirname "$0")/run.sh" "$@"
