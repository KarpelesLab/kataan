#!/bin/bash
# Compare kataan against node on the clean-room microbenchmarks in bench/js.
#
#   bench/run.sh [N] [filter]     best-of-N wall clock per engine (default 3)
#
# Each benchmark prints a checksum; a mismatch with node is reported as WRONG
# (a speedup that changes the answer does not count). Times are whole-process
# wall clock, so they include startup — see the `startup` row.
set -u
cd "$(dirname "$0")/.."
N=${1:-3}
FILTER=${2:-}
KATAAN=${KATAAN:-./target/release/kataan}
NODE=${NODE:-node}

best() { # best-of-N milliseconds for "$@"
  local best=999999999 t0 t1 ms
  for _ in $(seq "$N"); do
    t0=$(date +%s%N)
    "$@" > /dev/null 2>&1
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    (( ms < best )) && best=$ms
  done
  echo "$best"
}

echo 'void 0' > /tmp/kataan-bench-empty.js
printf '| benchmark | kataan ms | node ms | ratio | output |\n|---|---:|---:|---:|---|\n'
for f in /tmp/kataan-bench-empty.js bench/js/*.js; do
  name=$(basename "$f" .js)
  [ "$f" = /tmp/kataan-bench-empty.js ] && name=startup
  [ -n "$FILTER" ] && [[ "$name" != *$FILTER* ]] && continue
  kout=$("$KATAAN" run "$f" 2>&1 | tail -1)
  nout=$("$NODE" "$f" 2>&1 | tail -1)
  ok=ok; [ "$kout" != "$nout" ] && ok="WRONG (kataan: ${kout:0:40} / node: ${nout:0:40})"
  k=$(best "$KATAAN" run "$f")
  n=$(best "$NODE" "$f")
  ratio=$(awk -v k="$k" -v n="$n" 'BEGIN { printf "%.1fx", (n > 0 ? k / n : 0) }')
  printf '| %s | %s | %s | %s | %s |\n' "$name" "$k" "$n" "$ratio" "$ok"
done
