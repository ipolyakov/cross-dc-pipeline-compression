#!/usr/bin/env bash
# Median iteration time of a Megatron run from its output file, skipping the warm-up.
#
#   ./step_time.sh OUTPUT [SKIP]    SKIP: log lines to skip, default 5 (50 iterations at --log-interval 10)
#   ./step_time.sh $DEV_SDP4BIT_EXPERIMENTS/pp539M/step-500mbit-fp8/output
#
# Prints: median ms, number of log lines used, min and max ms.

set -euo pipefail
OUTPUT=${1:?output file of the run}
SKIP=${2:-5}
grep -o 'elapsed time per iteration (ms): [0-9.]*' "$OUTPUT" | awk '{print $NF}' | tail -n +$((SKIP + 1)) |
    sort -n | awk '{v[NR] = $1} END {
        if (NR == 0) { print "no iteration lines"; exit 1 }
        m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
        printf "median %.1f ms over %d log lines (min %.1f, max %.1f)\n", m, NR, v[1], v[NR] }'
