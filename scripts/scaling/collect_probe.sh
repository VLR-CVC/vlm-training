#!/bin/bash
# Peak reserved memory and step time per micro-batch size, from the probe jobs.
#   scripts/scaling/collect_probe.sh probe_20260918_101500
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
prefix="${1:?usage: collect_probe.sh <sweep prefix>}"
printf '%-10s %-10s %-9s %-9s %s\n' tokens state peak_GiB step_s note
for d in logs_scaling/${prefix}_t*/1n; do
    t="${d%/1n}"; t="${t##*_t}"
    job=$(awk '{print $1}' "$(dirname "$d")/jobs.txt" 2>/dev/null | head -1)
    state=$(sacct -j "${job:-0}" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    steps=$(sed 's/\x1b\[[0-9;]*m//g' "$d/log.out" 2>/dev/null | grep -E " - [0-9]+ - loss" || true)
    peak=$(echo "$steps" | grep -oE "mem [0-9.]+/[0-9.]+G" | cut -d/ -f2 | tr -d G | sort -n | tail -1)
    step=$(echo "$steps" | tail -n +5 | grep -oE "time [0-9.]+s" | awk '{s+=$2; n++} END{if(n) printf "%.3f", s/n}')
    note=$(grep -lE "OutOfMemoryError" "$d"/errors/*.err 2>/dev/null | head -1 | xargs -r -I{} echo OOM)
    printf '%-10s %-10s %-9s %-9s %s\n' "$t" "${state:-?}" "${peak:--}" "${step:--}" "$note"
done
