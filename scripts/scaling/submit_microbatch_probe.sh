#!/bin/bash
# Largest micro-batch that fits one node, before a scaling sweep. The micro-batch is
# ONE row, so this sweeps the row length (`seq_len`, with `tokens_per_microbatch`
# following it) and reads the peak reserved memory out of each log. Run from the repo
# root on JUPITER.
#
#   [PROBE_EXTRA='--training.data-parallel ddp'] scripts/scaling/submit_microbatch_probe.sh [config] [seq_len...]
#
# Read the results with:
#   scripts/scaling/collect_probe.sh <sweep prefix>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${1:-configs/jupiter/qwen3_vl_2b.toml}"
shift || true
TOKENS=("${@:-12288 16384 20480 24576}")
[ $# -gt 0 ] && TOKENS=("$@")
STAMP="probe_$(date +%Y%m%d_%H%M%S)"

for t in ${TOKENS[@]}; do
    LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 SCALING_TIME=00:25:00 \
        SCALING_SWEEP_ID="${STAMP}_t${t}" \
        SCALING_EXTRA_ARGS="--data.seq-len $t --data.tokens-per-microbatch $t --training.total-steps 12 --training.save-steps 100000 ${PROBE_EXTRA:-}" \
        SCALING_CONFIG="$CONFIG" scripts/scaling/submit_sweep.sh 1
done
echo
echo "sweep prefix: $STAMP"
