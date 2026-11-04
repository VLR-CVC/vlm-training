#!/bin/bash
# One single-node run at the settings a node can actually hold, to read MFU and
# tok/s off a short log. Wraps submit_sweep.sh so the environment (venv layering,
# NCCL, inductor cache keyed on world size) is the one every other job uses.
#
#   [SINGLE_EXTRA='--training.tp-size 1'] scripts/scaling/submit_single_node.sh [config] [steps]
#
# Read the result with:
#   scripts/scaling/collect_probe.sh <run id printed at the end>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${1:-configs/jupiter/qwen3_vl_2b.toml}"
STEPS="${2:-40}"
STAMP="single_$(date +%Y%m%d_%H%M%S)"

# save-steps above total-steps: a perf run should not spend minutes writing a
# checkpoint it is about to delete.
LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 SCALING_TIME=00:40:00 \
    SCALING_SWEEP_ID="$STAMP" \
    SCALING_RUN_NAME="$STAMP" \
    SCALING_EXTRA_ARGS="--training.total-steps $STEPS --training.save-steps 100000 ${SINGLE_EXTRA:-}" \
    SCALING_CONFIG="$CONFIG" scripts/scaling/submit_sweep.sh 1

echo
echo "run id: $STAMP"
