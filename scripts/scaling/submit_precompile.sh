#!/bin/bash
# Does the synthetic pre-compile collapse the warmup? (WARMUP_PLAN.md, Option A)
#
#   scripts/scaling/submit_precompile.sh          # 1 node, 300 steps
#   NODES=32 STEPS=500 scripts/scaling/submit_precompile.sh
#   PRECOMPILE=0 scripts/scaling/submit_precompile.sh   # control, same script
#
# Baselines, same shape (SHARD=0, TP=2, seq 32768):
#   1 node,  300 steps: job 1894616
#   32 node, 500 steps: job 1897238 -- warmup 416, 12,147 tok/s/GPU
#
# Success is `warmup ends` in collect_steady.py dropping toward single digits, and
# `scratchpad/cid_timeline.py` showing spread ~0s for every compile_id. It should
# NOT go to zero: rows outside the spec range still recompile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NODES="${NODES:-1}"
STEPS="${STEPS:-300}"
ON="${PRECOMPILE:-1}"
[ "$ON" = 1 ] && FLAG="--training.precompile" || FLAG=""
STAMP="${PRECOMPILE_ID:-precompile${ON}_${NODES}n_$(date +%Y%m%d_%H%M%S)}"

TRACE_ROOT="${TRACE_ROOT:-/e/scratch/open-sci-mm/ockier1}"
export QWEN_TORCH_LOGS_DIR="$TRACE_ROOT/torchlogs/precompile"
export QWEN_TORCH_TRACE_DIR="$TRACE_ROOT/torchtrace/precompile"
export TORCH_LOGS="recompiles"

SHARD=0 TP=2 STEPS="$STEPS" \
HSDP_CONFIG=configs/jupiter/qwen3_vl_2b.toml \
SCALING_TIME="${SCALING_TIME:-01:00:00}" \
HSDP_SWEEP_ID="$STAMP" \
HSDP_EXTRA="$FLAG" \
    scripts/scaling/submit_hsdp_sweep.sh "$NODES"

echo
echo "collect: python scripts/scaling/collect_steady.py logs_scaling/$STAMP"
