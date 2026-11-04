#!/bin/bash
# Gate run for the MoE arms: torchtitan's Qwen3.5 MoE debug model, 1 node.
# Run on the JUPITER login node.
#
#   scripts/scaling/submit_tt_moe_debug.sh
#   STEPS=40 scripts/scaling/submit_tt_moe_debug.sh --compile.enable
#
# Everything after the script name is forwarded to torchtitan.train.
#
# No dataset work: `qwen35_debugmodel*` reads the bundled
# tests/assets/cc12m_test tar, which is why phase 0 (job 1810612) ran offline.
set -euo pipefail

TT=${TT:-/e/project1/open-sci-mm/ockier1/torchtitan}
mkdir -p "$TT/logs"

# `--training.disable-cuda-graphs` is set by this config itself
# (TORCHTITAN_BENCHMARK.md section 8), unlike qwen35_debugmodel. Passed anyway:
# it costs nothing and the run is a gate, not a measurement.
cd "$TT/repo"
TT_MODULE=qwen3_5 TT_CONFIG="${TT_CONFIG:-qwen35_debugmodel_moe}" \
sbatch --nodes=1 --time="${TIME:-00:20:00}" \
    --output="$TT/logs/moe_debug_%j.out" \
    --error="$TT/logs/moe_debug_%j.err" \
    "$TT/tt_bench.sbatch" \
        --training.disable-cuda-graphs \
        --training.steps "${STEPS:-10}" \
        --metrics.log-freq 1 \
        "$@"
