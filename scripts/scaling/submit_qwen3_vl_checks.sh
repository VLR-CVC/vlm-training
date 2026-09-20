#!/bin/bash
# Qwen3-VL on the torchtitan components, one JUPITER node each (run from the repo root):
#   1. transformers parity + vision-only TP + TP/SP/FSDP parity (jup_s3_parity.sbatch)
#   2. 40-step training smoke, TP=4+SP, HF weights (configs/jupiter/qwen3_vl_2b.toml)
#
#   [SMOKE=0] [VISION_TP_CHECK=0] [TP_PARITY=0] scripts/scaling/submit_qwen3_vl_checks.sh [snapshot]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SNAPSHOT="${1:-/e/project1/reformo/ockier1/qwen_models/qwen3_vl_2b}"
VENV=/e/project1/open-sci-mm/ockier1/torchtitan/venv

mkdir -p logs_s3
sbatch --parsable --job-name=vl_parity \
    --export="ALL,QWEN3_5_9B_SNAPSHOT=$SNAPSHOT,HF_PARITY_TEST=models/tests/test_qwen3_vl_tt_parity.py,VISION_TP_CHECK=${VISION_TP_CHECK:-1},TP_PARITY=${TP_PARITY:-1},JUP_VENV=$VENV" \
    scripts/scaling/jup_s3_parity.sbatch

[ "${SMOKE:-1}" = 1 ] || exit 0   # SMOKE=0: parity job only
LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 JUP_VENV="$VENV" SCALING_TIME=00:40:00 \
    SCALING_SWEEP_ID="vl_smoke_$(date +%Y%m%d_%H%M%S)" \
    SCALING_EXTRA_ARGS="--training.total-steps 40 --training.save-steps 100000" \
    SCALING_CONFIG=configs/jupiter/qwen3_vl_2b.toml \
    scripts/scaling/submit_sweep.sh 1
