#!/bin/bash
# Full parity gate for BOTH models, one JUPITER node each (run from the repo root
# on JUPITER). Each job runs, on the model's own snapshot:
#
#   1. transformers parity, single GPU  (test_qwen3_{5,vl}_parity.py)
#   2. vision-only TP=2                 (every vision grad exactly 2x TP=1)
#   3. TP parity: tp1, tp2, tp2sp, dp2tp2sp, tp4sp   (test_tp_parity.py)
#   4. the same again with COMPILE=1
#
#   scripts/scaling/submit_parity_both.sh
#   ONLY=qwen3_5 scripts/scaling/submit_parity_both.sh    # one model
#   ONLY=qwen3_vl scripts/scaling/submit_parity_both.sh
#
# Prints the two job ids. Results land in logs_s3/<jobid>.{out,err}; the per-run
# gradient dumps go to $TP_PARITY_OUT (see jup_s3_parity.sbatch).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

VENV="${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}"
SNAP_ROOT="${SNAP_ROOT:-/e/project1/reformo/ockier1/qwen_models}"
QWEN3_5_SNAP="${QWEN3_5_SNAP:-$SNAP_ROOT/qwen3_5_9b}"
QWEN3_VL_SNAP="${QWEN3_VL_SNAP:-$SNAP_ROOT/qwen3_vl_2b}"

mkdir -p logs_s3

submit() {  # submit <name> <snapshot> <hf parity test>
    [ -d "$2" ] || { echo "no snapshot: $2" >&2; exit 1; }
    sbatch --parsable --job-name="parity_$1" \
        --export="ALL,PARITY_SNAPSHOT=$2,HF_PARITY_TEST=$3,VISION_TP_CHECK=${VISION_TP_CHECK:-1},TP_PARITY=${TP_PARITY:-1},JUP_VENV=$VENV" \
        scripts/scaling/jup_s3_parity.sbatch
}

only="${ONLY:-}"
if [ -z "$only" ] || [ "$only" = qwen3_5 ]; then
    echo "qwen3_5  $(submit qwen3_5 "$QWEN3_5_SNAP" models/tests/test_qwen3_5_parity.py)"
fi
if [ -z "$only" ] || [ "$only" = qwen3_vl ]; then
    echo "qwen3_vl $(submit qwen3_vl "$QWEN3_VL_SNAP" models/tests/test_qwen3_vl_parity.py)"
fi
