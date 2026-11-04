#!/bin/bash
# Validate MTP end to end on one free local card: with a depth, then without.
#
#   scripts/local_mtp_debug.sh              # both runs, GPU 3, 1.1B MoE
#   MODEL=2b scripts/local_mtp_debug.sh     # real 2B, real weights, 2 depths
#   GPU=1 scripts/local_mtp_debug.sh        # elsewhere
#   ONLY=mtp scripts/local_mtp_debug.sh     # skip the baseline
#
# MODEL=moe is the fast smoke (random init, ~40s/run); MODEL=2b is the one that
# loads an HF snapshot and chains two depths.
#
# Reuses scripts/local_moe_debug.sh as the launcher (same env, same torchrun
# line) with CONFIG pointed at the MTP config; this only adds the A/B and the
# logs. One rank is enough: FSDP still wraps and shards, and the thing being
# tested is the MTP path, not scaling.
#
# Logs land in /data/151-2 (151-1 is full). Read them with:
#   grep -E "mtp:|step " <log>
set -euo pipefail

GPU="${GPU:-3}"
MODEL="${MODEL:-moe}"
case "$MODEL" in
    moe) CONFIG=configs/local/qwen3_5_mtp_debug.toml ;;
    2b)  CONFIG=configs/local/qwen3_5_mtp_2b.toml ;;
    *)   echo "MODEL must be moe or 2b" >&2; exit 2 ;;
esac
export CONFIG
LOGDIR="${LOGDIR:-/data/151-2/users/tockier/tests/mtp_debug}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$LOGDIR"

run() {  # name, extra args
    local name="$1"; shift
    local log="$LOGDIR/${MODEL}_${name}_$(date +%m%d_%H%M).log"
    # per-run output_dir: both runs write a FINAL checkpoint and a shared
    # one would mean the second run silently overwrites the first
    set -- "$@" --training.output_dir \
        "/data/151-2/users/tockier/qwen_finetune/checkpoints/mtp_${MODEL}_$name"
    echo "=== $name -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" \
    TORCH_TRACE="/data/151-2/users/tockier/profiling/mtp_trace_${MODEL}_$name" \
        "$REPO/scripts/local_moe_debug.sh" "$@" >"$log" 2>&1 \
        && echo "ok  $name" || { echo "FAIL $name (tail below)"; tail -30 "$log"; return 1; }
    grep -E "^.*(mtp:|loss) " "$log" | tail -5 || true
}

# "$@" is forwarded so a one-off override (--data.seq_len 16384) reaches both runs
case "${ONLY:-both}" in
    mtp)  run mtp "$@" ;;
    base) run base --model.num_mtp_layers 0 "$@" ;;
    *)    run mtp "$@"; run base --model.num_mtp_layers 0 "$@" ;;
esac
