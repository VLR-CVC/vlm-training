#!/bin/bash
# Strong scaling: the global batch is FIXED in tokens, so adding nodes only splits the
# same work over more ranks and gradient accumulation absorbs the difference.
#
#   scripts/scaling/submit_strong_sweep.sh [nodes...]      # default 8 16 32 64 128
#
# Data parallel only: `data_parallel = ddp` (dp_replicate = world, dp_shard = 1), so
# the parameters are replicated and per-GPU memory does not move with the node count.
# The micro-batch is ONE row of MBS tokens per GPU, sized on a single node by raising
# `seq_len` until the card is full -- with the accumulation window of the SMALLEST
# point in the sweep, because `train_step` holds a whole window on the device at once.
# probe_20260917_214439 (tpi_multiplier 16): 8192 -> 78.3 GiB of 95, 12288 -> 91.2,
# 16384 -> OOM. 8192 keeps ~17 GiB for NCCL buffers, which grow with the rank count:
# sizing on a single micro-batch instead (85.1 GiB at 16384) OOM'd every point of
# sweep strong_20260917_213255.
#
#   global batch = MBS x ANCHOR_RANKS tokens, fixed
#   accumulation at N nodes = ANCHOR_RANKS / (GPUS_PER_NODE x N)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${STRONG_CONFIG:-configs/jupiter/qwen3_vl_2b.toml}"
MBS="${MBS:-8192}"
GPUS_PER_NODE=4
ANCHOR_RANKS="${ANCHOR_RANKS:-512}"   # 128 nodes x 4 GPUs: one micro-batch per rank there
STEPS="${STEPS:-40}"
NODES=("${@:-8 16 32 64 128}")
[ $# -gt 0 ] && NODES=("$@")
STAMP="${STRONG_SWEEP_ID:-strong_$(date +%Y%m%d_%H%M%S)}"

echo "global batch $((MBS * ANCHOR_RANKS)) tokens ($MBS x $ANCHOR_RANKS ranks), sweep $STAMP"
for n in ${NODES[@]}; do
    ranks=$((GPUS_PER_NODE * n))
    if [ $((ANCHOR_RANKS % ranks)) -ne 0 ]; then
        echo "skip ${n}n: $ranks ranks does not divide $ANCHOR_RANKS" >&2
        continue
    fi
    accum=$((ANCHOR_RANKS / ranks))
    echo "${n}n: $ranks ranks, $accum micro-batches per rank per step"
    LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 SCALING_TIME="${SCALING_TIME:-00:45:00}" \
        SCALING_SWEEP_ID="$STAMP" \
        SCALING_EXTRA_ARGS="--training.data-parallel ddp --training.tpi-multiplier $accum \
--data.seq-len $MBS --data.tokens-per-microbatch $MBS \
--training.total-steps $STEPS --training.save-steps 100000" \
        SCALING_CONFIG="$CONFIG" scripts/scaling/submit_sweep.sh "$n"
done
