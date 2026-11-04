#!/bin/bash
# Scaling sweeps with the parameters sharded INSIDE a node and replicated between
# nodes: TP=2 x FSDP=2 over the 4 GPUs of a node, and the node count becomes the
# replicate degree. Both sharding collectives stay on NVLink; the only inter-node
# traffic is the gradient all-reduce (HSDP).
#
#   scripts/scaling/submit_hsdp_sweep.sh 4 8 16 32                 # weak
#   ANCHOR_NODES=32 MBS=8192 scripts/scaling/submit_hsdp_sweep.sh 4 8 16 32   # strong
#
# HSDP_EXTRA is appended to the trainer's arguments, for anything this script does
# not model, e.g. HSDP_EXTRA='--wandb.log-topk --wandb.topk-interval 1'.
#
# WEAK (no ANCHOR_NODES): the micro-batch is whatever the config says and there is
# no accumulation, so the work per GPU is fixed and the global batch grows with the
# node count. Baseline is the 1-node run (weakhsdp_1n_20260918_210824, 22.6% MFU).
#
# STRONG (ANCHOR_NODES set): the global batch is fixed at the ANCHOR_NODES point,
# and smaller node counts make up the difference with gradient accumulation:
#
#   dp ranks at N nodes = 4 x N / TP          (a TP group shares one micro-batch)
#   accumulation        = ANCHOR_NODES / N
#   global batch        = MBS x dp_ranks(ANCHOR_NODES), constant
#
# MBS must be set well below the weak-sweep micro-batch: `train()` materialises the
# WHOLE accumulation window before `train_step` (train_qwen.py:871), so the point
# with the fewest nodes holds ANCHOR_NODES/N micro-batches at once. The weak sweep
# already peaks at 78.9 GiB of 95 with a single 32768-token micro-batch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${HSDP_CONFIG:-configs/jupiter/qwen3_vl_2b.toml}"
TP="${TP:-2}"
# Ranks that share one parameter copy; TP x SHARD = GPUs per node keeps both
# sharding collectives on NVLink. SHARD=0 means plain FSDP instead: one copy
# spread over the whole world, no replicate axis, inter-node all-gather.
SHARD="${SHARD:-2}"
STEPS="${STEPS:-40}"
GPUS_PER_NODE=4
ANCHOR_NODES="${ANCHOR_NODES:-}"   # empty = weak scaling
MBS="${MBS:-8192}"                 # tokens per micro-batch per DP rank, strong only
NODES=(1 2 4 8 16)
[ $# -gt 0 ] && NODES=("$@")
STAMP="${HSDP_SWEEP_ID:-${ANCHOR_NODES:+strong}${ANCHOR_NODES:-weak}hsdp_$(date +%Y%m%d_%H%M%S)}"

if [ -n "$ANCHOR_NODES" ]; then
    gbs=$((MBS * GPUS_PER_NODE * ANCHOR_NODES / TP))
    echo "strong: global batch $gbs tokens ($MBS x $((GPUS_PER_NODE * ANCHOR_NODES / TP)) DP ranks), sweep $STAMP"
else
    echo "weak: micro-batch from $CONFIG, no accumulation, sweep $STAMP"
fi
if [ "$SHARD" -le 0 ]; then
    echo "TP=$TP, plain FSDP over the whole world (no replicate axis)"
else
    echo "TP=$TP FSDP=$SHARD per node, replicate = node count"
fi

# SHARD=0 is plain FSDP: one parameter copy over everything, so exactly 1 replica.
replicas() { if [ "$SHARD" -le 0 ]; then echo 1; else echo $(( GPUS_PER_NODE * $1 / (TP * SHARD) )); fi; }

for n in "${NODES[@]}"; do
    extra=""
    if [ -n "$ANCHOR_NODES" ]; then
        if [ $((ANCHOR_NODES % n)) -ne 0 ]; then
            echo "skip ${n}n: does not divide anchor $ANCHOR_NODES" >&2
            continue
        fi
        accum=$((ANCHOR_NODES / n))
        echo "${n}n: $((GPUS_PER_NODE * n)) ranks, $(replicas "$n") replicas, accum $accum"
        extra="--training.tpi-multiplier $accum --data.seq-len $MBS --data.tokens-per-microbatch $MBS"
    else
        echo "${n}n: $((GPUS_PER_NODE * n)) ranks, $(replicas "$n") replicas"
    fi

    LOCAL_CACHE="${LOCAL_CACHE:-1}" QWEN_NCCL_TIMEOUT_S=1800 SCALING_TIME="${SCALING_TIME:-00:45:00}" \
        SCALING_SWEEP_ID="$STAMP" \
        SCALING_RUN_NAME="${STAMP}_${n}n" \
        SCALING_EXTRA_ARGS="--training.data-parallel fsdp --training.tp-size $TP \
--training.dp-shard-size $SHARD \
--training.total-steps $STEPS --training.save-steps 100000 $extra ${HSDP_EXTRA:-}" \
        SCALING_CONFIG="$CONFIG" scripts/scaling/submit_sweep.sh "$n"
done

echo
echo "sweep id: $STAMP"
