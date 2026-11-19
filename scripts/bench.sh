#!/bin/bash
# Throughput benchmark (train/bench.py): N steps, wall-clock tok/s, TFLOP/s, MFU,
# step breakdown and peak memory; no checkpoint, no export, wandb off.
#
#   CUDA_VISIBLE_DEVICES=0,1 BENCH_OUT=out.json scripts/bench.sh --config configs/bench/engram_0_8b.toml [overrides]
#
# Env: BENCH_WARMUP (steps left out of the summary, default 10), BENCH_OUT (JSON
# path), PROFILE_DIR / PROFILE_WAIT (torch.profiler trace of 2 steps), NNODES /
# NODE_RANK / MASTER_ADDR / MASTER_PORT for a multi-node (or simulated
# multi-node) run.
set -eo pipefail
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | grep -o '[^,]\+' | wc -l)
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-disabled}
unset TORCH_TRACE  # an empty value crashes torch._logging (makedirs(""))
torchrun --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
         --nproc_per_node="$NGPUS" \
         --master_addr="${MASTER_ADDR:-127.0.0.1}" \
         --master_port="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}" \
         -m train.bench "$@"
