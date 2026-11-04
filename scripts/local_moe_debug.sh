#!/bin/bash
# Debug MoE training run on local GPUs 4 and 6, P2P disabled.
#
#   scripts/local_moe_debug.sh [extra args passed to train.train_qwen]
#   CUDA_VISIBLE_DEVICES=4,6 STEPS=10 scripts/local_moe_debug.sh
#
# GPUs 4 and 6 are the two free L40S (46 GiB); 0-3 and 5,7 are usually taken by
# other tenants. `configs/local/qwen3_5_moe_debug.toml` is sized for one card.
#
# NCCL_P2P_DISABLE=1 forces every FSDP collective through host memory rather
# than card-to-card. Slower, and deliberate: it is how this box is run.
#
# Does NOT call scripts/finetune.sh: that runs convert_final_checkpoint.sh
# afterwards, and this config writes no checkpoint (save_steps > total_steps).
#
# `torch11_jup` is the env, same as scripts/vit_sweep.py. The `tt_port` venv the
# MoE unit tests run under has neither `tyro` nor a `torchrun` entry point, so it
# cannot launch the trainer at all. torch11_jup in turn has no `torch_remat`,
# which is exactly why the MoE port dropped the remat regions rather than
# vendoring them -- with them this run would not start.
#
# torch11_jup is the JUPITER reference env: run in it, never install into it.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,6}"
export NCCL_P2P_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
# keep dynamo/inductor traces off the repo and off 151-1 (full)
export TORCH_TRACE="${TORCH_TRACE:-/data/151-2/users/tockier/profiling/moe_debug_trace}"

ENV_DIR="${ENV_DIR:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
CONFIG="${CONFIG:-configs/local/qwen3_5_moe_debug.toml}"
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$TORCH_TRACE"
export PATH="$ENV_DIR/bin:$PATH"

echo "GPUs $CUDA_VISIBLE_DEVICES ($NGPUS ranks) | P2P disabled | wandb $WANDB_MODE | $CONFIG"
cd "$REPO"
exec torchrun --nproc_per_node="$NGPUS" \
              --master_addr=127.0.0.1 \
              --master_port="$(shuf -i 20000-29999 -n 1)" \
              -m train.train_qwen \
              --config "$CONFIG" "$@"
