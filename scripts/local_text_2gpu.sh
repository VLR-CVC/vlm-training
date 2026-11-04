#!/bin/bash
# Text-only Qwen3.5-2B on local GPUs 6 and 7, P2P disabled.
#
#   scripts/local_text_2gpu.sh [extra args passed to train.train_qwen]
#
# GPUs 6 and 7 are the 44 GiB cards; 0-5 are usually busy. `configs/text/
# qwen3_5_2b_2gpu.toml` is sized for that pair (seq_len 4096, op-level AC).
#
# NCCL_P2P_DISABLE=1 forces every FSDP collective through host memory rather
# than card-to-card. Slower, and deliberate: it is how this box is run.
#
# WANDB_MODE defaults to offline here. This run was not asked to publish
# anything, and offline keeps it off the network and out of the account until
# somebody chooses otherwise -- `scripts/sync_wandb_mn5.sh` is the uploader.
# Override with WANDB_MODE=online if you do want it live.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NCCL_P2P_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
# torchrun picks this up for the log dir; keep the traces off the repo
export TORCH_TRACE="${TORCH_TRACE:-/data/151-2/users/tockier/profiling/text_2gpu_trace}"

CONFIG="${CONFIG:-configs/text/qwen3_5_2b_2gpu.toml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "GPUs $CUDA_VISIBLE_DEVICES | P2P disabled | wandb $WANDB_MODE | $CONFIG"
exec "$SCRIPT_DIR/finetune.sh" --config "$CONFIG" "$@"
