#!/bin/bash
# Qwen3.5-2B text with attn_backend="dsa" on ONE local card (default 0).
#
#   scripts/local_dsa_1gpu.sh [extra args passed to train.train_qwen]
#   CUDA_VISIBLE_DEVICES=1 scripts/local_dsa_1gpu.sh --training.total_steps 5
#
# Single rank, so no collectives and nothing to say about NCCL. Unlike
# scripts/local_text_4gpu.sh this calls train.train_qwen directly rather than
# through scripts/finetune.sh: the smoke config writes no checkpoint, so the
# post-run HF conversion would have nothing to convert.
#
# WANDB_MODE defaults to offline: nothing here was asked to publish.
set -euo pipefail

TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"
python -c 'import spmd_types, torch' || {
    echo "env at $TORCH_ENV cannot import spmd_types/torch -- wrong env?" >&2
    exit 1
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The box is shared; a trace per smoke run is noise. Opt in by exporting it.
unset TORCH_TRACE

CONFIG="${CONFIG:-configs/local/qwen3_5_2b_dsa.toml}"

echo "GPU $CUDA_VISIBLE_DEVICES | wandb $WANDB_MODE | $CONFIG"
exec torchrun --nproc_per_node=1 --master_addr=127.0.0.1 \
    --master_port="$(shuf -i 20000-29999 -n 1)" \
    -m train.train_qwen --config "$CONFIG" "$@"
