#!/bin/bash
# Text-only Qwen3.5-2B on the 4 local 96 GiB cards (0-3).
#
#   scripts/local_text_4gpu.sh [extra args passed to train.train_qwen]
#
# UNLIKE scripts/local_text_2gpu.sh, this does NOT disable NCCL peer-to-peer.
# The first-all-gather hang is a property of the L40S pair (6,7), not of the
# box: a 16 MiB all_reduce across 0-3 with P2P enabled completes immediately.
# So these four run with real collectives, and a step time measured here is the
# model's rather than the host memory bus's.
#
# Set NCCL_P2P_DISABLE=1 if that ever stops being true.
#
# WANDB_MODE defaults to offline: nothing here was asked to publish. Upload
# later with scripts/sync_wandb.sh.
set -euo pipefail

# Pin the interpreter instead of inheriting whatever the shell has active.
# `torch11_jup` is this repo's env -- the one scripts/pack_env_mn5.sh ships to
# MN5. The older `torch11` (torch 2.11) is missing `spmd_types`, which
# models/common/attention.py imports at module scope, so a run started under it
# dies in `Trainer.__init__` on every rank with a bare ModuleNotFoundError.
TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"
python -c 'import spmd_types, torch' || {
    echo "env at $TORCH_ENV cannot import spmd_types/torch -- wrong env?" >&2
    exit 1
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_TRACE="${TORCH_TRACE:-/data/151-2/users/tockier/profiling/text_4gpu_trace}"

CONFIG="${CONFIG:-configs/text/qwen3_5_2b_pretrain.toml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "GPUs $CUDA_VISIBLE_DEVICES | P2P ${NCCL_P2P_DISABLE:-enabled} | wandb $WANDB_MODE | $CONFIG"
exec "$SCRIPT_DIR/finetune.sh" --config "$CONFIG" "$@"
