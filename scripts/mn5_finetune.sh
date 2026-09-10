#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

WANDB_MODE=offline
HF_HUB_OFFLINE=1
DOMAIN_BLACKLIST=github.com,huggingface.co

# environments
#source /gpfs/projects/ehpc391/env_variables.sh
source /gpfs/projects/ehpc543/envs/torch11_cuda12_6/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

module load cuda/12.8

NGPUS=4

export NCCL_P2P_LEVEL=NVL
export LOGLEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# pull `--config <path>` out of the passthrough args for the post-run conversion
CONFIG_FILE=""
_prev=""
for _a in "$@"; do
    case "$_prev" in --config) CONFIG_FILE="$_a";; esac
    case "$_a" in --config=*) CONFIG_FILE="${_a#--config=}";; esac
    _prev="$_a"
done

torchrun \
        --nnodes=1 \
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
	-m train.train_qwen \
	$@

# convert the final checkpoint of the finished run to an HF snapshot
bash "$SCRIPT_DIR/convert_final_checkpoint.sh" "$CONFIG_FILE"
