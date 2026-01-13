#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

WANDB_MODE=offline
HF_HUB_OFFLINE=1
DOMAIN_BLACKLIST=github.com,huggingface.co

source /gpfs/projects/ehpc391/env_variables.sh
source /gpfs/projects/ehpc391/envs/qwen3/bin/activate

NGPUS=4

export NCCL_P2P_LEVEL=NVL

export LOGLEVEL=INFO
export FI_PROVIDER="efa"

# debugging flags (optional)
#export NCCL_DEBUG=WARN
#export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

torchrun \
        --nnodes=1 \
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
	-m train.train_qwen \
	$@ \
