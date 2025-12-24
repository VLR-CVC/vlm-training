#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="/gpfs/scratch/ehpc391/qwen_finetune/checkpoints"
CACHE_DIR="/gpfs/scratch/ehpc391/qwen_finetune/cache"

WANDB_MODE=offline
HF_HUB_OFFLINE=1
DOMAIN_BLACKLIST=github.com,huggingface.co

source /gpfs/projects/ehpc391/env_variables.sh
source /gpfs/projects/ehpc391/envs/qwen3/bin/activate

#export NCCL_DEBUG=INFO

DATASETS="finevision_mn5"
NGPUS=2

export NCCL_P2P_LEVEL=NVL

export LOGLEVEL=INFO
export FI_PROVIDER="efa"

# debugging flags (optional)
#export NCCL_DEBUG=WARN
#export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

echo "finetuning qwen vl model from $MODEL_PATH on datasets: $DATASETS"

torchrun \
        --nnodes=1 \
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        /home/uab/uab210596/qwen3vl/qwen-vl-finetune/qwenvl/train/train_qwen.py \
