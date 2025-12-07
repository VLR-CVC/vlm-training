#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts

NGPUS=3

export DOMAIN_BLACKLIST=github.com,huggingface.co
export OMP_NUM_THREADS=16

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NGPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
