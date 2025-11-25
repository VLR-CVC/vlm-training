#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="/data/users/tockier/qwen_finetune/checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="/data/users/tockier/qwen_finetune/cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="finevision"
NGPUS=3
NCCL_P2P_DISABLE=2

DOMAIN_BLACKLIST=github.com,huggingface.co

echo "finetuning qwen vl model from $MODEL_PATH on datasets: $DATASETS"

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NGPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 1 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 2048 \
         --data_packing True \
         --max_pixels 451584 \
         --min_pixels 12544 \
         --weight_decay 0.01 \
         --save_steps 100 \
         --save_total_limit 3 \

         # Advanced Options
         #--deepspeed zero3.json \
