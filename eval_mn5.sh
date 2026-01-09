#!/bin/bash
CACHE='/gpfs/projects/ehpc391/evaluation/hf_cache'
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HUGGINGFACE_HUB_CACHE=$CACHE
export HF_HUB_OFFLINE=1

# Qwen3-VL Evaluation Script with vLLM Backend
# This script demonstrates how to evaluate Qwen3-VL models using vLLM for accelerated inference
#
# Requirements:
# - vllm>=0.11.0
# - qwen-vl-utils
# - CUDA-enabled GPU(s)
#
# Installation:
# uv add vllm qwen-vl-utils
# OR
# pip install vllm>=0.11.0 qwen-vl-utils

# ============================================================================
# Configuration
# ============================================================================

# Model Configuration
# Available Qwen3-VL models:
# - Qwen/Qwen3-VL-30B-A3B-Instruct
# - Qwen/Qwen3-VL-30B-A3B-Thinking
# - Qwen/Qwen3-VL-235B-A22B-Instruct
# - Qwen/Qwen3-VL-235B-A22B-Thinking
MODEL="/gpfs/scratch/ehpc391/qwen_finetune/checkpoints_32_packed/final_model"

# Parallelization Settings
# Adjust based on your GPU configuration
TENSOR_PARALLEL_SIZE=4 # Number of GPUs for tensor parallelism
DATA_PARALLEL_SIZE=1     # Number of GPUs for data parallelism

# Memory and Performance Settings
GPU_MEMORY_UTILIZATION=0.90  # Fraction of GPU memory to use (0.0 - 1.0)
BATCH_SIZE=32       # Batch size for evaluation
MAX_MODEL_LEN=$((2**15))
# Task Configuration
# Common tasks: mmmu_val, mme, mathvista, ai2d, etc.
TASKS="docvqa_test,infovqa_test"

# Output Configuration
OUTPUT_PATH="./logs/qwen3vl_vllm"
LOG_SAMPLES=true
LOG_SUFFIX="qwen3vl_vllm"

# Evaluation Limits (optional)
# LIMIT=100  # Uncomment to limit number of samples (for testing)

# ============================================================================
# NCCL Configuration (for multi-GPU setups)
# ============================================================================
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
# export NCCL_DEBUG=INFO  # Uncomment for debugging

# ============================================================================
# Run Evaluation
# ============================================================================

echo "=========================================="
echo "Qwen3-VL Evaluation with vLLM"
echo "=========================================="
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Data Parallel Size: $DATA_PARALLEL_SIZE"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Output Path: $OUTPUT_PATH"
echo "=========================================="

EVAL_REPO="/gpfs/projects/ehpc391/evaluation/lmms-eval"

# Build the command
CMD="PYTHONPATH=${EVAL_REPO} python -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}\
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}"

# Add optional arguments
if [ "$LOG_SAMPLES" = true ]; then
    CMD="$CMD --log_samples --log_samples_suffix ${LOG_SUFFIX}"
fi

if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

# Execute
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="
