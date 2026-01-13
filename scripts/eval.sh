#!/bin/bash
conda init
conda activate eval

CACHE_ROOT="/data-net/storage2/users/tockier"
HF_BASE_CACHE="${CACHE_ROOT}/hf_cache"
export HF_HOME="${HF_BASE_CACHE}/home"               # Used for config, tokens, etc.
export HF_XET_CACHE="${HF_BASE_CACHE}/xet"           # Cache for Xet data (if used)
export HF_ASSETS_CACHE="${HF_BASE_CACHE}/assets"     # Used for certain dataset assets
export HF_DATASETS_CACHE="${HF_BASE_CACHE}/datasets" # Cache for datasets
export HUGGINGFACE_HUB_CACHE="${HF_BASE_CACHE}/hub"   # Primary cache for model files and repos

export HF_HUB_OFFLINE=0

MODEL="/home-local/tockier/vlm-training/final_models/final_model"

TENSOR_PARALLEL_SIZE=1 # Number of GPUs for tensor parallelism
DATA_PARALLEL_SIZE=4     # Number of GPUs for data parallelism

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

EVAL_REPO="/home-local/tockier/lmms_eval"

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
