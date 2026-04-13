#!/bin/bash

DATA_PATH="/gpfs/scratch/ehpc543/c4_ca"
BASE_MODEL="/gpfs/scratch/ehpc391/cache/qwen3_1_7b/" # Update this
MODELS_DIR="/gpfs/scratch/ehpc543/qwen_finetune/text_catalan_init/models"
SAMPLES=10000
LOG_FILE="eval_results.txt"

echo "======================================" | tee -a $LOG_FILE
echo "Evaluating Base Model: $BASE_MODEL" | tee -a $LOG_FILE
PYTHONPATH=. torchrun --nproc_per_node=4 eval/val.py \
    --model_path $BASE_MODEL \
    --data_path $DATA_PATH \
    --limit_samples $SAMPLES >> $LOG_FILE 2>&1

for ckpt in $(ls -v $MODELS_DIR/step-* -d); do
    echo "======================================" | tee -a $LOG_FILE
    echo "Evaluating Checkpoint: $ckpt" | tee -a $LOG_FILE
    PYTHONPATH=. torchrun --nproc_per_node=4 eval/val.py \
        --model_path $ckpt \
        --data_path $DATA_PATH \
        --limit_samples $SAMPLES >> $LOG_FILE 2>&1
done

echo "Done. Check $LOG_FILE for results."