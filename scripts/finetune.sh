#!/bin/bash

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

export OMP_NUM_THREADS=16

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NGPUS=$(echo $CUDA_VISIBLE_DEVICES | grep -o '[^,]\+' | wc -l)
fi

torchrun --nproc_per_node=2 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         -m train.train_qwen \
         "$@"
