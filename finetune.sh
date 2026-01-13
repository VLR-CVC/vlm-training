#!/bin/bash

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

NGPUS=1

export DOMAIN_BLACKLIST=github.com,huggingface.co
export OMP_NUM_THREADS=16

# ======================
# Training Hyperparameters
# ======================

torchrun --nproc_per_node=$NGPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         -m train.train_qwen \
	 $@ \
