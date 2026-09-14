#!/bin/bash

MASTER_ADDR="127.0.0.1"                     
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co

export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

export NCCL_SOCKET_IFNAME="ib,eth"
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_BUFFSIZE=2097152

module load CUDA

conda activate torch11

ulimit -l unlimited
ulimit -s unlimited
# a crashing rank dumps its whole address space into the CWD, which is the
# repo; three ranks once left 52 GB each. `ulimit -s unlimited` above makes
# those dumps enormous, so disable them.
ulimit -c 0

torchrun \
        --nnodes=1 \
        --nproc_per_node=4 \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen $@