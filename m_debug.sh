#!/bin/bash
#SBATCH -D .
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --job-name=Q-VL-debug
#SBATCH --partition=acc
#SBATCH --mail-type=all
#SBATCH --mail-user=Tomas.Ockier@autonoma.cat

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

# load env
source /gpfs/projects/ehpc391/envs/qwen3/bin/activate
source /gpfs/projects/ehpc391/env_variables.sh

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_P2P_LEVEL=NVL

export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

lscpu | grep "NUMA"
taskset -cp $$
ulimit -l unlimited
ulimit -s unlimited

# paths for the model
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/gpfs/scratch/ehpc391/qwen_finetune/checkpoints"
CACHE_DIR="/gpfs/scratch/ehpc391/qwen_finetune/cache"

WANDB_MODE=offline
HF_HUB_OFFLINE=1
DOMAIN_BLACKLIST=github.com,huggingface.co

DATASETS="finevision_mn5"

echo "finetuning qwen vl model from $MODEL_PATH on datasets: $DATASETS"

wandb enabled
wandb offline

# *****
NGPUS=4
NNODES=1
# *****

srun --cpu-bind=none torchrun --nproc_per_node=$NGPUS \
                --nnodes=$NNODES \
                --rdzv_id 101 \
                --rdzv_backend c10d \
                --rdzv_endpoint "$head_node_ip:29500" \
                /home/uab/uab210596/qwen3vl/qwen-vl-finetune/qwenvl/train/train_qwen.py \
