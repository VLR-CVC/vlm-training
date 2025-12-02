#!/bin/bash
#SBATCH -D .
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --job-name=qwen3vl
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
source ../env_variables.sh

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
export CUDA_LAUNCH_BLOCKING=0

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

# *****
NGPUS=4
NNODES=2
# *****

EXEC_FILE=${EXEC_FILE:-"/home/uab/uab210596/qwen3vl/mn5_finetune.sh"}
srun --cpu-bind=none torchrun --nproc_per_node=$NGPUS \
                --nnodes=$NNODES \
                --rdzv_id 101 \
                --rdzv_backend c10d \
                --rdzv_endpoint "$head_node_ip:29500" \
                /home/uab/uab210596/qwen3vl/qwen-vl-finetune/qwenvl/train/train_qwen.py \
                --model_name_or_path $MODEL_PATH \
                --tune_mm_llm True \
                --tune_mm_vision True \
                --tune_mm_mlp True \
                --dataset_use $DATASETS \
                --output_dir $OUTPUT_DIR \
                --cache_dir $CACHE_DIR \
                --bf16 \
                --per_device_train_batch_size 6 \
                --gradient_accumulation_steps 1 \
                --learning_rate 2e-6 \
                --mm_projector_lr 1e-5 \
                --vision_tower_lr 1e-6 \
                --optim adamw_torch \
                --model_max_length 2048 \
                --data_packing False \
                --max_pixels 451584 \
                --min_pixels 12544 \
                --weight_decay 0.01 \
                --save_steps 1000 \
                --save_total_limit 3 \