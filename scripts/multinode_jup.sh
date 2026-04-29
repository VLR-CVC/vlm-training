#!/bin/bash -x
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=qwen_test

#SBATCH --output=logs_jup/%j/log_%x.out
#SBATCH --error=logs_jup/%j/errors/rank_%t.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

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

source /e/project1/reformo/ockier1/miniforge3/bin/activate
conda activate torch11

ulimit -l unlimited
ulimit -s unlimited

sleep 5

# *****
NGPUS=4
NNODES=1
# *****

srun --cpu-bind=none \
        torchrun \
        --nnodes=$NNODES\
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen --config configs/jupiter/qwen3_5_27b.toml