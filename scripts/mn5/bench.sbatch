#!/bin/bash
#SBATCH -D .
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=bench2b
#
# One arm of the Qwen3-VL-2B config sweep. Node count and config come from
# sbatch, so the same file serves every arm:
#
#   MN5_CONFIG=configs/mn5/bench/base.toml sbatch --nodes=4 scripts/mn5/bench.sbatch
#
# Environment mirrors scripts/multinode_mn5_instruct.sh exactly -- a throughput
# benchmark that runs in a different environment than production measures the
# wrong thing.

CONFIG_FILE="${MN5_CONFIG:?set MN5_CONFIG}"

export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export GPUS_PER_NODE=4
export PYTHONUNBUFFERED=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "[bench] head $head_node_ip  nodes $SLURM_NNODES  config $CONFIG_FILE"

# `set -u` is deliberately NOT on: the env's activate script reads CONDA_PREFIX
# unguarded and dies under it.
source /gpfs/projects/ehpc543/envs/torch11_cuda12_6/bin/activate
module load cuda/12.8

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

# per-section forward breakdown: the tower is 47% of forward at 2B and that is
# the whole reason this sweep exists
export QWEN_SECTION_TIMING=1

# compiled collectives bake process-group names into the inductor cache, so it
# has to stay keyed on world size
export TORCHINDUCTOR_CACHE_DIR="/gpfs/scratch/ehpc543/tockier/vlmtest/inductor/ws$((SLURM_NNODES * 4))"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

ulimit -l unlimited
mkdir -p slurm_output/$SLURM_JOB_ID

srun --cpu-bind=none torchrun --nproc_per_node=4 \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint "$head_node_ip:29500" \
        --redirects 2 \
        --log-dir slurm_output/$SLURM_JOB_ID \
        -m train.train_qwen --config "$CONFIG_FILE"
