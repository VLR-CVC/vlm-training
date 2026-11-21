#!/bin/bash
# Multinode training on Leonardo Booster (4x A100 64 GB per node, 32 cores, InfiniBand).
#   cd <repo> && sbatch --nodes=2 scripts/multinode_leo.sh
#   CONFIG_FILE=configs/leonardo/<x>.toml sbatch ... to pick another config.
#   TRAIN_ARGS="--data.seq-len 8192 ..." sbatch ... to override config fields.
# No numa_wrapper.sh: it binds rank N to NUMA node N, written for JUPITER's 4-NUMA GH200 nodes.
#SBATCH -D .
#SBATCH --job-name=qwen4_leo
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_scratch/fast/EUHPC_E07_125/ockier1/logs/%x_%j.out

set -eo pipefail
source scripts/leo_env.sh

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
# Rendezvous endpoint must be the hostname: torchrun's c10d backend hosts the store only on
# the node whose hostname matches it (an ib0 IP matched nothing; job 64103595 timed out).
echo "nodes ${nodes[*]} head $head_node job $SLURM_JOB_ID"

# Per-job compile cache: kernels compiled by another job are not safe to replay
# (numa_wrapper.sh, job 2259891 on JUPITER). Node-local /tmp, not Lustre: with
# several nodes writing one Lustre cache, inductor's mkdir(exist_ok=True) raced
# into FileExistsError (job 64171713). Each node compiles into its own copy.
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER-$SLURM_JOB_ID/inductor"
export TRITON_CACHE_DIR="/tmp/$USER-$SLURM_JOB_ID/triton"

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 LOGLEVEL=INFO
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=2097152
export QWEN_SECTION_TIMING=1
export QWEN_NCCL_TIMEOUT_S="${QWEN_NCCL_TIMEOUT_S:-1800}"
ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

CONFIG_FILE="${CONFIG_FILE:-configs/leonardo/qwen3_5_2b.toml}"
echo "config $CONFIG_FILE ${TRAIN_ARGS:-}"
nvidia-smi -i 0 --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'nccl', torch.cuda.nccl.version())"

srun --cpu-bind=none torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node:29500" \
    -m train.train_qwen --config "$CONFIG_FILE" $TRAIN_ARGS
