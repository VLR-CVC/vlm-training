#!/bin/bash
#SBATCH -D .
#SBATCH --nodes=32
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --job-name=midarm
#SBATCH --mail-type=all
#SBATCH --mail-user=Tomas.Ockier@autonoma.cat

#SBATCH --output=slurm_output/%x-%A/%n/%t.out
#SBATCH --error=slurm_output/%x-%A/%n/%t.err

# One midtraining arm. The config is the only thing that varies:
#
#   sbatch scripts/midtraining_arms.sh configs/mn5/qwen3_vl_2b_midtraining_vl_decoder.toml
#   sbatch scripts/midtraining_arms.sh configs/mn5/qwen3_vl_2b_midtraining_native_vit.toml
#   sbatch scripts/midtraining_arms.sh configs/mn5/qwen3_5_2b_midtraining.toml
#
# --nodes=32 is NOT a preference. The reference arm (wandb run 7t1don48,
# `all_epoched`) ran at world_size 128, and with `tpi_multiplier = 2` and
# `tokens_per_microbatch = 0` the global batch is world_size * seq_len * 2 =
# 1,572,864 tokens per optimizer step. Change the node count and every arm's
# loss-vs-step curve moves, and so does the step at which `repeat = false`
# exhausts the data -- the reference stopped at 3271 steps, not at total_steps.
# Nothing in the TOML pins this, which is why it is pinned here.
set -euo pipefail

CONFIG_FILE="${1:?usage: sbatch scripts/midtraining_arms.sh <config.toml>}"
[ -f "$CONFIG_FILE" ] || { echo "no such config: $CONFIG_FILE" >&2; exit 2; }

export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export GPUS_PER_NODE=4

export PYTHONUNBUFFERED=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node IP: $head_node_ip"
echo "[midarm] $SLURM_NNODES nodes / $((SLURM_NNODES * 4)) GPUs, config $CONFIG_FILE"

source /gpfs/projects/ehpc543/envs/qwen4_mn5/bin/activate
sleep 5

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

ulimit -l unlimited
ulimit -s unlimited

wandb enabled
wandb offline

mkdir -p slurm_output/$SLURM_JOB_ID

srun --cpu-bind=none torchrun --nproc_per_node=4 \
                --nnodes=$SLURM_JOB_NUM_NODES \
                --rdzv_id $SLURM_JOB_ID \
                --rdzv_backend c10d \
                --rdzv_endpoint "$head_node_ip:29500" \
                --redirects 2 \
                --log-dir slurm_errors/$SLURM_JOB_ID \
                -m train.train_qwen \
                --config "$CONFIG_FILE"
