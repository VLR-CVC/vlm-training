#!/bin/bash
#SBATCH -D .
#SBATCH --nodes=16
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --job-name=midtraining_qwen3vl
#SBATCH --partition=acc
#SBATCH --mail-type=all
#SBATCH --mail-user=Tomas.Ockier@autonoma.cat

#SBATCH --output=slurm_output/%x-%A/%n/%t.out
#SBATCH --error=slurm_output/%x-%A/%n/%t.err

export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export GPUS_PER_NODE=4

export PYTHONUNBUFFERED=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

# Configs from megatorn moe docs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0 # Disable NVLS to prevent memory overhead
export NCCL_CUMEM_ENABLE=0

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

# load env
source /gpfs/projects/ehpc543/envs/torch11_cuda12_6/bin/activate

module load cuda/12.8

sleep 5

which wandb

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_P2P_LEVEL=NVL

export LOGLEVEL=INFO

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

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

WANDB_MODE=offline
HF_HUB_OFFLINE=1
DOMAIN_BLACKLIST=github.com,huggingface.co

wandb enabled
wandb offline

mkdir -p slurm_output/$SLURM_JOB_ID

CONFIG_FILE=configs/mn5/instruct.toml
CONV_HELPER="$(dirname "${BASH_SOURCE[0]:-$0}")/convert_final_checkpoint.sh"

srun --cpu-bind=none torchrun --nproc_per_node=4 \
                --nnodes=$SLURM_JOB_NUM_NODES \
                --rdzv_id 101 \
                --rdzv_backend c10d \
                --rdzv_endpoint "$head_node_ip:29500" \
                --redirects 2 \
                --log-dir slurm_output/$SLURM_JOB_ID \
                -m train.train_qwen \
		--config "$CONFIG_FILE"

# batch-script body runs on the head node only -> convert final checkpoint once
if [ "$(hostname -s)" = "$head_node" ]; then
    bash "$CONV_HELPER" "$CONFIG_FILE"
fi
