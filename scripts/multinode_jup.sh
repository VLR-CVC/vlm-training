#!/bin/bash -x
#SBATCH --account=reformo
#SBATCH --nodes=128
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
# JUPITER's slurm rejects --gpus-per-task with "Invalid GRES specification";
# --ntasks-per-node=1 makes this the same request.
#SBATCH --gpus-per-node=4
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

# `/e/project1/jureap59/ockier1/miniforge` does not exist -- that `source` has
# been failing silently, and `conda activate torch11` only worked because
# --export=ALL inherited conda and CONDA_ENVS_DIRS from the submitting login
# shell. Activate the env by absolute prefix through the install that is
# actually there, so the job does not depend on how it was submitted.
CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"

# `module` is a shell function the login shell exports; a batch script is not
# interactive, so it only has it because --export=ALL inherited it. /etc/profile
# defines the function *and* MODULEPATH -- sourcing Lmod's init/bash alone gets
# the function with an empty MODULEPATH, which finds nothing.
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

sleep 5

# *****
NGPUS=4
NNODES=128
# *****

CONFIG_FILE=configs/jupiter/qwen3_5_9b.toml
CONV_HELPER="$(dirname "${BASH_SOURCE[0]:-$0}")/convert_final_checkpoint.sh"

srun --cpu-bind=none \
        torchrun \
        --nnodes=$NNODES\
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen --config "$CONFIG_FILE"

# batch-script body runs on the head node only -> convert final checkpoint once
if [ "$(hostname -s)" = "$head_node" ]; then
    bash "$CONV_HELPER" "$CONFIG_FILE"
fi

