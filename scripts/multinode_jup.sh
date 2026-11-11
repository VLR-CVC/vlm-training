#!/bin/bash -x
#SBATCH --account=open-sci-mm
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
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
# NOT inherited. The login shell exports WANDB_DIR and --export=ALL carries it in,
# and in the shell job 2268024 was submitted from that value was
# /e/project1/jureap59/ockier1/cache/wandb/runs -- a different project allocation,
# whose quota we do not have, so wandb died with
#   OSError: [Errno 122] Disk quota exceeded: '.../offline-run-20261010_231758-fgdsqyjl'
# before the run could log anything. Same trap as the jureap59 miniforge path below.
# Pinned to open-sci-mm's scratch, which has the room (25.9 of 214.7 TB used).
# wandb puts runs in $WANDB_DIR/wandb/offline-run-*, which is what sync_wandb_jup.sh expects.
export WANDB_DIR="${JUP_WANDB_DIR:-/e/scratch/open-sci-mm/ockier1/wandb}"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
echo "wandb dir: $WANDB_DIR"

# Same trap as WANDB_DIR, for every other cache. These are derived from the login
# shell's CACHE_ROOT/XDG_CACHE_HOME and carried in by --export=ALL already
# expanded, so a job inherits whichever project the submitting shell had active.
# Job 2268024 was submitted from a jureap59 shell and left
# /e/project1/jureap59/ockier1/cache/{attention_gym,nv} behind holding nothing but
# lock files -- we have no quota on that allocation. Pinned so where a job caches
# does not depend on who launched it.
CACHE_ROOT="${JUP_CACHE_ROOT:-/e/project1/open-sci-mm/ockier1/cache}"
export CACHE_ROOT
export XDG_CACHE_HOME="$CACHE_ROOT"
# attn_gym's CuTeDSL kernels. ATTN_GYM_CUTE_CACHE_DIR wins over XDG_CACHE_HOME and
# is used verbatim, so it carries the attention_gym/cute suffix itself -- see
# get_cache_path() in attn_gym/_backends/cute/cache.py.
export ATTN_GYM_CUTE_CACHE_DIR="$CACHE_ROOT/attention_gym/cute"
export CUDA_CACHE_PATH="$CACHE_ROOT/nv"
export TRITON_HOME="$CACHE_ROOT/triton"
export TILELANG_CACHE_DIR="$CACHE_ROOT/tilelang"
export TILELANG_TMP_DIR="$CACHE_ROOT/tilelang/tmp"
mkdir -p "$ATTN_GYM_CUTE_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_HOME" "$TILELANG_TMP_DIR"
# TRITON_CACHE_DIR is deliberately NOT set here: the per-job compile cache below
# owns it.

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

export QWEN_SECTION_TIMING=1

export QWEN_NCCL_TIMEOUT_S="${QWEN_NCCL_TIMEOUT_S:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# One cache per job by default: kernels another job compiled are not safe to
# replay (job 2259891, `misaligned address` on step 5). QWEN_COMPILE_CACHE points
# at an existing cache to reuse it deliberately -- numa_wrapper.sh reads it and
# appends the per-node suffix, so it has to reach the ranks, not just this shell.
COMPILE_CACHE="${QWEN_COMPILE_CACHE:-${JUP_COMPILE_CACHE:-/e/scratch/open-sci-mm/ockier1/compile_cache/${SLURM_JOB_ID}}}"
export QWEN_COMPILE_CACHE="$COMPILE_CACHE"
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE/inductor"
export TRITON_CACHE_DIR="$COMPILE_CACHE/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "compile cache: $COMPILE_CACHE"

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main_214}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv_214}/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

sleep 5

# *****
NGPUS=4
# *****

CONFIG_FILE="${CONFIG_FILE:-configs/jupiter/qwen3_5_35b_a3b.toml}"
# NOT `dirname $BASH_SOURCE`: inside a batch job that is SLURM's spool copy
# (/var/spool/parastation/jobs/<id>), not the checkout, so the helper was never
# found and the job exited 127 after training finished cleanly -- job 2260406.
CONV_HELPER="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/convert_final_checkpoint.sh"

srun --cpu-bind=none \
        torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES\
        --nproc_per_node=$NGPUS \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen --config "$CONFIG_FILE" \
            ${JUP_EXTRA_ARGS:-}   # unquoted, word-split on purpose, as jup_scaling.sbatch does

# batch-script body runs on the head node only -> convert final checkpoint once
if [ "$(hostname -s)" = "$head_node" ]; then
    bash "$CONV_HELPER" "$CONFIG_FILE"
fi

