#!/bin/bash -x
# One point on the scaling curve. The node count comes from sbatch, not from a
# directive here, so the same file serves every scale -- submit_sweep.sh drives
# it. Run it by hand with:
#
#   sbatch --nodes=32 --output=... --error=... scripts/scaling/jup_scaling.sbatch
#
# but prefer submit_sweep.sh, which creates the log directories (SLURM will not)
# and names things consistently for collect.py.

#SBATCH --account=open-sci-mm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=scale9b

set -u

CONFIG_FILE="${SCALING_CONFIG:-configs/jupiter/scaling/qwen3_5_9b.toml}"
OUTPUT_ROOT="${SCALING_OUTPUT_ROOT:-/e/scratch/open-sci-mm/$USER/scaling}"
KEEP_CHECKPOINT="${SCALING_KEEP_CHECKPOINT:-0}"

# `--ntasks-per-node=1` + whatever `--nodes` sbatch was given. Never hardcode
# this: a stale NNODES silently under-subscribes the allocation and the scaling
# curve bends for a reason that has nothing to do with the model.
NNODES=$SLURM_JOB_NUM_NODES
NGPUS=4
NGPU_TOTAL=$(( NNODES * NGPUS ))

nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "[scaling] ${NNODES} nodes / ${NGPU_TOTAL} GPUs, head ${head_node} (${head_node_ip})"

export WANDB_MODE=offline
export QWEN_SECTION_TIMING="${QWEN_SECTION_TIMING:-0}"

# Dynamo/inductor structured trace, for `tlparse`. Unset by default: it writes
# one file per rank and is only wanted for compile diagnostics. Give each job its
# own directory so runs do not interleave into one trace.
if [ -n "${TORCH_TRACE_ROOT:-}" ]; then
    export TORCH_TRACE="${TORCH_TRACE_ROOT}/${SLURM_JOB_ID}"
    mkdir -p "$TORCH_TRACE"
    echo "[scaling] TORCH_TRACE $TORCH_TRACE"
fi
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co

export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

export NCCL_SOCKET_IFNAME="ib,eth"
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export NCCL_BUFFSIZE=2097152

# flash_qla/tilelang JIT-load their cubins with `cuModuleLoadData`, which takes
# GPU memory *outside* PyTorch's caching allocator. Once the allocator has
# reserved almost everything the module load fails with
#   cuModuleLoadData(...) failed with error: CUDA_ERROR_OUT_OF_MEMORY
# and no tensor OOM is ever reported. Expandable segments keep the reservation
# from fragmenting into that space; torch's own OOM messages recommend it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The inductor FX graph cache bakes process-group *names* into compiled
# collectives, and those names are handed out in creation order -- so they
# depend on the mesh shape, not just on the code. A graph cached by a 4-GPU
# run replays "group 35" inside a 512-GPU job that never registered it:
#   RuntimeError: Could not resolve the process group registered under the name 35
# That killed the 16n and 32n points of sweep 20260912_223302 at ~18 min, after
# the 1-node probes had warmed the shared cache. Key the cache on the world
# size: each scale keeps its own warm cache, and no scale can read another's.
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT:-$HOME/.cache}/torch/inductor/ws${NGPU_TOTAL}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
echo "[scaling] inductor cache $TORCHINDUCTOR_CACHE_DIR"

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

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

sleep 5

# override for A/B runs at the same node count, where the default name
# would collide and make two runs indistinguishable in wandb
RUN_NAME="${SCALING_RUN_NAME:-scaling-9b-${NNODES}n}"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${SLURM_JOB_ID}_${NNODES}n"

srun --cpu-bind=none \
        torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPUS \
        --rdzv_id "$SLURM_JOB_ID" \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen \
            --config "$CONFIG_FILE" \
            --wandb.run-name "$RUN_NAME" \
            --training.output-dir "$RUN_OUTPUT_DIR"
rc=$?

# remove the final checkpoint
if [ "$KEEP_CHECKPOINT" = "0" ] && [ -d "$RUN_OUTPUT_DIR" ]; then
    echo "[scaling] removing throwaway checkpoint $RUN_OUTPUT_DIR"
    rm -rf "$RUN_OUTPUT_DIR"
fi

echo "[scaling] done: ${NNODES} nodes, exit ${rc}"
exit $rc
