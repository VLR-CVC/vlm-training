#!/bin/bash -x
# Runs one TorchTitan configuration on JUPITER, for TORCHTITAN_BENCHMARK.md.
# Node count comes from sbatch, never from a directive here, so the same file
# serves every phase:
#
#   # gate 0 -- debugmodel, 1 node
#   sbatch --nodes=1 --output=... --error=... scripts/scaling/tt_bench.sbatch \
#       --training.disable-cuda-graphs
#
#   # phase 1 -- 9B at our seq_len, from the ttbench module
#   TT_MODULE=ttbench TT_CONFIG=qwen35_9b_local \
#   sbatch --nodes=1 --output=... --error=... scripts/scaling/tt_bench.sbatch \
#       --training.disable-cuda-graphs --activation-checkpoint none ...
#
# Everything after the script name is forwarded to torchtitan.train verbatim.
#
# `--training.disable-cuda-graphs` is effectively mandatory for this model.
# Image patch counts vary per sample, so the pixel tensor changes shape every
# step and capture fails on step 2 with
#   ValueError: CUDA graph tensor inputs must keep the same shape, dtype, and
#   device, but input 4 changed from (1344, 1536) to (1040, 1536)
# Their own varlen and MoE debug configs set it; qwen35_9b does not.

#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=ttbench

set -u

TT_ROOT="${TT_ROOT:-/e/project1/open-sci-mm/ockier1/torchtitan}"
TT_CONFIG="${TT_CONFIG:-qwen35_debugmodel}"
TT_MODULE="${TT_MODULE:-qwen3_5}"
NGPUS="${TT_NGPUS:-4}"

# From the allocation, not from the directive above: a stale node count silently
# under-subscribes and bends the curve for a reason that is not the framework.
NNODES=$SLURM_JOB_NUM_NODES
NGPU_TOTAL=$(( NNODES * NGPUS ))

# Benchmark-only config functions, kept outside the pinned checkout so the
# reference stays the reference. `--module ttbench` resolves through this.
export PYTHONPATH="${TT_BENCH_PATH:-$TT_ROOT/bench}${PYTHONPATH:+:$PYTHONPATH}"

nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "[ttbench] ${NNODES} nodes / ${NGPU_TOTAL} GPUs, head ${head_node} (${head_node_ip})"

# No outbound network on the compute nodes. cc12m-test is a local directory of
# tars, so datasets must not try to reach the hub to resolve it.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NCCL_SOCKET_IFNAME="ib,eth"
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Compile caches live on node-local /tmp by default. Booster nodes are diskless:
# / (and so /tmp) is an overlay whose writable layer is the /run tmpfs, ~96 GiB of
# host RAM, 819k inodes -- per node, separate from GPU memory. With a shared GPFS cache,
# ~1,100 nodes starting at once (jobs 1828602-15) hung in their first step: ranks
# stalled past the NCCL timeouts while others waited in collectives. Local means
# every job compiles cold, but no rank waits on another node's file locks.
# The per-job directory is removed after training. LOCAL_CACHE=0 restores the
# shared cache: the inductor FX cache bakes process-group names in, so key it on
# world size, and keep it apart from ours -- a torchtitan graph has no business
# being served to train.train_qwen.
if [ "${LOCAL_CACHE:-1}" = 1 ]; then
    LOCAL_CACHE_DIR="/tmp/ttbench_${SLURM_JOB_ID}"
    export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_DIR/inductor"
    export TRITON_CACHE_DIR="$LOCAL_CACHE_DIR/triton"
    srun --ntasks-per-node=1 mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    echo "[ttbench] node-local compile cache $LOCAL_CACHE_DIR ($(df -hT /tmp | awk 'NR==2 {print $2, $5" free"}') on $(hostname))"
else
    export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT:-$HOME/.cache}/torch/inductor_tt/ws${NGPU_TOTAL}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
fi

# torch itself comes from torch_main; the venv only shadows it with torchtitan's
# own dependency set. Activating both in this order is what keeps torch_main
# unmodified -- see TORCHTITAN_BENCHMARK.md §2.2.
CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "$TT_ROOT/venv/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

cd "$TT_ROOT/repo"

# `torchrun` is a console script in torch_main/bin and its shebang hard-pins the
# interpreter to torch_main/bin/python -- which skips the venv's site-packages no
# matter what activate did to PATH, and the job dies with
#   ModuleNotFoundError: No module named 'spmd_types'
# after a pre-flight import in this same shell has just succeeded. Go through the
# venv interpreter explicitly and run the module, never the script.
PY_BIN="$TT_ROOT/venv/bin/python"
"$PY_BIN" -c "import torch, torchtitan, spmd_types, attn_gym; print('[ttbench] torch', torch.__version__, 'cuda', torch.version.cuda, 'spmd_types', spmd_types.__file__)"

srun --cpu-bind=none \
    "$PY_BIN" -m torch.distributed.run \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPUS \
        --rdzv_id "$SLURM_JOB_ID" \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        --local-ranks-filter 0 --role rank --tee 3 \
        -m torchtitan.train \
            --module "$TT_MODULE" \
            --config "$TT_CONFIG" \
            "$@"

# Capture srun's status before anything else runs: a trailing `echo` of its own
# `$?` makes the batch script exit 0 and slurm report COMPLETED for a job whose
# training died at import. Job 1805564 looked like a pass because of it.
rc=$?
if [ -n "${LOCAL_CACHE_DIR:-}" ]; then
    srun --ntasks-per-node=1 bash -c "echo [ttbench] \$(hostname) cache \$(du -sh $LOCAL_CACHE_DIR | cut -f1); rm -rf $LOCAL_CACHE_DIR" | sort
fi
echo "[ttbench] exit $rc"
exit $rc
