#!/bin/bash
# Run the trainer on ONE node inside an allocation you already hold.
#
#   salloc --account=open-sci-mm --partition=booster --nodes=1 \
#          --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 --time=02:00:00
#   ./scripts/scaling/debug_1node.sh                       # a few steps, memory report
#   ./scripts/scaling/debug_1node.sh --training.master-dtype bfloat16
#   ./scripts/scaling/debug_1node.sh --training.compile true
#
# Everything after the script name is passed straight to `train.train_qwen`, so
# one held allocation covers a whole afternoon of one-variable experiments
# without queueing for 16 nodes each time.
#
# WHAT THIS IS AND IS NOT REPRESENTATIVE OF
#
# Faithful: the model, tp_size=4 inside the node, the compiled path, dtypes, the
# optimizer, the dataloader. Every failure this repo hit at 16 nodes except one
# reproduces here.
#
# NOT faithful: absolute memory. The sweep runs dp=16, and FSDP shards optimizer
# state across those 16 ranks; here dp=1 shards it across nothing, so each GPU
# holds far more. Read memory numbers as *relative* -- "master_dtype=bfloat16
# cut allocated by X%" extrapolates, "it fits in 95 GiB" does not.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${DEBUG_CONFIG:-configs/jupiter/qwen3_5_9b.toml}"
STEPS="${DEBUG_STEPS:-6}"
NGPUS="${DEBUG_NGPUS:-4}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "no allocation held. Get one with:" >&2
    echo "  salloc --account=open-sci-mm --partition=booster --nodes=1 \\" >&2
    echo "         --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 --time=02:00:00" >&2
    exit 1
fi

export WANDB_MODE=disabled          # a throwaway run should not make wandb runs
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NCCL_SOCKET_IFNAME="ib,eth"
export NCCL_P2P_LEVEL=NVL
export PYTHONFAULTHANDLER=1
# the allocator message every OOM here prints recommends exactly this
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
# The trainer imports spmd_types and attn_gym (with its CuTeDSL backend), which
# torch_main does not carry: layer the torchtitan venv on top, as tt_bench.sbatch
# does, so torch_main stays unmodified. torchrun is --no-python, so ranks resolve
# `python` from PATH, i.e. the venv interpreter.
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

OUT_DIR="${DEBUG_OUTPUT_ROOT:-/e/scratch/open-sci-mm/$USER/debug}/${SLURM_JOB_ID}"
LOG_DIR="$REPO_ROOT/logs_scaling/debug/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR/errors"
STAMP=$(date +%H%M%S)
LOG="$LOG_DIR/run_${STAMP}.log"

echo "[debug] config    $CONFIG"
echo "[debug] steps     $STEPS  gpus $NGPUS  (dp=1, tp from the config)"
echo "[debug] overrides $*"
echo "[debug] log       $LOG"

srun --nodes=1 --ntasks=1 --cpu-bind=none \
    torchrun \
        --nnodes=1 \
        --nproc_per_node="$NGPUS" \
        --rdzv_id "$SLURM_JOB_ID" \
        --rdzv_backend c10d \
        --rdzv_endpoint="127.0.0.1:$(shuf -i 20000-29999 -n 1)" \
        --no-python \
        ./numa_wrapper.sh python -m train.train_qwen \
            --config "$CONFIG" \
            --training.total-steps "$STEPS" \
            --training.save-steps 100000 \
            --training.output-dir "$OUT_DIR" \
            "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

rm -rf "$OUT_DIR"          # `end_run` always checkpoints; none of it is read back

echo
echo "=== [debug] summary (exit $rc) ==="
sed -e 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "peak|tps |Number params" | tail -5
sed -e 's/\x1b\[[0-9;]*m//g' "$LOG" \
    | grep -oE "Tried to allocate [0-9.]+ GiB|[0-9.]+ GiB is allocated by PyTorch|[A-Za-z_.]*(Error|Exception): .{0,80}" \
    | sort -u | head -5
echo "  full log: $LOG"
exit $rc
