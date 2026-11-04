#!/bin/bash
# Run one torchtitan config on a single node, interactively -- no sbatch.
# Same environment as tt_bench.sbatch, minus the batch plumbing.
#
# Get a node first (this part still goes through slurm, there is no way around it):
#
#   salloc --account=open-sci-mm --partition=booster --nodes=1 \
#          --gpus-per-node=4 --cpus-per-task=288 --time=00:30:00
#   srun --pty bash -i
#
# then, on the compute node:
#
#   scripts/scaling/run_tt.sh                                  # qwen35_debugmodel_moe, 10 steps
#   STEPS=40 scripts/scaling/run_tt.sh --compile.enable
#   TT_CONFIG=qwen35_9b_local NGPUS=4 scripts/scaling/run_tt.sh
#   TT_TOML=$TT/repo/torchtitan/models/qwen3/train_configs/qwen3_moe_debug.toml \
#       scripts/scaling/run_tt.sh --training.dataset c4_test
#
# Everything after the script name is forwarded to torchtitan.train.
set -euo pipefail

TT_ROOT="${TT_ROOT:-/e/project1/open-sci-mm/ockier1/torchtitan}"
TT_CONFIG="${TT_CONFIG:-qwen35_debugmodel_moe}"
TT_MODULE="${TT_MODULE:-qwen3_5}"
NGPUS="${NGPUS:-4}"

export PYTHONPATH="${TT_BENCH_PATH:-$TT_ROOT/bench}${PYTHONPATH:+:$PYTHONPATH}"

# Compute nodes have no outbound network.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Kept across runs, unlike the batch script's per-job directory: interactively you
# rerun the same config over and over and a cold compile is ~5 minutes each time.
# Booster nodes are diskless, so this is the /run tmpfs -- it dies with the
# allocation either way.
CACHE_DIR="/tmp/ttbench_${USER}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# torch comes from torch_main; the venv only shadows it with torchtitan's own
# dependency set. This order is what leaves torch_main unmodified.
CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "$TT_ROOT/venv/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited

cd "$TT_ROOT/repo"

if [ -n "${TT_TOML:-}" ]; then
    CFG_ARGS=(--job.config_file "$TT_TOML")
else
    CFG_ARGS=(--module "$TT_MODULE" --config "$TT_CONFIG")
fi

# `torchrun` is a console script in torch_main/bin and its shebang hard-pins the
# interpreter to torch_main/bin/python, which skips the venv's site-packages no
# matter what activate did to PATH -- ModuleNotFoundError: No module named
# 'spmd_types'. Go through the venv interpreter and run the module.
PY_BIN="$TT_ROOT/venv/bin/python"
"$PY_BIN" -c "import torch, torchtitan, spmd_types, attn_gym; print('[run_tt] torch', torch.__version__, 'cuda', torch.version.cuda)"

exec "$PY_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node="$NGPUS" \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
        "${CFG_ARGS[@]}" \
        --training.disable-cuda-graphs \
        --training.steps "${STEPS:-10}" \
        --metrics.log-freq 1 \
        "$@"
