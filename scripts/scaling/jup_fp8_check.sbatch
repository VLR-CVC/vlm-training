#!/bin/bash -x
# Does the fp8 grouped GEMM agree with bf16? One GH200, a few seconds of compute.
#
#   sbatch --output=... --error=... scripts/scaling/jup_fp8_check.sbatch
#
# `torch._scaled_grouped_mm` needs compute capability 9.0/10.0. The dev box has
# sm_120 and sm_89, so this is the ONLY place the fp8 kernel can be exercised --
# every unit test around it mocks the kernel out. Until this passes,
# `moe_grouped_mm = "fp8_rowwise"` is plumbing with unverified numerics.
#
# Compares fp8 vs bf16 forward and backward at the real 35B-A3B expert shapes
# (2048 x 512), for balanced groups, a skew like the measured 3.9x router
# imbalance, and the minimum 16-token group.

#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=booster
#SBATCH --job-name=fp8_check

set -u

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
srun python "${FP8_SCRIPT:-models/tests/fp8_grouped_mm_check.py}"
rc=$?
echo "[fp8] exit $rc"
exit $rc
