#!/bin/bash
#SBATCH --account=open-sci-mm
#SBATCH --job-name=jup_convert_ckpt
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs_jup/%x_%j.out
#SBATCH --error=logs_jup/%x_%j.err

# DCP checkpoint -> HF safetensors snapshot, on a compute node.
#
#   sbatch scripts/jup_convert_ckpt.sbatch <ckpt-dir> <base-snapshot> <out-dir>
#
# Not a login-node job: `utils/dcp_to_hf.py` is single-process CPU and holds the
# whole 35B model plus the converted bf16 copy in host RAM. A booster node has
# 858 GB; a login node does not, and the OOM killer there takes other people's
# work with it. The GPU allocation is incidental -- booster has no CPU-only
# partition -- the conversion never touches it.
#
# The sbatch epilogue in multinode_jup.sh cannot do this: it resolves its helper
# through `dirname $BASH_SOURCE`, which inside a batch job points at SLURM's
# spool copy, not the checkout. That is the `FAILED 127:0` on job 2260406.

set -euo pipefail

CKPT="${1:?usage: sbatch $0 <ckpt-dir> <base-snapshot> <out-dir>}"
BASE="${2:?usage: sbatch $0 <ckpt-dir> <base-snapshot> <out-dir>}"
OUT="${3:?usage: sbatch $0 <ckpt-dir> <base-snapshot> <out-dir>}"

[ -f "$CKPT/.metadata" ] || { echo "not a DCP checkpoint (no .metadata): $CKPT" >&2; exit 2; }
[ -f "$BASE/config.json" ] || { echo "no config.json in base snapshot: $BASE" >&2; exit 2; }

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

echo "=== $CKPT -> $OUT (base: $BASE)"
python utils/dcp_to_hf.py "$CKPT" "$BASE" "$OUT"

# The index is what `from_pretrained` reads; without it the conversion wrote
# nothing usable, however cleanly python exited.
[ -f "$OUT/model.safetensors.index.json" ] || { echo "FAILED: no index written" >&2; exit 1; }
ls -la "$OUT"
