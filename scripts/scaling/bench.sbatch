#!/bin/bash
#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --partition=booster
#SBATCH --job-name=benchlayers

set -u
cd /e/project1/open-sci-mm/ockier1/vlm-training

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM THREADS=16 2>/dev/null || true
export OMP_NUM_THREADS=16

source /e/project1/open-sci-mm/ockier1/envs/miniforge3/etc/profile.d/conda.sh
conda activate /e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

PYTHONPATH=. srun python models/tests/bench_layers.py "$@"
