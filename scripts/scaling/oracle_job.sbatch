#!/bin/bash -x
#SBATCH --account=open-sci-mm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --exclusive
set -u
ENV=/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main
command -v module >/dev/null 2>&1 || source /etc/profile
# tilelang's check_cuda_availability() probes for the CUDA *toolkit* (nvcc),
# not the runtime, so flash_qla's GDN JIT dies without this even though torch
# sees the GPUs fine.
module load CUDA/13
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29599
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torch/inductor/oracle1"
cd /e/project1/open-sci-mm/ockier1/vlm-training
"$ENV/bin/python" models/tests/oracle_mask.py \
    configs/jupiter/scaling/qwen3_5_9b_oracle_clevr_16384.toml 2
