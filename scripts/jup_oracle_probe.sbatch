#!/bin/bash -x
#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --partition=booster
#SBATCH --job-name=oracle_probe
#SBATCH --output=logs_jup/%j/log_%x.out
#SBATCH --error=logs_jup/%j/log_%x.err

# Oracle sparse-attention recall on long documents. Single GPU, no sharding.
# Overridable:
#   PROBE_MODEL   HF snapshot (default: the 2B)
#   PROBE_DATA    jsonl, one doc per line with a "text" field (raw-text mode)
#   PROBE_SOURCE  overrides the input flag entirely, e.g.
#                 "--metadataset /path/to.yaml" for the production loader path
#   PROBE_SEQLEN  context length to measure at
#   PROBE_ARGS    extra flags for utils/oracle_sparse_recall.py

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"

command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

REPO="${PROBE_REPO:-/e/project1/open-sci-mm/ockier1/titan-port}"
MODEL="${PROBE_MODEL:-/e/project1/open-sci-mm/ockier1/models/Qwen3.5-2B}"
DATA="${PROBE_DATA:-/e/scratch/open-sci-mm/ockier1/longctx/longctx_top.jsonl}"
SEQLEN="${PROBE_SEQLEN:-65536}"
OUT_DIR="${PROBE_OUT_DIR:-/e/scratch/open-sci-mm/ockier1/oracle_probe}"
mkdir -p "$OUT_DIR"

cd "$REPO"
SOURCE="${PROBE_SOURCE:---data $DATA}"

python utils/oracle_sparse_recall.py \
    --model-dir "$MODEL" \
    $SOURCE \
    --seq-len "$SEQLEN" \
    --budgets 256,512,1024,2048,4096 \
    ${PROBE_MIN_TOKENS:+--min-tokens $PROBE_MIN_TOKENS} \
    --out "$OUT_DIR/oracle_${SLURM_JOB_ID}_${SEQLEN}.json" \
    ${PROBE_ARGS:-}
