#!/bin/bash
# Length index of an energon dataset on a CPU-only allocation: one array task per node,
# 80 worker processes each, tars split round-robin across tasks (utils/length_index.py).
#
#   sbatch --array=0-3 scripts/length_index_mn5.sh                     # OV-Instruct, 4 nodes
#   DATA=/gpfs/scratch/ehpc543/finevision_energon COOKER=finevision \
#     OUT=/gpfs/scratch/ehpc543/length_index/finevision sbatch --array=0-3 scripts/length_index_mn5.sh
#
# FineVisionMax (shuffled mixture, 10k shards named data-NNNNN.tar): subsets and quality
# ratings live in each sample's json, not in the tar name:
#   DATA=/gpfs/scratch/ehpc543/finevision_energon COOKER=finevision SUBSET_JSON_KEY=source \
#   EXTRA_FIELDS=num_images,relevance_min,visual_dependency_min,image_correspondence_min,formatting_min \
#   OUT=/gpfs/scratch/ehpc543/length_index/finevision sbatch --array=0-7 scripts/length_index_mn5.sh
# Nemotron (repacked; <key>/<subset>/shard-NNNNNN.tar, one dir per subset, multi-image):
#   DATA=/gpfs/scratch/ehpc543/nemotron_valid GLOB='*/*/*.tar' SUBSET_DIRS=2 COOKER=nemotron \
#   OUT=/gpfs/scratch/ehpc543/length_index/nemotron sbatch --array=0-3 scripts/length_index_mn5.sh
# Mid-training (<source>_<lang>_..._train-N-of-M.tar; subset = "<source>_<lang>", e.g. coyo_EN):
#   DATA=/gpfs/scratch/ehpc543/mid_training COOKER=onevision_midtraining \
#   SUBSET_REGEX='^([^_]+_[^_]+)_' OUT=/gpfs/scratch/ehpc543/length_index/mid_training \
#   sbatch --array=0-7 scripts/length_index_mn5.sh
# IDL (page-level tars, .jpg/.txt/.json per page; json has page/n_pages/score):
#   DATA=/gpfs/scratch/ehpc543/idl_wds COOKER=idl_ocr SUBSET_CONST=idl EXTRA_FIELDS=page,n_pages,score \
#   OUT=/gpfs/scratch/ehpc543/length_index/idl sbatch --array=0-7 scripts/length_index_mn5.sh
# COOKER must be the type_dataset subflavor your dataset actually trains with.
#
# Rerunning is safe: tars that already have a parquet in $OUT are skipped.
# Check the fast path against the real processor first (a minute, one node):
#   VALIDATE=500 sbatch --array=0-0 scripts/length_index_mn5.sh
#SBATCH -D .
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=len_index
#SBATCH --output=slurm_output/%x-%A_%a.out
#SBATCH --error=slurm_output/%x-%A_%a.err

set -euo pipefail
source /gpfs/projects/ehpc543/envs/qwen4_mn5/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1        # 80 processes, not 80 x n threads

DATA=${DATA:-/gpfs/scratch/ehpc543/onevision_instruct}
OUT=${OUT:-/gpfs/scratch/ehpc543/length_index/ov_instruct}
COOKER=${COOKER:-onevision_instruct}
: "${MODEL_DIR:?set MODEL_DIR to the local Qwen3-VL checkpoint dir (the one training.model_dir points at)}"
CHAT_TEMPLATE=${CHAT_TEMPLATE:-assets/chat_template_sft.jinja}
SEQ_LEN=${SEQ_LEN:-24576}

# FineVisionMax is a shuffled mixture: without these the whole index collapses into one
# "unknown" subset and drops the quality-rating columns. Default them for that cooker.
if [ "$COOKER" = "finevision" ]; then
  SUBSET_JSON_KEY=${SUBSET_JSON_KEY:-source}
  EXTRA_FIELDS=${EXTRA_FIELDS:-num_images,relevance_min,visual_dependency_min,image_correspondence_min,formatting_min}
fi

ARGS=(--data "$DATA" --out "$OUT" --model-dir "$MODEL_DIR" --chat-template "$CHAT_TEMPLATE"
      --seq-len "$SEQ_LEN" --cooker "$COOKER" --workers "$SLURM_CPUS_PER_TASK")
[ -n "${SUBSET_REGEX:-}" ] && ARGS+=(--subset-regex "$SUBSET_REGEX")
[ -n "${SUBSET_CONST:-}" ] && ARGS+=(--subset-const "$SUBSET_CONST")
[ -n "${GLOB:-}" ] && ARGS+=(--glob "$GLOB")
[ -n "${SUBSET_DIRS:-}" ] && ARGS+=(--subset-dirs "$SUBSET_DIRS")
[ -n "${SUBSET_JSON_KEY:-}" ] && ARGS+=(--subset-json-key "$SUBSET_JSON_KEY")
[ -n "${EXTRA_FIELDS:-}" ] && ARGS+=(--extra-json-fields "$EXTRA_FIELDS")
[ -n "${VALIDATE:-}" ] && exec python utils/length_index.py "${ARGS[@]}" --validate "$VALIDATE"

python utils/length_index.py "${ARGS[@]}" --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
       --num-tasks "${SLURM_ARRAY_TASK_COUNT:-1}"
