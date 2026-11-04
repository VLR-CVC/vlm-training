#!/bin/bash
# Split OV-Instruct into per-subset energon datasets (utils/build_ov_instruct_groups.py).
# 110 groups (109 standalone + 1 'small_subsets' bucket), symlinks only -- no tar is read
# or rewritten, original onevision_instruct/onevision_instruct_split are untouched.
#
# CPU + GPFS I/O only (indexing shard headers), no GPU. Run as its own allocation since
# the largest original tars are tens of GB and indexing all 110 groups serially took
# ~30 min for one group with a 36 GB tar in it -- expect this to run for hours.
#
#   sbatch scripts/build_ov_instruct_groups_mn5.sh
#   WORKERS=8 SMALL_THRESHOLD=5000 sbatch scripts/build_ov_instruct_groups_mn5.sh
#
# Idempotent: a group with .nv-meta/.info.json already present is skipped; a group left
# mid-prepare by a kill has its stale .nv-meta cleared and is retried from scratch.
#SBATCH -D .
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --job-name=ov_groups
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --error=slurm_output/%x-%j.err

set -euo pipefail
source /gpfs/projects/ehpc543/envs/qwen4_mn5/bin/activate
export PYTHONUNBUFFERED=1

SRC=${SRC:-/gpfs/scratch/ehpc543/onevision_instruct}
SRC_CHUNKS=${SRC_CHUNKS:-/gpfs/scratch/ehpc543/onevision_instruct_split}
DST=${DST:-/gpfs/scratch/ehpc543/onevision_instruct_grouped}
SMALL_THRESHOLD=${SMALL_THRESHOLD:-5000}
GIANT_THRESHOLD=${GIANT_THRESHOLD:-10000}
WORKERS=${WORKERS:-6}   # concurrent `energon prepare` subprocesses; GPFS-bound, not CPU-bound -
                       # more than ~6-8 tends to fight itself on shared filesystem I/O rather than help

python3 utils/build_ov_instruct_groups.py --src "$SRC" --src-chunks "$SRC_CHUNKS" --dst "$DST" \
        --small-threshold "$SMALL_THRESHOLD" --giant-threshold "$GIANT_THRESHOLD" --workers "$WORKERS"

echo "groups with .nv-meta: $(find "$DST" -maxdepth 2 -name .info.json | wc -l) / 110 expected"
