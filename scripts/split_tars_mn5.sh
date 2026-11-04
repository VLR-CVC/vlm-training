#!/bin/bash
# Split oversized tars of an energon dataset into ~2000-sample tars (utils/split_tars.py).
# Resumable: tars already split (a .done.<stem> marker in $DST) are skipped.
#
#   sbatch scripts/split_tars_mn5.sh                                  # OV-Instruct, tars > 10K samples
#   sbatch --cpus-per-task=4 scripts/split_tars_mn5.sh                # a rerun with only a few tars left
#   SRC=... DST=... MIN_SAMPLES=50000 sbatch scripts/split_tars_mn5.sh
#SBATCH -D .
#SBATCH --account=ehpc543
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00
#SBATCH --job-name=split_tars
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --error=slurm_output/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
SRC=${SRC:-/gpfs/scratch/ehpc543/onevision_instruct}
DST=${DST:-/gpfs/scratch/ehpc543/onevision_instruct_split}
MIN_SAMPLES=${MIN_SAMPLES:-10000}
CHUNK=${CHUNK:-2000}

python3 utils/split_tars.py --src "$SRC" --dst "$DST" --min-samples "$MIN_SAMPLES" --chunk "$CHUNK" \
        --workers "$SLURM_CPUS_PER_TASK"
echo "chunk tars in $DST: $(ls "$DST" | grep -c '\.tar$'); done markers: $(ls -A "$DST" | grep -c '^\.done\.')"
