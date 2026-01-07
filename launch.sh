#!/bin/bash

TARGET_SLURM_SCRIPT="$1"

if [ ! -f "$TARGET_SLURM_SCRIPT" ]; then
    echo "Error: Slurm script '$TARGET_SLURM_SCRIPT' not found!"
    exit 1
fi

shift 

COMMIT_HASH=$(git rev-parse --short HEAD)
RUN_ID="${COMMIT_HASH}_$(date +%d_%m_%H_%M)"

EXP_BASE_DIR="/gpfs/scratch/ehpc391/experiments" 
DEST_DIR="${EXP_BASE_DIR}/${RUN_ID}"

echo "Creating snapshot in: $DEST_DIR"
mkdir -p "$DEST_DIR"

git archive --format=tar HEAD | (cd "$DEST_DIR" && tar xf -)

cd "$DEST_DIR"
mkdir -p logs

echo "Submitting $TARGET_SLURqwen3vlM_SCRIPT..."
echo "Passing arguments to "sbatch": $@"

sbatch -A ehpc391 --qos acc_ehpc $TARGET_SLURM_SCRIPT $@
