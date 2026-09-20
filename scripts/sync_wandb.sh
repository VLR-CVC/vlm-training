#!/bin/bash
# Push offline wandb runs from JUPITER to the server. Run it ON a login node:
# compute nodes have no route out, which is why the trainer runs WANDB_MODE=offline.
#
#   scripts/sync_wandb.sh                    # every run from today
#   scripts/sync_wandb.sh 20260917           # one day
#   scripts/sync_wandb.sh '2026091[78]'      # a glob over run directories
#
# WANDB_DIR sends the runs to a cache outside the checkout, so they are not next
# to the job logs -- hence the explicit root here.
set -euo pipefail

RUN_ROOT="${WANDB_RUN_ROOT:-/e/project1/open-sci-mm/$USER/cache/wandb/runs/wandb}"
PATTERN="${1:-$(date +%Y%m%d)}"

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"

shopt -s nullglob
runs=( "$RUN_ROOT"/offline-run-${PATTERN}* )
(( ${#runs[@]} )) || { echo "no runs matching offline-run-${PATTERN}* in $RUN_ROOT" >&2; exit 1; }

echo "syncing ${#runs[@]} run(s) from $RUN_ROOT"
# One at a time: a single unsyncable run should not abort the rest, and the
# output says which one failed.
rc=0
for r in "${runs[@]}"; do
    echo "--- $(basename "$r")"
    wandb sync "$r" || { echo "FAILED: $(basename "$r")" >&2; rc=1; }
done
exit $rc
