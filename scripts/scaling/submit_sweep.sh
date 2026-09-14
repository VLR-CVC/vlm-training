#!/bin/bash
# Submit one job per point on the scaling curve.
#
#   ./scripts/scaling/submit_sweep.sh                  # 16 32 64 128 256
#   ./scripts/scaling/submit_sweep.sh 4 8 16           # custom points
#   SCALING_DRY_RUN=1 ./scripts/scaling/submit_sweep.sh   # print, do not submit
#
# Jobs are independent, so SLURM will start the small ones long before the
# 256-node point clears the queue. That is fine -- collect.py reads whatever has
# finished.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ $# -gt 0 ]; then
    NODE_COUNTS=("$@")
else
    NODE_COUNTS=(16 32 64 128 256 512)
fi

CONFIG="${SCALING_CONFIG:-configs/jupiter/scaling/qwen3_5_9b.toml}"
ACCOUNT="${SCALING_ACCOUNT:-open-sci-mm}"
SWEEP_ID="${SCALING_SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SCALING_LOG_ROOT:-$REPO_ROOT/logs_scaling}/$SWEEP_ID"
DRY_RUN="${SCALING_DRY_RUN:-0}"
# Overrides the --time directive in jup_scaling.sbatch. That directive is sized
# for a cold inductor cache (the first step compiles every shape and costs
# ~5 min). Once the cache is warm for these shapes, 25 min is plenty.
WALLTIME="${SCALING_TIME:-}"

[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

echo "sweep   $SWEEP_ID"
echo "config  $CONFIG"
echo "account $ACCOUNT"
echo "logs    $LOG_ROOT"
echo "nodes   ${NODE_COUNTS[*]}"
echo "time    ${WALLTIME:-<sbatch default>}"
echo

for n in "${NODE_COUNTS[@]}"; do
    # SLURM does not create the directories in an --output path; a missing one
    # kills the job at launch with no log to explain it.
    mkdir -p "$LOG_ROOT/${n}n/errors"

    args=(
        --account="$ACCOUNT"
        --nodes="$n"
        --ntasks="$n"
        ${WALLTIME:+--time="$WALLTIME"}
        --job-name="scale9b_${n}n"
        --output="$LOG_ROOT/${n}n/log.out"
        --error="$LOG_ROOT/${n}n/errors/rank_%t.err"
        --export="ALL,SCALING_CONFIG=$CONFIG"
        scripts/scaling/jup_scaling.sbatch
    )

    if [ "$DRY_RUN" != "0" ]; then
        echo "sbatch ${args[*]}"
        continue
    fi

    jobid=$(sbatch --parsable "${args[@]}")
    echo "${n}n -> job $jobid"
    echo "$jobid $n" >> "$LOG_ROOT/jobs.txt"
done

[ "$DRY_RUN" != "0" ] && exit 0

cat <<EOF

submitted. watch with:
    squeue -u \$USER
collect when done:
    python scripts/scaling/collect.py $LOG_ROOT
EOF
