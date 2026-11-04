#!/bin/bash
# Submit a sequence-length feasibility probe: one short run of an existing
# config, named after it, on a fixed node count.
#   scripts/jup_seqlen_probe.sh <config-under-configs/jupiter> <nodes> <job-name> [minutes]
set -euo pipefail
CFG="$1"; NODES="$2"; NAME="$3"; MINUTES="${4:-60}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sbatch --nodes="$NODES" --ntasks="$NODES" --time="0:${MINUTES}:00" \
       --job-name="$NAME" --export=ALL,CONFIG_FILE="configs/jupiter/$CFG" \
       scripts/multinode_jup.sh
