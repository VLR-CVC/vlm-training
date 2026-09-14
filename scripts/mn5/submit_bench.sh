#!/bin/bash
# Submit the Qwen3-VL-2B config sweep.
#   ./scripts/mn5/submit_bench.sh 4 base vis_off seq8192 ...
#   ./scripts/mn5/submit_bench.sh 4 all
set -eu
N=${1:?node count}; shift
ARMS="${*:?arm names, or 'all'}"
[ "$ARMS" = "all" ] && ARMS=$(ls configs/mn5/bench/*.toml | xargs -n1 basename | sed 's/\.toml$//')
mkdir -p logs_bench
for a in $ARMS; do
    C="configs/mn5/bench/$a.toml"
    [ -f "$C" ] || { echo "missing $C"; exit 1; }
    J=$(sbatch --parsable --nodes="$N" --job-name="b2b-$a" \
        --output="logs_bench/%j_${a}_${N}n.out" --error="logs_bench/%j_${a}_${N}n.err" \
        --export=ALL,MN5_CONFIG="$C" scripts/mn5/bench.sbatch)
    echo "$J  $a  ${N}n"
done
