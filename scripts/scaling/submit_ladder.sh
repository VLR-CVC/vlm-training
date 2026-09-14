#!/bin/bash
# Weak-scaling ladder for the attention benchmark (ATTN_PLAN.md).
#
#   submit_ladder.sh <dataset> <seq_len> <nodes...>
#   submit_ladder.sh clevr 24576 16 32 64 128 256
#
# One job per (variant, node count). Variants differ only in model_dir, so the
# four arms see identical parallelism, data and step count -- the only thing
# that moves is the attention implementation.
set -eu
cd /e/project1/open-sci-mm/ockier1/vlm-training

DS=${1:?dataset: clevr|plotqa}; SL=${2:?seq_len}; shift 2
NODES=${*:?node counts}
L=logs_scaling/attnbench; mkdir -p $L

for n in $NODES; do
    for v in hybrid fullattn qsa512 allqsa512; do
        C=configs/jupiter/scaling/qwen3_5_9b_${v}_${DS}_${SL}.toml
        [ -f "$C" ] || { echo "missing $C"; exit 1; }
        # 256 nodes queue slowly and compile once per node; give the tail room.
        T=00:25:00; [ "$n" -ge 128 ] && T=00:40:00
        J=$(sbatch --parsable --nodes="$n" --time=$T \
            --job-name="ab-${v}-${n}n" \
            --output="$L/%j_${v}_${DS}_${SL}_${n}n.out" \
            --error="$L/%j_${v}_${DS}_${SL}_${n}n.err" \
            --export=ALL,QWEN_SECTION_TIMING=1,SCALING_CONFIG=$C,SCALING_RUN_NAME=attn-${v}-${DS}-${SL}-${n}n \
            scripts/scaling/jup_scaling.sbatch)
        echo "$J  $v  $DS  $SL  ${n}n"
    done
done
