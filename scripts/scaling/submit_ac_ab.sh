#!/bin/bash
# Activation checkpointing A/B/C on JUPITER: off vs whole-block vs per-op SAC.
#
#   scripts/scaling/submit_ac_ab.sh                 # 1 node, 300 steps
#   NODES=2 STEPS=500 scripts/scaling/submit_ac_ab.sh
#
# Read with:
#   python scripts/scaling/collect_steady.py logs_scaling/<id>_{off,layer,op}
#   grep -h "activation checkpointing" logs_scaling/<id>_{layer,op}/<N>n/log.out
#
# WHY THIS RUNS HERE AND NOT LOCALLY. The local A/B is unusable: the workstation
# is shared, and a co-tenant's process on the same card time-slices the SMs. Two
# `off` arms running identical code came out 31% apart on throughput (1,877 vs
# 2,458 tok/s/GPU), which is far wider than the effect being measured. Peak
# memory was stable across those runs (60.9 GiB both times) and is the more
# robust of the two metrics, but the throughput column decided nothing. A
# JUPITER node is exclusive (`#SBATCH --exclusive` in jup_scaling.sbatch), so
# both columns mean something here.
#
# Three arms, one flag apart:
#   off    no AC
#   layer  whole-block recompute on every 2nd block (upstream's default)
#   op     per-op SAC on every block: keep attention / GDN / reduce-scatter and
#          every other matmul, recompute the rest
#
# `selective_ac_op` ignores the frequency by design -- op-level applies to every
# block, as upstream does -- so the `op` arm still passes freq=2 only to switch
# AC on.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NODES="${NODES:-1}"
STEPS="${STEPS:-300}"
CONFIG="${CONFIG:-configs/jupiter/qwen3_vl_2b.toml}"
STAMP="${AC_AB_ID:-ac_$(date +%Y%m%d_%H%M%S)}"

submit() {  # submit <arm> [extra flags]
    LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 \
    SCALING_TIME="${SCALING_TIME:-00:40:00}" \
    SCALING_SWEEP_ID="${STAMP}_$1" \
    SCALING_RUN_NAME="${STAMP}_$1" \
    SCALING_CONFIG="$CONFIG" \
    SCALING_EXTRA_ARGS="--training.total-steps $STEPS --training.save-steps 100000 \
--training.precompile ${2:-} ${AC_EXTRA:-}" \
        scripts/scaling/submit_sweep.sh "$NODES"
}

echo "### arm off"
submit off
echo
echo "### arm layer (whole-block, every 2nd)"
submit layer "--training.selective-ac-freq 2"
echo
echo "### arm op (per-op SAC, every block)"
submit op "--training.selective-ac-freq 2 --training.selective-ac-op"

cat <<EOF

run ids: ${STAMP}_off  ${STAMP}_layer  ${STAMP}_op

    python scripts/scaling/collect_steady.py \\
        logs_scaling/${STAMP}_off logs_scaling/${STAMP}_layer logs_scaling/${STAMP}_op

Confirm each arm did what it claims before reading any delta -- the "off" arm
must print nothing, and the other two must print their wrapping line:

    grep -h "activation checkpointing" logs_scaling/${STAMP}_{off,layer,op}/${NODES}n/log.out
EOF
