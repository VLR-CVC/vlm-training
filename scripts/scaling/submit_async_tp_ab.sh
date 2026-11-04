#!/bin/bash
# Does async TP actually buy anything on GH200? Two identical runs, one flag apart.
#
#   scripts/scaling/submit_async_tp_ab.sh              # 1 node, 300 steps
#   NODES=4 STEPS=500 scripts/scaling/submit_async_tp_ab.sh
#
# Read the result with (NOT collect_probe.sh -- see below):
#   python scripts/scaling/collect_steady.py logs_scaling/<id>_off logs_scaling/<id>_on
#
# Async TP micro-pipelines the TP collectives into the GEMMs that feed them, so it
# only has anything to work with when TP>1 and `sequence_parallel=true` -- without
# SP those layers emit an all-reduce instead of the all-gather/reduce-scatter pair
# the pass rewrites. configs/jupiter/qwen3_vl_2b.toml is tp_size=2 and SP defaults
# to true, so it qualifies as-is.
#
# Numerics are already settled: async TP passed the TP parity gate on 2 local
# Blackwell cards (top1 1.0000, loss rel 1.4e-03, grad norm rel 3.6e-03) with
# `_micro_pipeline_tp=True` and 169 fused_all_gather_matmul / 198
# fused_matmul_reduce_scatter in the generated code. THIS script is only about
# speed, which that run could not answer: those cards are PCIe-only, and
# micro-pipelining is a bet on fast intra-node links.
#
# Why collect_steady.py and not collect_probe.sh: collect_probe averages every step
# after the 4th, which on this workload is still deep in warmup. collect_steady
# finds where the stall rate settles and reports the mean after that, keeping the
# residual stalls in. See METRICS.md.
#
# precompile=true on BOTH arms. It moves the compile cost before the loop so the
# steady-state window is reached in tens of steps instead of hundreds. It is not a
# thumb on the scale -- both arms get it -- but it does mean this measures steady
# state, not time-to-first-useful-step. Async TP changes the compiled graph, so the
# two arms compile separately no matter what.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NODES="${NODES:-1}"
STEPS="${STEPS:-300}"
CONFIG="${CONFIG:-configs/jupiter/qwen3_vl_2b.toml}"
STAMP="${ASYNC_AB_ID:-asynctp_$(date +%Y%m%d_%H%M%S)}"

# Booleans are bare flags, not `--flag true`: tyro renders `async_tp: bool` as a
# switch, so `--training.async-tp true` is a parse error. It does not fail
# usefully -- the config sets `adam_betas = [0.9, 0.95]`, a TOML list where the
# dataclass declares tuple[float, float], and tyro asserts on that while building
# the help text it prints for the *real* error. So a wrong override here reports
# adam_betas and nothing about the flag you got wrong. Arm A passes no flag at
# all (async_tp defaults to false).
submit() {  # submit <arm> [extra flag]
    LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 \
    SCALING_TIME="${SCALING_TIME:-00:40:00}" \
    SCALING_SWEEP_ID="${STAMP}_$1" \
    SCALING_RUN_NAME="${STAMP}_$1" \
    SCALING_CONFIG="$CONFIG" \
    SCALING_EXTRA_ARGS="--training.total-steps $STEPS --training.save-steps 100000 \
--training.precompile ${2:-} ${ASYNC_EXTRA:-}" \
        scripts/scaling/submit_sweep.sh "$NODES"
}

echo "### arm A: async_tp = false (no flag)"
submit off
echo
echo "### arm B: async_tp = true"
submit on --training.async-tp

cat <<EOF

run ids: ${STAMP}_off  ${STAMP}_on

    python scripts/scaling/collect_steady.py logs_scaling/${STAMP}_off logs_scaling/${STAMP}_on

Check arm B really enabled it before believing any delta -- the trainer logs
"Async TP is enabled" once per run, and _maybe_enable_async_tp returns silently
when there is no TP mesh:

    grep -c "Async TP is enabled" logs_scaling/${STAMP}_on/${NODES}n/log.out
EOF
