#!/bin/bash
# Does a persistent inductor cache remove the cold-compile floor at 32 nodes?
#
#   scripts/scaling/submit_cache_ab.sh            # seeded tmpfs cache (use this)
#   MODE=gpfs scripts/scaling/submit_cache_ab.sh  # compile straight against GPFS
#
# MODE=seed (default) copies the node's cache in from GPFS before the run and back
# out after, so compiles hit tmpfs and the cache still survives the job. Run it
# twice: the first invocation writes the cache, the second is the measurement.
#
# MODE=gpfs is kept only to document the dead end. Job 1896752 ran that way and
# spent 21 min without reaching step 1, where the identical shape on tmpfs
# (job 1891361) ran all 500 steps in 21.5 min -- GPFS cache writes stall the
# compile itself, which is the whole reason LOCAL_CACHE exists.
#
# Baseline is job 1894341 (weak_vl2b_fsdp shape: SHARD=0, TP=2, seq 32768, 500
# steps), which ran with LOCAL_CACHE=1 -- a node-local /tmp cache wiped at the end
# of the job, so every run compiles cold. `tlparse` on the 1-node trace (job
# 1894616) put 31.2s of the ~73s per-rank compile budget in compile_id 0/0, a
# single cold-start compile that every rank pays in lockstep. A pre-compile pass
# cannot touch it; only a warm cache can. See WARMUP_PLAN.md, Option C.
#
# READ THIS BEFORE INTERPRETING THE RESULT. `numa_wrapper.sh` appends
# /n${GROUP_RANK} to TORCHINDUCTOR_CACHE_DIR unconditionally, so nothing is shared
# BETWEEN the 32 nodes in either mode -- each node keeps its own directory. The win
# is persistence across jobs, not sharing within one:
#
#   run 1 (cold):  populates ws128/n0..n31. No speedup expected; this run pays the
#                  full cold compile and exists to write the cache.
#   run 2 (warm):  the measurement. Node i re-reads node i's cache from run 1, so
#                  compile_id 0/0 should come back a cache hit.
#
# Run this script twice. The first invocation is the cache populator.
#
# The per-node split is deliberate and load-bearing: inductor bakes process-group
# *names* into compiled collectives, and sharing one cache across TP groups gave
#   RuntimeError: Could not resolve the process group registered under the name 16
# (job 1782272). Flattening the split so one node's compile serves the other 31
# would be the bigger win and is the next thing to try, but it needs the cache
# keyed on TP group identity first -- this config runs TP=2 on 4 GPUs/node, i.e.
# two TP groups per node, which is exactly the case numa_wrapper.sh warns about.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NODES="${NODES:-32}"
MODE="${MODE:-seed}"
STAMP="${CACHE_AB_ID:-cache_ab_${MODE}_$(date +%Y%m%d_%H%M%S)}"

# Pinned, NOT `${CACHE_ROOT:-...}`: the login profile already exports a CACHE_ROOT,
# and the whole point of this A/B is that run 2 reads what run 1 wrote. Inheriting
# the path would silently break that the moment the script runs from a context with
# a different profile. Not $HOME either -- one directory per node per world size is
# not what the GPFS home quota is sized for. Project, not scratch, so a purge
# between the two runs cannot quietly turn the measurement back into a cold run.
CACHE_BASE=/e/project1/open-sci-mm/ockier1/cache/inductor_ab

# The two modes must not share a directory: `seed` stores <gpfs>/wsN/n<K>/inductor
# while `gpfs` stores <gpfs>/torch/inductor/wsN/n<K>, and crossing them would make
# one mode silently read a cold cache and look like a null result.
if [ "$MODE" = seed ]; then
    export SEED_CACHE="$CACHE_BASE/seeded"
    CACHE_PROBE="$SEED_CACHE/ws$((NODES * 4))"
else
    export LOCAL_CACHE=0
    export CACHE_ROOT="$CACHE_BASE/direct"
    CACHE_PROBE="$CACHE_ROOT/torch/inductor/ws$((NODES * 4))"
fi

# Per-rank recompile events and the structured compile trace. 1894341 has the
# recompile logs but no trace, so the 32-node spread in WARMUP_PLAN.md is
# extrapolated from 4 ranks; this closes that. ~27 MB/rank => ~3.5 GB.
TRACE_ROOT="${TRACE_ROOT:-/e/scratch/open-sci-mm/ockier1}"
export QWEN_TORCH_LOGS_DIR="$TRACE_ROOT/torchlogs/cache_ab"
export QWEN_TORCH_TRACE_DIR="$TRACE_ROOT/torchtrace/cache_ab"
export TORCH_LOGS="recompiles"

echo "cache A/B: ${NODES}n, MODE=$MODE, cache $CACHE_PROBE"
if [ -z "$(ls -A "$CACHE_PROBE" 2>/dev/null)" ]; then
    echo "  cache empty: THIS RUN IS THE COLD POPULATOR, run the script again after it"
else
    echo "  cache present ($(du -sh "$CACHE_PROBE" 2>/dev/null | cut -f1)): this run reads it, so this is the measurement"
fi

SHARD=0 TP=2 STEPS=500 \
HSDP_CONFIG=configs/jupiter/qwen3_vl_2b.toml \
SCALING_TIME="${SCALING_TIME:-01:30:00}" \
HSDP_SWEEP_ID="$STAMP" \
    scripts/scaling/submit_hsdp_sweep.sh "$NODES"

cat <<EOF

compare against job 1894341 (same shape, LOCAL_CACHE=1) with:
    python scripts/scaling/collect_steady.py logs_scaling/$STAMP
    python scratchpad/cid_timeline.py $QWEN_TORCH_TRACE_DIR/<jobid>
success = compile_id 0/0 gone (cache hit) on the SECOND run of this script.
EOF
