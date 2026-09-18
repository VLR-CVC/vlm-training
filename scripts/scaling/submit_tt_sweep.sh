#!/bin/bash
# Submit torchtitan Qwen3.5-9B scaling points on JUPITER (runs on the login node).
#
#   submit_tt_sweep.sh <weak|grow|strong> <nodes>...
#   submit_tt_sweep.sh <mode>:<nodes>...      mixed sweeps in one staggered chain,
#                                            e.g. strong:4 grow:8 weak:64 strong:512
#
#   weak    FSDP, 32768 tokens/rank at every size, no accumulation   (logs weak_<n>n_<job>)
#   grow    FSDP, seq_len grown into the memory FSDP frees            (logs grow_<n>n_<job>)
#   strong  DDP, global batch fixed at STRONG_GBS_NODES x 16384 tokens,
#           gradient accumulation absorbs the scale                   (logs strong<G>_<n>n_<job>)
#
# env:
#   STAGGER_MIN=N       each job starts >= N minutes after the previous one *starts*
#                       (--dependency=after:<prev>+N). 0 = let slurm start them together;
#                       ~1,100 nodes launching at once hit ParaStation spawn timeouts.
#   AFTER_JOB=ID        chain the first job after an already-submitted job
#   STRONG_GBS_NODES=G  node count that runs one microbatch per step (default 512)
#   STEPS=N             override the step count (short smoke tests)
#   LOCAL_CACHE=0       use the shared GPFS inductor cache instead of node-local /tmp
#   DRY_RUN=1           print the sbatch commands instead of submitting
#
# Every job's config comes from bench/ttbench/config_registry.py; tt_bench.sbatch
# forwards everything after its own path to torchtitan.train.
set -euo pipefail

TT=${TT:-/e/project1/open-sci-mm/ockier1/torchtitan}
CACHE=${CACHE:-/e/project1/open-sci-mm/ockier1/cache}
STAGGER_MIN=${STAGGER_MIN:-0}
STRONG_GBS_NODES=${STRONG_GBS_NODES:-512}
MICRO=16384

[ $# -gt 0 ] || { echo "usage: submit_tt_sweep.sh <weak|grow|strong> <nodes>... | <mode>:<nodes>..." >&2; exit 2; }
if [[ $1 != *:* ]]; then
  m=$1; shift
  [ $# -gt 0 ] || { echo "no node counts given" >&2; exit 2; }
  set -- "${@/#/$m:}"
fi

# Timeouts: torchtitan defaults (init 300 s, train 100 s) killed 256n/512n while ranks
# were still compiling their first step (jobs 1828665/66: ncclUniqueId wait 300000ms).
COMMON=(--comm.init-timeout-seconds 1800 --comm.train-timeout-seconds 900
        --training.disable-cuda-graphs --compile.enable
        --parallelism.tensor-parallel-degree 4
        --parallelism.fsdp-reshard-after-forward never
        --metrics.log-freq 1)

fsdp_time() { if [ "$1" -le 16 ]; then echo 01:10:00; elif [ "$1" -le 64 ]; then echo 01:20:00; else echo 01:30:00; fi; }

prev=${AFTER_JOB:-}
for item in "$@"; do
  mode=${item%%:*} n=${item#*:}
  case $mode in
    weak)
      tag=weak; config=qwen35_9b_plotqa_16384; time=$(fsdp_time "$n")
      args=(--parallelism.data-parallel-shard-degree -1
            --training.num-tokens-per-microbatch-per-dp-rank 32768 --training.steps "${STEPS:-500}") ;;
    grow)
      # seq_len per node count; calibration in config_registry.py (target peak ~87.1 GiB)
      case $n in
        4) seq=16896 ;; 8) seq=17728 ;; 16|32) seq=18176 ;; 64) seq=18432 ;; 128) seq=18496 ;;
        *) echo "grow: no calibrated seq_len for ${n}n" >&2; exit 2 ;;
      esac
      tag=grow; config=qwen35_9b_plotqa_$seq; time=$(fsdp_time "$n")
      args=(--parallelism.data-parallel-shard-degree -1
            --training.num-tokens-per-microbatch-per-dp-rank $((2 * seq)) --training.steps "${STEPS:-500}") ;;
    strong)
      # steps keep each run to ~30 min of training; small n carries many microbatches/step
      case $n in
        4) steps=20 time=01:00:00 ;; 8) steps=30 time=00:50:00 ;; 16) steps=50 time=00:45:00 ;;
        32) steps=80 time=00:45:00 ;; 64) steps=100 time=00:50:00 ;; 128) steps=150 time=00:55:00 ;;
        256) steps=200 time=01:15:00 ;; 512) steps=300 time=01:30:00 ;;
        *) echo "strong: no step/time budget for ${n}n" >&2; exit 2 ;;
      esac
      [ $((STRONG_GBS_NODES % n)) -eq 0 ] || { echo "strong: ${n}n does not divide ${STRONG_GBS_NODES}" >&2; exit 2; }
      tag=strong$STRONG_GBS_NODES; config=qwen35_9b_plotqa_16384
      args=(--parallelism.data-parallel-replicate-degree "$n" --parallelism.data-parallel-shard-degree 1
            --training.num-tokens-per-microbatch-per-dp-rank $MICRO
            --training.num-tokens-per-train-step $((STRONG_GBS_NODES * MICRO)) --training.steps "${STEPS:-$steps}") ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac

  cache=$CACHE/tt_${mode}
  dep=()
  [ "$STAGGER_MIN" -gt 0 ] && [ -n "$prev" ] && dep=(--dependency="after:$prev+$STAGGER_MIN")

  cmd=(env TT_MODULE=ttbench TT_CONFIG="$config" CACHE_ROOT="$cache"
       sbatch --parsable --nodes="$n" --time="$time" "${dep[@]}"
       --output="$TT/logs/${tag}_${n}n_%j.out" --error="$TT/logs/${tag}_${n}n_%j.err"
       "$TT/tt_bench.sbatch" "${COMMON[@]}" "${args[@]}" activation-checkpoint:none)

  if [ -n "${DRY_RUN:-}" ]; then
    printf '%q ' "${cmd[@]}"; echo
    prev=DRYRUN_${n}n
  else
    mkdir -p "$cache/torch" "$TT/logs"
    prev=$(cd "$TT/repo" && "${cmd[@]}")
    echo "$tag ${n}n $config $prev"
  fi
done
