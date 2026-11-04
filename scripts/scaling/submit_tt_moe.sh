#!/bin/bash
# Submit the torchtitan Qwen3.5 MoE arms on JUPITER (runs on the login node).
#
#   scripts/scaling/submit_tt_moe.sh 35b 122b 397b
#   NODES=32 TIME=01:30:00 STEPS=100 scripts/scaling/submit_tt_moe.sh 122b
#   DRY_RUN=1 scripts/scaling/submit_tt_moe.sh 35b
#
# env: NODES TIME STEPS SEQ_LEN PP MICRO WANDB_PROJECT DRY_RUN
#
# Metrics go to wandb *offline* -- compute nodes have no outbound network, so the
# run directory has to be replayed from the login node afterwards:
#
#   wandb sync /e/scratch/open-sci-mm/$USER/ttbench/<tag>/tb/*/wandb/offline-run-*
#
# Nothing before this script logged to wandb at all: MetricsProcessor.Config has
# enable_wandb = False by default and no ttbench config overrode it, so every
# earlier number came from grepping the .out file.
set -euo pipefail

TT=${TT:-/e/project1/open-sci-mm/ockier1/torchtitan}
NODES=${NODES:-16}
TIME=${TIME:-00:45:00}
STEPS=${STEPS:-60}
SEQ_LEN=${SEQ_LEN:-16384}
WANDB_PROJECT=${WANDB_PROJECT:-ttbench_moe}
SCRATCH=${SCRATCH:-/e/scratch/open-sci-mm/$USER/ttbench}

[ $# -gt 0 ] || { echo "usage: submit_tt_moe.sh <35b|122b|397b>..." >&2; exit 2; }

for arm in "$@"; do
  case $arm in
    35b)   config=qwen35_35b_a3b_plotqa;    pp=${PP:-4} ;;
    122b)  config=qwen35_122b_a10b_plotqa;  pp=${PP:-4} ;;
    397b)  config=qwen35_397b_a17b_plotqa;  pp=${PP:-8} ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac

  # PP, not TP. These decoders have n_kv_heads = 2, so TP above 2 is rejected
  # outright ("tensor_parallel_degree (4) must divide n_kv_heads (2)") -- the
  # stock tp=4 / tp=8 recipes cannot run as shipped. PP splits by layer instead,
  # so it has no such divisor, and it keeps every collective off InfiniBand that
  # TP would have put there (4 GH200 per node, NVLink stops at the node).
  #
  # ep is 8 on every arm. With tp = cp = etp = 1 the constraints in
  # job_config.py collapse to `dp_shard % ep == 0`, so dp_shard -- which is
  # NODES*4/pp -- has to stay a multiple of 8. That is what picks pp:
  #   16 nodes ->  64 GPUs, pp=4 -> dp_shard=16
  #   64 nodes -> 256 GPUs, pp=8 -> dp_shard=32
  ep=8
  gpus=$((NODES * 4))
  dp_shard=$((gpus / pp))
  if [ $((gpus % pp)) -ne 0 ] || [ $((dp_shard % ep)) -ne 0 ]; then
    echo "$arm: pp=$pp over $gpus GPUs gives dp_shard=$dp_shard, not a multiple of ep=$ep" >&2
    exit 2
  fi

  # 1F1B needs at least pp microbatches to have anything to pipeline, and a
  # microbatch cannot be less than one packed row -- so the rank's token budget
  # is pp rows, split pp ways.
  micro=${MICRO:-$((pp * SEQ_LEN))}

  par=(--parallelism.data-parallel-shard-degree -1
       --parallelism.tensor-parallel-degree 1
       --parallelism.expert-parallel-degree "$ep"
       --parallelism.pipeline-parallel-degree "$pp"
       --parallelism.num-pp-microbatches "$pp")

  tag="moe${arm}_${NODES}n"
  dump="$SCRATCH/$tag"
  mkdir -p "$dump" "$TT/logs"

  cmd=(env TT_MODULE=ttbench TT_CONFIG="$config"
       WANDB_PROJECT="$WANDB_PROJECT" WANDB_RUN_NAME="$tag" WANDB_RUN_GROUP="moe_${NODES}n"
       sbatch --parsable --nodes="$NODES" --time="$TIME"
       --output="$TT/logs/${tag}_%j.out" --error="$TT/logs/${tag}_%j.err"
       "$TT/tt_bench.sbatch"
         --dump-folder "$dump"
         --metrics.enable-wandb --metrics.log-freq 1
         --training.disable-cuda-graphs --compile.enable
         # stock recipes ask for 4 x max_context_length per rank: a pretraining
         # batch, not a benchmark microbatch, and it OOMs at these sizes.
         --training.num-tokens-per-microbatch-per-dp-rank "$micro"
         --training.steps "$STEPS"
         --comm.init-timeout-seconds 1800 --comm.train-timeout-seconds 900
         "${par[@]}")

  if [ -n "${DRY_RUN:-}" ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    jobid=$(cd "$TT/repo" && "${cmd[@]}")
    echo "$tag $config $jobid"
  fi
done
