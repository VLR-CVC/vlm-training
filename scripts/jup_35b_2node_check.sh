#!/bin/bash
# Final check of a branch on JUPITER: Qwen3.5-35B-A3B on 2 nodes at production
# parallelism (tp=2, ep=4, FSDP, MBS 1, 32K rows, Base checkpoint, decay blend).
#
# This is the setup of the 2026-10-21 probe, job 2434497 (archived config
# qwen3_5_35b_a3b_32k.toml differs from the 128K config only in the row length,
# so the overrides below reproduce it). Compare the per-step loss, memory and
# MFU against logs_jup/2434497/; that run saw loss 0.4136 at step 1 and
# 9-11% MFU at 68.4 GiB reserved.
#
#   scripts/jup_35b_2node_check.sh             # submit
#   DRY_RUN=1 scripts/jup_35b_2node_check.sh   # print the sbatch line only
#   STEPS=100 scripts/jup_35b_2node_check.sh
#
# No WSD decay: the baseline ran total_steps=2600 and was at peak LR from step 5
# to 2300. A 50-step run keeps the 300-step decay and drops to ~5% LR right
# after warmup (job 2700935: loss up to 0.06 above the baseline from step 8).
#
# The run writes one final checkpoint (~180 GB) to OUT; delete it afterwards.
# The post-run conversion reads output_dir from the TOML, not from the override,
# so it finds no checkpoint there and skips.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEPS="${STEPS:-50}"
OUT="${OUT:-/e/scratch/open-sci-mm/ockier1/checkpoints/final_check_35b_2n}"

export CONFIG_FILE=configs/jupiter/qwen3_5_35b_a3b_128k.toml
export JUP_EXTRA_ARGS="--data.seq_len 32768 --data.tokens_per_microbatch 32768 \
--training.total_steps $STEPS --training.wsd_decay_steps 0 --training.wsd_decay_ratio 0 \
--training.output_dir $OUT \
--wandb.run_name final_check_35b_2n_32k"

cmd=(sbatch --nodes=2 --ntasks=2 --time=01:00:00 --job-name=final_35b_2n scripts/multinode_jup.sh)
echo "CONFIG_FILE=$CONFIG_FILE"
echo "JUP_EXTRA_ARGS=$JUP_EXTRA_ARGS"
echo "${cmd[*]}"
[ -n "${DRY_RUN:-}" ] && exit 0
"${cmd[@]}"
