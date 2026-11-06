#!/bin/bash
# Throughput / memory probe: Qwen3.5-9B on one JUPITER node (4x GH200), the 2B /
# 35B-A3B production blend, 8K rows, 50 steps. Benchmark only.
#
# The GH200 has 95 GiB; the caching allocator is capped at 60 GiB per GPU
# (per_process_memory_fraction 0.63) so the run stays under 64 GB on the card,
# CUDA context and NCCL buffers included, i.e. would fit an MN5 H100 64GB.
# Allocations past the cap raise OOM instead of spilling into the rest of the HBM.
#
#   scripts/jup_9b_8k_probe.sh                          # MBS 1
#   MBS=2 scripts/jup_9b_8k_probe.sh
#   MBS=4 RESHARD=always scripts/jup_9b_8k_probe.sh     # FSDP frees params after forward
#   DRY_RUN=1 scripts/jup_9b_8k_probe.sh                # print the sbatch line only
#
# The trainer always writes a final checkpoint (~75 GB) to OUT; delete it afterwards.
# No WSD decay, so the 50 steps stay at peak LR after the warmup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEPS="${STEPS:-50}"
MBS="${MBS:-1}"
RESHARD="${RESHARD:-never}"
SEQ=8192
NAME="probe_9b_8k_mbs${MBS}_${RESHARD}"
OUT="${OUT:-/e/scratch/open-sci-mm/ockier1/checkpoints/$NAME}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,per_process_memory_fraction:${MEM_FRACTION:-0.63}"
export CONFIG_FILE=configs/jupiter/qwen3_5_9b.toml
export JUP_EXTRA_ARGS="--data.seq_len $SEQ --data.microbatch_size $MBS \
--training.reshard_after_forward $RESHARD \
--training.total_steps $STEPS --training.wsd_decay_steps 0 --training.wsd_decay_ratio 0 \
--training.save_steps 100000 --training.output_dir $OUT \
--wandb.run_name $NAME"

cmd=(sbatch --nodes=1 --ntasks=1 --time=01:00:00 --job-name="$NAME" scripts/multinode_jup.sh)
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "JUP_EXTRA_ARGS=$JUP_EXTRA_ARGS"
echo "${cmd[*]}"
[ -n "${DRY_RUN:-}" ] && exit 0
"${cmd[@]}"
