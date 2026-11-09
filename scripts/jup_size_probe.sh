#!/bin/bash
# Throughput / memory probe across Qwen3.5 sizes on one JUPITER node (4x GH200,
# 96 GB, no memory cap): the 2B / 35B-A3B production recipe and blend
# (configs/jupiter/qwen3_5_9b.toml), 16K rows, reshard never, 50 steps. Benchmark only.
#
# The default MBS per size aims at ~80 GiB allocated (estimated from the 9B 8K
# probe), leaving allocator headroom on the 94.5 GiB the GH200 exposes. The 0.8B at
# MBS 22 peaked at 79.6 GiB but its host RSS reached 442 GB by step 31 and the
# node OOM killer stopped it (job 2788457).
#
#   SIZE=0_8b scripts/jup_size_probe.sh     # FSDP 4, MBS 22
#   SIZE=2b   STEPS=30 scripts/jup_size_probe.sh  # FSDP 4, MBS 11 (~8 GiB static +
#                                            # 6.9 GiB per row, from job 2274993)
#   SIZE=4b  scripts/jup_size_probe.sh     # FSDP 4, MBS 8 (10 OOMs in step 1, job 2788458)
#   SIZE=9b   STEPS=30 scripts/jup_size_probe.sh  # FSDP 4, MBS 4: 77.5 GiB, 15% faster
#                                            # than TP 2 x FSDP 2 MBS 8 (same GBS);
#                                            # MBS 6 OOMs in step 1 (job 2789484)
#   MBS=4 SIZE=9b scripts/jup_size_probe.sh # override the micro-batch
#   TP=2 MBS=8 SIZE=9b scripts/jup_size_probe.sh  # TP 2 x FSDP 2 (MBS 12 OOMs, job 2788459)
#   DRY_RUN=1 SIZE=4b scripts/jup_size_probe.sh
#
# Weights are randomly initialised: throughput and memory do not depend on them,
# and only the 2B, 9B and 35B Base checkpoints are on JUPITER. The 2B-Base
# directory supplies the tokenizer and processor, which the family shares (same
# patch 16, merge 2, temporal patch 2). Losses are therefore meaningless.
# The trainer writes a final checkpoint to OUT; delete it afterwards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SIZE="${SIZE:?SIZE=0_8b|2b|4b|9b}"
case "$SIZE" in
    0_8b) DEFAULT_TP=1; DEFAULT_MBS=22 ;;
    2b)   DEFAULT_TP=1; DEFAULT_MBS=11 ;;
    4b)   DEFAULT_TP=1; DEFAULT_MBS=8 ;;
    9b)   DEFAULT_TP=1; DEFAULT_MBS=4 ;;
    *) echo "unknown SIZE $SIZE" >&2; exit 1 ;;
esac
MBS="${MBS:-$DEFAULT_MBS}"
TP="${TP:-$DEFAULT_TP}"
STEPS="${STEPS:-50}"
SEQ=16384
# INIT=hf loads the Base checkpoint (2B and 9B only), as production does
INIT="${INIT:-random}"
if [ "$INIT" = hf ]; then
    INIT_ARGS="--training.model_dir /e/project1/open-sci-mm/ockier1/models/Qwen3.5-${SIZE^^}-Base"
else
    INIT_ARGS="--training.random_init --training.model_dir /e/project1/open-sci-mm/ockier1/models/Qwen3.5-2B-Base"
fi
NAME="probe_${SIZE}_16k_tp${TP}_mbs${MBS}_never"; [ "$INIT" = hf ] && NAME="${NAME}_hf"
OUT="${OUT:-/e/scratch/open-sci-mm/ockier1/checkpoints/$NAME}"

export CONFIG_FILE=configs/jupiter/qwen3_5_9b.toml
export JUP_EXTRA_ARGS="--model.model_name Qwen/Qwen3.5-$SIZE --model.model_config configs/models/qwen3_5_$SIZE.json \
$INIT_ARGS \
--training.tp_size $TP --training.reshard_after_forward never \
--data.seq_len $SEQ --data.microbatch_size $MBS \
--training.total_steps $STEPS --training.wsd_decay_steps 0 --training.wsd_decay_ratio 0 \
--training.save_steps 100000 --training.output_dir $OUT \
--wandb.run_name $NAME"

cmd=(sbatch --nodes=1 --ntasks=1 --time=01:00:00 --job-name="$NAME" scripts/multinode_jup.sh)
echo "CONFIG_FILE=$CONFIG_FILE"
echo "JUP_EXTRA_ARGS=$JUP_EXTRA_ARGS"
echo "${cmd[*]}"
[ -n "${DRY_RUN:-}" ] && exit 0
"${cmd[@]}"
