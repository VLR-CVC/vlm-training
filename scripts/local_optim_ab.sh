#!/bin/bash
# Low-precision optimizer A/B on local cards: does bf16 master + stochastic
# rounding track an fp32 master, and does round-to-nearest fail the way the
# `train/adamw_sr.py` docstring says it does?
#
#   scripts/local_optim_ab.sh                 # 2 GPUs, 120 steps, 3 arms
#   GPUS=6,7 STEPS=300 scripts/local_optim_ab.sh
#
# Arms:
#   fp32     master fp32, plain foreach AdamW          -- the reference
#   bf16_sr  master bf16, foreach_sr, SR on            -- what we want to ship
#   bf16_rn  master bf16, foreach_sr, SR off           -- the control that stalls
#
# `set_determinism` pins seed 42 unconditionally, so all three arms see the same
# init and the same data order. Any divergence in the loss is the optimizer.
#
# Defaults to the L40S pair (6,7): GPUs 0-3 are usually busy. The text config is
# random_init, so this needs no checkpoint -- only the tokenizer and the dataset.
#
# NCCL_P2P_DISABLE=1 is not optional on this box. Peer-to-peer between these
# cards hangs the first FSDP all-gather: `WorkNCCL(SeqNum=1, OpType=COALESCED)`
# sits for the full 600 s watchdog and the job dies with no step logged. Same
# reason `scripts/local_text_2gpu.sh` sets it. Every collective goes through host
# memory instead, so read step time as this box's, not the model's.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-6,7}"
STEPS="${STEPS:-120}"
CONFIG="${CONFIG:-configs/text/qwen3_5_2b_2gpu.toml}"
OUT="${OUT:-/data/151-2/users/tockier/tests/optim_ab}"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$OUT"

run() {  # run <arm> <extra args...>
    local arm="$1"; shift
    echo ">> arm $arm: $*"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    OMP_NUM_THREADS=16 WANDB_MODE=offline HF_HUB_OFFLINE=1 \
    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node="$NGPUS" --master_port=$(shuf -i 20000-29999 -n 1) \
            -m train.train_qwen \
            --config "$CONFIG" \
            --training.total-steps "$STEPS" \
            --training.save-steps 1000000 \
            "$@" \
            > "$OUT/$arm.log" 2>&1 || { echo "   arm $arm FAILED, see $OUT/$arm.log"; return 1; }
}

run fp32 \
    --training.master-dtype float32 \
    --training.adamw-impl foreach \
    --training.no-adamw-stochastic-round || true

run bf16_sr \
    --training.master-dtype bfloat16 \
    --training.adamw-impl foreach_sr \
    --training.adamw-stochastic-round || true

run bf16_rn \
    --training.master-dtype bfloat16 \
    --training.adamw-impl foreach_sr \
    --training.no-adamw-stochastic-round || true

python - "$OUT" <<'EOF'
import re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
row = re.compile(r" - (\d+) - loss ([\d.]+).*?time ([\d.]+)s.*?mem [\d.]+/([\d.]+)G")
print(f"\n{'arm':9} {'steps':>6} {'loss@10':>8} {'loss_end':>9} {'drop':>7}"
      f" {'peak_GiB':>9} {'step_s':>8}")
first = None
for arm in ("fp32", "bf16_sr", "bf16_rn"):
    log = out / f"{arm}.log"
    if not log.exists():
        print(f"{arm:9} {'no log':>6}")
        continue
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    hits = row.findall(text)
    if not hits:
        bad = [l for l in text.splitlines() if "Error" in l or "error" in l]
        print(f"{arm:9} {'-':>6}  no steps parsed. {bad[-1][:80] if bad else ''}")
        continue
    losses = [float(h[1]) for h in hits]
    early = losses[min(9, len(losses) - 1)]
    end = statistics.fmean(losses[-10:])
    tail = hits[5:] or hits
    peak = max(float(h[3]) for h in tail)
    step = statistics.median(float(h[2]) for h in tail)
    print(f"{arm:9} {len(hits):>6} {early:>8.4f} {end:>9.4f} {early - end:>7.4f}"
          f" {peak:>9.1f} {step:>8.3f}")
EOF
