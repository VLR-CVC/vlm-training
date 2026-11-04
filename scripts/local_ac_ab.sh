#!/bin/bash
# Selective activation checkpointing A/B on local cards.
#
#   scripts/local_ac_ab.sh                  # 2 GPUs, 40 steps, freq 0 vs 2
#   GPUS=0,3 STEPS=60 AC_FREQ=2 scripts/local_ac_ab.sh
#
# AC is the only lever that touches ACTIVATION memory; reshard_after_forward and
# dp_shard_size only move parameter and optimizer state. So the metric that
# matters here is peak reserved memory, and the price is step time.
#
# TOKENS_PER_MICROBATCH defaults to 4 rows rather than the config's 1. At one
# 8192-token row a 2B model barely touches a 96 GB card and both arms would read
# the same -- the A/B needs enough activations to be measuring anything. This is
# the same knob whose value drove the 9B OOM on JUPITER.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-0,3}"
STEPS="${STEPS:-40}"
AC_FREQ="${AC_FREQ:-2}"
# AC_OP=1 switches the "on" arm from whole-block to per-op SAC.
AC_OP="${AC_OP:-0}"
CONFIG="${CONFIG:-configs/local/qwen3_5_2b.toml}"
TPM="${TOKENS_PER_MICROBATCH:-32768}"
OUT="${OUT:-/data/151-2/users/tockier/tests/ac_ab}"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$OUT"

run() {  # run <arm> <freq>
    echo ">> arm $1: selective_ac_freq=$2"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    OMP_NUM_THREADS=16 WANDB_MODE=offline HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node="$NGPUS" --master_port=$(shuf -i 20000-29999 -n 1) \
            -m train.train_qwen \
            --config "$CONFIG" \
            --training.total-steps "$STEPS" \
            --training.save-steps 100000 \
            --data.tokens-per-microbatch "$TPM" \
            --training.selective-ac-freq "$2" ${3:-} \
            > "$OUT/$1.log" 2>&1 || { echo "   arm $1 FAILED, see $OUT/$1.log"; return 1; }
}

run off 0 || true
[ "$AC_OP" = 1 ] && ON_EXTRA="--training.selective-ac-op" || ON_EXTRA=""
run on "$AC_FREQ" $ON_EXTRA || true

python - "$OUT" <<'EOF'
import re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
row = re.compile(r" - (\d+) - loss .*?time ([\d.]+)s.*?mem [\d.]+/([\d.]+)G")
print(f"\n{'arm':6} {'steps':>6} {'peak_GiB':>9} {'step_s':>8} {'tok/s/GPU':>10}")
for arm in ("off", "on"):
    log = out / f"{arm}.log"
    if not log.exists():
        print(f"{arm:6} {'no log':>6}")
        continue
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    hits = row.findall(text)
    if not hits:
        bad = [l for l in text.splitlines() if "Error" in l or "error" in l]
        print(f"{arm:6} {'-':>6}  no steps parsed. {bad[-1][:90] if bad else ''}")
        continue
    # skip the first 5 steps: compile and allocator warmup, not steady state
    tail = hits[5:] or hits
    peak = max(float(h[2]) for h in tail)
    step = statistics.median(float(h[1]) for h in tail)
    tps = re.findall(r"\((\d+)/gpu\)", text)
    gpu = statistics.median(int(t) for t in tps[5:]) if len(tps) > 5 else float("nan")
    print(f"{arm:6} {len(hits):>6} {peak:>9.1f} {step:>8.3f} {gpu:>10.0f}")
EOF
