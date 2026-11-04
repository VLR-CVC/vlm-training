#!/bin/bash
# Attention-operator A/B on the 4 local 96 GiB cards: is the hybrid schedule
# (GatedDeltaNet on 3 of every 4 layers) what makes the text model slow?
#
#   scripts/local_attn_ab.sh                  # 4 GPUs, 30 steps, 3 arms
#   GPUS=0,1,2,3 STEPS=60 scripts/local_attn_ab.sh
#
# Arms -- `model.attention_kind` is the ONLY thing that changes:
#   hybrid   every 4th layer softmax, rest GatedDeltaNet (the real Qwen3.5)
#   softmax  all 24 layers softmax attention
#   gdn      all 24 layers GatedDeltaNet
#
# Params differ slightly by arm (1.882B / 1.767B / 1.920B measured), so compare
# tokens/s and step time, not raw TFLOPs: the FLOPs model bills each schedule
# differently and `train/flops_estimation.py` is the thing under test otherwise.
#
# `random_init` is set by the config, and required: a layer that trained as
# GatedDeltaNet has no softmax weights to load.
#
# Steps 0-5 are dropped from the summary -- torch.compile warmup dominates them
# (step 1 measured 66 s against a steady 8 s).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"

GPUS="${GPUS:-0,1,2,3}"
STEPS="${STEPS:-30}"
CONFIG="${CONFIG:-configs/text/qwen3_5_2b_pretrain.toml}"
OUT="${OUT:-/data/151-2/users/tockier/tests/attn_ab}"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$OUT"

run() {  # run <arm> <attention_kind>
    local arm="$1" kind="$2"
    echo ">> arm $arm: attention_kind=${kind:-<snapshot hybrid>}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    OMP_NUM_THREADS=16 WANDB_MODE=offline HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node="$NGPUS" --master_port=$(shuf -i 20000-29999 -n 1) \
            -m train.train_qwen \
            --config "$CONFIG" \
            --model.attention-kind "$kind" \
            --training.total-steps "$STEPS" \
            --training.save-steps 1000000 \
            > "$OUT/$arm.log" 2>&1 || { echo "   arm $arm FAILED, see $OUT/$arm.log"; return 1; }
}

run hybrid  ""        || true
run softmax "softmax" || true
run gdn     "gdn"     || true

python - "$OUT" <<'EOF'
import re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
# the log is ANSI-coloured: strip before matching or none of this hits
row = re.compile(
    r" - (\d+) - loss ([\d.]+) tps (\d+).*?tflops ([\d.]+).*?"
    r"time ([\d.]+)s fwd ([\d.]+)s mem [\d.]+/([\d.]+)G"
)
print(f"\n{'arm':9} {'steps':>6} {'loss_end':>9} {'tps':>7} {'tflops':>7}"
      f" {'step_s':>7} {'fwd_s':>7} {'GiB':>6}")
for arm in ("hybrid", "softmax", "gdn"):
    log = out / f"{arm}.log"
    if not log.exists():
        print(f"{arm:9} {'no log':>6}")
        continue
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    hits = row.findall(text)
    if not hits:
        bad = [l for l in text.splitlines() if "Error" in l or "error" in l]
        print(f"{arm:9} {'-':>6}  no steps parsed. {bad[-1][:70] if bad else ''}")
        continue
    tail = hits[5:] or hits          # drop compile warmup
    med = lambda i: statistics.median(float(h[i]) for h in tail)
    print(f"{arm:9} {len(hits):>6} {float(hits[-1][1]):>9.4f} {med(2):>7.0f}"
          f" {med(3):>7.1f} {med(4):>7.3f} {med(5):>7.3f}"
          f" {max(float(h[6]) for h in tail):>6.1f}")
EOF
