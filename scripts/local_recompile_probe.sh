#!/bin/bash
# Why did the 16-GPU MN5 pretrain run keep recompiling until it hit the limit,
# when the 4-GPU local run generalised and settled at step ~100?
#
#   scripts/local_recompile_probe.sh
#
# `TORCH_LOGS=recompiles` prints the guard that failed on every recompile, which
# names the varying quantity instead of us inferring it from step times.
#
# Two arms, because `precompile = true` is the suspect: it warms the graph with
# TEXT_SPECS (2, 8 and 16 documents) before any real batch arrives, and real
# batches carry 90-469. Seeding dynamo with three small specialised shapes may
# be what stops `automatic_dynamic_shapes` from generalising the document
# dimension -- the MN5 run had precompile on, the local run that settled did not.
#
# Runs on the L40S quad (4-7) so the Blackwell run on 0-3 is untouched. P2P is
# disabled: the 6,7 pair hangs the first all-gather otherwise. Step times here
# are therefore meaningless -- this probe is about recompiles, not throughput.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"

GPUS="${GPUS:-4,5,6,7}"
STEPS="${STEPS:-60}"
CONFIG="${CONFIG:-configs/text/qwen3_5_2b_pretrain.toml}"
OUT="${OUT:-/data/151-2/users/tockier/tests/recompile_probe}"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$OUT"

run() {  # run <arm> <extra args...>
    local arm="$1"; shift
    echo ">> arm $arm: $*"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    OMP_NUM_THREADS=8 WANDB_MODE=offline HF_HUB_OFFLINE=1 \
    NCCL_P2P_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_LOGS=recompiles \
        torchrun --nproc_per_node="$NGPUS" --master_port=$(shuf -i 20000-29999 -n 1) \
            -m train.train_qwen \
            --config "$CONFIG" \
            --training.total-steps "$STEPS" \
            --training.save-steps 1000000 \
            "$@" \
            > "$OUT/$arm.log" 2>&1 || { echo "   arm $arm FAILED (may be the recompile limit -- that is a result, not an error); see $OUT/$arm.log"; }
}

run precompile_on  --training.precompile
run precompile_off --training.no-precompile

python - "$OUT" <<'EOF'
import re, sys
from pathlib import Path
from collections import Counter

out = Path(sys.argv[1])
for arm in ("precompile_on", "precompile_off"):
    log = out / f"{arm}.log"
    if not log.exists():
        print(f"\n=== {arm}: no log"); continue
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    # one "Recompiling function X" block per recompile, with the failed guard below
    recompiles = re.findall(r"Recompiling function (\S+) in (\S+)", text)
    reasons = re.findall(r"triggered by the following guard failure\(s\):\n(.*?)\n\n", text, re.S)
    guards = Counter()
    for block in reasons:
        for line in block.splitlines():
            line = line.strip().lstrip("- ")
            if line:
                guards[line[:110]] += 1
    steps = len(re.findall(r" - \d+ - loss ", text))
    limit = "cache_size_limit" in text or "recompile_limit" in text
    print(f"\n=== {arm}: {steps} steps logged, {len(recompiles)} recompiles"
          f"{', HIT THE LIMIT' if limit else ''}")
    for fn, cnt in Counter(f for f, _ in recompiles).most_common(5):
        print(f"    {cnt:>4}x  {fn}")
    print("  top failed guards:")
    for g, cnt in guards.most_common(8):
        print(f"    {cnt:>4}x  {g}")
EOF
