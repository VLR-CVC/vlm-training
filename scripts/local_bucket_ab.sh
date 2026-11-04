#!/bin/bash
# A/B the varlen bucketing on the 4 local 96 GiB cards: the same config run once
# with QWEN_VARLEN_BUCKETS=1 (current default) and once with 0 (pre-bucketing).
#
#   scripts/local_bucket_ab.sh
# Step count lives in the config, not a flag: the trainer's tyro parser rejects
# a --training.total_steps override and then dies formatting the error.
#
# Mirrors configs/mn5/qwen3_vl_2b_instruct_32k.toml, the run that went from a
# uniform 0.954 s/step to a bimodal 3.592 s after the bucketing landed
# (MN5 jobs 46510294 vs 46615198).
set -euo pipefail

TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"
python -c 'import spmd_types, torch' || {
    echo "env at $TORCH_ENV cannot import spmd_types/torch -- wrong env?" >&2
    exit 1
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 4-7, not 0-3: the Blackwell cards are usually busy with other people's work,
# and a step time measured against someone else's inference server is noise.
# The L40S pair 6-7 hang the first all-gather with peer-to-peer on.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/local/qwen3_vl_2b_bucket_ab.toml}"
OUT="${OUT:-/data/151-2/users/tockier/profiling/bucket_ab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

for buckets in 1 0; do
    log="$OUT/buckets_${buckets}.log"
    echo "=== QWEN_VARLEN_BUCKETS=$buckets -> $log"
    # let the previous arm's ranks release their memory before the next starts
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$(pgrep -u "$USER" -f train.train_qwen | head -1)" 2>/dev/null; do sleep 2; done
    QWEN_VARLEN_BUCKETS=$buckets \
        "$SCRIPT_DIR/finetune.sh" --config "$CONFIG" >"$log" 2>&1 || echo "  arm exited $?"
done

python - "$OUT" <<'PY'
import re, statistics, sys, pathlib

out = pathlib.Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
step = re.compile(r" - (\d+) - .*?time ([0-9.]+)s")

for buckets in (1, 0):
    log = out / f"buckets_{buckets}.log"
    if not log.exists():
        print(f"buckets={buckets}: no log"); continue
    times = [(int(a), float(b)) for a, b in step.findall(ansi.sub("", log.read_text()))]
    # drop compile/precompile warmup: the first steps are not steady state
    warm = [t for _, t in times[15:]]
    if not warm:
        print(f"buckets={buckets}: {len(times)} steps, none past warmup"); continue
    print(f"buckets={buckets}: n={len(warm)} mean={statistics.mean(warm):.3f}s "
          f"median={statistics.median(warm):.3f}s min={min(warm):.3f}s max={max(warm):.3f}s")
PY
