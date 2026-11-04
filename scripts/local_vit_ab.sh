#!/bin/bash
# What does the vision encoder cost in tokens/s? Three arms, GPU 0, 30 steps each.
#
#   scripts/local_vit_ab.sh
#
#   A vlm      Qwen3.5-0.8B, 12-block ViT, 853.0M   on plotqa_cot
#   B decoder  depth-0 tower + 28 decoder layers, 850.95M (-0.24%)  on plotqa_cot
#   C text     the same 850.95M model              on fineweb_edu_text
#
# A vs B is the measurement: same data, same tokens per row, same processor, same
# decoder width -- the only difference is the 12 ViT blocks. B still patchifies
# images and still runs patch_embed + merger, which is exactly what a decoder-only
# model fed patch embeddings would do.
#
# C is the upper bound if the image pipeline disappears too. It is NOT the answer
# to "what does the ViT cost": it also drops image decode, smart_resize and
# patchify, and it changes the varlen document mix. Read it as a separate number.
#
# seq_len is 4096, not 8192: GPU 5 is a 44 GiB L40S that another job already holds
# 17.5 GiB of, and 8192 peaked at 24.9 GiB then OOMed at step 7. Buffers are 100,
# not 1000 -- filling 1000 cost 75 s of startup per arm for a 30-step run. Both are
# identical across the three arms, so neither touches the comparison.
#
# mfu is NOT a real utilization figure here: peak_tflops_per_gpu is hardcoded to
# 989.4 (H100 bf16) at train_qwen.py:313, and an L40S peaks near 362. Same constant
# every arm, so read tps / step time / data_pct and ignore mfu.
#
# Steps live in the configs, not flags: the tyro parser rejects a
# --training.total_steps override and dies formatting the error (see
# scripts/local_bucket_ab.sh).
set -euo pipefail

TORCH_ENV="${TORCH_ENV:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
export PATH="$TORCH_ENV/bin:$PATH"
python -c 'import spmd_types, torch' || {
    echo "env at $TORCH_ENV cannot import spmd_types/torch -- wrong env?" >&2
    exit 1
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU 5, an L40S: idle, ~28 GiB free. The Blackwell cards 0-3 carry other
# people's work and a step time measured against someone else's job is noise.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-/data/151-2/users/tockier/profiling/vit_ab}"
mkdir -p "$OUT"

# torchrun directly, not finetune.sh: that one sets TORCH_TRACE and then tries to
# convert a final checkpoint these 30-step runs never write.
for arm in vlm decoder text; do
    log="$OUT/$arm.log"
    echo "=== $arm -> $log"
    torchrun --nproc_per_node=1 --master_port=$(shuf -i 20000-29999 -n 1) \
        -m train.train_qwen \
        --config "configs/local/qwen3_5_0_8b_vit_ab_${arm}.toml" \
        >"$log" 2>&1 || echo "  arm exited $?"
done

python - "$OUT" <<'PY'
import pathlib, re, statistics, sys

out = pathlib.Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
# "12 - loss 6.9 tps 4321 (4321/gpu) mfu 31.2% tflops 180.4 ... time 1.894s ...
#  data_pct 3.11% ..."
step = re.compile(
    r" - (\d+) - .*?tps (\d+) .*?mfu ([0-9.]+)% tflops ([0-9.]+).*?"
    r"time ([0-9.]+)s.*?data_pct ([0-9.]+)%"
)
WARMUP = 10   # compile happens on the first step; the next few are still settling

rows = {}
for arm in ("vlm", "decoder", "text"):
    log = out / f"{arm}.log"
    if not log.exists():
        print(f"{arm:8s} no log"); continue
    hits = step.findall(ansi.sub("", log.read_text(errors="replace")))
    warm = hits[WARMUP:]
    if not warm:
        print(f"{arm:8s} {len(hits)} steps logged, none past warmup -- check {log}")
        continue
    med = lambda i: statistics.median(float(h[i]) for h in warm)
    rows[arm] = (len(warm), med(1), med(4), med(2), med(3), med(5))
    n, tps, t, mfu, tflops, dpct = rows[arm]
    print(f"{arm:8s} n={n:2d} tps={tps:7.0f} step={t:.3f}s mfu={mfu:4.1f}% "
          f"tflops={tflops:6.1f} data_pct={dpct:5.2f}%")

if "vlm" in rows and "decoder" in rows:
    a, b = rows["vlm"][1], rows["decoder"][1]
    print(f"\nViT cost: decoder-only is {b / a:.3f}x the VLM's tps "
          f"({(b / a - 1) * 100:+.1f}%) -- same data, same tokens")
if "vlm" in rows and "text" in rows:
    a, c = rows["vlm"][1], rows["text"][1]
    print(f"ViT + image pipeline: text-only is {c / a:.3f}x ({(c / a - 1) * 100:+.1f}%)")
PY
