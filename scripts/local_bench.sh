#!/bin/bash
# Generation benchmarks (lmms-eval on vLLM, one local GPU) for the SFT vs KD
# comparison (docs/OPD.md, "SFT vs KD"): base, SFT, KD and teacher, same tasks,
# greedy decoding, identical image preprocessing.
#
#   GPU=3 nohup scripts/local_bench.sh > /data/151-2/users/tockier/opd_cmp/bench.log 2>&1 &
#   GPU=3 LIMIT=8 ONLY=sft scripts/local_bench.sh       # smoke test
#
# Prompt format. The students (base, SFT, KD) answer after `<|im_start|>assistant\n`,
# exactly as the training template rendered every answer of this data (no think
# block). The eval template is the training template with that generation prompt;
# the teacher keeps its official template (non-thinking by default). Exports from
# `*-Base` stop on `<|endoftext|>`, so their tokenizer is first pointed at
# `<|im_end|>` with `scripts/finalize_instruct_tokenizer.py` (the base is
# evaluated from a copy whose weights are symlinked).
#
# Tasks need no API judge: mmbench / hallusion_bench would call api.openai.com.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD

ROOT=/data/151-2/users/tockier/opd_cmp
LMMS=/data/151-2/users/tockier/eval/lmms-eval
BASE=/data/151-2/users/tockier/models/Qwen3.5-2B-Base
TEACHER=/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b
TASKS="${TASKS:-ai2d,chartqa,ocrbench,docvqa_val,textvqa_val,infovqa_val,realworldqa,mme}"
ONLY="${ONLY:-base sft kd teacher}"
PY=/data/151-1/users/tockier/conda_install/envs/vllm/bin/python
OUT="$ROOT/bench${LIMIT:+_limit$LIMIT}"
TPL="$ROOT/chat_template_eval.jinja"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-3}"
export HF_HOME=/data/151-2/users/tockier/hf_cache
export LMMS_EVAL_DATASETS_CACHE=$HF_HOME/datasets
export LMMS_EVAL_ANON_HF=1 WANDB_MODE=disabled VLLM_WORKER_MULTIPROC_METHOD=spawn

# training template minus the think block of the generation prompt
sed '/enable_thinking is defined and enable_thinking is false/,/endif/d' \
    assets/chat_template_sft.jinja > "$TPL"
grep -q "enable_thinking" "$TPL" && { echo "eval template still has a think block" >&2; exit 1; }

finalize() {  # <snapshot>: stop on <|im_end|>, once
    [ -f "$1/generation_config.json" ] && return
    cp "$TPL" "$1/chat_template.jinja"
    "$PY" scripts/finalize_instruct_tokenizer.py "$1" "$TPL"
}

if [[ " $ONLY " == *" base "* && ! -d "$ROOT/hf_base" ]]; then
    mkdir -p "$ROOT/hf_base"
    for f in "$BASE"/*; do
        case "$f" in
            *.safetensors) ln -s "$f" "$ROOT/hf_base/" ;;
            *) cp "$f" "$ROOT/hf_base/" ;;
        esac
    done
fi

for name in $ONLY; do
    case $name in
        teacher) model=$TEACHER; tpl_arg="" ;;
        *) model="$ROOT/hf_$name"; finalize "$model"; tpl_arg=",chat_template=$TPL" ;;
    esac
    echo "$(date '+%F %T') start $name"
    (cd "$LMMS" && "$PY" -m lmms_eval --model vllm \
        --model_args "model=$model,gpu_memory_utilization=${MEM:-0.85},max_model_len=${MAX_LEN:-32768}$tpl_arg" \
        --tasks "$TASKS" --batch_size 64 --output_path "$OUT/$name" --log_samples \
        ${LIMIT:+--limit $LIMIT}) > "$ROOT/logs/bench_$name.log" 2>&1
    echo "$(date '+%F %T') end $name rc=$?"
done

# summary: one row per model, the headline metric of each task
"$PY" - "$OUT" "$TASKS" $ONLY <<'EOF'
import glob, json, os, sys
out, tasks, names = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
# headline metric per task (accuracy-like ones x100); MME reports two scores
HEAD = {"ai2d": "exact_match,flexible-extract", "chartqa": "relaxed_overall,none",
        "ocrbench": "ocrbench_accuracy,none", "docvqa_val": "anls,none",
        "textvqa_val": "exact_match,none", "infovqa_val": "anls,none",
        "realworldqa": "exact_match,none"}
cols = [c for t in tasks for c in ((t,) if t != "mme" else ("mme_perception", "mme_cognition"))]
print(f"{'model':8s} " + " ".join(f"{c[:13]:>13s}" for c in cols))
for n in names:
    files = sorted(glob.glob(f"{out}/{n}/**/*_results.json", recursive=True), key=os.path.getmtime)
    if not files:
        print(f"{n:8s} no results"); continue
    res = json.load(open(files[-1]))["results"]
    cells = []
    for c in cols:
        if c.startswith("mme_"):
            v = res.get("mme", {}).get(f"{c}_score,none")
        else:
            v = res.get(c, {}).get(HEAD.get(c, ""))
            v = None if v is None else 100 * v
        cells.append("-" if v is None else f"{v:.1f}")
    print(f"{n:8s} " + " ".join(f"{c:>13s}" for c in cells))
EOF
