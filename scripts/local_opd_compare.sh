#!/bin/bash
# SFT vs distillation (KD) on one local GPU, same student, data order and tokens
# (docs/OPD.md, "SFT vs KD"). About 3.7 h on one RTX PRO 6000:
#   1. held-out split of onevision_instruct_samples (symlinks, built once)
#   2. held-out eval of the base student and of the teacher
#   3. SFT arm (configs/local/opd_cmp_sft.toml), HF export, held-out eval
#   4. KD arm  (configs/local/opd_cmp_kd.toml),  HF export, held-out eval
#   5. summary: held-out CE on the dataset's answers and KL to the teacher
#
#   GPU=3 nohup scripts/local_opd_compare.sh > /data/151-2/users/tockier/opd_cmp/run.log 2>&1 &
#   ONLY="sft kd" scripts/local_opd_compare.sh      # a subset of: base teacher sft kd
#
# A held-out eval is `train_opd` at lr 0 for EVAL_STEPS rows of 8K on the held-out
# tars: the logged `loss: ce .. kl ..` per row is the model's CE on the answers and
# its forward KL to the teacher, on the same rows for every model. Every final DCP
# checkpoint is deleted once converted (or, for evals, right away): /data/151-2 is
# nearly full.
set -uo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=/data/151-2/users/tockier/opd_cmp
SRC=/data/151-1/datasets/onevision_instruct_samples
BASE=/data/151-2/users/tockier/models/Qwen3.5-2B-Base
TEACHER=/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b
EVAL_STEPS="${EVAL_STEPS:-100}"
ONLY="${ONLY:-base teacher sft kd}"
PY=/data/151-1/users/tockier/conda_install/envs/torch11_jup/bin
export CUDA_VISIBLE_DEVICES="${GPU:-3}" OMP_NUM_THREADS=16
export WANDB_MODE=offline WANDB_DIR="$ROOT"
mkdir -p "$ROOT/logs"

# --- 1. data: same tars, two split.yaml files ---------------------------------
make_split() {  # <name> <split list, one "  - <tar pattern>" per line>
    local dir="$ROOT/data/$1"
    [ -f "$dir/.nv-meta/split.yaml" ] && return
    mkdir -p "$dir/.nv-meta"
    ln -sf "$SRC"/*.tar "$SRC"/*.tar.idx "$dir"/
    cp "$SRC"/.nv-meta/{.info.json,dataset.yaml,index.sqlite,index.uuid,sample_loader.py} "$dir/.nv-meta/"
    printf 'split_parts:\n  train:\n%s\n  val: []\n  test: []\nexclude: []\n' "$2" > "$dir/.nv-meta/split.yaml"
}
make_split ov_train "$(sed -e 's/coco_train-{00027..00029}/coco_train-{00027..00028}/' \
    -e 's/ocr_train-{00000..00029}/ocr_train-{00000..00027}/' \
    "$SRC/.nv-meta/split.yaml" | sed -n '/^  train:/,/^  val:/p' | sed '1d;$d')"
make_split ov_heldout "  - coco_train-00029-of-00032.tar
  - ocr_train-{00028..00029}-of-00190.tar"

run() {  # <name> <module> <args...>; the log is $ROOT/logs/<name>.log
    local name=$1 mod=$2; shift 2
    echo "$(date '+%F %T') start $name"
    "$PY/torchrun" --nproc_per_node=1 --master_addr=127.0.0.1 \
        --master_port="$(shuf -i 20000-29999 -n 1)" -m "$mod" "$@" \
        > "$ROOT/logs/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end $name rc=$rc"
    return $rc
}

evaluate() {  # <name> <HF snapshot>
    run "eval_$1" train.train_opd --config configs/local/opd_cmp_kd.toml \
        --training.model_dir "$2" --data.data_path "$ROOT/data/ov_heldout" \
        --training.lr_llm 0 --training.lr_mlp 0 --training.lr_vit 0 \
        --training.total_steps "$EVAL_STEPS" --training.tpi_multiplier 1 \
        --training.warmup_steps 0 --data.no-save-dataloader-state \
        --training.output_dir "$ROOT/ckpt_eval_$1" --wandb.run_name "opd_cmp_eval_$1"
    rm -rf "$ROOT/ckpt_eval_$1"
}

train_arm() {  # <sft|kd> <module>
    local ckpt="$ROOT/ckpt_$1/checkpoint-step-1220"
    run "train_$1" "$2" --config "configs/local/opd_cmp_$1.toml" || return 1
    "$PY/python" utils/dcp_to_hf.py "$ckpt" "$BASE" "$ROOT/hf_$1" \
        > "$ROOT/logs/convert_$1.log" 2>&1 || return 1
    rm -rf "$ROOT/ckpt_$1"
    evaluate "$1" "$ROOT/hf_$1"
}

for arm in $ONLY; do
    case $arm in
        base)    evaluate base "$BASE" ;;
        teacher) evaluate teacher "$TEACHER" ;;
        sft)     train_arm sft train.train_qwen ;;
        kd)      train_arm kd train.train_opd ;;
    esac
done

# --- 5. summary -----------------------------------------------------------------
"$PY/python" - "$ROOT/logs" <<'EOF'
import pathlib, re, statistics, sys
ansi = re.compile(r"\x1b\[[0-9;]*m")
loss = re.compile(r"  loss: ce ([\d.]+)  kl ([\d.]+)")
for name in ("base", "teacher", "sft", "kd"):
    p = pathlib.Path(sys.argv[1]) / f"eval_{name}.log"
    if not p.exists():
        continue
    rows = loss.findall(ansi.sub("", p.read_text()))
    if not rows:
        print(f"{name:8s} no eval rows")
        continue
    ce = statistics.mean(float(c) for c, _ in rows)
    kl = statistics.mean(float(k) for _, k in rows)
    print(f"{name:8s} held-out rows {len(rows):4d}  CE {ce:.4f}  KL(teacher||model) {kl:.4f}")
EOF
