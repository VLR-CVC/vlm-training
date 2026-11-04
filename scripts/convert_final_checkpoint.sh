#!/bin/bash
# Post-training hook: convert the LAST DCP checkpoint

set -u

CONFIG_FILE="${1:-}"
[ $# -gt 0 ] && shift

if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "[convert-final] no config file '$CONFIG_FILE'; skipping checkpoint conversion"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

read_training_key() {
    python3 - "$CONFIG_FILE" "$1" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:  # py < 3.11
    import tomli as tomllib
cfg, key = sys.argv[1], sys.argv[2]
try:
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    print(data.get("training", {}).get(key, ""))
except Exception:
    print("")
PY
}

BASE_MODEL="$(read_training_key model_dir)"
CHECKPOINT_DIR="$(read_training_key output_dir)"
CHAT_TEMPLATE="$(read_training_key chat_template)"

# QWEN_FINALIZE_INSTRUCT=1 turns the export into a usable instruct snapshot.
#
# `dcp_to_hf.py` copies every non-weight file from the BASE snapshot, so a run
# finetuned off `*-Base` inherits Base's tokenizer settings: `eos_token` is
# `<|endoftext|>` (248044) while the SFT chat template closes every turn with
# `<|im_end|>` (248046), and Base ships no `generation_config.json` at all.
# `generate` then never sees a stop token and runs to `max_new_tokens`. Same
# Base-vs-Instruct mismatch that made the dataloader skip 100% of samples, only
# on the inference side.
#
# Off by default: it rewrites the tokenizer config, and a mid-training export is
# not necessarily meant to be served. Point QWEN_INSTRUCT_REF at an official
# instruct snapshot to copy its `generation_config.json` verbatim (sampling
# defaults included); without it a minimal one is written from the token ids,
# which is the only option for a model whose instruct release ships none --
# Qwen3.5-2B, for instance.
FINALIZE="${QWEN_FINALIZE_INSTRUCT:-0}"

if [ -z "$BASE_MODEL" ] || [ -z "$CHECKPOINT_DIR" ]; then
    echo "[convert-final] could not read model_dir/output_dir from $CONFIG_FILE; skipping"
    exit 0
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[convert-final] checkpoint dir '$CHECKPOINT_DIR' does not exist; skipping"
    exit 0
fi

LAST_CKPT="$(for d in "$CHECKPOINT_DIR"/checkpoint-step-*; do [ -d "$d" ] && echo "${d##*-} $d"; done | sort -n | tail -1 | cut -d' ' -f2-)"
if [ -z "$LAST_CKPT" ]; then
    echo "[convert-final] no checkpoint-step-* in '$CHECKPOINT_DIR'; skipping"
    exit 0
fi
STEP="${LAST_CKPT##*-}"
OUT="$CHECKPOINT_DIR/models/step-$STEP"

echo "[convert-final] converting $LAST_CKPT -> $OUT (base: $BASE_MODEL)"
python "$REPO_ROOT/utils/dcp_to_hf.py" "$LAST_CKPT" "$BASE_MODEL" "$OUT" || exit $?

# The index is what `from_pretrained` reads; without it nothing above is usable.
if [ ! -f "$OUT/model.safetensors.index.json" ]; then
    echo "[convert-final] no index written to '$OUT'" >&2
    exit 1
fi

# Match the other model dirs: `save_file` inherits the umask and leaves the
# weights mode 600, unreadable to the rest of the project group.
chmod 644 "$OUT/model.safetensors" 2>/dev/null || true

[ "$FINALIZE" = "1" ] || exit 0

TPL="$CHAT_TEMPLATE"
[ -f "$TPL" ] || TPL="$REPO_ROOT/$CHAT_TEMPLATE"
if [ ! -f "$TPL" ]; then
    echo "[convert-final] chat_template '$CHAT_TEMPLATE' not found; not finalizing" >&2
    exit 1
fi

echo "[convert-final] finalizing instruct snapshot in $OUT (template: $TPL)"
cp -n "$OUT/chat_template.jinja" "$OUT/chat_template.jinja.base-orig" 2>/dev/null || true
cp -n "$OUT/tokenizer_config.json" "$OUT/tokenizer_config.json.base-orig"
cp "$TPL" "$OUT/chat_template.jinja"

python "$REPO_ROOT/scripts/finalize_instruct_tokenizer.py" \
    "$OUT" "$TPL" "${QWEN_INSTRUCT_REF:-}" || exit $?

chmod 644 "$OUT/generation_config.json"
