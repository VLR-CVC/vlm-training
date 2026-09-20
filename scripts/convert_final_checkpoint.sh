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
python "$REPO_ROOT/utils/dcp_to_hf.py" "$LAST_CKPT" "$BASE_MODEL" "$OUT"
