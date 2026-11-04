#!/bin/bash
# Download an HF asset into the torchtitan checkout's assets/hf/.
# Run on the JUPITER **login node** -- compute nodes have no outbound network.
#
#   scripts/scaling/fetch_hf_asset.sh                        # Qwen3.5-35B-A3B
#   scripts/scaling/fetch_hf_asset.sh Qwen/Qwen3-30B-A3B
#   ASSETS="tokenizer config safetensors index" scripts/scaling/fetch_hf_asset.sh <repo>
#
# Lands in $TT_ROOT/repo/assets/hf/<model-name>, which is the one directory the
# pinned checkout is allowed to grow (TORCHTITAN_BENCHMARK.md section 8) and what
# --model.hf_assets_path expects.
#
# Tokenizer + config only by default: torchtitan inits on meta device and calls
# init_weights(), so a throughput benchmark needs no safetensors. For 35B that is
# a few MB instead of ~70 GB.
set -euo pipefail

TT_ROOT="${TT_ROOT:-/e/project1/open-sci-mm/ockier1/torchtitan}"
REPO_ID="${1:-Qwen/Qwen3.5-35B-A3B}"
ASSETS="${ASSETS:-tokenizer config}"
DEST="$TT_ROOT/repo/assets/hf"

# HF_HUB_OFFLINE is exported by the run scripts and may be live in this shell;
# with it set every download is a cache miss that raises instead of fetching.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE

PY_BIN="$TT_ROOT/venv/bin/python"
mkdir -p "$DEST"
cd "$TT_ROOT/repo"

"$PY_BIN" scripts/download_hf_assets.py \
    --repo_id "$REPO_ID" \
    --local_dir "$DEST" \
    --assets $ASSETS

echo "--- $DEST/${REPO_ID##*/} ---"
ls -la "$DEST/${REPO_ID##*/}"
