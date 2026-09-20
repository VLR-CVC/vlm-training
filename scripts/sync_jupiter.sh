#!/usr/bin/env bash
# Push this checkout to JUPITER. Extra args go to rsync, so:
#   scripts/sync_jupiter.sh -n          dry run
#   scripts/sync_jupiter.sh --delete    drop remote files that are gone here
set -euo pipefail

REMOTE=${JUP_REMOTE:-jupiter}
DEST=${JUP_DEST:-/e/project1/open-sci-mm/ockier1/titan-port}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec rsync -az --info=stats1,name0 \
  --filter=':- .gitignore' \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$@" "$SRC/" "$REMOTE:$DEST/"
