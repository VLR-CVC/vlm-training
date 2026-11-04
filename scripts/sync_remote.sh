#!/usr/bin/env bash
# Push this checkout to a remote cluster. Extra args go to rsync, so:
#   REMOTE=jupiter DEST=/e/project1/open-sci-mm/ockier1/vlm-training scripts/sync_remote.sh -n   dry run
#   ... scripts/sync_remote.sh --delete    drop remote files that are gone here
#   FORCE=1 ... scripts/sync_remote.sh     push even over newer remote files
#
# Code goes through the login node; bulk data does not -- see scripts/pack_env_mn5.sh
# for MN5's transfer-node convention, which doesn't apply here.
set -euo pipefail

REMOTE=${REMOTE:?set REMOTE to the ssh host, e.g. mnlogin1 or jupiter}
DEST=${DEST:?set DEST to the repo path on that host, e.g. /gpfs/projects/ehpc543/tockier/vlm-training}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RSYNC_FILTERS=(
  --filter=':- .gitignore'
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc'
)

# Preflight: refuse to clobber files that are NEWER on the remote.
#
# This is not hypothetical -- it already cost edits on the JUPITER checkout.
# Configs get changed directly on the cluster between syncs, and a plain push
# overwrites them silently: rsync's itemized output marks a file that merely
# *differs* exactly like one that is stale, so reading the dry run is not a
# safety check. Only files present on BOTH sides are at risk; a push without
# --delete leaves remote-only files alone.
if [ -z "${FORCE:-}" ]; then
    newer=$(rsync -azn --update --out-format='%n' "${RSYNC_FILTERS[@]}" \
        "$REMOTE:$DEST/" "$SRC/" 2>/dev/null | grep -v '/$' \
        | while IFS= read -r f; do [ -e "$SRC/$f" ] && echo "$f"; done || true)
    if [ -n "$newer" ]; then
        echo "refusing to push: newer on $REMOTE than here" >&2
        echo "$newer" | sed 's/^/  /' >&2
        cat >&2 <<EOF

Pull them first, or re-apply the change locally. FORCE=1 overrides -- but the
remote copy is then gone for good.
EOF
        exit 1
    fi
fi

exec rsync -az --info=stats1,name0 "${RSYNC_FILTERS[@]}" \
  "$@" "$SRC/" "$REMOTE:$DEST/"
