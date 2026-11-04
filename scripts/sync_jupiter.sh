#!/usr/bin/env bash
# Push this checkout to JUPITER. Extra args go to rsync, so:
#   scripts/sync_jupiter.sh -n          dry run
#   scripts/sync_jupiter.sh --delete    drop remote files that are gone here
#   FORCE=1 scripts/sync_jupiter.sh     push even over newer remote files
set -euo pipefail

REMOTE=${JUP_REMOTE:-jupiter}
DEST=${JUP_DEST:-/e/project1/open-sci-mm/ockier1/titan-port}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RSYNC_FILTERS=(
  --filter=':- .gitignore'
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc'
)

# Preflight: refuse to clobber files that are NEWER on JUPITER.
#
# This exists because it already happened. Configs get edited directly on
# JUPITER between syncs, and a plain push overwrites them with no warning --
# rsync's itemized output marks a file that merely *differs* the same way as one
# that is stale, so "it showed up in the dry run" is not a safety check. The
# edits are unrecoverable: the checkout's working tree is what gets overwritten,
# and GPFS .snapshots is not readable from here.
#
# `rsync --update` on a reverse dry run lists exactly the files the remote would
# send back because its copy is newer -- i.e. the ones a push would destroy.
if [ -z "${FORCE:-}" ]; then
    # Only files that exist on BOTH sides are at risk: a push without --delete
    # leaves remote-only files alone, and --update reports those too because a
    # missing local copy counts as older.
    newer=$(rsync -azn --update --out-format='%n' "${RSYNC_FILTERS[@]}" \
        "$REMOTE:$DEST/" "$SRC/" 2>/dev/null | grep -v '/$' \
        | while IFS= read -r f; do [ -e "$SRC/$f" ] && echo "$f"; done || true)
    if [ -n "$newer" ]; then
        echo "refusing to push: newer on JUPITER than here" >&2
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
