#!/usr/bin/env bash
# Push this checkout to MareNostrum 5. Thin wrapper over sync_remote.sh -- see it for
# the preflight/flags. The env tarball does NOT go this way -- BSC wants bulk data on
# the transfer nodes (mntransfer1/2), which is what the `mn5:` rclone remote points at.
# See scripts/pack_env_mn5.sh.
set -euo pipefail
export REMOTE=${MN5_REMOTE:-mnlogin1}
export DEST=${MN5_DEST:?set MN5_DEST to the repo path on MN5, e.g. /gpfs/projects/ehpc543/tockier/vlm-training}
exec "$(dirname "${BASH_SOURCE[0]}")/sync_remote.sh" "$@"
