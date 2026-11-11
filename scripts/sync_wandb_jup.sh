#!/bin/bash
# Sync offline wandb runs from JUPITER. Run it HERE, not on JUPITER.
#
#   scripts/sync_wandb_jup.sh                    # every run that grew since last time
#   scripts/sync_wandb_jup.sh 20260921           # restrict to one day
#   scripts/sync_wandb_jup.sh '2026092[01]'      # a glob over run directories
#   FORCE_ALL=1 scripts/sync_wandb_jup.sh        # ignore the manifest, sync everything
#   FETCH_ONLY=1 scripts/sync_wandb_jup.sh       # copy them down, do not upload
#   LIVE=1 scripts/sync_wandb_jup.sh             # tail runs (see the warning below)
#
# MN5 HAS NO OUTBOUND INTERNET -- not even from a login node, which is why the
# env is shipped as a tarball in the first place. `scripts/sync_wandb.sh` runs
# `wandb sync` directly on JUPITER's login node; that cannot work here. So this
# copies the run directories down first and uploads them from this machine.
#
# ONLY RUNS THAT GREW ARE TOUCHED. The remote `.wandb` transaction log is the
# run: it only ever gets appended to, so its size is a sufficient "has new data"
# signal. One ssh call stats all of them, the sizes are diffed against
# $STAGE/.sizes, and everything else is skipped -- no transfer, no upload.
#
# NOT `--live`. `--live` waits for the transaction log's final exit record. A
# run killed by SLURM (walltime, OOM, node failure) never writes one, so `--live`
# tails it forever -- that is the "syncing to eternity". Without it, sync replays
# what is on disk and returns. The run shows up as crashed, which is what it is.
# Re-running after it grows replays the log again into the SAME run id (the id
# lives in the log), so history is rewritten, not duplicated.
#
# Runs are left in place on MN5: they are small, and deleting the only copy of a
# training log because an upload half-failed is not a trade worth making.
set -euo pipefail

REMOTE="${MN5_REMOTE:-jupiter}"
# multinode_jup.sh pins WANDB_DIR here. It used to be inherited from the login
# shell, which pointed at /e/project1/jureap59/... in some shells and killed job
# 2268024 with `Errno 122 Disk quota exceeded` -- a project allocation we do not
# have. The 344 runs written before the move are still under
# /e/project1/open-sci-mm/ockier1/cache/wandb/runs; sync those with
#   MN5_DEST=/e/project1/open-sci-mm/ockier1/cache/wandb/runs scripts/sync_wandb_jup.sh
REPO="${MN5_DEST:-/e/scratch/open-sci-mm/ockier1/wandb}"
# wandb always nests runs one level down, as $WANDB_DIR/wandb/offline-run-*.
RUN_ROOT="${MN5_WANDB_ROOT:-$REPO/wandb}"
# Staging on 151-2: 151-1 is full.
STAGE="${WANDB_STAGE:-/data/151-2/users/tockier/wandb_mn5}"
PATTERN="${1:-*}"
WANDB_BIN="${WANDB_BIN:-/data/151-2/users/tockier/envs/qwen4_mn5/bin/wandb}"
MANIFEST="$STAGE/.sizes"
# A backstop, not the mechanism. Without --live a sync terminates on its own;
# this only catches a hung upload.
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"

mkdir -p "$STAGE"
touch "$MANIFEST"

# --- what is on MN5, and how big ------------------------------------------
# %s is the size of the .wandb file; %h gives its run directory. One round trip.
echo "stat-ing offline-run-${PATTERN} on $REMOTE, path: $RUN_ROOT"
remote_sizes=$(ssh "$REMOTE" \
    "find '$RUN_ROOT' -maxdepth 2 -name '*.wandb' -path '*/offline-run-${PATTERN}*/*' -printf '%s\t%h\n'" \
    2>/dev/null || true)
[ -n "$remote_sizes" ] || { echo "no runs matching offline-run-${PATTERN} in $REMOTE:$RUN_ROOT" >&2; exit 1; }

declare -A seen_size
while IFS=$'\t' read -r size dir; do
    if [ -n "${dir:-}" ]; then seen_size["$(basename "$dir")"]=$size; fi
done <<< "$remote_sizes"

declare -A known_size
while IFS=$'\t' read -r size name; do
    if [ -n "${name:-}" ]; then known_size["$name"]=$size; fi
done < "$MANIFEST"

# --- which of them have new data ------------------------------------------
changed=()
for name in "${!seen_size[@]}"; do
    if [ -n "${FORCE_ALL:-}" ] || [ "${known_size[$name]:-0}" != "${seen_size[$name]}" ]; then
        changed+=("$name")
    fi
done

if (( ! ${#changed[@]} )); then
    echo "${#seen_size[@]} run(s) on $REMOTE, none with new data. Nothing to do."
    exit 0
fi
# Oldest first, so a partial failure leaves the newest runs as the ones to retry.
mapfile -t changed < <(printf '%s\n' "${changed[@]}" | sort)
echo "${#changed[@]}/${#seen_size[@]} run(s) with new data:"
for name in "${changed[@]}"; do
    printf '  %-44s %s -> %s\n' "$name" "${known_size[$name]:-none}" "${seen_size[$name]}"
done

# --- fetch just those -----------------------------------------------------
# NO --ignore-existing. That was the bug: a run already staged was never
# refreshed, so a growing run's local copy stayed frozen at its first snapshot
# and every later sync re-uploaded the same stale data.
printf '%s\n' "${changed[@]}" | rsync -azr --info=stats1,name0 \
    --files-from=- "$REMOTE:$RUN_ROOT/" "$STAGE/"

if [ -n "${FETCH_ONLY:-}" ]; then
    echo "fetched ${#changed[@]} run(s) to $STAGE (FETCH_ONLY, not uploading)"
    exit 0
fi

# --- upload ---------------------------------------------------------------
# One at a time: a single unsyncable run should not abort the rest, and the
# output says which one failed.
#
# --no-skip-synced: a run synced before carries a .wandb.synced marker, and we
# already know from the size diff that it has new data. Skipping it here is
# exactly what we do not want.
live_flag=()
if [ -n "${LIVE:-}" ]; then live_flag=(--live); fi

rc=0
synced=0
for name in "${changed[@]}"; do
    echo "--- $name"
    code=0
    # `|| code=$?` rather than a bare call: `set -e` would abort the loop.
    timeout -k 5 "$SYNC_TIMEOUT" "$WANDB_BIN" beta sync "$STAGE/$name" \
        --no-skip-synced --yes "${live_flag[@]}" || code=$?
    case "$code" in
        0)
            # Record the size only on success, so a failed run is retried next
            # time instead of being written off as done.
            printf '%s\t%s\n' "${seen_size[$name]}" "$name" >> "$MANIFEST"
            synced=$((synced + 1))
            ;;
        124|137)
            echo "TIMEOUT: $name after ${SYNC_TIMEOUT}s -- raise SYNC_TIMEOUT, or LIVE=1 was set on a dead run" >&2
            rc=1
            ;;
        *) echo "FAILED: $name (exit $code)" >&2; rc=1 ;;
    esac
done

# Last line per run wins on read-back, so appending is enough; compact anyway so
# the manifest does not grow without bound.
if [ -s "$MANIFEST" ]; then
    tac "$MANIFEST" | sort -u -k2,2 | sort -k2,2 > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
fi

echo "synced $synced/${#changed[@]} run(s)"
exit $rc
