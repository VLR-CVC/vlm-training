#!/bin/bash
# Sync offline wandb runs from Leonardo. Run it HERE, not on Leonardo.
#
#   scripts/sync_wandb_leo.sh                    # every run that grew since last time
#   scripts/sync_wandb_leo.sh 20261110           # restrict to one day
#   scripts/sync_wandb_leo.sh '2026111[01]'      # a glob over run directories
#   FORCE_ALL=1 scripts/sync_wandb_leo.sh        # ignore the manifest, sync everything
#   FETCH_ONLY=1 scripts/sync_wandb_leo.sh       # copy them down, do not upload
#   LIVE=1 scripts/sync_wandb_leo.sh             # tail runs (see the warning below)
#
# Same as sync_wandb_jup.sh: copy the run directories down, upload from this
# machine (with this machine's wandb login). Leonardo's compute nodes have no
# internet, so runs are written offline (WANDB_MODE=offline in leo_env.sh). See
# sync_wandb_jup.sh for why only grown runs are touched and why not --live.
#
# Runs are left in place on Leonardo.

set -euo pipefail

REMOTE="${LEO_REMOTE:-leonardo}"
# leo_env.sh pins WANDB_DIR here; wandb nests runs one level down, as
# $WANDB_DIR/wandb/offline-run-*.
RUN_ROOT="${LEO_WANDB_ROOT:-/leonardo_scratch/fast/EUHPC_E07_125/ockier1/wandb/wandb}"
STAGE="${WANDB_STAGE:-/data/151-2/users/tockier/wandb_leo}"
PATTERN="${1:-*}"
WANDB_BIN="${WANDB_BIN:-/data/151-2/users/tockier/envs/qwen4_mn5/bin/wandb}"
MANIFEST="$STAGE/.sizes"
# A backstop, not the mechanism. Without --live a sync terminates on its own;
# this only catches a hung upload.
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"

mkdir -p "$STAGE"
touch "$MANIFEST"

# --- what is on Leonardo, and how big ------------------------------------------
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
