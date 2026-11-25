#!/bin/bash
set -euo pipefail

module --force purge
module load Stages/2025 GCCcore/.13.3.0 Apptainer-Tools/2025

IMAGE="${IMAGE:-$PROJECT_ccstdl/$USER/containers/vlm-training-ngc26.11-arm64.sif}"
[[ ! -e "$IMAGE" ]] || { echo "Image already exists: $IMAGE" >&2; exit 1; }

# Keep the unpacked rootfs and SquashFS work on the login node's local disk.
APPTAINER_TMPDIR=$(mktemp -d /tmp/vlm-training-ngc26.11.XXXXXX)
export APPTAINER_TMPDIR
trap 'rmdir "$APPTAINER_TMPDIR"' EXIT

cd "$(dirname "$0")/.."
apptainer build --fakeroot --arch arm64 --disable-cache --mksquashfs-args "-processors 8 -comp gzip -Xcompression-level 1" \
    "$IMAGE" containers/vlm-training.def
