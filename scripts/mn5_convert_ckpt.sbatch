#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=mn5_convert_ckpt
#SBATCH --account=ehpc543
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

BASE="${1:?usage: sbatch $0 <base-snapshot> <ckpt-dir> [...]}"
shift
[ $# -gt 0 ] || { echo "no checkpoint directories given" >&2; exit 2; }

source scripts/mn5_env.sh

[ -f "$BASE/config.json" ] || { echo "no config.json in base snapshot $BASE" >&2; exit 2; }

rc=0
for ckpt in "$@"; do
    if [ ! -f "$ckpt/.metadata" ]; then
        echo "SKIP $ckpt: not a DCP checkpoint (no .metadata)" >&2
        rc=1
        continue
    fi
    step="${ckpt##*-}"
    out="$(dirname "$ckpt")/models/step-$step"
    echo "=== $ckpt -> $out"
    if python utils/dcp_to_hf.py "$ckpt" "$BASE" "$out"; then
        # The index is what `from_pretrained` reads; if it is missing the
        # conversion wrote nothing usable.
        [ -f "$out/model.safetensors.index.json" ] || { echo "FAILED $ckpt: no index written" >&2; rc=1; }
    else
        echo "FAILED $ckpt" >&2
        rc=1
    fi
done
exit $rc
