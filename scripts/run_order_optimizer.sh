#!/bin/bash
# Run utils/optimize_split_order.py over OV-Instruct (giants excluded), the split-chunk
# dataset, and FineVision held fixed, and write orders/{ov,ov_chunks}.split.yaml.
#
# Run from the vlm-training checkout on MN5 (no sbatch needed -- this is CPU + a few GB
# RAM, minutes to tens of minutes depending on SECONDS):
#   bash scripts/run_order_optimizer.sh
#   SEARCH_SECONDS=300 WORKERS=64,128,256 bash scripts/run_order_optimizer.sh
#
# @10000 on the `ov` dataset keeps only tars with <= 10000 samples -- the ones still
# listed in onevision_instruct/.nv-meta/split.yaml.no_giants. Must match the
# --min-samples split_tars_mn5.sh was run with, or `ov` and `ov_chunks` overlap/gap.
#
# NB: the search-time variable is SEARCH_SECONDS, not SECONDS -- bash reserves
# SECONDS as its own auto-incrementing elapsed-time counter, so `${SECONDS:-180}`
# silently reads bash's counter (already "set", just usually 0 this early) instead
# of ever falling back to 180. That bug shipped in the first version of this
# script and gave a 0-second hill-climb budget with no error.

set -euo pipefail
LI=${LI:-/gpfs/scratch/ehpc543/length_index}
SECONDS_BUDGET=${SEARCH_SECONDS:-180}
WORKERS=${WORKERS:-64,128,256}
GIANT_THRESHOLD=${GIANT_THRESHOLD:-10000}
OUT=${OUT:-orders}

python3 utils/optimize_split_order.py \
  --dataset "ov=${LI}/ov_instruct@${GIANT_THRESHOLD}" \
  --dataset "ov_chunks=${LI}/ov_instruct_split" \
  --fixed "fv=${LI}/finevision" \
  --workers "$WORKERS" \
  --seconds "$SECONDS_BUDGET" \
  --out "$OUT"

echo
echo "Next: review the printed coverage, then to apply an order:"
echo "  cp $OUT/ov.split.yaml        onevision_instruct/.nv-meta/split.yaml"
echo "  cp $OUT/ov_chunks.split.yaml onevision_instruct_split/.nv-meta/split.yaml"
echo "(back up the live split.yaml first -- it's already been overwritten once before,"
echo " see .nv-meta/split.yaml.orig_20260926 / .shuffled_20260926 / .no_giants)"
