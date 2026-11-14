#!/bin/bash
# Convert the newest COMPLETE DCP checkpoint of a run to an HF snapshot, then verify it.
#   RUN_DIR=$SCRATCH/checkpoints/<run> BASE=<HF snapshot the run started from> \
#   OUT=$RUN_DIR/models/final FROZEN_PREFIX=model.language_model. sbatch scripts/dcp_to_hf_leo.sh
# "Complete" = has the `.metadata` DCP writes last, so a save cut by the walltime is skipped.
# Fails (non-zero exit, so `afterok` dependents never start) unless the export has every key of
# BASE except MTP, tensors under FROZEN_PREFIX equal BASE's bit for bit (in bf16), the ViT patch
# embed differs from BASE's, and the merger is present, finite and differs from BASE's.
#SBATCH -D .
#SBATCH --job-name=dcp_to_hf
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_scratch/fast/EUHPC_E07_125/ockier1/logs/%x_%j.out

set -eo pipefail
source scripts/leo_env.sh

CKPT=""
for d in $(ls -d "$RUN_DIR"/checkpoint-step-* 2>/dev/null | sort -rV); do
    [ -f "$d/.metadata" ] && { CKPT=$d; break; }
    echo "skipping incomplete $d (no .metadata)"
done
[ -n "$CKPT" ] || { echo "no complete checkpoint in $RUN_DIR" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT already exists; refusing to overwrite" >&2; exit 1; }

echo "converting $CKPT -> $OUT (base $BASE)"
python utils/dcp_to_hf.py "$CKPT" "$BASE" "$OUT"
chmod 644 "$OUT/model.safetensors"
echo "$CKPT" > "$OUT/SOURCE_CHECKPOINT"

python - "$BASE" "$OUT" "${FROZEN_PREFIX:-model.language_model.}" <<'PY'
import json, sys, torch
from pathlib import Path
from safetensors import safe_open
base, out, frozen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

def load(d):
    files = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())
    t = {}
    for f in files:
        with safe_open(str(d / f), "pt") as h:
            t.update({k: h.get_tensor(k) for k in h.keys()})
    return t

b, o = load(base), load(out)
want = {k for k in b if not k.startswith("mtp.")}
assert set(o) == want, f"key mismatch: missing {sorted(want - set(o))[:5]} extra {sorted(set(o) - want)[:5]}"
assert all(torch.isfinite(v.float()).all() for v in o.values()), "non-finite tensor in export"
fz = [k for k in o if k.startswith(frozen)]
assert fz, f"no keys under {frozen}"
bad = [k for k in fz if not torch.equal(o[k], b[k].to(o[k].dtype))]
assert not bad, f"frozen tensors changed: {bad[:5]}"
pe = "model.visual.patch_embed.proj.weight"
assert not torch.equal(o[pe], b[pe].to(o[pe].dtype)), "ViT patch embed equals BASE's (SigLIP not loaded?)"
mg = [k for k in o if ".merger." in k]
assert len(mg) == 6, f"expected 6 merger tensors, got {mg}"
same = [k for k in mg if torch.equal(o[k], b[k].to(o[k].dtype))]
assert not same, f"merger tensors equal BASE's: {same}"
print(f"OK: {len(o)} tensors, {len(fz)} frozen identical, ViT differs from base, merger trained")
PY
