#!/bin/bash -x
# Gradient accumulation reduces once per window, not once per micro-batch
# (train/step.py). Two checks on one JUPITER node:
#
#   1. models/tests/test_fsdp_nosync.py at 1 and 2 ranks -- the mechanism, torch only
#   2. models/tests/test_dp_parity.py at 1 rank (4 micro-batches) and 2 ranks (2 each)
#      -- the numerics: same loss and grad norm, or the deferred reduction changed
#      the step
#
#   mkdir -p logs_dp && sbatch scripts/scaling/jup_dp_parity.sbatch
#
# Compare the two JSON lines in $PARITY_OUT. `replicate` (data_parallel = ddp) is
# excluded on purpose: it has the same torch bug and no patch (train/step.py).

#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=dpparity
#SBATCH --output=logs_dp/%j.out
#SBATCH --error=logs_dp/%j.err

set -u

# 2B, not 9B: at world=1 FSDP shards over nothing, so the whole model sits on one
# GPU. The 9B OOMs there (91.7 of 95 GiB, job 1889557) while world=2 passes, which
# leaves exactly one row in $PARITY_OUT and nothing to compare.
export QWEN3_5_SNAPSHOT="${QWEN3_5_SNAPSHOT:-/e/project1/reformo/ockier1/qwen_models/qwen3_5_2b}"
export PARITY_OUT="${PARITY_OUT:-/e/scratch/open-sci-mm/$USER/dp_parity/$SLURM_JOB_ID.jsonl}"
mkdir -p "$(dirname "$PARITY_OUT")"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="/tmp/dpparity_${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/dpparity_${SLURM_JOB_ID}/triton"

source "${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}/etc/profile.d/conda.sh"
conda activate "${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

fail=0
run() {  # run <nproc> <script> [args...]
    local n=$1; shift
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( n - 1 ))) \
        python -m torch.distributed.run --standalone --nproc_per_node="$n" "$@" || fail=1
}

# 1. the mechanism, plus its control (reduce every micro-batch): all four must pass
run 1 models/tests/test_fsdp_nosync.py
run 2 models/tests/test_fsdp_nosync.py
NOSYNC=0 run 1 models/tests/test_fsdp_nosync.py
NOSYNC=0 run 2 models/tests/test_fsdp_nosync.py

# 2. the numerics, on the real model
run 1 models/tests/test_dp_parity.py fsdp
run 2 models/tests/test_dp_parity.py fsdp

echo "[dpparity] $PARITY_OUT"
cat "$PARITY_OUT" || fail=1
python - "$PARITY_OUT" <<'EOF' || fail=1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by_world = {r["world"]: r for r in rows}
assert set(by_world) >= {1, 2}, f"need world 1 and 2, got {sorted(by_world)}"
a, b = by_world[1], by_world[2]
assert a["global_tokens"] == b["global_tokens"], (a, b)
# bf16 reduction order differs with the rank count, so this is a tolerance, not
# equality; 1e-4 relative is the spread seen between two runs of the same config
for key, tol in (("loss", 1e-4), ("grad_norm", 1e-3)):
    rel = abs(a[key] - b[key]) / max(abs(a[key]), 1e-12)
    print(f"{key}: world1={a[key]!r} world2={b[key]!r} rel={rel:.2e} tol={tol:.0e}")
    assert rel < tol, f"{key} differs by {rel:.2e} > {tol:.0e}"
print("[dpparity] loss and grad norm agree across DP size")
EOF

rm -rf "/tmp/dpparity_${SLURM_JOB_ID}"
echo "[dpparity] done, fail=$fail"
exit $fail
