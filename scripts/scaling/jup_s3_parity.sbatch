#!/bin/bash -x
# S3 gate on one JUPITER node (TITAN_MIGRATION_v2.md): TP=2, TP=2+SP, dp2 x TP2+SP
# (FSDP combined with TP) and TP=4+SP against one GPU, 9B, HF weights, fp32 GEMMs;
# then TP=4+SP and dp2 x TP2+SP again compiled (fullgraph=True) against compiled
# TP=1. Runs models/tests/test_titan_tp_parity.py; results in $TP_PARITY_OUT.
#
#   mkdir -p logs_s3 && sbatch scripts/scaling/jup_s3_parity.sbatch
#
# Other models: QWEN3_5_9B_SNAPSHOT=<snapshot>. HF_PARITY_TEST=<test script> first
# runs that single-GPU transformers parity test on the same snapshot (it reads
# QWEN3_VL_SNAPSHOT / QWEN3_5_SNAPSHOT), and VISION_TP_CHECK=1 adds the vision-only
# TP=2 run (every vision grad exactly 2x TP=1). See submit_qwen3_vl_checks.sh.

#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=s3parity
#SBATCH --output=logs_s3/%j.out
#SBATCH --error=logs_s3/%j.err

set -u

export QWEN3_5_9B_SNAPSHOT="${QWEN3_5_9B_SNAPSHOT:-/e/project1/reformo/ockier1/qwen_models/qwen3_5_9b}"
export TP_PARITY_OUT="${TP_PARITY_OUT:-/e/scratch/open-sci-mm/$USER/s3_tp_parity/$SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PARAM_DTYPE=fp32
export TORCHINDUCTOR_CACHE_DIR="/tmp/s3parity_${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/s3parity_${SLURM_JOB_ID}/triton"

source "${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}/etc/profile.d/conda.sh"
conda activate "${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

T=models/tests/test_titan_tp_parity.py
run() {  # run <nproc> <mode>
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( $1 - 1 ))) \
        python -m torch.distributed.run --standalone --nproc_per_node="$1" "$T" run "$2" || fail=1
}
fail=0

if [ -n "${HF_PARITY_TEST:-}" ]; then
    QWEN3_VL_SNAPSHOT="$QWEN3_5_9B_SNAPSHOT" QWEN3_5_SNAPSHOT="$QWEN3_5_9B_SNAPSHOT" \
        CUDA_VISIBLE_DEVICES=0 PARAM_DTYPE= python "$HF_PARITY_TEST" || fail=1
fi

if [ "${VISION_TP_CHECK:-0}" = 1 ]; then
    VISION_ONLY=1 run 1 tp1
    VISION_ONLY=1 run 2 tp2
    VISION_ONLY=1 python "$T" compare || fail=1
fi

# TP_PARITY=0: only the checks above
[ "${TP_PARITY:-1}" = 1 ] || { echo "[s3parity] done, fail=$fail"; exit $fail; }

run 1 tp1
run 2 tp2
run 2 tp2sp
run 4 dp2tp2sp
run 4 tp4sp
python "$T" compare || fail=1

export COMPILE=1
run 1 tp1
run 4 dp2tp2sp
run 4 tp4sp
python "$T" compare || fail=1

rm -rf "/tmp/s3parity_${SLURM_JOB_ID}"
echo "[s3parity] done, fail=$fail"
exit $fail
