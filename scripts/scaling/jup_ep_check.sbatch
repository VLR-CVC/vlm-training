#!/bin/bash -x
# Expert parallelism on real hardware: ONE node, 4 GH200, three stages.
#
#   sbatch --output=.../ep_%j.out --error=.../ep_%j.err scripts/scaling/jup_ep_check.sbatch
#
# Everything EP has been verified on so far ran two ranks on ONE card, where
# NCCL turns the all-to-all into local copies and `efsdp` is pinned to 1. This
# job is the first run where neither is true:
#
#   world 4, tp_size 1, ep_size 2
#     -> dp_shard 4, sparse region dp_shard*cp*tp = 4, efsdp = 4/2 = 2
#     -> expert weights split across 2 EP ranks AND FSDP-sharded within each
#        EP group, with a real device-to-device all-to-all between them
#
# Stages, cheapest first, so a failure tells you where it broke:
#   1. dispatch parity  -- the a2a against exact arithmetic (no model, seconds)
#   2. wiring probe     -- dispatcher class, ep_mesh, expert/dense shard shapes
#                          at ep 2 (efsdp 2) and ep 4 (efsdp 1)
#   3. trainer          -- ep_size 2 then 1, same steps, losses must track
#
# All three run even if an earlier one fails; the summary at the end reports
# each, and the job exits non-zero if any did. Nothing here resumes or keeps a
# checkpoint.

#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --partition=booster
#SBATCH --exclusive
#SBATCH --job-name=ep_check

set -u

CONFIG_FILE="${EP_CONFIG:-configs/jupiter/qwen3_5_moe_ep_debug.toml}"
OUTPUT_ROOT="${EP_OUTPUT_ROOT:-/e/scratch/open-sci-mm/$USER/ep_check}"
STEPS="${EP_STEPS:-20}"
# ep degrees for the trainer stage: 2 gives efsdp 2, 1 is the reference curve
TRAIN_EPS="${EP_TRAIN_EPS:-2 1}"
# ep degrees for the wiring probe: 4 also covers efsdp 1 at 4 ranks
PROBE_EPS="${EP_PROBE_EPS:-2 4}"

NNODES=$SLURM_JOB_NUM_NODES
NGPUS=4
NGPU_TOTAL=$(( NNODES * NGPUS ))
[ "$NNODES" = 1 ] || echo "[ep] NOTE: written for one node, got $NNODES"

nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "[ep] ${NNODES} node / ${NGPU_TOTAL} GPUs, head ${head_node} (${head_node_ip})"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NCCL_SOCKET_IFNAME="ib,eth"
export NCCL_P2P_LEVEL=NVL
export LOGLEVEL="${LOGLEVEL:-INFO}"
# WARN by default; set NCCL_DEBUG=INFO to see which transport the all-to-all
# actually picked, which is the one thing a single-card run could never show
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export NCCL_BUFFSIZE=2097152
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Node-local compile cache: this job compiles cold on purpose. The shared GPFS
# cache keys compiled collectives by process-group name, and an EP job registers
# groups a dense job never did -- see the long note in jup_scaling.sbatch.
LOCAL_CACHE_DIR="/tmp/ep_check_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_CACHE_DIR/triton"
srun --ntasks-per-node=1 mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

CONDA_ROOT="${JUP_CONDA_ROOT:-/e/project1/open-sci-mm/ockier1/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-/e/project1/open-sci-mm/ockier1/cache/conda/envs/torch_main}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
# spmd_types and attn_gym live in the torchtitan venv, layered on top so
# torch_main stays unmodified (same as jup_scaling.sbatch / tt_bench.sbatch)
source "${JUP_VENV:-/e/project1/open-sci-mm/ockier1/torchtitan/venv}/bin/activate"
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13

ulimit -l unlimited
ulimit -s unlimited
ulimit -c 0

# `models/tests/*.py` are run by path, so the repo has to be importable
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

launch() {  # entrypoint + args, 4 ranks on this node
    srun --cpu-bind=none \
        torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPUS \
        --rdzv_id "${SLURM_JOB_ID}_${RDZV_SEQ:-0}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="$head_node_ip:29500" \
        "$@"
}

declare -A RC

echo "=============== stage 1: dispatch parity (4 ranks) ==============="
RDZV_SEQ=1 launch models/tests/ep_dispatch_parity.py
RC[dispatch]=$?

for ep in $PROBE_EPS; do
    echo "=============== stage 2: wiring probe ep=$ep ==============="
    # MODEL_CONFIG matches the toml so the probe checks the same expert count
    RDZV_SEQ="2$ep" EP=$ep TP=1 \
        MODEL_CONFIG=configs/models/qwen3_5_moe_tp_ep.json \
        launch models/tests/ep_wiring_probe.py
    RC[probe_ep$ep]=$?
done

# The processor/tokenizer has to exist for the dataloader; the weights do not
# (random_init). Skipping beats failing four ranks deep in the dataloader.
MODEL_DIR="${EP_MODEL_DIR:-/e/project1/open-sci-mm/ockier1/qwen_models/qwen35_2b}"
if [ ! -f "$MODEL_DIR/preprocessor_config.json" ]; then
    cat <<MSG
[ep] SKIPPING stage 3: no processor at $MODEL_DIR
     Stages 1 and 2 are the EP proof and ran above. To enable the trainer
     stage, stage the 22 MB of processor json (not the 4.3 GB snapshot) from
     the local box:
       rsync -az --include='*/' --include='*.json' --include='*.txt' \
           --include='*.jinja' --exclude='*' \
           /data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b/ \
           jupiter:$MODEL_DIR/
MSG
    RC[train_skipped]=0
    TRAIN_EPS=""
fi

for ep in $TRAIN_EPS; do
    echo "=============== stage 3: trainer ep_size=$ep ==============="
    RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_ID}_ep${ep}"
    RDZV_SEQ="3$ep" launch --no-python \
        python -m train.train_qwen \
            --config "$CONFIG_FILE" \
            --training.model-dir "$MODEL_DIR" \
            --training.ep-size "$ep" \
            --training.total-steps "$STEPS" \
            --training.output-dir "$RUN_DIR" \
            --training.no-resume-checkpoint \
            --wandb.run-name "ep_check-ep${ep}-${SLURM_JOB_ID}" \
            ${EP_EXTRA_ARGS:-}
    RC[train_ep$ep]=$?
    # throwaway: the final checkpoint is written whatever save_steps says
    [ "${EP_KEEP_CHECKPOINT:-0}" = 0 ] && rm -rf "$RUN_DIR"
done

srun --ntasks-per-node=1 rm -rf "$LOCAL_CACHE_DIR"

echo "=============== summary ==============="
fail=0
for k in "${!RC[@]}"; do
    printf '  %-14s exit %s\n' "$k" "${RC[$k]}"
    [ "${RC[$k]}" = 0 ] || fail=1
done
cat <<'NOTE'

Next, by hand, from the two trainer logs:
  grep -E "^.*[0-9]+ - .loss" <log>   # ep2 and ep1 losses must track step for step
The wiring probe lines are the proof EP engaged at all -- a silent EP=1
fallback produces the SAME loss curve, which is why the probe exists.
NOTE
echo "[ep] done, fail=$fail"
exit $fail
