#!/bin/bash
# Two EP ranks on ONE local card: does the all-to-all route every token right?
#
#   scripts/local_ep_parity.sh              # dispatcher parity, GPU 3
#   MODE=train scripts/local_ep_parity.sh   # trainer, ep_size 2 then 1, 2 ranks
#   MODE=train TRAIN_CONFIG=configs/local/qwen3_5_moe_tp_ep.toml \
#       scripts/local_ep_parity.sh          # TP=2 x EP=2, same two ranks
#   GPU=1 scripts/local_ep_parity.sh
#
# NCCL refuses two ranks per device by default ("Multiple Ranks are using the
# same GPU/Partition"); NCCL_MULTI_RANK_GPU_ENABLE=1 (NCCL >= 2.30) allows it.
# Verified on this box. What this buys: a real process group and a real
# all-to-all with uneven splits, on one free card. What it does NOT buy: a
# same-device a2a is local copies, so nothing here says anything about
# inter-device bandwidth or about perf. Confirm those on two real cards.
#
# torch11_jup is the JUPITER reference env: run in it, never install into it.
set -euo pipefail

GPU="${GPU:-3}"
RANKS="${RANKS:-2}"
ENV_DIR="${ENV_DIR:-/data/151-1/users/tockier/conda_install/envs/torch11_jup}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="$GPU"
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PATH="$ENV_DIR/bin:$PATH"
# torchrun runs the file by path, so the repo is not on sys.path by itself
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"

launch() {  # entrypoint args...
    torchrun --nproc_per_node="$RANKS" \
             --master_addr=127.0.0.1 \
             --master_port="$(shuf -i 20000-29999 -n 1)" \
             "$@"
}

if [ "${MODE:-dispatch}" = dispatch ]; then
    launch models/tests/ep_dispatch_parity.py "$@"
    exit $?
fi

if [ "${MODE}" = wiring ]; then
    # asserts the dispatcher class, its ep_mesh and the expert shard shapes --
    # the things a silent EP=1 fallback would get wrong while still matching
    # the ep_size=1 loss exactly
    launch models/tests/ep_wiring_probe.py "$@"
    exit $?
fi

# Trainer mode: the same MoE config at ep_size 2 and 1, two ranks either way, so
# the only difference is whether the experts are split. The losses track rather
# than match to the bit -- the two runs draw their random init on different
# meshes -- so what this proves is that EP trains at all and lands in the same
# place, not bitwise equality. `local_ep_parity.sh` (dispatch mode) is the exact
# check.
LOGDIR="${LOGDIR:-/data/151-2/users/tockier/tests/ep_debug}"
mkdir -p "$LOGDIR"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_TRACE="${TORCH_TRACE:-/data/151-2/users/tockier/profiling/ep_debug_trace}"
mkdir -p "$TORCH_TRACE"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/local/qwen3_5_moe_debug.toml}"
TAG="${TAG:-$(basename "$TRAIN_CONFIG" .toml)}"
for ep in ${EPS:-2 1}; do
    log="$LOGDIR/${TAG}_ep${ep}_$(date +%m%d_%H%M).log"
    echo "=== $TAG ep_size $ep -> $log"
    launch -m train.train_qwen \
           --config "$TRAIN_CONFIG" \
           --training.ep_size "$ep" \
           --training.output_dir "/data/151-2/users/tockier/qwen_finetune/checkpoints/${TAG}_ep${ep}" \
           "$@" >"$log" 2>&1 \
        && echo "ok  ep_size $ep" \
        || { echo "FAIL ep_size $ep"; tail -25 "$log"; exit 1; }
done
