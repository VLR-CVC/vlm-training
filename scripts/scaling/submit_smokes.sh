#!/bin/bash
# One-node, 40-step training smoke runs of both models from their JUPITER example
# configs (run from the repo root on JUPITER):
#   Qwen3.5-9B  HF weights, one 16384-token row (two do not fit one node's FSDP shard)
#   Qwen3-VL-2B HF weights, two 8192-token rows, dataloader state saved
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
smoke="--training.total-steps 40 --training.save-steps 100000"
submit() {  # submit <model> <extra trainer args>
    LOCAL_CACHE=1 QWEN_NCCL_TIMEOUT_S=1800 SCALING_TIME=00:40:00 \
        SCALING_SWEEP_ID="smoke_$1_$(date +%Y%m%d_%H%M%S)" SCALING_EXTRA_ARGS="$smoke $2" \
        SCALING_CONFIG="configs/jupiter/$1.toml" scripts/scaling/submit_sweep.sh 1
}
submit qwen3_5_9b "--data.tokens-per-microbatch 16384"
submit qwen3_vl_2b ""
