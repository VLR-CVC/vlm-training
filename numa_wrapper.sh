#!/bin/bash

# this script is used to BIND the arm cores in the GB200 (JUPITER) to each GPU
# you do not want pytorch/python to do whatever it wants, each instance of the code should
# have access to a specific set of ARM cores with their respective single GPU

# it is meant to be used as a wrapper with our own bash scripts
# example:
#
#    torchrun \
#    --nnodes=1 \
#    --nproc_per_node=$NGPUS \
#    --no-python \
#    ./numa_wrapper.sh python -m train.train_qwen $@

NUMA_NODE=${LOCAL_RANK:-0}

# Inductor bakes process-group *names* into compiled collectives, and a group is
# only registered on the ranks that belong to it. Two ranks in different TP
# groups therefore compile byte-different kernels from an identical graph, and a
# shared FX cache lets one load the other's:
if [ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/n${GROUP_RANK:-${SLURM_NODEID:-0}}"
fi

# Per-rank torch logs (compile, recompiles, graph breaks). `TORCH_LOGS_OUT` names
# ONE file per process and has no rank template, and the sbatch cannot build the
# path because `RANK` only exists inside torchrun -- hence here. Pair it with
# TORCH_LOGS, e.g.
#   QWEN_TORCH_LOGS_DIR=$RUN/torchlogs TORCH_LOGS=recompiles,graph_breaks
# Every rank writes its own file, so this is how two ranks that compiled
# different kernels from the same graph become visible.
if [ -n "${QWEN_TORCH_LOGS_DIR:-}" ]; then
    # Keyed on the job too: a sweep points every node count at one QWEN_TORCH_LOGS_DIR,
    # and without this the 32-node job's rank_0.log overwrites the 4-node job's.
    dir="${QWEN_TORCH_LOGS_DIR}/${SLURM_JOB_ID:-0}"
    mkdir -p "$dir"
    export TORCH_LOGS_OUT="${dir}/rank_${RANK:-${SLURM_PROCID:-0}}.log"
fi

# Structured compile trace for `tlparse`: every compile with its guards, symbolic-shape
# decisions and duration -- far more than TORCH_LOGS=recompiles gives. One directory per
# rank for the same reason, and `RANK` again only exists inside torchrun.
if [ -n "${QWEN_TORCH_TRACE_DIR:-}" ]; then
    export TORCH_TRACE="${QWEN_TORCH_TRACE_DIR}/${SLURM_JOB_ID:-0}/rank_${RANK:-${SLURM_PROCID:-0}}"
    mkdir -p "$TORCH_TRACE"
fi

exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"