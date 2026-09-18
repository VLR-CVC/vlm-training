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
#   RuntimeError: Could not resolve the process group registered under the name 16
# (job 1782272, rank 15 -- TP group 3 -- running TP group 15's kernel). The cache
# dir the sbatch exports is keyed on world size only, which does not separate
# these. Verified: compilecache/d16/.../ws64 holds kernels for groups 7, 16 and
# 17 side by side.
# ponytail: one cache per node == one per TP group only while tp_size == NGPUS
# (TP is intra-node here). If tp_size ever drops below the GPUs-per-node count,
# key this on the TP group index instead.
if [ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/n${GROUP_RANK:-${SLURM_NODEID:-0}}"
fi

exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"