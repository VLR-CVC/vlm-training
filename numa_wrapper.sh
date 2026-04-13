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
exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"