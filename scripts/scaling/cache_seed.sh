#!/bin/bash
# Move one node's inductor/triton cache between GPFS and node-local /tmp.
#
#   srun --ntasks-per-node=1 scripts/scaling/cache_seed.sh seed <gpfs_dir> <local_dir>
#   srun --ntasks-per-node=1 scripts/scaling/cache_seed.sh save <gpfs_dir> <local_dir>
#
# Compiling straight against a GPFS cache stalls the job before step 1 (job
# 1896752: 21 min without reaching step 1, where the same shape on a tmpfs cache
# ran 500 steps in 21.5 min). Compiling with no cache at all costs ~31s of
# cold-start per rank (compile_id 0/0, WARMUP_PLAN.md). `seed` before the run and
# `save` after gets both: tmpfs speed during the run, persistence between runs.
#
# Layout. `numa_wrapper.sh` appends /n${GROUP_RANK} to TORCHINDUCTOR_CACHE_DIR, so
# on disk a node uses <local>/inductor/n<K> and <local>/triton. The GPFS side
# mirrors that as <gpfs>/n<K>/{inductor,triton}.
#
# K is SLURM_NODEID here but GROUP_RANK inside torchrun. They are both the node's
# index in the same allocation and should agree; if they ever do not, a node just
# gets seeded with another node's cache. That is harmless -- inductor entries are
# content-keyed, so a non-matching entry is a miss, never a wrong kernel. It costs
# a recompile, not correctness.
set -euo pipefail

mode="$1"; gpfs="$2"; local_dir="$3"
n="${SLURM_NODEID:-0}"

case "$mode" in
seed)
    mkdir -p "$local_dir/inductor/n$n" "$local_dir/triton"
    # cp -a of a missing source is an error under `set -e`; a cold cache is normal.
    [ -d "$gpfs/n$n/inductor" ] && cp -a "$gpfs/n$n/inductor/." "$local_dir/inductor/n$n/" || true
    [ -d "$gpfs/n$n/triton" ] && cp -a "$gpfs/n$n/triton/." "$local_dir/triton/" || true
    [ "$n" = 0 ] && echo "[cache] seeded node 0 from $gpfs/n0: $(du -sh "$local_dir" 2>/dev/null | cut -f1)"
    ;;
save)
    mkdir -p "$gpfs/n$n/inductor" "$gpfs/n$n/triton"
    [ -d "$local_dir/inductor/n$n" ] && cp -a "$local_dir/inductor/n$n/." "$gpfs/n$n/inductor/" || true
    [ -d "$local_dir/triton" ] && cp -a "$local_dir/triton/." "$gpfs/n$n/triton/" || true
    [ "$n" = 0 ] && echo "[cache] saved node 0 to $gpfs/n0: $(du -sh "$gpfs/n$n" 2>/dev/null | cut -f1)"
    ;;
*)
    echo "usage: $0 seed|save <gpfs_dir> <local_dir>" >&2
    exit 2
    ;;
esac
exit 0
