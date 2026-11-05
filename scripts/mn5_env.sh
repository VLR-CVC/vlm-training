# Shared MN5 environment setup. Sourced by mn5_finetune.sh (multinode_mn5.sh
# activates the same env directly).
#
# The env is a conda-pack tarball built on the workstation and unpacked here --
# MN5 has no internet. INSTALL.md, "MareNostrum 5", has how it is produced.

MN5_ENV="${MN5_ENV:-/gpfs/projects/ehpc543/envs/qwen4_mn5}"

[ -f "$MN5_ENV/bin/activate" ] || {
    echo "no env at $MN5_ENV -- build and unpack it first (INSTALL.md, MareNostrum 5)" >&2
    exit 1
}
# conda-pack's activate reads CONDA_PREFIX before it sets it, so under `set -u`
# it dies with "CONDA_PREFIX: unbound variable". Drop -u across the source only,
# and put it back only if the caller had it.
case $- in *u*) _had_nounset=1;; *) _had_nounset=;; esac
set +u
source "$MN5_ENV/bin/activate"
if [ -n "$_had_nounset" ]; then set -u; fi
unset _had_nounset

# No `module load cuda/...` here, deliberately. torch 2.14 ships its own CUDA 13
# runtime in the nvidia/* wheels, and a 12.x module puts an older libcudart and
# libnccl ahead of them on LD_LIBRARY_PATH. What the module was actually needed
# for is nvcc, which tilelang calls at run time (INSTALL.md, "Why the job scripts
# load CUDA") -- that now ships inside the env.
CUDA_HOME="$(python -c 'import nvidia.cu13 as m; print(m.__path__[0])')"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

command -v nvcc >/dev/null || {
    echo "nvcc not found under $CUDA_HOME -- tilelang/flash_qla will take the" >&2
    echo "gated-delta-rule forward down. Rebuild the env, or export" >&2
    echo "FLA_DISABLE_BACKEND_DISPATCH=1 to force FLA's Triton kernels." >&2
    exit 1
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export LOGLEVEL=INFO
