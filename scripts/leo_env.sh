# Shared Leonardo (CINECA, A100 sm_80) environment setup. Source it from a job script.
#
# Env: uv venv with requirements.txt (torch 2.14.0+cu130) + the INSTALL.md resolution
# (flash_qla out; apache-tvm-ffi 0.1.12, tilelang 0.1.14, attn_gym[linear] 0.0.9,
# spmd_types 0.2.5, cuda-toolkit[nvcc] 13.0 inside the env).

LEO_BASE="${LEO_BASE:-/leonardo_scratch/fast/EUHPC_E07_125/ockier1}"
LEO_ENV="${LEO_ENV:-$LEO_BASE/envs/qwen4_leo}"

[ -x "$LEO_ENV/bin/python" ] || { echo "no env at $LEO_ENV" >&2; exit 1; }

# The kernel driver is 535 (CUDA 12.2); torch cu130 needs the CUDA 13.0 forward-compat
# user-mode driver (NVIDIA cuda-compat-13-0 580.196.07 rpm, unpacked). Without it:
# "The NVIDIA driver on your system is too old". Triton finds libcuda on its own, so
# point it there too.
COMPAT="$LEO_BASE/cuda_compat/13.0_580.196.07"
export LD_LIBRARY_PATH="$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH="$COMPAT"

# The system g++ is 8.5 and rejects -std=c++20, which torch Inductor's CPU codegen passes
# (torchao's compiled _AdamW, any CPU torch.compile). Loaded before the env so the env's
# python comes first on PATH. A batch shell may not have `module` defined yet.
command -v module >/dev/null 2>&1 || source /etc/profile
module load gcc/12.2.0

export PATH="$LEO_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

# tilelang calls nvcc at run time (INSTALL.md, "Why the job scripts load CUDA"); it ships
# inside the env. No `module load cuda`: Leonardo's modules are 12.x.
CUDA_HOME="$(python -c 'import nvidia.cu13 as m; print(m.__path__[0])')"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# Compiler caches on personal scratch ($SCRATCH = $CINECA_SCRATCH, owner-only, purged
# after ~40 days unused: fine for caches). Everything else on $FAST, not $HOME (50 GB
# quota). TILELANG_TMP_DIR must share a filesystem with TILELANG_CACHE_DIR.
VLM_CACHE="${VLM_CACHE:-${SCRATCH:-/leonardo_scratch/large/userexternal/$USER}/vlm_cache}"
export VLM_CACHE
export XDG_CACHE_HOME="$LEO_BASE/cache"
export HF_HOME="$LEO_BASE/cache/hf"
export TRITON_CACHE_DIR="$VLM_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$VLM_CACHE/inductor"
export TILELANG_CACHE_DIR="$VLM_CACHE/tilelang"
export TILELANG_TMP_DIR="$VLM_CACHE/tilelang/tmp"
export ATTN_GYM_CUTE_CACHE_DIR="$VLM_CACHE/attention_gym/cute"
export CUDA_CACHE_PATH="$VLM_CACHE/nv"
export WANDB_DIR="$LEO_BASE/wandb"
mkdir -p "$TILELANG_TMP_DIR" "$ATTN_GYM_CUTE_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

# Compute nodes have no internet.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
