#!/bin/bash -x
#SBATCH --account=open-sci-mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=booster
#SBATCH --job-name=build_fa
#SBATCH --output=logs_jup/%j/log_%x.out
#SBATCH --error=logs_jup/%j/log_%x.err

# Build flash_attn against torch 2.14 and verify the env, as a JOB.
#
#   sbatch scripts/jup_build_fa.sbatch
#
# WHY A JOB. INSTALL.md builds this on the login node, which does not survive
# contact with automation: `login.jupiter.fz-juelich.de` is round-robin DNS over
# several login nodes, they cannot ssh to each other, and a detached build is
# therefore unfindable from the next connection -- which lands somewhere else and
# reports it missing. Two builds were lost that way. Under Slurm the node is
# fixed, the log is kept, and nothing reaps it.
#
# A GPU is requested for the call check: the tvm-ffi/attn_gym break is a
# call-time failure that no import test can see, and login nodes have no GPU.
#
# Reuses /e/project1/open-sci-mm/ockier1/build/flash-attention if it is there, so
# a partially finished compile resumes instead of restarting.

set -euo pipefail

OWNER=/e/project1/open-sci-mm/ockier1
CONDA_ROOT="${JUP_CONDA_ROOT:-$OWNER/envs/miniforge3}"
TORCH_ENV="${JUP_TORCH_ENV:-$OWNER/cache/conda/envs/torch_main_214}"
VENV="${JUP_VENV:-$OWNER/torchtitan/venv_214}"
BUILD="$OWNER/build"

export CACHE_ROOT="$OWNER/cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton/cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch/inductor"
export HF_HOME="$CACHE_ROOT/huggingface"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$TORCH_ENV"
source "$VENV/bin/activate"
command -v module >/dev/null 2>&1 || source /etc/profile
module load CUDA/13
# flash_attn's own setup.py shells out to git while generating metadata, so the
# job needs it on PATH even though none of the steps below call it: job 2559184
# died with `FileNotFoundError: [Errno 2] No such file or directory: 'git'` and
# `error: metadata-generation-failed`. Compute nodes do not have it by default.
module load git
nvcc --version | tail -1
git --version

python -c "import torch; assert torch.__version__.startswith('2.14.'), torch.__version__; print('torch', torch.__version__)"

# The clone, the checkout and the C++20 patch happen on a login node, where they
# take seconds; this job only compiles, and asserts the tree rather than
# assuming it. (`git` is still loaded above, for flash_attn's setup.py.)
cd "$BUILD/flash-attention" || {
    echo "no prepared source tree. On a login node:" >&2
    echo "  module load git && cd $BUILD" >&2
    echo "  git clone https://github.com/Dao-AILab/flash-attention" >&2
    echo "  cd flash-attention && git checkout v2.8.3.post1" >&2
    echo "  sed -i 's/-std=c++17/-std=c++20/g' setup.py" >&2
    exit 1
}
grep -q '__version__ = "2.8.3.post1"' flash_attn/__init__.py || {
    echo "source tree is not v2.8.3.post1" >&2; exit 1; }
# torch 2.14's headers need C++20 (std::strong_ordering in c10/util/intrusive_ptr.h);
# setup.py ships -std=c++17 in five places, so an unpatched tree fails late.
! grep -q 'std=c++17' setup.py || {
    echo "setup.py still asks for C++17; apply the sed on a login node" >&2; exit 1; }
# 72 cores here, but INSTALL.md's note stands: MAX_JOBS=48 dies with "Cannot
# allocate memory" partway through, 8 finishes.
MAX_JOBS=8 pip install . --no-build-isolation

echo "=== verify"
python - <<'PY'
import inspect, pathlib, torch
assert torch.__version__.startswith("2.14."), torch.__version__
from torch.distributed.fsdp import fully_shard
assert "dp_mesh_dims" in inspect.signature(fully_shard).parameters
# The 2.13 check that killed jobs 2557458/60/61.
root = pathlib.Path(torch.__file__).parent / "distributed"
hits = [f for f in root.rglob("*.py")
        if "all parameters must be DTensors" in f.read_text(errors="ignore")]
assert not hits, f"this torch still rejects plain annotated tensors: {hits}"

from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn  # noqa: F401
import causal_conv1d as cc, flash_attn, spmd_types, triton, torchtitan  # noqa: F401
import tilelang, tvm_ffi  # dist apache-tvm-ffi, module tvm_ffi
try:
    import flash_qla  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("flash_qla is installed; it breaks attn_gym.linear at call time")
print("torch", torch.__version__, "| triton", triton.__version__,
      "| flash_attn", flash_attn.__version__,
      "| tvm-ffi", tvm_ffi.__version__, "| tilelang", tilelang.__version__)

# The call that an import check cannot stand in for.
x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
w = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16)
print("causal_conv1d call ok", tuple(causal_conv1d(x, w).shape))
PY

echo ENV_214_VERIFIED
