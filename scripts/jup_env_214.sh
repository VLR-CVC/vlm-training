#!/bin/bash
# Build a JUPITER environment at the pinned torch 2.14.0, beside the current one.
#
#   scripts/jup_env_214.sh go          # run it (the argument is deliberate)
#
# WHY. `torch_main` drifted to torch 2.13.0 / triton 3.7.1 on 2026-10-23, against
# `requirements.txt`'s torch==2.14.0 / triton==3.8.0. torch 2.13's `fully_shard`
# rejects plain tensors when `dp_mesh_dims` is passed:
#
#   Error: When dp_mesh_dims is provided, all parameters must be DTensors on the
#   full SPMD mesh (e.g. via distribute_module). Got plain tensor for parameter 'weight'.
#
# which is exactly what `train/parallel/fsdp.py` relies on -- TP-local plain
# tensors carrying SPMD annotations, which `fully_shard` is supposed to convert.
# Every tp_size > 1 run dies in `Trainer.__init__` (jobs 2557458/60/61). 2.14
# dropped that check, so this is an upgrade back onto the lock, not a rollback.
#
# NOTHING EXISTING IS MUTATED. A new conda env and a new venv are created; the
# old pair stays bootable, so backing out is a matter of not passing the two
# variables printed at the end.
#
# A fresh `conda create` rather than `--clone torch_main`: a clone would inherit
# the extensions built against 2.13 (`causal_conv1d` there already fails to
# import with `undefined symbol`) and the drifted triton. This is INSTALL.md's
# documented procedure instead.
set -euo pipefail
[ "${1:-}" = "go" ] || { sed -n '2,8p' "$0"; exit 2; }

OWNER=/e/project1/open-sci-mm/ockier1
CONDA_ROOT="${JUP_CONDA_ROOT:-$OWNER/envs/miniforge3}"
OLD_CONDA="$OWNER/cache/conda/envs/torch_main"
OLD_VENV="$OWNER/torchtitan/venv"
NEW_CONDA="${NEW_CONDA:-$OWNER/cache/conda/envs/torch_main_214}"
NEW_VENV="${NEW_VENV:-$OWNER/torchtitan/venv_214}"
TITAN_REPO="$OWNER/torchtitan/repo"
BUILD="$OWNER/build"
CU130=https://download.pytorch.org/whl/cu130

# INSTALL.md section 0: $HOME has a small quota and every one of these caches
# will blow through it on its own.
export CACHE_ROOT="$OWNER/cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton/cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch/inductor"
export HF_HOME="$CACHE_ROOT/huggingface"

for p in "$NEW_CONDA" "$NEW_VENV"; do
    [ -e "$p" ] && { echo "$p exists already -- remove it or set NEW_CONDA/NEW_VENV" >&2; exit 1; }
done
[ -d "$OLD_VENV" ] && [ -d "$TITAN_REPO" ] || { echo "old venv or torchtitan repo missing" >&2; exit 1; }

# Needed at build time for both extensions (and at run time for tilelang).
module load CUDA/13
nvcc --version | tail -1

echo "=== 1/6 conda env at $NEW_CONDA"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -y -p "$NEW_CONDA" python=3.13.15

echo "=== 2/6 requirements.txt (torch 2.14.0, triton 3.8.0) into the conda env"
conda activate "$NEW_CONDA"
pip install -r requirements.txt --extra-index-url "$CU130"
python -c "import torch; assert torch.__version__.startswith('2.14.'), torch.__version__; print('torch', torch.__version__)"

echo "=== 3/6 venv at $NEW_VENV"
# --system-site-packages, like the old one: a venv's pyvenv.cfg binds it to the
# conda env it was made from, which is why the new conda env needs its own.
python -m venv --system-site-packages "$NEW_VENV"
# shellcheck disable=SC1091
source "$NEW_VENV/bin/activate"

echo "=== 4/6 the old venv's own packages"
# Everything the old venv held locally, minus torchtitan: that one is an
# editable install pointing at the PINNED reference checkout, and `pip install
# -e` would write egg metadata into it. A .pth gives the same imports and
# writes nothing there.
"$OLD_VENV/bin/pip" freeze --local | grep -v '^-e ' > /tmp/venv_local_$$.txt
grep -c . /tmp/venv_local_$$.txt
pip install -r /tmp/venv_local_$$.txt
echo "$TITAN_REPO" > "$NEW_VENV/lib/python3.13/site-packages/torchtitan_reference.pth"
rm -f /tmp/venv_local_$$.txt

echo "=== 4b/6 the flash_qla / attn_gym.linear resolution"
# INSTALL.md, "flash_qla and attn_gym.linear cannot both be installed":
# requirements.txt still locks the flash_qla trio (apache-tvm-ffi==0.1.9,
# tilelang==0.1.9, flash_qla==0.1.2) and that freeze predates attn_gym.linear.
# attn_gym[linear] needs apache-tvm-ffi>=0.1.12; on 0.1.9 its causal_conv1d dies
# at CALL time with `make_kwargs_wrapper() got an unexpected keyword argument
# 'map_dataclass_to_tuple'`, and models/qwen3_5/gdn.py imports those kernels
# unguarded, so that is the trainer failing. attn_gym wins; flash_qla goes.
# 0.1.12 is the one version leaving both tilelang and attn_gym importable.
pip uninstall -y flash_qla || true
pip install "apache-tvm-ffi==0.1.12" "tilelang==0.1.14"

echo "=== 5/6 CUDA extensions, rebuilt against 2.14"
# INSTALL.md: never reuse a wheel across a torch upgrade -- torch 2.14 added a
# sixth parameter to c10::cuda::c10_cuda_check_implementation, so anything built
# earlier fails at import with `undefined symbol` while pip list looks fine.
mkdir -p "$BUILD" && cd "$BUILD"
[ -d causal-conv1d ] || git clone https://github.com/Dao-AILab/causal-conv1d
(cd causal-conv1d && git checkout v1.7.0 && MAX_JOBS=8 pip install . --no-build-isolation)

[ -d flash-attention ] || git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout v2.8.3.post1
git checkout -- setup.py
# torch 2.14's headers need C++20 (std::strong_ordering in c10/util/intrusive_ptr.h);
# setup.py still asks for C++17. MAX_JOBS=48 dies with "Cannot allocate memory"
# on the shared login node; 8 finishes.
sed -i 's/-std=c++17/-std=c++20/g' setup.py
MAX_JOBS=8 pip install . --no-build-isolation

echo "=== 6/6 verify"
cd - >/dev/null
python - <<'PY'
import torch
print("torch", torch.__version__)
assert torch.__version__.startswith("2.14."), torch.__version__
import inspect
from torch.distributed.fsdp import fully_shard
assert "dp_mesh_dims" in inspect.signature(fully_shard).parameters
# The import check INSTALL.md insists on: a stale extension shows up here and
# nowhere in `pip list`.
from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn  # noqa: F401
import causal_conv1d as cc, spmd_types, triton, torchtitan  # noqa: F401
import apache_tvm_ffi, tilelang
print("triton", triton.__version__, "| tvm-ffi", apache_tvm_ffi.__version__,
      "| tilelang", tilelang.__version__)
try:
    import flash_qla  # noqa: F401
    raise SystemExit("flash_qla is installed; it breaks attn_gym.linear at call time")
except ImportError:
    pass
try:
    import flash_attn  # noqa: F401
    print("flash_attn", flash_attn.__version__)
except ImportError:
    print("flash_attn ABSENT -- optional, attention falls back to varlen_attn")
print("imports clean")
PY

# The tvm-ffi break is a CALL-time failure, so an import check cannot see it.
# Needs a GPU, which the login node does not have, so it runs on a compute node.
echo "=== call check on a compute node"
srun --nodes=1 --ntasks=1 --gres=gpu:1 --time=0:05:00 \
    "$NEW_VENV/bin/python" -c "
import torch
from attn_gym.linear import causal_conv1d
x = torch.randn(2, 16, 64, device='cuda', dtype=torch.bfloat16)
w = torch.randn(16, 4, device='cuda', dtype=torch.bfloat16)
print('causal_conv1d call ok', tuple(causal_conv1d(x, w).shape))
" || echo "CALL CHECK FAILED -- do not queue training"

cat <<MSG

Built. Nothing existing was touched; run against it with:

  JUP_TORCH_ENV=$NEW_CONDA JUP_VENV=$NEW_VENV

both of which scripts/multinode_jup.sh already reads.
MSG
