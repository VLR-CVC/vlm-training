#!/bin/bash
# Build and pack the MN5 environment.
#
# MN5 has no internet, so the env is built here and shipped as a conda-pack
# tarball. The source is the local x86 env that INSTALL.md calls the byte-checked
# twin of JUPITER's `torch_main` (torch 2.14.0+cu130, python 3.13, attn_gym,
# spmd_types, causal_conv1d built with sm_90 device code -- H100 is sm_90, so the
# extension carries over without a rebuild).
#
# The source env is NOT mutated: it is the reference for JUPITER. Everything
# MN5-specific happens in the clone.
#
# Usage:
#   ./scripts/pack_env_mn5.sh              # clone, patch, verify, pack
#   SKIP_CLONE=1 ./scripts/pack_env_mn5.sh # re-pack an existing clone
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/data/151-1/users/tockier/conda_install}"
SRC_ENV="${SRC_ENV:-$CONDA_ROOT/envs/torch11_jup}"
DST_ENV="${DST_ENV:-/data/151-2/users/tockier/envs/qwen4_mn5}"
OUT_DIR="${OUT_DIR:-/data/151-2/users/tockier/envs}"
OUT_TAR="$OUT_DIR/$(basename "$DST_ENV").tar.gz"

# conda-pack pads the new prefix with nulls inside binaries, so the deployment
# path must be no longer than the build path. Checked below, not assumed.
MN5_PREFIX="${MN5_PREFIX:-/gpfs/projects/ehpc543/envs/$(basename "$DST_ENV")}"

if [ ${#MN5_PREFIX} -gt ${#DST_ENV} ]; then
    echo "MN5 prefix ($(echo -n "$MN5_PREFIX" | wc -c) chars) is longer than the" >&2
    echo "build prefix ($(echo -n "$DST_ENV" | wc -c) chars); conda-pack cannot" >&2
    echo "rewrite binaries into it. Shorten MN5_PREFIX or lengthen DST_ENV." >&2
    exit 1
fi

# 1. clone ---------------------------------------------------------------
if [ -z "${SKIP_CLONE:-}" ]; then
    [ -e "$DST_ENV" ] && { echo "$DST_ENV exists; SKIP_CLONE=1 to reuse it" >&2; exit 1; }
    echo ">> cloning $SRC_ENV -> $DST_ENV"
    "$CONDA_ROOT/bin/conda" create --yes --clone "$SRC_ENV" --prefix "$DST_ENV"
fi

PY="$DST_ENV/bin/python"

# 2. MN5-specific changes ------------------------------------------------
# wandb: 0.29.0 has compatibility problems; take the current release.
echo ">> upgrading wandb"
"$PY" -m pip install --upgrade wandb

# flash_qla cannot coexist with attn_gym.linear, so it goes.
#
#   attn_gym[linear] requires apache-tvm-ffi >= 0.1.12
#   flash_qla 0.1.2   requires apache-tvm-ffi == 0.1.9 and tilelang == 0.1.9
#
# There is no version that satisfies both. On 0.1.9 the CuTeDSL short-conv path
# dies at call time with
#   TypeError: make_kwargs_wrapper() got an unexpected keyword argument
#              'map_dataclass_to_tuple'
# and models/qwen3_5/gdn.py imports those kernels unguarded, so that is the
# trainer, not a slow path. flash_qla is only worth 20% fwd / 30% fwd+bwd, and
# FLA falls back to its Triton kernels without it.
#
# tilelang 0.1.14 wants apache-tvm-ffi <0.1.13,>=0.1.11, so 0.1.12 is the single
# version that keeps tilelang and attn_gym both importable.
#
# requirements.txt still locks the flash_qla trio (apache-tvm-ffi==0.1.9,
# tilelang==0.1.9, flash_qla==0.1.2). That lock predates attn_gym.linear and is
# wrong for this repo -- see INSTALL.md.
echo ">> resolving the attn_gym / flash_qla conflict"
"$PY" -m pip uninstall -y flash_qla
"$PY" -m pip install "apache-tvm-ffi==0.1.12" "tilelang==0.1.14"

# nvcc: tilelang resolves its compile target with `nvcc.find_cuda_path()`, and
# FLA dispatches the gated delta rule to flash_qla the moment it imports -- so a
# missing nvcc is not a slow path, it takes the forward down (INSTALL.md, "Why
# the job scripts load CUDA"). JUPITER solves this with `module load CUDA/13`.
# MN5's modules are cuda/12.x, so ship nvcc inside the env instead of depending
# on a module that may not match.
echo ">> installing nvcc into the env"
"$PY" -m pip install "cuda-toolkit[nvcc]==13.0.3.0"

# 3. verify --------------------------------------------------------------
echo ">> verifying"
CU13="$("$PY" -c "import nvidia.cu13 as m; print(m.__path__[0])")"
"$CU13/bin/nvcc" --version | tail -2
CUDA_HOME="$CU13" PATH="$CU13/bin:$PATH" "$PY" - <<'EOF'
import shutil
import torch, wandb, tilelang, fla  # noqa: F401
from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn  # noqa: F401
import spmd_types  # noqa: F401

# H100 is sm_90. The prebuilt torch wheel and the locally built causal_conv1d
# both have to carry device code for it -- neither is rebuilt on MN5.
assert "sm_90" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
assert shutil.which("nvcc"), "nvcc not on PATH; tilelang will fail at run time"
print(f"torch {torch.__version__}  wandb {wandb.__version__}  tilelang {tilelang.__version__}")
EOF

# 4. pack ----------------------------------------------------------------
echo ">> packing -> $OUT_TAR"
rm -f "$OUT_TAR"
"$CONDA_ROOT/bin/conda-pack" --prefix "$DST_ENV" --output "$OUT_TAR" \
    --ignore-missing-files --n-threads -1

ls -lh "$OUT_TAR"
cat <<EOF

Next, on MN5 (no internet -- the tarball is the whole env):

  mkdir -p $MN5_PREFIX
  tar -xzf $(basename "$OUT_TAR") -C $MN5_PREFIX
  source $MN5_PREFIX/bin/activate
  conda-unpack            # rewrites the remaining absolute paths

EOF
