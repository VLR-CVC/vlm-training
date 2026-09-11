# Installation

`requirements.txt` is the lock: a `pip freeze` of an environment that passes
`models/tests/test_dependencies.py`, taken on JUPITER and byte-checked against
the x86 workstation env (130 of 131 packages match exactly). `requirements.in`
lists only the direct dependencies and the reason each pin exists; it is
documentation, not the install path.

Two packages are deliberately absent from the lock. `causal_conv1d` and
`flash_attn` are CUDA extensions linked against the installed libtorch, so a
pinned wheel is wrong the moment torch moves — see
[CUDA extensions](#cuda-extensions-built-from-source).

Verify any install with:

```bash
pytest models/tests/test_dependencies.py -v -rs
```

That suite asserts on imported symbols and running kernels, never on version
strings, because both failure modes this repo has hit leave `pip list` looking
correct: a stale extension ABI, and `compile_ops.py` swallowing an ImportError
and silently falling back to a much slower path.

---

## JUPITER (GH200, aarch64/sbsa, sm_90)

### 0. Caches and quota

`$HOME` on JUPITER has a small quota that pip, conda, triton and tilelang will
blow through on their own. Point every cache at project storage **before**
installing anything — the reference `.bashrc` already does this via
`CACHE_ROOT`:

```bash
export CACHE_ROOT="/e/project1/<project>/$USER/cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export CONDA_ENVS_DIRS="$CACHE_ROOT/conda/envs"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton/cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch/inductor"
export TILELANG_CACHE_DIR="$CACHE_ROOT/tilelang"
export TILELANG_TMP_DIR="$CACHE_ROOT/tilelang/tmp"
export HF_HOME="$CACHE_ROOT/huggingface"
```

`TILELANG_TMP_DIR` must sit on the **same filesystem** as `TILELANG_CACHE_DIR`.
tilelang compiles into the tmp dir and then hardlinks the result into the cache;
across filesystems that fails with `OSError: [Errno 18] Invalid cross-device
link` and no kernel is ever cached.

### 1. Toolchain

```bash
module load CUDA/13     # nvcc 13.0.48, GCCcore 14.3.0, Stages/2026
```

Needed at **build time** for the CUDA extensions, and at **run time** for
tilelang — see [Why the job scripts load CUDA](#why-the-job-scripts-load-cuda).

### 2. Environment

```bash
conda create -n torch11 python=3.13.15
conda activate torch11
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`torch`, `torchvision` and `triton` come from the cu130 index (there is an sbsa
`manylinux_2_28_aarch64` wheel for each); everything else resolves from PyPI.

### 3. CUDA extensions, built from source

Both are optional — the repo runs without them — but both are worth having on a
GH200. Build them **after** torch is installed and never reuse a wheel across a
torch upgrade.

**Why they cannot be pinned.** torch 2.14 added a sixth parameter to
`c10::cuda::c10_cuda_check_implementation`, which is what the `C10_CUDA_CHECK`
macro expands to. Every extension compiled against an earlier libtorch now fails
at import with `undefined symbol: _ZN3c104cuda19c10_cuda_check_implementationE...`
while still showing up in `pip list`. `test_optional_dependency` fails rather
than skips on that message, precisely so an upgrade cannot leave a half-broken
env behind.

#### causal_conv1d

Used by the Qwen3.5/Qwen4 short-conv path. Without it `causal_conv1d_torch`
stands in, at a real cost.

```bash
cd /e/project1/<project>/$USER/build
git clone https://github.com/Dao-AILab/causal-conv1d && cd causal-conv1d
git checkout v1.7.0
MAX_JOBS=8 pip install . --no-build-isolation
```

#### flash_attn

Qwen3-VL and the Qwen3.5 vision tower dispatch to it when it imports, otherwise
they fall back to `torch.nn.attention.varlen.varlen_attn`. On a GH200 at 9B
shapes the kernels are worth 7-8% forward / 4-6% forward+backward.

Two things have to be worked around:

```bash
cd /e/project1/<project>/$USER/build
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
git checkout v2.8.3.post1

# torch 2.14's headers need C++20 (`std::strong_ordering` in
# c10/util/intrusive_ptr.h); setup.py still asks for C++17 in five places.
sed -i 's/-std=c++17/-std=c++20/g' setup.py

# the login node is shared: 72 cores, 572 GB RAM. MAX_JOBS=48 dies with
# "Cannot allocate memory" partway through. 8 finishes.
MAX_JOBS=8 pip install . --no-build-isolation
```

This produces sm_90 kernels only.

### 4. flash_qla

Installs from PyPI as a pure-Python wheel (`py3-none-any`) — the kernels are
JIT-compiled by tilelang on first use, so there is nothing to build. It is
already in `requirements.txt`.

It needs no repo code to take effect: FLA 0.5.2 ships a priority-based backend
dispatcher that hands `chunk_gated_delta_rule` to `flash_qla` whenever the module
imports. So `qwen3_5::gated_delta_rule`, which calls FLA's top-level function,
picks it up automatically. Measured on a GH200 at Qwen3.5-9B per-rank shapes
(T=10240, H=8, K=V=128, tp=4): **-20% forward, -30% forward+backward**, agreeing
with the native Triton kernels to 7.3e-4 max / 1.0e-5 mean.

### Why the job scripts load CUDA

That automatic dispatch has a sharp edge. tilelang locates the compile target
through `nvcc.find_cuda_path()`, so with no `nvcc` on `PATH` it raises

```
ValueError: No CUDA or HIP or MPS available on this system.
```

— even on a warm kernel cache, and even though the process has working CUDA
devices. Because FLA has already dispatched to flash_qla by then, this does not
degrade to the Triton path: it takes the whole gated-delta-rule forward down
with it, so a Qwen3.5 job that ran fine before flash_qla was installed will now
crash. `scripts/jup_finetune.sh` and `scripts/multinode_jup.sh` therefore
`module load CUDA/13` after activating the env.

The escape hatch, if you would rather not depend on the module:

```bash
export FLA_DISABLE_BACKEND_DISPATCH=1   # force FLA's native Triton kernels
```

### First run

The tilelang JIT compiles on first use and caches; expect the first training
step after a fresh env to take an extra minute or two. Subsequent runs hit the
cache at `$TILELANG_CACHE_DIR`.