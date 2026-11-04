#!/bin/bash
# Wait for the flash_attn build, verify the torch 2.14 env, and only then
# resubmit the three 35B long-document probes against it.
#
#   scripts/jup_resubmit_longdoc.sh            # waits for the build marker
#   NO_WAIT=1 scripts/jup_resubmit_longdoc.sh  # verify and submit now
#
# Jobs 2557458/60/61 died in `Trainer.__init__` under torch 2.13 with
# "all parameters must be DTensors on the full SPMD mesh". These are the same
# three configs, unchanged, on the 2.14 env from scripts/jup_env_214.sh.
#
# The wait is on a marker the build prints, NOT on pgrep: `pgrep -f <pattern>`
# matches the polling command's own command line, so it reports the build alive
# when it is dead and dead when the poller is phrased differently. That cost a
# round already -- the first version of this script fired while the build was
# still compiling and failed on a flash_attn that did not exist yet.
set -euo pipefail

OWNER=/e/project1/open-sci-mm/ockier1
REPO="$OWNER/titan-port"
NEW_CONDA="${JUP_TORCH_ENV:-$OWNER/cache/conda/envs/torch_main_214}"
NEW_VENV="${JUP_VENV:-$OWNER/torchtitan/venv_214}"
FA_LOG="${FA_LOG:-$OWNER/cache/jup_fa_build.log}"
NODES="${NODES:-2}"
MINUTES="${MINUTES:-60}"

if [ -z "${NO_WAIT:-}" ]; then
    echo "waiting for FLASH_ATTN_BUILD_DONE in $FA_LOG"
    for _ in $(seq 1 240); do   # 240 x 30s = 2h ceiling
        grep -q FLASH_ATTN_BUILD_DONE "$FA_LOG" 2>/dev/null && break
        if grep -qE "^ERROR:|error: command .* failed|Cannot allocate memory" "$FA_LOG" 2>/dev/null; then
            echo "flash_attn build failed; submitting nothing:" >&2
            tail -20 "$FA_LOG" >&2
            exit 1
        fi
        sleep 30
    done
    grep -q FLASH_ATTN_BUILD_DONE "$FA_LOG" || { echo "timed out waiting on the build" >&2; exit 1; }
fi

echo "=== verifying $NEW_VENV"
"$NEW_VENV/bin/python" - <<'PY'
import inspect, pathlib, torch
assert torch.__version__.startswith("2.14."), torch.__version__
from torch.distributed.fsdp import fully_shard
assert "dp_mesh_dims" in inspect.signature(fully_shard).parameters
# The 2.13 check that killed the first round. Searched over the tree rather than
# through a private module path, which moves between releases.
root = pathlib.Path(torch.__file__).parent / "distributed"
hits = [f for f in root.rglob("*.py")
        if "all parameters must be DTensors" in f.read_text(errors="ignore")]
assert not hits, f"this torch still rejects plain annotated tensors: {hits}"

from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn  # noqa: F401
import causal_conv1d as cc, flash_attn, spmd_types, triton, torchtitan  # noqa: F401
import tilelang, tvm_ffi  # dist apache-tvm-ffi, module tvm_ffi
# INSTALL.md: flash_qla and attn_gym.linear cannot coexist. flash_qla pins
# apache-tvm-ffi==0.1.9, on which attn_gym.linear.causal_conv1d raises
# `make_kwargs_wrapper() got an unexpected keyword argument` when CALLED -- which
# no import check can see. attn_gym wins, so flash_qla must be absent.
try:
    import flash_qla  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("flash_qla is installed; it breaks attn_gym.linear at call time")
print("torch", torch.__version__, "| triton", triton.__version__,
      "| flash_attn", flash_attn.__version__,
      "| tvm-ffi", tvm_ffi.__version__, "| tilelang", tilelang.__version__)
PY

# The tvm-ffi break is call-time, so it needs a GPU; the login node has none.
echo "=== call check on a compute node"
# --account and --partition are not optional here: the site's job_submit_filter
# rejects an srun without them ("please specify the job's account with -A").
srun --account="${SLURM_ACCOUNT:-open-sci-mm}" --partition="${SLURM_PARTITION:-booster}" \
    --nodes=1 --ntasks=1 --gres=gpu:1 --time=0:05:00 \
    "$NEW_VENV/bin/python" -c "
import torch
from attn_gym.linear import causal_conv1d
# Shapes and keywords as models/qwen3_5/gdn.py:47 calls it: x is [1, T, C],
# weight is [C, W] (the Conv1d weight with its middle dim squeezed out), and
# cu_seqlens carries the per-document resets.
C, T, W = 64, 16, 4
x = torch.randn(1, T, C, device='cuda', dtype=torch.bfloat16)
w = torch.randn(C, W, device='cuda', dtype=torch.bfloat16)
cu = torch.tensor([0, T], dtype=torch.int32, device='cuda')
out = causal_conv1d(x, w, activation='silu', cu_seqlens=cu)
assert tuple(out.shape) == (1, T, C), tuple(out.shape)
print('causal_conv1d call ok', tuple(out.shape))
"

echo "=== submitting"
cd "$REPO"
export JUP_TORCH_ENV="$NEW_CONDA" JUP_VENV="$NEW_VENV"   # --export=ALL carries these
for pair in \
    "qwen3_5_35b_a3b_64k_longdoc.toml q35b_64k_ld" \
    "qwen3_5_35b_a3b_64k_longdoc_reshard.toml q35b_64k_ld_rs" \
    "qwen3_5_35b_a3b_128k_longdoc.toml q35b_128k_ld"; do
    set -- $pair
    bash scripts/jup_seqlen_probe.sh "$1" "$NODES" "$2" "$MINUTES"
done
squeue -u "$USER" -o "%.10i %.18j %.9T %.6D" | head -6
