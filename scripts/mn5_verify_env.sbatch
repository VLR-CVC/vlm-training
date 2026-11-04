#!/bin/bash
# Verify the shipped MN5 environment on a real H100. Submit from the repo root:
#   sbatch scripts/mn5_verify_env.sbatch
#
# Everything here fails loudly. The point is to find a broken env now rather
# than three hours into an allocation.
#SBATCH -D .
#SBATCH --job-name=mn5_verify_env
#SBATCH --account=ehpc543
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# `-D .` above means cwd is wherever this was submitted from, i.e. the repo root.
source scripts/mn5_env.sh

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

python - <<'EOF'
import torch

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device {name}  capability {cap}  torch {torch.__version__}")

# The env was built on a Blackwell workstation and never ran on sm_90 before
# this job. Nothing here is rebuilt on MN5, so the prebuilt device code has to
# cover the H100.
assert cap == (9, 0), f"expected sm_90, got {cap}"
assert "sm_90" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()

# 1. causal_conv1d -- a locally built CUDA extension. If its fatbin lacks sm_90
#    this raises "no kernel image is available for execution on the device".
from attn_gym.linear import causal_conv1d
B, T, C, W = 1, 256, 128, 4  # x is [B, T, C], weight is [C, W]
x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)
w = torch.randn(C, W, device="cuda", dtype=torch.bfloat16)
# packed varlen, the shape the trainer actually runs: two documents in one row
cu = torch.tensor([0, 100, T], device="cuda", dtype=torch.int32)
out = causal_conv1d(x, w, activation="silu", cu_seqlens=cu)
assert out.shape == x.shape and out.isfinite().all()
print("causal_conv1d OK")

# 2. the gated delta rule. flash_qla is deliberately NOT installed -- it pins
#    apache-tvm-ffi==0.1.9, which breaks attn_gym.linear above (see
#    scripts/pack_env_mn5.sh). This is FLA's Triton path.
import importlib.util
assert importlib.util.find_spec("flash_qla") is None, (
    "flash_qla is installed; it pins apache-tvm-ffi==0.1.9 and breaks attn_gym.linear"
)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

B, T, H, K, V = 1, 256, 4, 64, 64
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)

o, _ = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
assert o.isfinite().all(), "gated delta rule produced non-finite output"
print("chunk_gated_delta_rule OK (FLA Triton path)")
EOF

echo "=== repo test suite ==="
# `python -m pytest`, not `pytest`: only the module form puts the cwd on
# sys.path, and the tests import `train.*`. Bare `pytest` collects 3 errors with
# "ModuleNotFoundError: No module named 'train'".
python -m pytest models/tests -q

echo "=== MN5 ENV VERIFY OK ==="
