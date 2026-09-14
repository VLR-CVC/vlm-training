"""Pre-download the Qwen4-Exp (Qwen3.8-Flash-Next) snapshot to shared storage.

Run on a login node. ~335 GiB across 144 files.
"""
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.8-Flash-Next",
    local_dir="/data/151-2/users/tockier/models/qwen4",
    max_workers=8,
)
