"""Forward-pass parity test: our Qwen3 impl vs HF Qwen3ForCausalLM.

Run:
    pytest models/test_qwen3_parity.py -s
    # or
    python models/test_qwen3_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

try:
    import pytest
except ModuleNotFoundError:  # allow running as a plain script without pytest
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.qwen3.model_qwen3 import Qwen3ForCausalLM

SNAPSHOT = os.environ.get(
    "QWEN3_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_1_7b",
)


if pytest is not None:

    @pytest.fixture(scope="module")
    def device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="module")
    def input_ids(device: torch.device) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randint(0, 100_000, (1, 32), device=device)

    @pytest.fixture(scope="module")
    def our_model(device: torch.device) -> Qwen3ForCausalLM:
        model = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32, device=device)
        model.eval()
        return model

    @pytest.fixture(scope="module")
    def hf_model(device: torch.device):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)
        model.eval()
        return model


@torch.no_grad()
def test_logits_match(our_model, hf_model, input_ids):
    ours = our_model(input_ids)
    theirs = hf_model(input_ids).logits

    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)

    max_abs = (ours - theirs).abs().max().item()
    mean_abs = (ours - theirs).abs().mean().item()
    print(f"\nmax_abs_diff={max_abs:.3e}  mean_abs_diff={mean_abs:.3e}")

    # argmax should match for every position (greedy parity).
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1)), "argmax mismatch"

    # Tight numeric tolerance in float32.
    torch.testing.assert_close(ours, theirs, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ids = torch.randint(0, 100_000, (1, 32), device=dev)

    print(f"Loading our Qwen3 from {SNAPSHOT} ...")
    ours = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32, device=dev).eval()

    print("Loading HF Qwen3 ...")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(dev).eval()

    with torch.no_grad():
        a = ours(ids)
        b = hf(ids).logits

    print(f"ours shape={tuple(a.shape)}  hf shape={tuple(b.shape)}")
    print(f"max abs diff  = {(a - b).abs().max().item():.3e}")
    print(f"mean abs diff = {(a - b).abs().mean().item():.3e}")
    print(f"argmax equal  = {torch.equal(a.argmax(-1), b.argmax(-1))}")
