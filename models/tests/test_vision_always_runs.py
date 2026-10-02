"""The vision tower must stay in the autograd graph even with no images.

It is one FSDP unit, and FSDP posts its collectives from autograd hooks on the
parameters. A rank whose micro-batch has no images used to skip the tower, so
those hooks never fired while the other ranks sat in the matching collective --
a deadlock at 100% SM and low power, at whichever step the first image-free
micro-batch landed (hence "always the same step, and it moves with the dataset").

The property that prevents it is narrow and easy to regress: vision parameters
must receive gradients that are **zero**, not `None`. `None` means autograd never
reached them, which is exactly the hang.

Needs a snapshot:
    QWEN3_5_SNAPSHOT=<dir> python -m pytest models/tests/test_vision_always_runs.py
"""

import os

import pytest
import torch

SNAPSHOT = os.environ.get("QWEN3_5_SNAPSHOT")

pytestmark = pytest.mark.skipif(
    not SNAPSHOT or not torch.cuda.is_available(),
    reason="needs QWEN3_5_SNAPSHOT and a GPU",
)

def _text_only_batch(device, n=64):
    tokens = torch.randint(0, 1000, (n,), device=device)
    positions = torch.arange(n, device=device)
    return tokens, positions

def test_vision_params_get_zero_grads_without_images():
    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize

    device = torch.device("cuda")
    model = build_meta(SNAPSHOT, seq_len=512, tp=1, enable_sp=False)
    materialize(model, device)
    load_hf(model, SNAPSHOT)
    assert model.vision_encoder is not None, "snapshot has no vision tower"

    tokens, positions = _text_only_batch(device)
    attention_masks = model.get_attention_masks(positions)
    model._skip_lm_head = True

    out = model(
        tokens,
        attention_masks=attention_masks,
        positions=positions,
        pixel_values=None,
        grid_thw=None,
        special_tokens={"image_id": 151655},
    )
    out.sum().backward()

    vision = [(n, p) for n, p in model.named_parameters() if n.startswith("vision_encoder")]
    assert vision, "no vision parameters found"

    missing = [n for n, p in vision if p.grad is None]
    assert not missing, (
        f"{len(missing)}/{len(vision)} vision params have grad=None -- autograd "
        f"never reached the tower, which is what deadlocks FSDP. e.g. {missing[:3]}"
    )
    nonzero = [n for n, p in vision if p.grad.abs().sum().item() != 0.0]
    assert not nonzero, f"a text-only batch moved vision weights: {nonzero[:3]}"
