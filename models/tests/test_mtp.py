"""Multi-Token Prediction (MTP) tests for Qwen3-VL and Qwen3.5.

Covers:
  * boundary masking of MTP targets over a packed varlen row (CPU, no kernels);
  * the MTP module is absent and the loss path is unchanged when disabled;
  * a depth-1 MTP forward/backward produces finite losses and grads (CUDA);
  * the native Qwen3.5 loader maps the checkpoint's ``mtp.*`` weights exactly.

Run:
    pytest models/tests/test_mtp.py
    python models/tests/test_mtp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_vl.model import (
    Qwen3VLConfig,
    Qwen3VLForCausalLM,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    build_mtp_targets,
)

QWEN35_SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
)


def _vl_cfg(mtp_layers: int) -> Qwen3VLConfig:
    text = Qwen3VLTextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0,
        tie_word_embeddings=True, mtp_num_hidden_layers=mtp_layers,
    )
    vision = Qwen3VLVisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=64, out_hidden_size=64, hidden_act="gelu",
        deepstack_visual_indexes=[],
    )
    return Qwen3VLConfig(
        text=text, vision=vision, image_token_id=200, video_token_id=201,
        vision_start_token_id=202, vision_end_token_id=203, tie_word_embeddings=True,
    )


def test_build_mtp_targets_no_cross_segment_leak():
    # Two packed samples: [0,6) and [6,10).
    cu = torch.tensor([0, 6, 10], dtype=torch.int32)
    labels = torch.arange(100, 110).view(1, 10)
    tgt = build_mtp_targets(labels, cu, offset=2)
    # offset=2: position i predicts token i+2; last two positions of each
    # segment have no in-segment target and must be masked.
    expected = [102, 103, 104, 105, -100, -100, 108, 109, -100, -100]
    assert tgt[0].tolist() == expected


def test_build_mtp_targets_inherits_ignore_mask():
    cu = torch.tensor([0, 8], dtype=torch.int32)
    labels = torch.tensor([[10, -100, 12, 13, -100, 15, 16, 17]])
    tgt = build_mtp_targets(labels, cu, offset=2)
    # tgt[i] = labels[i+2]; last two positions masked by the boundary rule.
    assert tgt[0].tolist() == [12, 13, -100, 15, 16, 17, -100, -100]


def test_mtp_disabled_is_noop():
    model = Qwen3VLForCausalLM(_vl_cfg(0))
    assert model.mtp is None
    assert not any(n.startswith("mtp.") for n, _ in model.named_parameters())


@pytest.mark.cuda_only
def test_mtp_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA (varlen flash kernel)")
    dev = "cuda"
    ids = torch.randint(0, 200, (1, 10), device=dev)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=dev)

    base = Qwen3VLForCausalLM(_vl_cfg(0)).to(dev).to(torch.bfloat16)
    out0 = base(input_ids=ids, attention_mask=cu, labels=ids.clone())
    assert out0.mtp_loss is None

    model = Qwen3VLForCausalLM(_vl_cfg(1)).to(dev).to(torch.bfloat16)
    model.mtp.init_weights()
    out = model(input_ids=ids, attention_mask=cu, labels=ids.clone())
    assert out.mtp_loss is not None and torch.isfinite(out.mtp_loss)
    out.loss.backward()
    g = model.mtp.fc.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0

    # MTP param layout matches Qwen3.5/Qwen3-Next checkpoints.
    names = {n for n, _ in model.named_parameters() if n.startswith("mtp.")}
    for required in (
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    ):
        assert required in names, required


@pytest.mark.cuda_only
def test_qwen35_loads_checkpoint_mtp_weights():
    import json

    from safetensors import safe_open

    if not os.path.isdir(QWEN35_SNAPSHOT):
        pytest.skip(f"no Qwen3.5 snapshot at {QWEN35_SNAPSHOT}")
    from models.qwen3_5.model import Qwen3_5ForCausalLM

    model, cfg = Qwen3_5ForCausalLM.from_pretrained(
        QWEN35_SNAPSHOT, dtype=torch.bfloat16, device="cpu", load_vision=True
    )
    assert model.mtp is not None and cfg.text.mtp_num_hidden_layers > 0

    sd = dict(model.state_dict())
    wm = json.load(
        open(os.path.join(QWEN35_SNAPSHOT, "model.safetensors.index.json"))
    )["weight_map"]
    mtp_keys = [k for k in wm if k.startswith("mtp.")]
    assert len(mtp_keys) > 0
    for k in mtp_keys:
        with safe_open(os.path.join(QWEN35_SNAPSHOT, wm[k]), framework="pt") as f:
            ref = f.get_tensor(k).to(torch.bfloat16)
        assert torch.equal(sd[k], ref), k


@pytest.mark.cuda_only
def test_qwen35_mtp_head_drafts_usefully():
    """Teacher-forced guard on the trained head: it must predict token i+2 well
    above chance. Catches mis-invocation (e.g. wrong mtp.fc concat order, which
    previously dropped this from ~0.55 to ~0.05 yet passed every other test)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not os.path.isdir(QWEN35_SNAPSHOT):
        pytest.skip(f"no Qwen3.5 snapshot at {QWEN35_SNAPSHOT}")
    from transformers import AutoTokenizer

    from models.qwen3_5.model import Qwen3_5ForCausalLM

    model, _ = Qwen3_5ForCausalLM.from_pretrained(
        QWEN35_SNAPSHOT, dtype=torch.bfloat16, device="cuda", load_vision=False
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(QWEN35_SNAPSHOT, use_fast=False)
    ids = tok(
        "The history of the Roman Empire is a long and complex story that spans "
        "many centuries, from the founding of the city to the fall of the west.",
        return_tensors="pt",
    ).input_ids.to("cuda")
    L = ids.shape[1]
    dtype = model.lm_head.weight.dtype
    lm = model.model.language_model
    pos = torch.arange(L, device="cuda").view(1, 1, -1).expand(3, 1, -1)
    cos, sin = model._compute_cos_sin(pos)
    cos, sin = cos.to(dtype), sin.to(dtype)
    cu = torch.tensor([0, L], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        h = lm(lm.embed_tokens(ids), cos, sin, cu, L)
        next_embeds = lm.embed_tokens(torch.roll(ids, -1, dims=1))
        mtp_h = model.mtp(h, next_embeds, cos, sin, cu, L)
        pred = model.lm_head(mtp_h)[0].argmax(-1)
        tgt = torch.roll(ids, -2, dims=1)[0]
        acc = (pred[: L - 2] == tgt[: L - 2]).float().mean().item()
    assert acc > 0.3, f"MTP target+2 top1={acc:.3f} too low — head mis-invoked"


if __name__ == "__main__":
    test_build_mtp_targets_no_cross_segment_leak()
    test_build_mtp_targets_inherits_ignore_mask()
    test_mtp_disabled_is_noop()
    print("CPU tests passed")
    if torch.cuda.is_available():
        test_mtp_forward_backward()
        test_qwen35_loads_checkpoint_mtp_weights()
        test_qwen35_mtp_head_drafts_usefully()
        print("CUDA tests passed")
    else:
        print("CUDA unavailable; skipped GPU tests")
