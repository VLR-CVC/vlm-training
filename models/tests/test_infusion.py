"""In-fusion vision capacity: zero at birth, routed by modality, sized to budget.

The recipe is from *Unbundling Multimodal Vision Encoding* (arXiv 2611.17334):
vision-only modules added to the backbone's first layers, their outputs added to
the shared ones (Eqs. 2 and 3), zero-initialised so training starts from the
shared model.

Two invariants carry the whole design and are what these tests pin:

1. **Zero at birth.** A freshly built in-fusion model is numerically identical to
   the same model without it. If that ever fails, every run that resumes from text
   weights starts from a different model than it believes.
2. **Routed by modality.** Once the deltas are non-zero they must move vision
   tokens and leave text tokens exactly alone. A mask that leaks is invisible in
   the loss curve and shows up only as text regression weeks later.
"""

import json

import pytest
import torch

from models.qwen3_5.configs import InfusionConfig, qwen35_config_from_hf

BASE = "configs/models/qwen3_5_2b.json"

# Every attention path in this model is a CUDA kernel: flash varlen for full
# attention and a CuTeDSL short conv inside GatedDeltaNet, which rejects CPU
# tensors outright. So the structural tests run anywhere and the numeric ones need
# a GPU. The numeric ones also use an all-full-attention schedule, because a GDN
# layer has no Q/K/V/O to twin and contributes nothing to what they check; the
# real hybrid schedule is what the budget test builds.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
N_LAYERS = 8
INFUSION_LAYERS = (0, 1, 2, 3)
FFN_WIDTH = 128


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> str:
    """A small dense Qwen3.5 that still has the real head and RoPE geometry.

    Only width, depth and vocabulary shrink: `head_dim`, the head counts and
    `mrope_section` are validated against each other inside the config builder, so
    they are left exactly as shipped.
    """
    raw = json.loads(open(BASE).read())
    tc = raw["text_config"]
    tc["num_hidden_layers"] = N_LAYERS
    tc["hidden_size"] = 128
    tc["intermediate_size"] = 256
    tc["vocab_size"] = 64
    tc["layer_types"] = [
        "full_attention" if (i + 1) % tc["full_attention_interval"] == 0
        else "linear_attention"
        for i in range(N_LAYERS)
    ]
    path = tmp_path_factory.mktemp("infusion") / "config.json"
    path.write_text(json.dumps(raw))
    return str(path)


def build(path: str, infusion: InfusionConfig | None, *, device: str = "cpu"):
    cfg = qwen35_config_from_hf(
        path,
        seq_len=256,
        with_vision=False,
        infusion=infusion,
        layer_types=["full_attention"] * N_LAYERS if device != "cpu" else None,
    )
    torch.manual_seed(0)
    model = cfg.build()
    model.init_states(buffer_device=torch.device("cpu"))
    return model.to(device).eval()


def infusion_config() -> InfusionConfig:
    return InfusionConfig(layers=INFUSION_LAYERS, ffn_hidden_dim=FFN_WIDTH)


IMAGE_ID, VIDEO_ID = 7, 8
SPECIAL = {"image_id": IMAGE_ID, "video_id": VIDEO_ID}


def batch(n: int = 32, device: str = "cuda"):
    """Tokens with a contiguous run of image ids, plus the inputs a forward needs."""
    tokens = torch.arange(n, device=device) % 5 + 10  # never an image or video id
    tokens[8:20] = IMAGE_ID
    positions = torch.arange(n, device=device)
    return tokens, positions


def run(model, tokens, positions, special=SPECIAL):
    masks = model.get_attention_masks(positions)
    with torch.no_grad():
        return model(tokens, attention_masks=masks, positions=positions,
                     special_tokens=special)


def test_the_budget_matches_the_formula(tiny):
    """`P_FFN = L_a(3DF + D)` and `P_QKVO = 2 L_a D d_h (H_Q + H_KV)`, Appendix E.1.

    The Q term is doubled against the paper because Qwen3.5 fuses the attention
    output gate into `wq`, so its twin carries a gate delta too.
    """
    model = build(tiny, infusion_config())
    tc = json.loads(open(tiny).read())["text_config"]
    D, dh = tc["hidden_size"], tc["head_dim"]
    HQ, HKV = tc["num_attention_heads"], tc["num_key_value_heads"]
    full = [i for i in INFUSION_LAYERS if (i + 1) % tc["full_attention_interval"] == 0]

    expected = len(INFUSION_LAYERS) * 3 * D * FFN_WIDTH
    expected += len(full) * (D * 2 * HQ * dh + 2 * D * HKV * dh + HQ * dh * D)
    actual = sum(p.numel() for n, p in model.named_parameters() if "vision_" in n)
    assert actual == expected

    assert sum(1 for l in model.layers.values() if l.vision_feed_forward is not None) \
        == len(INFUSION_LAYERS)
    assert sum(1 for l in model.layers.values() if l.full_attn and l.attn.infusion) \
        == len(full)


@needs_cuda
def test_a_fresh_model_computes_as_if_it_had_no_capacity(tiny):
    """Invariant 1, as one model against itself with the capacity switched off.

    Not two separately built models: the vision branches draw from the same
    generator during `init_states`, so building them shifts every later layer's
    weights and a same-seed comparison would compare two different backbones.
    With the deltas at zero, routing the mask must change nothing at all.
    """
    model = build(tiny, infusion_config(), device="cuda")
    tokens, positions = batch()
    torch.testing.assert_close(
        run(model, tokens, positions),
        run(model, tokens, positions, special=None),
        rtol=0, atol=0,
    )


def test_the_branch_is_not_born_dead(tiny):
    """Only `w2` is zeroed.

    Zeroing the gate/up pair as well would make the branch unrecoverable: with
    `up = 0` the gradient of `w2` is zero and with `w2 = 0` the gradient of `w13`
    is zero, so it would stay a no-op for the whole run.
    """
    model = build(tiny, infusion_config())
    branch = model.layers["0"].vision_feed_forward
    assert torch.count_nonzero(branch.w2.weight) == 0
    assert torch.count_nonzero(branch.w13.weight) > 0

    x = torch.randn(4, branch.w13.weight.shape[1])
    branch.w2.weight.requires_grad_(True)
    branch(x).sum().backward()
    assert torch.count_nonzero(branch.w2.weight.grad) > 0, "no gradient reaches w2"


@needs_cuda
@pytest.mark.parametrize("deltas", ["ffn", "attention"])
def test_capacity_moves_vision_tokens_and_only_vision_tokens(tiny, deltas):
    """Invariant 2, for each of Eq. 3 and Eq. 2 on its own."""
    tokens, positions = batch()
    model = build(tiny, infusion_config(), device="cuda")
    before = run(model, tokens, positions)

    with torch.no_grad():
        for layer in model.layers.values():
            if deltas == "ffn" and layer.vision_feed_forward is not None:
                layer.vision_feed_forward.w2.weight.normal_(std=0.5)
            if deltas == "attention" and layer.full_attn and layer.attn.infusion:
                for name in ("vision_wq", "vision_wk", "vision_wv", "vision_wo"):
                    getattr(layer.attn, name).weight.normal_(std=0.5)
    after = run(model, tokens, positions)

    vision = tokens == IMAGE_ID
    assert not torch.allclose(after[vision], before[vision]), "vision tokens unmoved"

    # Text tokens *before* the first image cannot see it under a causal mask, so
    # they are the clean probe: anything downstream may legitimately change
    # through attention to a moved vision token.
    head = slice(0, int(vision.nonzero()[0]))
    torch.testing.assert_close(after[head], before[head], rtol=0, atol=0)


@needs_cuda
def test_without_special_tokens_the_branches_are_skipped(tiny):
    """A text-only batch carries no `special_tokens`, so the mask is None and the
    branches are skipped entirely rather than multiplied by a zero mask."""
    tokens, positions = batch()
    model = build(tiny, infusion_config(), device="cuda")
    before = run(model, tokens, positions, special=None)
    with torch.no_grad():
        for layer in model.layers.values():
            if layer.vision_feed_forward is not None:
                layer.vision_feed_forward.w2.weight.normal_(std=0.5)
    assert model._vision_mask(tokens, None) is None
    torch.testing.assert_close(
        run(model, tokens, positions, special=None), before, rtol=0, atol=0
    )


def test_layers_outside_the_backbone_are_rejected(tiny):
    with pytest.raises(ValueError, match="outside the backbone"):
        qwen35_config_from_hf(
            tiny, seq_len=256, with_vision=False,
            infusion=InfusionConfig(layers=(0, N_LAYERS), ffn_hidden_dim=FFN_WIDTH),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
