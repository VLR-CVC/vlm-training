"""Loss parity against HF `Qwen3VLForConditionalGeneration`.

The existing `test_qwen3_vl_*_parity.py` scripts compare logits. This compares
the number training actually optimises, which is the one that has to agree: a
per-logit bf16 discrepancy that shifts an argmax on a tied pair is harmless to
a gradient, and a systematic bias in the mean is not.

Both sides are bf16 -- our varlen kernel (`torch.nn.attention.varlen`) and HF's
flash-attention-2 path both refuse fp32 -- so the floor here is bf16 reduction
noise across the decoder, not exactness.

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> python -m pytest models/tests/test_qwen3_vl_hf_loss_parity.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

from models.qwen3_vl.model import Qwen3VLForCausalLM, set_loss_chunk_mb

SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)

pytestmark = pytest.mark.cuda_only


def _load(dtype, attn):
    import transformers

    device = torch.device("cuda")
    ours, cfg = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=dtype, device=device, load_vision=True
    )
    hf = (
        transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=dtype, attn_implementation=attn
        )
        .to(device)
        .eval()
    )
    return ours.eval(), hf, cfg, device


@pytest.fixture(scope="module")
def models_fp32():
    """fp32 pair. Reachable since `dispatch_varlen_attention` gained an SDPA
    fallback, and the only configuration where a loss difference means
    something other than bf16 reduction order."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not os.path.isdir(SNAPSHOT):
        pytest.skip(f"no snapshot at {SNAPSHOT}")
    pytest.importorskip("transformers")
    ours, hf, cfg, device = _load(torch.float32, "sdpa")
    yield ours, hf, cfg, device
    del ours, hf
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def models():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: both attention paths are bf16-only")
    if not os.path.isdir(SNAPSHOT):
        pytest.skip(f"no snapshot at {SNAPSHOT}")
    transformers = pytest.importorskip("transformers")
    device = torch.device("cuda")

    ours, cfg = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=True
    )
    hf = (
        transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to(device)
        .eval()
    )
    yield ours.eval(), hf, cfg, device
    del ours, hf
    torch.cuda.empty_cache()


def _report(name, ours_loss, hf_loss):
    a, b = float(ours_loss), float(hf_loss)
    print(f"\n{name}: ours={a:.6f}  hf={b:.6f}  abs={abs(a-b):.3e}  "
          f"rel={abs(a-b)/max(abs(b), 1e-9):.3e}")
    return a, b


def _text_batch(device):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids = tok(
        "The capital of France is Paris, and the capital of Germany is Berlin. "
        "Rome is the capital of Italy and Madrid is the capital of Spain.",
        return_tensors="pt",
    ).input_ids.to(device)
    labels = ids.clone()
    labels[:, :4] = -100  # a prompt region, as the task encoder produces
    return ids, labels


def test_text_loss_parity(models):
    ours, hf, _cfg, device = models
    ids, labels = _text_batch(device)
    cu = torch.tensor([0, ids.shape[1]], device=device, dtype=torch.int32)

    with torch.no_grad():
        a = ours(input_ids=ids, attention_mask=cu, labels=labels).loss
        b = hf(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss

    a, b = _report("text bf16", a, b)
    # Loose on purpose: at 25 tokens the mean has too few terms for per-logit
    # bf16 noise to cancel. `test_text_loss_parity_fp32` is the tight one.
    assert abs(a - b) < 0.1


def test_text_loss_parity_fp32(models_fp32):
    """The assertion with no noise budget in it."""
    ours, hf, _cfg, device = models_fp32
    ids, labels = _text_batch(device)
    cu = torch.tensor([0, ids.shape[1]], device=device, dtype=torch.int32)

    with torch.no_grad():
        a = ours(input_ids=ids, attention_mask=cu, labels=labels).loss
        b = hf(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss

    a, b = _report("text fp32", a, b)
    assert abs(a - b) < 1e-4


def test_long_sequence_loss_parity(models):
    """The assertion that matters: at a training-sized row the noise cancels.

    Per-token bf16 noise is unbiased, so the gap falls off like 1/sqrt(N):
    ~1e-2 over a 25-token prompt against 8e-5 here.
    """
    ours, hf, _cfg, device = models
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = " ".join(
        [
            "The capital of France is Paris and the capital of Germany is Berlin.",
            "Mathematics is the language in which the universe is written.",
            "In computer science a linked list is a linear data structure.",
        ]
        * 30
    )
    ids = tok(text, return_tensors="pt").input_ids[:, :4096].to(device)
    cu = torch.tensor([0, ids.shape[1]], device=device, dtype=torch.int32)

    with torch.no_grad():
        a = ours(input_ids=ids, attention_mask=cu, labels=ids).loss
        b = hf(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids).loss

    a, b = _report(f"long ({ids.shape[1]} tokens)", a, b)
    assert abs(a - b) < 1e-3


def test_chunked_loss_matches_whole_row(models):
    """`loss_chunk_mb` must not move the number it is trading memory for."""
    ours, _hf, _cfg, device = models
    ids, labels = _text_batch(device)
    cu = torch.tensor([0, ids.shape[1]], device=device, dtype=torch.int32)

    with torch.no_grad():
        set_loss_chunk_mb(0)
        whole = float(ours(input_ids=ids, attention_mask=cu, labels=labels).loss)
        try:
            set_loss_chunk_mb(1)  # forces several chunks at this vocab size
            chunked = float(ours(input_ids=ids, attention_mask=cu, labels=labels).loss)
        finally:
            set_loss_chunk_mb(0)

    print(f"\nchunked: whole={whole:.6f}  chunked={chunked:.6f}  "
          f"abs={abs(whole-chunked):.3e}")
    assert abs(whole - chunked) < 1e-5


def _image_batch(cfg, device):
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
    merge = cfg.vision.spatial_merge_size
    n_vis = int((grid[:, 0] * (grid[:, 1] // merge) * (grid[:, 2] // merge)).sum())
    patch_dim = (
        cfg.vision.in_channels * cfg.vision.temporal_patch_size * cfg.vision.patch_size ** 2
    )
    n_patch = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    torch.manual_seed(0)
    pixels = torch.randn(n_patch, patch_dim, device=device, dtype=torch.bfloat16)

    ids = torch.cat([
        torch.tensor([10, 11], device=device),
        torch.tensor([cfg.vision_start_token_id], device=device),
        torch.full((n_vis,), cfg.image_token_id, device=device),
        torch.tensor([cfg.vision_end_token_id], device=device),
        torch.tensor([20, 21, 22], device=device),
    ]).unsqueeze(0)
    labels = ids.clone()
    labels[ids == cfg.image_token_id] = -100
    return ids, labels, pixels, grid


def test_multimodal_loss_parity(models):
    ours, hf, cfg, device = models
    ids, labels, pixels, grid = _image_batch(cfg, device)
    mm_type = torch.zeros_like(ids, dtype=torch.int32)
    mm_type[ids == cfg.image_token_id] = 1
    cu = torch.tensor([0, ids.shape[1]], device=device, dtype=torch.int32)

    with torch.no_grad():
        a = ours(input_ids=ids, attention_mask=cu, labels=labels,
                 pixel_values=pixels, image_grid_thw=grid).loss
        b = hf(input_ids=ids, labels=labels, pixel_values=pixels,
               image_grid_thw=grid, mm_token_type_ids=mm_type).loss

    a, b = _report("multimodal", a, b)
    # The image is random noise, so the loss sits up around 11 and the absolute
    # gap scales with it; ~0.1% relative, in line with the text case.
    assert abs(a - b) < 5e-2


def test_packed_loss_matches_unpacked(models):
    """Two samples in one packed row vs the same two run separately.

    This is our own varlen contract rather than HF's, but it is the piece the
    trainer actually exercises, and a packing bug shows up as a loss offset
    exactly where an HF comparison would not look.
    """
    ours, _hf, _cfg, device = models
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    # Long enough that bf16 scatter is well under the tolerance; at 7 and 13
    # tokens the noise alone is ~3e-2 and the test would say nothing.
    a_ids = tok(
        "The capital of France is Paris. " * 20, return_tensors="pt"
    ).input_ids.to(device)
    b_ids = tok(
        "Mathematics is the language of the universe and of physics. " * 20,
        return_tensors="pt",
    ).input_ids.to(device)
    la, lb = a_ids.shape[1], b_ids.shape[1]

    packed = torch.cat([a_ids, b_ids], dim=1)
    cu = torch.tensor([0, la, la + lb], device=device, dtype=torch.int32)
    labels = packed.clone()
    # `causal_lm_loss` shifts once over the whole row, so position la-1 would
    # otherwise be scored against b's first token. Masking that target leaves
    # the packed run scoring exactly the tokens the two separate runs do.
    labels[0, la] = -100

    with torch.no_grad():
        together = float(ours(input_ids=packed, attention_mask=cu, labels=labels).loss)
        alone_a = float(ours(input_ids=a_ids,
                             attention_mask=torch.tensor([0, la], device=device,
                                                         dtype=torch.int32),
                             labels=a_ids).loss)
        alone_b = float(ours(input_ids=b_ids,
                             attention_mask=torch.tensor([0, lb], device=device,
                                                         dtype=torch.int32),
                             labels=b_ids).loss)

    blend = (alone_a * (la - 1) + alone_b * (lb - 1)) / (la + lb - 2)
    print(f"\npacked={together:.6f}  separate(length-weighted)={blend:.6f}  "
          f"abs={abs(together - blend):.3e}")
    assert abs(together - blend) < 2e-2
