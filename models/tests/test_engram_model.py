"""Engram inside Qwen3.5, on the real Qwen3.5-0.8B snapshot (CUDA).

    CUDA_VISIBLE_DEVICES=0 pytest models/tests/test_engram_model.py -q

`test_engram_hash.py` pins the hash and the module alone (CPU). This file pins
what only shows up with the decoder around it:

  * zero init: an Engram model loaded from the snapshot gives the SAME logits as
    the plain model, bit for bit;
  * causality and packed-document isolation of the whole model, with a non-zero
    memory so the Engram path actually contributes;
  * frozen backbone: only `engram.*` gets gradients, W_V a non-zero one;
  * the VLM builds, runs, and gives image placeholders no memory;
  * DCP save -> load round trip;
  * HF export -> `trust_remote_code` model matches the training model, and its
    cached `generate()` agrees with a full forward.

The snapshot defaults to the shared HF cache; QWEN3_5_0_8B_SNAPSHOT overrides it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(os.environ.get(
    "QWEN3_5_0_8B_SNAPSHOT",
    "/data/151-2/users/allabres/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17",
))
TEXT_CONFIG = REPO / "configs/models/qwen3_5_0_8b_text.json"
VLM_CONFIG = REPO / "configs/models/qwen3_5_0_8b.json"
SEQ_LEN = 512
ROWS = 4099  # small tables: the tests are about wiring, not capacity

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GatedDeltaNet needs CUDA"),
    pytest.mark.skipif(not SNAPSHOT.is_dir(), reason=f"no Qwen3.5-0.8B snapshot at {SNAPSHOT}"),
]


def _engram_args(**kw):
    from models.qwen3_5.engram import EngramArgs

    return EngramArgs(rows_per_head=ROWS, **kw)


def _model(engram=None, config=TEXT_CONFIG, *, load=True, dtype=torch.bfloat16):
    from models.qwen3_5.checkpoint import build_meta, init_engram, load_hf, materialize

    model = build_meta(config, seq_len=SEQ_LEN, tp=1, engram=engram)
    materialize(model, "cuda")
    if load:
        load_hf(model, SNAPSHOT)
        init_engram(model, "cuda")
    else:
        with torch.no_grad():
            model.init_states(buffer_device=torch.device("cuda"))
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.eval()


def _wake_engram(model, seed=0):
    """Give W_V (and the table) real values, so the memory changes the output."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for module in model.engram.values():
            w = module.value_proj.weight
            w.copy_(torch.randn(w.shape, generator=g, device="cuda", dtype=torch.float32).to(w.dtype) * 0.02)


def _batch(lengths=(96, 64), seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1000, 100_000, (sum(lengths),), generator=g)
    positions = torch.cat([torch.arange(n) for n in lengths])
    return {"input": tokens.cuda(), "labels": tokens.clone().cuda(), "positions": positions.cuda()}


@torch.no_grad()
def _logits(model, batch):
    inputs, _, extra = model.preprocess_inputs(dict(batch))
    model._skip_lm_head = False
    return model(inputs, **extra).float()


def test_zero_init_gives_the_base_model_bit_for_bit():
    batch = _batch()
    base = _logits(_model(None), batch)
    model = _model(_engram_args())
    assert torch.count_nonzero(model.engram["2"].value_proj.weight) == 0
    assert torch.equal(_logits(model, batch), base)


def test_the_memory_changes_the_output_once_awake():
    """The control for the test above: a live W_V must move the logits, or
    'identical logits' would also pass with the module never called."""
    batch = _batch()
    model = _model(_engram_args())
    before = _logits(model, batch)
    _wake_engram(model)
    after = _logits(model, batch)
    assert (after - before).abs().max() > 1e-2


def test_whole_model_is_causal():
    model = _model(_engram_args())
    _wake_engram(model)
    batch = _batch(lengths=(128,))
    ref = _logits(model, batch)
    for t in (3, 40, 100):
        changed = dict(batch)
        changed["input"] = batch["input"].clone()
        changed["input"][t + 1] += 1
        out = _logits(model, changed)
        torch.testing.assert_close(out[: t + 1], ref[: t + 1], rtol=0, atol=1e-3)
        assert (out[t + 1] - ref[t + 1]).abs().max() > 0


def test_nothing_crosses_a_packed_document_boundary():
    """Changing the END of document A (the tokens B's n-grams would read if the
    reset were missing) leaves every logit of document B unchanged."""
    model = _model(_engram_args())
    _wake_engram(model)
    batch = _batch(lengths=(96, 64))
    ref = _logits(model, batch)
    changed = dict(batch)
    changed["input"] = batch["input"].clone()
    changed["input"][93:96] += 1
    out = _logits(model, changed)
    torch.testing.assert_close(out[96:], ref[96:], rtol=0, atol=1e-3)
    # and B's ids are exactly those of B alone in a row
    _, _, extra = model.preprocess_inputs(dict(batch))
    alone = _batch(lengths=(64,))
    alone["input"] = batch["input"][96:].clone()
    _, _, extra_alone = model.preprocess_inputs(alone)
    assert torch.equal(extra["engram_ids"][96:], extra_alone["engram_ids"])


def test_frozen_backbone_trains_only_engram():
    from train.config import Model
    from train.step import forward_backward
    from train.utils import set_trainable_parts

    model = _model(_engram_args()).train()
    set_trainable_parts(Model(engram=True, train_llm=False, train_mlp=False, train_engram=True), model)
    batch = _batch()
    batch["num_valid_tokens"] = int(batch["labels"].numel())
    forward_backward(model, [batch], dp_group=None, special_tokens={"image_id": -1}, loss_chunks=4)
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad and all(n.startswith("engram.") for n in with_grad), sorted(with_grad)[:5]
    assert model.engram["2"].value_proj.weight.grad.abs().sum() > 0


def test_vlm_builds_runs_and_gives_images_no_memory():
    """The VLM (`qwen3_5`, with the tower) with images in the row: image
    placeholders are blocked, so the Engram module leaves them untouched."""
    import json

    raw = json.loads(VLM_CONFIG.read_text())
    image_id = raw["image_token_id"]
    model = _model(_engram_args(), config=VLM_CONFIG, load=False)
    _wake_engram(model)
    vc = model.vision_encoder
    # one 8x8-patch image -> 64 patches -> 16 merged tokens
    grid = torch.tensor([[1, 8, 8]], device="cuda")
    n_img = 16
    patch_dim = vc.patch_embed.weight.shape[1]
    pixels = torch.randn(64, patch_dim, device="cuda", dtype=torch.bfloat16)
    batch = _batch(lengths=(80,))
    batch["input"][10 : 10 + n_img] = image_id
    seen = {}
    module = model.engram["2"]
    handle = module.register_forward_hook(lambda m, a, o: seen.update(x=a[0], out=o))
    try:
        inputs, _, extra = model.preprocess_inputs(dict(batch))
        model._skip_lm_head = False
        with torch.no_grad():
            logits = model(inputs, pixel_values=pixels, grid_thw=grid,
                           special_tokens={"image_id": image_id}, **extra)
    finally:
        handle.remove()
    assert torch.isfinite(logits.float()).all()
    assert not extra["engram_mask"][10 : 10 + n_img].any()
    assert torch.equal(seen["out"][10 : 10 + n_img], seen["x"][10 : 10 + n_img])
    assert not torch.equal(seen["out"][30:], seen["x"][30:])


def test_dcp_round_trip(tmp_path):
    import torch.distributed.checkpoint as dcp

    model = _model(_engram_args(), dtype=torch.float32)
    _wake_engram(model)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(tmp_path))
    fresh = _model(_engram_args(), dtype=torch.float32)
    state = {"model": fresh.state_dict()}
    dcp.load(state, checkpoint_id=str(tmp_path))
    fresh.load_state_dict(state["model"])
    for name, p in model.named_parameters():
        if name.startswith("engram."):
            assert torch.equal(p, dict(fresh.named_parameters())[name]), name
    batch = _batch()
    assert torch.equal(_logits(fresh, batch), _logits(model, batch))


def _write_run_config(path: Path, out_dir: Path) -> Path:
    path.write_text(f"""
[model]
model_config = "{TEXT_CONFIG}"
engram = true
engram_layers = [2]
engram_rows_per_head = {ROWS}

[training]
model_dir = "{SNAPSHOT}"
output_dir = "{out_dir}"
""")
    return path


def test_hf_export_matches_and_generates(tmp_path):
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM

    import importlib.util

    # `utils/` is a script directory, not a package
    spec = importlib.util.spec_from_file_location("dcp_to_hf", REPO / "utils" / "dcp_to_hf.py")
    dcp_to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dcp_to_hf)
    convert = dcp_to_hf.convert

    model = _model(_engram_args(), dtype=torch.float32)
    _wake_engram(model)
    with torch.no_grad():  # a table that is not the init, so the export must carry it
        for m in model.engram.values():
            m.table.weight.normal_(std=0.5)
            m.key_proj.weight.normal_(std=0.05)
    ckpt = tmp_path / "ckpt"
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(ckpt))
    out = tmp_path / "hf"
    convert(ckpt, SNAPSHOT, out, run_config=_write_run_config(tmp_path / "run.toml", tmp_path))

    hf = AutoModelForCausalLM.from_pretrained(
        str(out), trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()
    assert type(hf).__name__ == "Qwen3_5EngramForCausalLM"
    torch.testing.assert_close(
        hf.model.engram["2"].table.weight.float(),
        model.engram["2"].table.weight.to(torch.bfloat16).float(),
    )

    batch = _batch(lengths=(128,))
    ours = _logits(model.to(torch.bfloat16), batch)
    with torch.no_grad():
        theirs = hf(input_ids=batch["input"][None]).logits[0].float()
    diff = (ours - theirs).abs()
    top1 = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    print(f"[hf export] mean|dlogit| {diff.mean().item():.3e} max {diff.max().item():.3e} top1 {top1:.4f}")
    # the repo's parity gates against transformers (README): mean |dlogit| < 0.1,
    # top-1 agreement; bf16 on both sides
    assert diff.mean() < 0.1, diff.mean()
    assert top1 > 0.95, top1

    # cached generation must see the same n-grams as a full forward
    prompt = batch["input"][None, :40]
    with torch.no_grad():
        gen = hf.generate(prompt, max_new_tokens=12, do_sample=False, use_cache=True)
        full = hf(input_ids=gen).logits[0].float()
    predicted = full[39:-1].argmax(-1)
    assert torch.equal(predicted, gen[0, 40:]), (predicted, gen[0, 40:])
