"""LoRA (`models/common/lora.py`) on the real Qwen3.5-0.8B snapshot (CUDA).

    CUDA_VISIBLE_DEVICES=0 pytest models/tests/test_lora.py -q

  * zero init: B = 0, so a LoRA model loaded from the snapshot gives the SAME
    logits as the plain model, bit for bit (and a live B moves them: control);
  * only the adapters train under `set_trainable_parts`;
  * `merge_lora` gives the unmerged model's logits (fp32);
  * DCP checkpoint -> `utils/dcp_to_hf.py` (merges) -> plain HF model matches
    the training model (unmerged, bf16) within the repo's HF parity gates.
"""

from __future__ import annotations

import importlib.util
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
SEQ_LEN = 512
RANK = 8

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GatedDeltaNet needs CUDA"),
    pytest.mark.skipif(not SNAPSHOT.is_dir(), reason=f"no Qwen3.5-0.8B snapshot at {SNAPSHOT}"),
]


def _model(lora=True, *, dtype=torch.bfloat16):
    from models.common.lora import apply_lora, init_lora
    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize

    model = build_meta(TEXT_CONFIG, seq_len=SEQ_LEN, tp=1)
    if lora:
        apply_lora(model, rank=RANK, alpha=2 * RANK)
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)
    if lora:
        init_lora(model)
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.eval()


def _wake_lora(model, seed=0, std=0.02):
    from models.common.lora import lora_modules

    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_modules(model):
            b = m.lora_B
            b.copy_((torch.randn(b.shape, generator=g, device="cuda") * std).to(b.dtype))


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


def test_targets_cover_every_decoder_linear_kind():
    from models.common.lora import lora_modules

    model = _model()
    kinds = {n.rsplit(".", 1)[-1] for n, _ in lora_modules(model)}
    assert kinds == {"wq", "wk", "wv", "wo", "in_proj_q", "in_proj_k", "in_proj_v",
                     "in_proj_z", "out_proj", "w13", "w2"}
    # 6 full-attention layers x 4 + 18 GDN x 5 + 24 MLP x 2
    assert len(lora_modules(model)) == 6 * 4 + 18 * 5 + 24 * 2


def test_zero_init_gives_the_base_model_bit_for_bit():
    batch = _batch()
    base = _logits(_model(lora=False), batch)
    model = _model()
    assert torch.equal(_logits(model, batch), base)


def test_live_adapters_change_the_output():
    batch = _batch()
    model = _model()
    before = _logits(model, batch)
    _wake_lora(model)
    assert (_logits(model, batch) - before).abs().max() > 1e-2


def test_only_adapters_train():
    from train.config import Model as ModelArgs
    from train.utils import set_trainable_parts

    model = _model()
    set_trainable_parts(ModelArgs(train_llm=False, lora_rank=RANK), model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".lora_A", ".lora_B")) for n in trainable)


def test_merge_matches_unmerged():
    from models.common.lora import LoRALinear, merge_lora

    batch = _batch()
    model = _model(dtype=torch.float32)
    base = _logits(model, batch)  # B = 0
    _wake_lora(model)
    unmerged = _logits(model, batch)
    assert merge_lora(model) == 6 * 4 + 18 * 5 + 24 * 2
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert not any(n.endswith((".lora_A", ".lora_B")) for n, _ in model.named_parameters())
    merged = _logits(model, batch)
    diff = (merged - unmerged).abs()
    effect = (unmerged - base).abs().mean()
    print(f"[merge fp32] mean|dlogit| {diff.mean().item():.3e} max {diff.max().item():.3e}; "
          f"LoRA effect mean|dlogit| {effect.item():.3e}")
    # the fused GatedDeltaNet kernel runs in bf16 even in an fp32 model, so the
    # two orders of summation (x W' vs x W + x A B) differ by bf16 noise, not
    # by fp32 rounding: gate the merge error against the adapters' own effect
    assert diff.mean() < 0.05 * effect, (diff.mean(), effect)
    top2 = unmerged.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) >= 1.0
    assert decisive.any()
    assert torch.equal(merged.argmax(-1)[decisive], unmerged.argmax(-1)[decisive])


def _write_run_config(path: Path, out_dir: Path) -> Path:
    path.write_text(f"""
[model]
model_config = "{TEXT_CONFIG}"
train_llm = false
lora_rank = {RANK}
lora_alpha = {2 * RANK}

[training]
model_dir = "{SNAPSHOT}"
output_dir = "{out_dir}"
""")
    return path


def test_hf_export_is_merged_and_matches(tmp_path):
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM

    spec = importlib.util.spec_from_file_location("dcp_to_hf", REPO / "utils" / "dcp_to_hf.py")
    dcp_to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dcp_to_hf)

    model = _model(dtype=torch.float32)
    _wake_lora(model)
    ckpt = tmp_path / "ckpt"
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(ckpt))
    out = tmp_path / "hf"
    dcp_to_hf.convert(ckpt, SNAPSHOT, out, run_config=_write_run_config(tmp_path / "run.toml", tmp_path))

    hf = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).cuda().eval()
    base = AutoModelForCausalLM.from_pretrained(str(SNAPSHOT), dtype=torch.bfloat16).cuda().eval()
    assert not any("lora" in n for n, _ in hf.named_parameters())

    batch = _batch(lengths=(128,))
    ours = _logits(model.to(torch.bfloat16), batch)
    with torch.no_grad():
        theirs = hf(input_ids=batch["input"][None]).logits[0].float()
        stock = base(input_ids=batch["input"][None]).logits[0].float()
    diff = (ours - theirs).abs()
    top1 = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    moved = (stock - theirs).abs().mean().item()
    print(f"[hf export] mean|dlogit| {diff.mean().item():.3e} max {diff.max().item():.3e} "
          f"top1 {top1:.4f}; export vs stock mean|dlogit| {moved:.3e}")
    # the export carries the adapters: it is far from the stock model ...
    assert moved > 5 * diff.mean().item()
    # ... and within the repo's HF parity gates of the training model (bf16 both)
    assert diff.mean() < 0.1, diff.mean()
    top2 = ours.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) >= 1.0
    assert decisive.any()
    assert torch.equal(ours.argmax(-1)[decisive], theirs.argmax(-1)[decisive])
