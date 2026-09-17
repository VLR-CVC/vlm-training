"""S1 gate (TITAN_MIGRATION_v2.md): torchtitan-port Qwen3.5 vs transformers.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
        python models/tests/test_qwen3_5_tt_parity.py

Checks, in order: config vs config.json, state-dict round trip, meta-init memory,
text logits, multimodal logits, a packed row of two documents against the same
documents run separately, masked loss, and eager vs per-block fullgraph compile.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
from models.qwen3_5_tt.state_dict_adapter import Qwen35StateDictAdapter

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
)
IMAGE = Path(__file__).resolve().parents[2] / "test_images" / "horse.png"
SEQ_LEN = 8192
# bf16 compute on both sides: attn_gym's fused GDN kernel takes fp16/bf16 only,
# and varlen attention is bf16 by construction. That is how training runs too
# (FSDP param_dtype=bf16), so fp32 parity is not a meaningful target. Gates are on
# top-1 agreement and mean |dlogit|; max |dlogit| is reported. Scale of bf16 noise,
# measured: HF fp32 vs HF bf16 on the multimodal prompt's text positions is 0.050
# mean |dlogit|; two independent bf16 kernel paths should sit near sqrt(2) of that.
MEAN_TOL = float(os.environ.get("MEAN_TOL", "1e-1"))
TOP1_MIN = float(os.environ.get("TOP1_MIN", "0.97"))
dev = torch.device("cuda")
# TF32 is on by default for cuDNN convolutions; HF's Conv3d patch embed then differs
# from our equivalent Linear by ~1e-4 relative, which the ViT amplifies ~100x.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def report(name: str, ours: torch.Tensor, ref: torch.Tensor,
           mean_tol: float = MEAN_TOL, top1_min: float = TOP1_MIN) -> None:
    ours, ref = ours.float(), ref.float()
    d = (ours - ref).abs()
    top1 = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    ok = d.mean().item() <= mean_tol and top1 >= top1_min
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: mean|dlogit|={d.mean().item():.3e} "
          f"max={d.max().item():.3e} top1={top1:.4f} (n={ours.shape[0]})")
    assert ok, name


def cast_params(model, dtype) -> None:
    """Parameters only, buffers (RoPE caches) stay fp32 -- what FSDP's
    MixedPrecisionPolicy(param_dtype=...) does."""
    for p in model.parameters():
        p.data = p.data.to(dtype)


def hf_model():
    from transformers import Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return m.to(dev).eval()


def ours_forward(model, input_ids, positions, mrope=None, pixel_values=None, grid_thw=None):
    batch = {"input": input_ids, "labels": input_ids, "positions": positions}
    if mrope is not None:
        batch["mrope_positions"] = mrope
    inputs, _, extra = model.preprocess_inputs(batch)
    return model(
        inputs,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        special_tokens={"image_id": model_image_id()},
        **extra,
    )


def model_image_id() -> int:
    return json.loads((Path(SNAPSHOT) / "config.json").read_text())["image_token_id"]


def main() -> None:
    torch.manual_seed(0)
    raw = json.loads((Path(SNAPSHOT) / "config.json").read_text())

    # --- config + meta init ------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    model = build_meta(SNAPSHOT, seq_len=SEQ_LEN)
    assert model.enable_weight_tying == raw["text_config"]["tie_word_embeddings"]
    n_params = sum(p.numel() for p in model.parameters())
    materialize(model, dev)
    load_hf(model, SNAPSHOT)
    model.eval()
    peak = torch.cuda.max_memory_allocated() / 2**30
    param_gib = n_params * 4 / 2**30
    print(f"[{'PASS' if peak < 1.5 * param_gib else 'FAIL'}] meta init + load: "
          f"peak {peak:.2f} GiB for {param_gib:.2f} GiB of fp32 params")
    assert peak < 1.5 * param_gib

    # --- state dict round trip ----------------------------------------------
    adapter = Qwen35StateDictAdapter(model.config, SNAPSHOT)
    sd = {k: v for k, v in model.state_dict().items()}
    back = adapter.from_hf(adapter.to_hf(sd))
    keys_ok = set(back) >= set(sd) - {"tok_embeddings.weight"}
    same = all(torch.equal(back[k], sd[k]) for k in sd if k in back)
    print(f"[{'PASS' if keys_ok and same else 'FAIL'}] from_hf(to_hf(sd)) round trip")
    assert keys_ok and same

    index = json.loads((Path(SNAPSHOT) / "model.safetensors.index.json").read_text())
    hf_keys = set(index["weight_map"])
    covered = set(adapter.to_hf(sd))
    unused = sorted(k for k in hf_keys - covered if not k.startswith("mtp."))
    print(f"[{'PASS' if not unused else 'FAIL'}] every non-MTP HF tensor is consumed"
          + (f": unused {unused[:5]}" if unused else ""))
    assert not unused

    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(SNAPSHOT)
    tok = proc.tokenizer
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(IMAGE).convert("RGB").resize((448, 448))
    mm = proc(text=[text], images=[image], return_tensors="pt").to(dev)

    # --- vision tower, fp32 (no GDN here, so fp32 is possible) ------------------
    hf32 = Qwen3_5ForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    with torch.no_grad():
        v_ref = hf32.model.visual(mm["pixel_values"].float(), grid_thw=mm["image_grid_thw"])
        v_ref = getattr(v_ref, "pooler_output", v_ref)
        v_ours = model.vision_encoder(mm["pixel_values"].float(), grid_thw=mm["image_grid_thw"])
    rel = ((v_ours - v_ref).abs().mean() / v_ref.abs().mean()).item()
    print(f"[{'PASS' if rel < 1e-4 else 'FAIL'}] vision tower fp32: rel mean|d|={rel:.2e}")
    assert rel < 1e-4
    del hf32
    torch.cuda.empty_cache()

    cast_params(model, torch.bfloat16)
    model.retie_weights()
    hf = hf_model()

    # --- text ------------------------------------------------------------
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        ref = hf(input_ids=ids).logits[0]
        pos = torch.arange(ids.shape[1], device=dev)
        ours = ours_forward(model, ids[0], pos)
    report("text logits", ours, ref)

    # --- multimodal ------------------------------------------------------
    with torch.no_grad():
        ref_mm = hf(**mm).logits[0]
        mrope, _ = hf.model.get_rope_index(
            mm["input_ids"], mm["mm_token_type_ids"], image_grid_thw=mm["image_grid_thw"]
        )
        mrope = mrope[:, 0].transpose(0, 1).contiguous()  # (T, 3)
        mm_ids = mm["input_ids"][0]
        pos_mm = torch.arange(mm_ids.shape[0], device=dev)
        ours_mm = ours_forward(
            model, mm_ids, pos_mm, mrope=mrope,
            pixel_values=mm["pixel_values"].to(torch.bfloat16), grid_thw=mm["image_grid_thw"],
        )
    # Image-pad positions are never supervised, and bf16 alone flips ~11% of their
    # argmaxes (HF fp32 vs HF bf16: 88.8% top-1 there, 100% on text). Gate on text
    # positions; the vision tower is checked on its own in fp32 below.
    img = mm["input_ids"][0] == raw["image_token_id"]
    report("multimodal logits, text positions", ours_mm[~img], ref_mm[~img])
    print(f"[INFO] multimodal image positions: top1="
          f"{(ours_mm[img].argmax(-1) == ref_mm[img].argmax(-1)).float().mean().item():.4f}")

    # --- packed row: [multimodal doc | text doc] ---------------------------
    with torch.no_grad():
        text_mrope = pos.unsqueeze(-1).expand(-1, 3)
        packed_ids = torch.cat([mm_ids, ids[0]])
        packed_pos = torch.cat([pos_mm, pos])
        packed_mrope = torch.cat([mrope, text_mrope])
        ours_packed = ours_forward(
            model, packed_ids, packed_pos, mrope=packed_mrope,
            pixel_values=mm["pixel_values"].to(torch.bfloat16), grid_thw=mm["image_grid_thw"],
        )
    n_mm = mm_ids.shape[0]
    # Against our own unpacked runs. Doc 1 cannot see doc 2 (causal), so any gap
    # there is bf16 kernel noise from a different total length -- it sets the
    # scale. Doc 2 is the leakage check: a boundary bug in the varlen metadata or
    # the GDN conv/state reset shows up as a gap far above that noise.
    report("packed doc 1 (multimodal, text positions) vs ours alone",
           ours_packed[:n_mm][~img], ours_mm[~img], top1_min=1.0)
    report("packed doc 2 (text) vs ours alone", ours_packed[n_mm:], ours, top1_min=1.0)

    # --- masked loss --------------------------------------------------------
    labels = packed_ids.clone()
    labels[: n_mm // 2] = -100  # mask a prefix, as SFT masking does
    shifted = torch.cat([labels[1:], labels.new_tensor([-100])])
    shifted[n_mm - 1] = -100  # no prediction across the document boundary
    loss_ours = F.cross_entropy(ours_packed.float(), shifted, ignore_index=-100)
    ref_logits = torch.cat([ref_mm, ref]).float()
    loss_ref = F.cross_entropy(ref_logits, shifted, ignore_index=-100)
    rel = abs(loss_ours.item() - loss_ref.item()) / loss_ref.item()
    print(f"[{'PASS' if rel < 1e-2 else 'FAIL'}] masked loss: ours {loss_ours.item():.6f} "
          f"hf {loss_ref.item():.6f} rel {rel:.2e}")
    assert rel < 1e-2
    del hf
    torch.cuda.empty_cache()

    # --- eager vs per-block fullgraph compile, bf16 --------------------------
    from train.parallel.compile import apply_compile

    with torch.no_grad():
        eager = ours_forward(model, packed_ids, packed_pos, mrope=packed_mrope,
                             pixel_values=mm["pixel_values"].to(torch.bfloat16),
                             grid_thw=mm["image_grid_thw"])
    apply_compile(model)
    with torch.no_grad():
        compiled = ours_forward(model, packed_ids, packed_pos, mrope=packed_mrope,
                                pixel_values=mm["pixel_values"].to(torch.bfloat16),
                                grid_thw=mm["image_grid_thw"])
    text_pos = packed_ids != raw["image_token_id"]
    report("eager vs compiled (bf16, text positions)", compiled[text_pos], eager[text_pos], top1_min=1.0)

    # backward through the compiled blocks: custom-op autograd under fullgraph
    model.train()
    batch = {"input": packed_ids, "labels": packed_ids, "positions": packed_pos,
             "mrope_positions": packed_mrope}
    inputs, _, extra = model.preprocess_inputs(batch)
    logits = model(inputs, pixel_values=mm["pixel_values"].to(torch.bfloat16),
                   grid_thw=mm["image_grid_thw"],
                   special_tokens={"image_id": model_image_id()}, **extra)
    F.cross_entropy(logits.float(), shifted, ignore_index=-100).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    finite = all(g is not None and torch.isfinite(g).all() for g in grads)
    print(f"[{'PASS' if finite else 'FAIL'}] compiled backward: {len(grads)} grads, all finite")
    assert finite
    print("S1 parity: all checks passed")


if __name__ == "__main__":
    main()
