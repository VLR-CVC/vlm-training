"""Qwen3-VL on the torchtitan components (`models/qwen3_vl_tt`) vs transformers.

    QWEN3_VL_SNAPSHOT=<HF snapshot> python models/tests/test_qwen3_vl_tt_parity.py

The S1 checks of `test_qwen3_5_tt_parity.py` for Qwen3-VL: config, state-dict round
trip and HF coverage, meta-init memory, the vision tower with its DeepStack features
(fp32), MRoPE positions from `data/model_batch.py` against HF's `get_rope_index`,
text and multimodal logits (bf16), a packed row against its documents run alone,
masked loss, and eager vs per-block fullgraph compile with backward. One GPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.model_batch import mrope_positions
from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
from models.qwen3_5_tt.state_dict_adapter import Qwen35StateDictAdapter

SNAPSHOT = os.environ.get(
    "QWEN3_VL_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_vl_4b"
)
IMAGE = Path(__file__).resolve().parents[2] / "test_images" / "horse.png"
SEQ_LEN = 8192
# Same gates as the Qwen3.5 test: bf16 on both sides (varlen attention is bf16 by
# construction), top-1 agreement and mean |dlogit| on text positions.
MEAN_TOL = float(os.environ.get("MEAN_TOL", "1e-1"))
TOP1_MIN = float(os.environ.get("TOP1_MIN", "0.97"))
# A reference top-2 logit gap below this is a coin flip at bf16: `max|dlogit|` on the
# passing runs is ~0.2 (text) to 3.5 (multimodal), and mean ~0.07.
NEAR_TIE = float(os.environ.get("NEAR_TIE", "1.0"))
dev = torch.device("cuda")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def report(name, ours, ref, mean_tol=MEAN_TOL, top1_min=TOP1_MIN) -> None:
    ours, ref = ours.float(), ref.float()
    d = (ours - ref).abs()
    # Top-1 only means something where the reference is not a near-tie: with two
    # independent bf16 kernel paths, positions whose top-2 gap is inside the logit
    # noise flip for no reason (job 1861476 scored 16/16 on the prompt job 1861527
    # scored 15/16, same code). Gate on the decisive positions, report the rest.
    top2 = ref.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > NEAR_TIE
    agree = ours.argmax(-1) == ref.argmax(-1)
    top1 = agree[decisive].float().mean().item() if decisive.any() else 1.0
    ok = d.mean().item() <= mean_tol and top1 >= top1_min
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: mean|dlogit|={d.mean().item():.3e} "
          f"max={d.max().item():.3e} top1={top1:.4f} "
          f"(n={int(decisive.sum())} decisive of {ours.shape[0]}, "
          f"ties agree {int(agree[~decisive].sum())}/{int((~decisive).sum())})")
    assert ok, name


def check(name, ok, detail="") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    assert ok, name


def ours_forward(model, image_id, input_ids, positions, mrope=None, pixel_values=None, grid_thw=None):
    batch = {"input": input_ids, "labels": input_ids, "positions": positions}
    if mrope is not None:
        batch["mrope_positions"] = mrope
    inputs, _, extra = model.preprocess_inputs(batch)
    return model(inputs, pixel_values=pixel_values, grid_thw=grid_thw,
                 special_tokens={"image_id": image_id}, **extra)


def main() -> None:
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    raw = json.loads((Path(SNAPSHOT) / "config.json").read_text())
    image_id = raw["image_token_id"]
    vc = raw["vision_config"]

    # --- config + meta init ------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    model = build_meta(SNAPSHOT, seq_len=SEQ_LEN)
    tied = raw["text_config"].get("tie_word_embeddings", raw.get("tie_word_embeddings", False))
    check("config: model class and weight tying",
          type(model).__name__ == "Qwen3VLModel" and model.enable_weight_tying == bool(tied),
          f"{type(model).__name__}, tying {model.enable_weight_tying} (config says {bool(tied)}), "
          f"snapshot {SNAPSHOT}")
    check("config: DeepStack mergers",
          len(model.vision_encoder.deepstack_mergers) == len(vc["deepstack_visual_indexes"]))
    n_params = sum(p.numel() for p in model.parameters())
    materialize(model, dev)
    load_hf(model, SNAPSHOT)
    model.eval()
    peak = torch.cuda.max_memory_allocated() / 2**30
    param_gib = n_params * 4 / 2**30
    check("meta init + load", peak < 1.5 * param_gib, f"peak {peak:.2f} GiB for {param_gib:.2f} GiB fp32")

    # --- state dict round trip and HF coverage --------------------------------
    adapter = Qwen35StateDictAdapter(model.config, SNAPSHOT)
    sd = dict(model.state_dict())
    back = adapter.from_hf(adapter.to_hf(sd))
    check("from_hf(to_hf(sd)) round trip",
          set(back) >= set(sd) - {"tok_embeddings.weight"}
          and all(torch.equal(back[k], sd[k]) for k in sd if k in back))
    index_path = Path(SNAPSHOT) / "model.safetensors.index.json"
    if index_path.exists():
        hf_keys = set(json.loads(index_path.read_text())["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(str(Path(SNAPSHOT) / "model.safetensors"), "pt") as f:
            hf_keys = set(f.keys())
    unused = sorted(hf_keys - set(adapter.to_hf(sd)))
    check("every HF tensor is consumed", not unused, f"unused {unused[:5]}" if unused else "")

    proc = AutoProcessor.from_pretrained(SNAPSHOT)
    tok = proc.tokenizer
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image in detail: the animal, the background, "
                                 "the colours, and what the weather looks like."}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(IMAGE).convert("RGB").resize((448, 448))
    mm = proc(text=[text], images=[image], return_tensors="pt").to(dev)
    mm_ids = mm["input_ids"][0]

    # --- vision tower + DeepStack, fp32 -----------------------------------------
    hf32 = Qwen3VLForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    with torch.no_grad():
        out = hf32.model.visual(mm["pixel_values"].float(), grid_thw=mm["image_grid_thw"])
        v_ref = torch.cat([out.pooler_output, *out.deepstack_features])
        v_ours = model.vision_encoder(mm["pixel_values"].float(), grid_thw=mm["image_grid_thw"])
    check("vision tower shape (merged + DeepStack)", v_ours.shape == v_ref.shape,
          f"{tuple(v_ours.shape)} vs {tuple(v_ref.shape)}")
    rel = ((v_ours - v_ref).abs().mean() / v_ref.abs().mean()).item()
    check("vision tower + DeepStack fp32", rel < 1e-4, f"rel mean|d|={rel:.2e}")

    # --- MRoPE: the dataloader's positions against HF -------------------------
    with torch.no_grad():
        hf_mrope, _ = hf32.model.get_rope_index(
            mm["input_ids"], mm["mm_token_type_ids"], image_grid_thw=mm["image_grid_thw"])
    hf_mrope = hf_mrope[:, 0].transpose(0, 1).contiguous()  # (T, 3)
    cu = torch.tensor([0, mm_ids.shape[0]], dtype=torch.int32)
    mrope = mrope_positions(
        mm_ids.cpu(), cu, mm["image_grid_thw"].cpu(), image_token_id=image_id,
        video_token_id=raw["video_token_id"], spatial_merge_size=vc["spatial_merge_size"],
    ).to(dev)
    check("mrope_positions (data/model_batch.py) == HF get_rope_index", torch.equal(mrope, hf_mrope))
    del hf32
    torch.cuda.empty_cache()

    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.retie_weights()
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    # --- text ----------------------------------------------------------------
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt").input_ids.to(dev)
    pos = torch.arange(ids.shape[1], device=dev)
    with torch.no_grad():
        ref = hf(input_ids=ids).logits[0]
        ours = ours_forward(model, image_id, ids[0], pos)
    report("text logits", ours, ref)

    # --- multimodal (DeepStack active) ---------------------------------------
    pv = mm["pixel_values"].to(torch.bfloat16)
    pos_mm = torch.arange(mm_ids.shape[0], device=dev)
    with torch.no_grad():
        ref_mm = hf(**mm).logits[0]
        ours_mm = ours_forward(model, image_id, mm_ids, pos_mm, mrope=mrope,
                               pixel_values=pv, grid_thw=mm["image_grid_thw"])
    img = mm_ids == image_id
    report("multimodal logits, text positions", ours_mm[~img], ref_mm[~img])
    print(f"[INFO] multimodal image positions: top1="
          f"{(ours_mm[img].argmax(-1) == ref_mm[img].argmax(-1)).float().mean().item():.4f}")

    # --- packed row: [multimodal doc | text doc] ------------------------------
    with torch.no_grad():
        packed_ids = torch.cat([mm_ids, ids[0]])
        packed_pos = torch.cat([pos_mm, pos])
        packed_mrope = torch.cat([mrope, pos.unsqueeze(-1).expand(-1, 3)])
        ours_packed = ours_forward(model, image_id, packed_ids, packed_pos, mrope=packed_mrope,
                                   pixel_values=pv, grid_thw=mm["image_grid_thw"])
    n_mm = mm_ids.shape[0]
    report("packed doc 1 (multimodal, text positions) vs ours alone",
           ours_packed[:n_mm][~img], ours_mm[~img], top1_min=1.0)
    report("packed doc 2 (text) vs ours alone", ours_packed[n_mm:], ours, top1_min=1.0)

    # --- masked loss --------------------------------------------------------
    labels = packed_ids.clone()
    labels[: n_mm // 2] = -100
    shifted = torch.cat([labels[1:], labels.new_tensor([-100])])
    shifted[n_mm - 1] = -100
    loss_ours = F.cross_entropy(ours_packed.float(), shifted, ignore_index=-100)
    loss_ref = F.cross_entropy(torch.cat([ref_mm, ref]).float(), shifted, ignore_index=-100)
    rel = abs(loss_ours.item() - loss_ref.item()) / loss_ref.item()
    check("masked loss", rel < 1e-2, f"ours {loss_ours.item():.6f} hf {loss_ref.item():.6f} rel {rel:.2e}")
    del hf
    torch.cuda.empty_cache()

    # --- eager vs per-block fullgraph compile ------------------------------------
    # Gated in fp32 GEMMs (the varlen attention kernel is bf16 on both sides). In
    # bf16, inductor's fused kernels round differently from eager: job 1856053 saw
    # mean|dlogit| 5.8e-2 and one top-1 flip in 27 -- the size of the ours-vs-HF bf16
    # gap -- which cannot separate a compile bug from rounding.
    from train.parallel.compile import apply_compile

    def set_dtype(dtype):
        for p in model.parameters():
            p.data = p.data.to(dtype)
        model.retie_weights()

    def packed_forward():
        with torch.no_grad():
            return ours_forward(model, image_id, packed_ids, packed_pos, mrope=packed_mrope,
                                pixel_values=pv, grid_thw=mm["image_grid_thw"])

    text_mask = packed_ids != image_id
    set_dtype(torch.float32)
    eager32 = packed_forward()
    set_dtype(torch.bfloat16)
    eager16 = packed_forward()
    apply_compile(model)
    compiled16 = packed_forward()
    d16 = (compiled16[text_mask].float() - eager16[text_mask].float()).abs()
    top16 = (compiled16[text_mask].argmax(-1) == eager16[text_mask].argmax(-1)).float().mean().item()
    print(f"[INFO] eager vs compiled (bf16, text positions): mean|dlogit|={d16.mean().item():.3e} top1={top16:.4f}")
    set_dtype(torch.float32)
    compiled32 = packed_forward()
    # the attention kernel stays bf16: job 1856437 measured 1.2e-2 here, and the TP
    # parity runs sit at 1.1e-2 between layouts for the same reason; gate as they do
    report("eager vs compiled (fp32 GEMMs, text positions)",
           compiled32[text_mask], eager32[text_mask], mean_tol=5e-2, top1_min=1.0)
    set_dtype(torch.bfloat16)

    model.train()
    logits = ours_forward(model, image_id, packed_ids, packed_pos, mrope=packed_mrope,
                          pixel_values=pv, grid_thw=mm["image_grid_thw"])
    F.cross_entropy(logits.float(), shifted, ignore_index=-100).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    check("compiled backward", all(g is not None and torch.isfinite(g).all() for g in grads),
          f"{len(grads)} grads, all present and finite")
    ds = [p.grad for p in model.vision_encoder.deepstack_mergers.parameters()]
    check("DeepStack mergers receive gradient", all(g is not None and g.abs().sum() > 0 for g in ds))
    print("Qwen3-VL parity: all checks passed")


if __name__ == "__main__":
    main()
