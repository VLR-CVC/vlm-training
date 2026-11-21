"""`models/elliot` against ElliotVL's own HF code, on a released checkpoint.

    CUDA_VISIBLE_DEVICES=0 ELLIOT_SNAPSHOT=<local copy of elliot-mllm/elliot-vl-2b-v1.2-sft-25k-step> \
        python models/tests/test_elliot_hf_parity.py

The reference is the checkpoint's remote code (`modeling_elliot.py`, loaded with
`trust_remote_code`), run the way its `generate` runs: `_prepare_multimodal_inputs`
(vision tower + projector scattered into the embeddings), then `Qwen3_5ForCausalLM` with
`position_ids=None`. Ours is `ElliotModel` with `image_positions="1d"`, the setting that
reproduces that inference.

Both models get the same pixels: one normalised image tensor, fed to the reference as
the 4D image (single-image prompts) or as its flattened native-resolution patches with
`spatial_shapes` (the multi-image path), and to ours in the Qwen processor's patch layout
(2x2 block order, each image as two identical frames).

Checks: bitwise weight round trip, vision tower + projector and logits in fp32 with the
leaf kernels shared (see `test_hf_inference_parity.py`), per-layer hidden states, and
logits with the deployed bf16 kernels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.tests.test_hf_inference_parity import (  # noqa: E402
    check, check_bf16, check_fp32, reference_kernels, rel_err, FP32_REL_TOL,
)

SNAPSHOT = os.environ.get("ELLIOT_SNAPSHOT")
IMAGE = ROOT / "assets" / "horse.png"
PATCH, MERGE, TEMPORAL = 16, 2, 2
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"


def to_tensor(img) -> torch.Tensor:
    """SigLIP normalisation: (x / 255 - 0.5) / 0.5, (3, H, W)."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32)).permute(2, 0, 1)
    return (x / 255.0 - 0.5) / 0.5


def ours_patches(x: torch.Tensor) -> torch.Tensor:
    """(3, H, W) -> (h*w, 3*T*p*p) in the Qwen processor's order: 2x2 blocks, then
    (C, T, p, p) per patch, the image repeated over T frames."""
    c, H, W = x.shape
    h, w = H // PATCH, W // PATCH
    p = x.view(c, h // MERGE, MERGE, PATCH, w // MERGE, MERGE, PATCH)
    p = p.permute(1, 4, 2, 5, 0, 3, 6)  # (bh, bw, ih, iw, C, p, p)
    p = p.unsqueeze(-3).expand(*p.shape[:5], TEMPORAL, PATCH, PATCH)
    return p.reshape(h * w, c * TEMPORAL * PATCH * PATCH)


def theirs_flat(xs: list[torch.Tensor]):
    """Their flattened-patch input: (B, max_len, C*p*p) row-major (Conv2d weight order),
    with spatial_shapes and pixel_attention_mask."""
    flats, shapes = [], []
    for x in xs:
        c, H, W = x.shape
        h, w = H // PATCH, W // PATCH
        flats.append(x.view(c, h, PATCH, w, PATCH).permute(1, 3, 0, 2, 4).reshape(h * w, -1))
        shapes.append([h, w])
    n = max(f.shape[0] for f in flats)
    pv = torch.zeros(len(flats), n, flats[0].shape[1])
    mask = torch.zeros(len(flats), n, dtype=torch.int32)
    for i, f in enumerate(flats):
        pv[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = 1
    return pv, torch.tensor(shapes), mask


@pytest.mark.skipif(not SNAPSHOT or not torch.cuda.is_available(),
                    reason="needs ELLIOT_SNAPSHOT (a local ElliotVL HF checkpoint) and a GPU")
def test_elliot_hf_parity():
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize, state_dict_adapter

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snap = Path(SNAPSHOT)

    model = build_meta(snap, seq_len=16384, image_positions="1d")
    materialize(model, dev)
    load_hf(model, snap)
    model.eval()
    hf_sd = state_dict_adapter(model)(model.config, str(snap)).to_hf(model.state_dict())
    bad, stored = [], set()
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for key in s.keys():
                stored.add(key)
                t = s.get_tensor(key)
                if key not in hf_sd or not torch.equal(hf_sd[key].to(t.dtype).cpu(), t):
                    bad.append(key)
    extra = sorted(set(hf_sd) - stored)
    check("weights: to_hf(ours) == checkpoint, bitwise", not bad and not extra,
          f"{len(stored)} tensors; mismatched/missing {bad[:3]} extra {extra[:3]}")

    # `dtype=float32` does not reach the LM: ElliotVL builds it from `text_config`,
    # whose dtype is bfloat16, so cast the whole model explicitly
    ref = AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True,
                                               dtype=torch.float32).float().to(dev).eval()
    assert all(p.dtype == torch.float32 for p in ref.parameters())
    from models.elliot.configs import POST_LAYERNORM_EPS
    ref.vision_model.post_layernorm.eps = POST_LAYERNORM_EPS  # as trained, see configs.py
    tok = AutoTokenizer.from_pretrained(snap)
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    cfg = ref.config

    horse = Image.open(IMAGE).convert("RGB")
    cases = {
        "text only": ([], "The capital of France is Paris, and the capital of Germany is"),
        "512x512 (native PE grid)": ([horse.resize((512, 512), Image.BICUBIC)],
                                     "Describe this image in detail."),
        "640x384 (bicubic PE)": ([horse.resize((640, 384), Image.BICUBIC)], "What animal is this?"),
        "1024x768": ([horse.resize((1024, 768), Image.BICUBIC)], "What colour is the animal?"),
        "two images 448x448 + 320x704": (
            [horse.resize((448, 448), Image.BICUBIC),
             horse.crop((0, 0, horse.width // 2, horse.height)).resize((320, 704), Image.BICUBIC)],
            "Compare the two images."),
    }

    def build(images, text):
        xs = [to_tensor(im) for im in images]
        user = "".join("<|vision_start|>" + "<|image_pad|>" * ((x.shape[1] // 32) * (x.shape[2] // 32))
                       + "<|vision_end|>" for x in xs)
        prompt = (SYSTEM + "<|im_start|>user\n" + user + text + "<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        return xs, ids

    def reference(xs, ids, dtype):
        kw = {}
        if len(xs) == 1:
            kw["pixel_values"] = xs[0][None].to(dev, dtype)
        elif xs:
            pv, shapes, mask = theirs_flat(xs)
            kw = dict(pixel_values=pv.to(dev, dtype), spatial_shapes=shapes.to(dev),
                      pixel_attention_mask=mask.to(dev))
        if xs:
            prep = ref._prepare_multimodal_inputs(input_ids=ids, **kw)
            feats = ref._extract_image_features(image_mask=ids == pad_id, **kw)[0]
        else:
            prep = {"inputs_embeds": ref.get_input_embeddings()(ids)}
            feats = None
        logits = ref.language_model(inputs_embeds=prep["inputs_embeds"], use_cache=False).logits[0]
        return logits, feats

    def ours(xs, ids, dtype, pos=None):
        kw = {}
        if xs:
            kw["pixel_values"] = torch.cat([ours_patches(x) for x in xs]).to(dev, dtype)
            kw["grid_thw"] = torch.tensor([[1, x.shape[1] // PATCH, x.shape[2] // PATCH] for x in xs],
                                          device=dev)
        t = ids[0]
        pos = torch.arange(t.shape[0], device=dev) if pos is None else pos
        inputs, _, extra = model.preprocess_inputs({"input": t, "labels": t, "positions": pos})
        logits = model(inputs, special_tokens={"image_id": cfg.image_pad_token_id}, **kw, **extra)
        feats = model.vision_encoder(kw["pixel_values"], grid_thw=kw["grid_thw"]) if xs else None
        return logits, feats

    taps_ours, taps_ref = [], []
    hooks = [l.register_forward_hook(lambda m, i, o: taps_ours.append(o.detach()))
             for l in model.layers.values()]
    hooks += [l.register_forward_hook(lambda m, i, o: taps_ref.append((o[0] if isinstance(o, tuple) else o)[0].detach()))
              for l in ref.language_model.model.layers]

    # --- fp32, shared leaf kernels: definition parity ---------------------------
    with reference_kernels(), torch.no_grad():
        for name, (images, text) in cases.items():
            xs, ids = build(images, text)
            r_logits, r_feats = reference(xs, ids, torch.float32)
            taps_ref_case = list(taps_ref); taps_ref.clear()
            o_logits, o_feats = ours(xs, ids, torch.float32)
            taps_ours_case = list(taps_ours); taps_ours.clear()
            if xs:
                check_fp32(f"fp32 vision tower + projector ({name})", o_feats, r_feats)
            check_fp32(f"fp32 logits ({name})", o_logits, r_logits)
            errs = [rel_err(a, b)[0] for a, b in zip(taps_ours_case, taps_ref_case)]
            first_bad = next((i for i, e in enumerate(errs) if e > FP32_REL_TOL), None)
            print(f"[INFO]   per-layer: worst rel {max(errs):.2e} (layer {errs.index(max(errs))}); "
                  f"first layer over tolerance: {first_bad}")
    for h in hooks:
        h.remove()

    # --- packed row: decoder_mask "causal" is their training packing ---------------
    # ElliotVL trains packed rows with position_ids=None: one sequence per row. Ours gets
    # per-document positions (as the dataloader emits) and must reproduce it.
    _, ids1 = build([], cases["text only"][1])
    xs2, ids2 = build(*cases["640x384 (bicubic PE)"])
    ids, n1 = torch.cat([ids1, ids2], 1), ids1.shape[1]
    pos = torch.cat([torch.arange(n1), torch.arange(ids2.shape[1])]).to(dev)
    with reference_kernels(), torch.no_grad():
        r_logits, _ = reference(xs2, ids, torch.float32)
        model.config.decoder_mask, model.config.row_len = "causal", ids.shape[1]
        o_causal, _ = ours(xs2, ids, torch.float32, pos)
        model.config.decoder_mask = "causal_doc"
        o_doc, _ = ours(xs2, ids, torch.float32, pos)
    check_fp32("fp32 logits, packed row with decoder_mask causal", o_causal, r_logits)
    check_fp32("fp32 logits, packed row causal_doc: 1st document unchanged", o_doc[:n1], r_logits[:n1])
    err = rel_err(o_doc[n1:], r_logits[n1:])[0]
    check("packed row causal_doc: 2nd document differs (documents isolated)", err > 1e-3, f"rel {err:.2e}")

    # --- bf16, deployed kernels ----------------------------------------------------
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.retie_weights()
    ref = ref.to(torch.bfloat16)
    with torch.no_grad():
        for name, (images, text) in cases.items():
            xs, ids = build(images, text)
            r_logits, _ = reference(xs, ids, torch.bfloat16)
            o_logits, _ = ours(xs, ids, torch.bfloat16)
            text_pos = ids[0] != pad_id
            check_bf16(f"bf16 logits, text positions ({name})", o_logits[text_pos], r_logits[text_pos])
    print("ElliotVL HF parity: all checks passed")


if __name__ == "__main__":
    test_elliot_hf_parity()
