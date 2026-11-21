"""Inference parity of our Qwen3.5 VLM against transformers, on a trained export.

    CUDA_VISIBLE_DEVICES=0 PARITY_SNAPSHOT=<HF export, e.g. utils/dcp_to_hf.py output> \
        python models/tests/test_hf_inference_parity.py

`Qwen3_5ForConditionalGeneration` is the reference: our exports load as that class
and vLLM mirrors it, so it is what evaluation runs. The question is whether our model
definition computes the same function from the same weights.

Two passes:

1. Definition parity, fp32. The three leaf kernels whose implementations must differ
   (softmax attention, the gated delta rule recurrence, the GDN causal conv) are
   swapped, on BOTH sides, for one shared fp32 torch reference. Everything else runs
   as written: weight conversion, projections, norms, gates, partial RoPE, MRoPE,
   vision position-embedding interpolation, 2D vision RoPE, patch merger, packing
   masks. Any gap here is a definition difference, not kernel rounding, so the gates
   are tight. Per-layer hidden states are compared to localise a failure.
2. Deployed kernels, bf16. Our real kernels (varlen flash attention, attn_gym GDN,
   causal_conv1d) against HF's in bf16, gated on top-1 agreement at decisive
   positions as in `test_qwen3_5_parity.py`.

Also checked: the export's config against the training model config (including
`use_rope`, which HF ignores and always treats as on), bitwise weight round trip,
MRoPE positions from `data/model_batch.py` against HF's `get_rope_index`, and a packed
row of two documents against each document run alone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SNAPSHOT = os.environ.get("PARITY_SNAPSHOT")
TRAIN_CONFIG = Path(os.environ.get(
    "PARITY_TRAIN_CONFIG", ROOT / "configs" / "models" / "qwen3_5_2b_so400m.json"))
IMAGE = ROOT / "assets" / "horse.png"
SEQ_LEN = 8192

# fp32 definition pass: relative to the reference's mean |value|
FP32_REL_TOL = float(os.environ.get("FP32_REL_TOL", "1e-4"))
# bf16 deployed pass, as in test_qwen3_5_parity.py
MEAN_TOL = float(os.environ.get("MEAN_TOL", "1e-1"))
TOP1_MIN = float(os.environ.get("TOP1_MIN", "0.97"))
NEAR_TIE = float(os.environ.get("NEAR_TIE", "1.0"))

# config keys that do not change the function the model computes
_NON_FUNCTIONAL = {"dtype", "torch_dtype", "transformers_version", "_name_or_path",
                   "architectures", "use_cache", "initializer_range", "id2label",
                   "label2id"}


# --------------------------------------------------------------------------
# shared fp32 reference kernels (used by both models in the definition pass)
# --------------------------------------------------------------------------

def ref_attention(q_THK, k_THK, v_THV, cu_seqlens, *, causal, scale):
    """Softmax attention inside each [cu[i], cu[i+1]) segment, fp32."""
    T = q_THK.shape[0]
    idx = torch.arange(T, device=q_THK.device)
    doc = torch.searchsorted(cu_seqlens[1:].to(idx.dtype), idx, right=True)
    mask = doc[:, None] == doc[None, :]
    if causal:
        mask &= idx[None, :] <= idx[:, None]
    q, k, v = (t.float().transpose(0, 1).unsqueeze(0) for t in (q_THK, k_THK, v_THV))
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out.squeeze(0).transpose(0, 1)


def ref_conv(x_CT, w_CW):
    """Depthwise causal conv + SiLU over one sequence, fp32."""
    C, T = x_CT.shape
    W = w_CW.shape[-1]
    out = F.conv1d(x_CT.float().unsqueeze(0), w_CW.float().unsqueeze(1),
                   padding=W - 1, groups=C)[0, :, :T]
    return F.silu(out)


def ref_gdn(q_THK, k_THK, v_THV, g_TH, beta_TH):
    """Gated delta rule, token by token, fp32 (l2-normalised q/k, scale K**-0.5)."""
    def l2(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q, k, v, g, beta = (t.float() for t in (q_THK, k_THK, v_THV, g_TH, beta_TH))
    H, K = v.shape[1], q.shape[-1]
    if q.shape[1] != H:
        q, k = q.repeat_interleave(H // q.shape[1], 1), k.repeat_interleave(H // k.shape[1], 1)
    q, k = l2(q) * K ** -0.5, l2(k)
    S = q.new_zeros(H, K, v.shape[-1])
    out = []
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        u = (v[t] - torch.einsum("hk,hkv->hv", k[t], S)) * beta[t][:, None]
        S = S + k[t][:, :, None] * u[:, None, :]
        out.append(torch.einsum("hk,hkv->hv", q[t], S))
    return torch.stack(out)


def _segments(cu):
    cu = cu.tolist()
    return [(s, e) for s, e in zip(cu[:-1], cu[1:]) if e > s]


class reference_kernels:
    """Context manager: route both models' leaf kernels to the shared references."""

    def __enter__(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_mod
        from models.common import attention as our_attn
        from models.qwen3_5 import gdn as our_gdn

        def our_varlen(mod, q, k, v, *, attention_masks, scale=None, enable_gqa=False):
            causal = mod.window_size[1] == 0
            s = q.shape[-1] ** -0.5 if scale is None else scale
            return ref_attention(q, k, v, attention_masks.cu_seq_q, causal=causal,
                                 scale=s).to(q.dtype)

        def our_conv(x_TD, weight, cu_seqlens):
            w = weight.squeeze(1)
            out = torch.empty_like(x_TD, dtype=torch.float32)
            for s, e in _segments(cu_seqlens):
                out[s:e] = ref_conv(x_TD[s:e].T, w).T
            return out.to(x_TD.dtype)

        def our_kernel(mod, q, k, v, g, beta, *, cu_seqlens=None):
            out = q.new_empty(v.shape, dtype=torch.float32)
            segs = _segments(cu_seqlens) if cu_seqlens is not None else [(0, q.shape[0])]
            for s, e in segs:
                out[s:e] = ref_gdn(q[s:e], k[s:e], v[s:e], g[s:e], beta[s:e])
            return out.to(q.dtype)

        def hf_conv(hidden_states, weight, bias=None, activation=None, **kwargs):
            assert bias is None and activation == "silu"
            return torch.stack([ref_conv(x, weight) for x in hidden_states]).to(hidden_states.dtype)

        def hf_gdn(query, key, value, g, beta, initial_state=None, output_final_state=False,
                   use_qk_l2norm_in_kernel=False, **kwargs):
            assert use_qk_l2norm_in_kernel and initial_state is None
            out = torch.stack([ref_gdn(*(t[b] for t in (query, key, value, g, beta)))
                               for b in range(query.shape[0])])
            return out.to(query.dtype), None

        self._saved = [
            (our_attn.VarlenInnerAttention, "forward", our_attn.VarlenInnerAttention.forward),
            (our_gdn, "_causal_conv1d_varlen", our_gdn._causal_conv1d_varlen),
            (our_gdn.GatedDeltaKernel, "forward", our_gdn.GatedDeltaKernel.forward),
            (hf_mod, "causal_conv1d_fn", hf_mod.causal_conv1d_fn),
            (hf_mod, "torch_chunk_gated_delta_rule", hf_mod.torch_chunk_gated_delta_rule),
        ]
        our_attn.VarlenInnerAttention.forward = our_varlen
        our_gdn._causal_conv1d_varlen = our_conv
        our_gdn.GatedDeltaKernel.forward = our_kernel
        hf_mod.causal_conv1d_fn = hf_conv
        hf_mod.torch_chunk_gated_delta_rule = hf_gdn
        return self

    def __exit__(self, *exc):
        for obj, name, value in self._saved:
            setattr(obj, name, value)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def rel_err(ours, ref):
    ours, ref = ours.float(), ref.float()
    return ((ours - ref).abs().mean() / ref.abs().mean()).item(), (ours - ref).abs().max().item()


def check(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


def check_fp32(name, ours, ref, tol=FP32_REL_TOL):
    rel, mx = rel_err(ours, ref)
    top1 = ""
    if ours.ndim == 2 and ours.shape[-1] > 100_000:  # logits
        flips = ours.argmax(-1) != ref.argmax(-1)
        top2 = ref.float().topk(2, dim=-1).values
        gaps = (top2[:, 0] - top2[:, 1])[flips].tolist()
        # a flip is only a definition error if the reference was decisive; fp32
        # reordering noise is ~max|d| above, so ties inside that are coin flips
        bad = [g for g in gaps if g > 10 * mx]
        msg = (f"rel mean|d|={rel:.2e} max|d|={mx:.2e} top1 flips={int(flips.sum())}/"
               f"{flips.numel()} (top-2 gaps at flips: {[f'{g:.1e}' for g in gaps[:5]]})")
        check(name, rel <= tol and not bad, msg)
    else:
        check(name, rel <= tol, f"rel mean|d|={rel:.2e} max|d|={mx:.2e}")


def check_bf16(name, ours, ref):
    ours, ref = ours.float(), ref.float()
    d = (ours - ref).abs()
    top2 = ref.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > NEAR_TIE
    agree = ours.argmax(-1) == ref.argmax(-1)
    top1 = agree[decisive].float().mean().item() if decisive.any() else 1.0
    check(name, d.mean().item() <= MEAN_TOL and top1 >= TOP1_MIN,
          f"mean|dlogit|={d.mean().item():.3e} max={d.max().item():.3e} top1={top1:.4f} "
          f"(n={int(decisive.sum())} decisive of {ours.shape[0]})")


def functional(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _NON_FUNCTIONAL}


class LayerTaps:
    """Record each decoder layer's output on both models."""

    def __init__(self, ours, hf):
        self.ours, self.hf, self.handles = [], [], []
        for layer in ours.layers.values():
            self.handles.append(layer.register_forward_hook(
                lambda m, i, o: self.ours.append(o.detach())))
        for layer in hf.model.language_model.layers:
            self.handles.append(layer.register_forward_hook(
                lambda m, i, o: self.hf.append((o[0] if isinstance(o, tuple) else o)[0].detach())))

    def report(self):
        worst = max(range(len(self.ours)), key=lambda i: rel_err(self.ours[i], self.hf[i])[0])
        rel, mx = rel_err(self.ours[worst], self.hf[worst])
        first_bad = next((i for i in range(len(self.ours))
                          if rel_err(self.ours[i], self.hf[i])[0] > FP32_REL_TOL), None)
        print(f"[INFO]   per-layer: worst layer {worst} rel {rel:.2e} max {mx:.2e}; "
              f"first layer over tolerance: {first_bad}")
        self.ours.clear(), self.hf.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT or not torch.cuda.is_available(),
                    reason="needs PARITY_SNAPSHOT (an HF export of our model) and a GPU")
def test_hf_inference_parity():
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from data.model_batch import mrope_positions
    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize
    from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snap = Path(SNAPSHOT)
    raw = json.loads((snap / "config.json").read_text())
    image_id, video_id = raw["image_token_id"], raw["video_token_id"]

    # --- config: what evaluation reads vs what training built -----------------
    vc = raw["vision_config"]
    check("export vision RoPE on", vc.get("use_rope", True) is True,
          f"use_rope={vc.get('use_rope', True)} (HF/vLLM ignore the key and always apply "
          "the 2D RoPE, so a use_rope=false model cannot be served as trained)")
    if TRAIN_CONFIG.is_file():
        train = json.loads(TRAIN_CONFIG.read_text())
        for part in ("text_config", "vision_config"):
            a, b = functional(raw[part]), functional(train[part])
            b.pop("use_rope", None), a.pop("use_rope", None)
            diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
            check(f"export {part} == training config", not diff,
                  f"differs on {diff}" if diff else str(TRAIN_CONFIG.name))
    else:
        print(f"[SKIP] training config {TRAIN_CONFIG} not found")
    pp = json.loads((snap / "preprocessor_config.json").read_text())
    proc_geom = (pp.get("patch_size"), pp.get("merge_size"), pp.get("temporal_patch_size"))
    model_geom = (vc["patch_size"], vc["spatial_merge_size"], vc["temporal_patch_size"])
    check("processor geometry == vision config", proc_geom == model_geom,
          f"processor {proc_geom}, model {model_geom}")

    # --- our model, fp32, and bitwise weight round trip ------------------------
    model = build_meta(snap, seq_len=SEQ_LEN)
    materialize(model, dev)
    load_hf(model, snap)
    model.eval()
    from safetensors import safe_open
    hf_sd = Qwen35StateDictAdapter(model.config, str(snap)).to_hf(model.state_dict())
    mismatched, stored = [], set()
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for key in s.keys():
                stored.add(key)
                if key in hf_sd and not torch.equal(hf_sd[key].to(s.get_tensor(key).dtype).cpu(),
                                                    s.get_tensor(key)):
                    mismatched.append(key)
    unused = sorted(k for k in stored - set(hf_sd) if not k.startswith("mtp."))
    check("weights: to_hf(ours) == export, bitwise", not mismatched and not unused,
          f"{len(hf_sd)} tensors; mismatched {mismatched[:3]} unused {unused[:3]}")

    hf = Qwen3_5ForConditionalGeneration.from_pretrained(
        snap, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    proc = AutoProcessor.from_pretrained(snap)

    def ours_forward(ids, positions, mrope, pixel_values=None, grid_thw=None):
        inputs, _, extra = model.preprocess_inputs(
            {"input": ids, "labels": ids, "positions": positions, "mrope_positions": mrope})
        return model(inputs, pixel_values=pixel_values, grid_thw=grid_thw,
                     special_tokens={"image_id": image_id}, **extra)

    def prompt(images, text):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        chat = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True)
        return proc(text=[chat], images=images or None, return_tensors="pt").to(dev)

    horse = Image.open(IMAGE).convert("RGB")
    cases = {
        "text only": prompt([], "The capital of France is Paris, and the capital of Germany is"),
        "one image 448x448": prompt([horse.resize((448, 448))],
                                    "Describe this image in detail."),
        "two images, non-square": prompt(
            [horse.resize((640, 384)), horse.crop((0, 0, horse.width // 2, horse.height)).resize((320, 704))],
            "Compare the two images."),
        "one image 1024x768": prompt([horse.resize((1024, 768))], "What animal is this?"),
    }

    def our_positions(mm):
        ids = mm["input_ids"][0]
        grid = mm.get("image_grid_thw")
        mrope = mrope_positions(ids.cpu(), torch.tensor([0, ids.shape[0]]),
                                grid.cpu() if grid is not None else None,
                                image_token_id=image_id, video_token_id=video_id,
                                spatial_merge_size=vc["spatial_merge_size"]).to(dev)
        return ids, torch.arange(ids.shape[0], device=dev), mrope

    # --- MRoPE positions: our data pipeline vs HF get_rope_index ---------------
    for name, mm in cases.items():
        ids, _, mrope = our_positions(mm)
        ref, _ = hf.model.get_rope_index(mm["input_ids"], mm["mm_token_type_ids"],
                                         image_grid_thw=mm.get("image_grid_thw"))
        ref = ref[:, 0].transpose(0, 1)
        check(f"MRoPE positions ({name})", torch.equal(mrope, ref),
              f"{int((mrope != ref).any(-1).sum())} of {ids.shape[0]} tokens differ")

    # --- pass 1: definition parity, fp32, shared reference kernels --------------
    taps = LayerTaps(model, hf)
    with reference_kernels(), torch.no_grad():
        for name, mm in cases.items():
            if "pixel_values" in mm:
                v_ref = hf.model.visual(mm["pixel_values"], grid_thw=mm["image_grid_thw"])
                v_ref = getattr(v_ref, "pooler_output", v_ref)
                v_ours = model.vision_encoder(mm["pixel_values"], grid_thw=mm["image_grid_thw"])
                check_fp32(f"fp32 vision tower ({name})", v_ours, v_ref)
            ref = hf(**mm, use_cache=False).logits[0]
            ids, pos, mrope = our_positions(mm)
            ours = ours_forward(ids, pos, mrope, mm.get("pixel_values"), mm.get("image_grid_thw"))
            check_fp32(f"fp32 logits ({name})", ours, ref)
            taps.report()

        # packed row [two-image doc | text doc] vs each document alone
        a, b = cases["two images, non-square"], cases["text only"]
        ia, pa, ma = our_positions(a)
        ib, pb, mb = our_positions(b)
        packed = ours_forward(torch.cat([ia, ib]), torch.cat([pa, pb]), torch.cat([ma, mb]),
                              a["pixel_values"], a["image_grid_thw"])
        ref_a, ref_b = hf(**a, use_cache=False).logits[0], hf(**b, use_cache=False).logits[0]
        n = ia.shape[0]
        check_fp32("fp32 packed row, doc 1 vs HF alone", packed[:n], ref_a)
        check_fp32("fp32 packed row, doc 2 vs HF alone", packed[n:], ref_b)
    taps.remove()

    # --- pass 2: deployed kernels, bf16 -----------------------------------------
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.retie_weights()
    hf = hf.to(torch.bfloat16)
    with torch.no_grad():
        for name, mm in cases.items():
            pv = mm.get("pixel_values")
            pv = pv.to(torch.bfloat16) if pv is not None else None
            ref = hf(**{**mm, **({"pixel_values": pv} if pv is not None else {})},
                     use_cache=False).logits[0]
            ids, pos, mrope = our_positions(mm)
            ours = ours_forward(ids, pos, mrope, pv, mm.get("image_grid_thw"))
            text = ids != image_id
            check_bf16(f"bf16 logits, text positions ({name})", ours[text], ref[text])
    print("HF inference parity: all checks passed")


if __name__ == "__main__":
    test_hf_inference_parity()
