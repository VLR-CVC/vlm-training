from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor, distribute_tensor

from train.logger import logger

from models.qwen3_5.model import Qwen35Model
from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


@torch.no_grad()
def load_text_weights(model: Qwen35Model, snapshot: str | Path) -> None:
    """Fill the decoder from a text-only HF snapshot (e.g. Qwen3-1.7B)"""
    adapter = Qwen35StateDictAdapter(replace(model.config, vision_encoder=None), str(snapshot))
    decoder = {k: v for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}

    hf_state_dict = adapter.to_hf(decoder)
    dcp.load(hf_state_dict, storage_reader=HuggingFaceStorageReader(str(snapshot)))
    loaded = adapter.from_hf(hf_state_dict)

    missing = set(decoder) - set(loaded)
    if model.enable_weight_tying:
        missing.discard("tok_embeddings.weight")
    if missing:
        raise KeyError(
            f"text snapshot {snapshot} did not provide: {sorted(missing)[:10]} ... "
            "(is it the matching architecture? the decoder must have the same depth, "
            "width and head layout as the VLM's text config)"
        )
    model.load_state_dict(loaded, strict=False)
    model.retie_weights()
    logger.info(f"loaded {len(loaded)} decoder tensors from text-only snapshot {snapshot}")


# SigLIP2 block -> our ViT block. q/k/v stay separate on both sides, so unlike the
# pre-port version there is nothing to fuse.
_BLOCK_MAP = {
    "layer_norm1.weight": "norm1.weight",
    "layer_norm1.bias": "norm1.bias",
    "layer_norm2.weight": "norm2.weight",
    "layer_norm2.bias": "norm2.bias",
    "self_attn.q_proj.weight": "attn.wq.weight",
    "self_attn.q_proj.bias": "attn.wq.bias",
    "self_attn.k_proj.weight": "attn.wk.weight",
    "self_attn.k_proj.bias": "attn.wk.bias",
    "self_attn.v_proj.weight": "attn.wv.weight",
    "self_attn.v_proj.bias": "attn.wv.bias",
    "self_attn.out_proj.weight": "attn.proj.weight",
    "self_attn.out_proj.bias": "attn.proj.bias",
    "mlp.fc1.weight": "mlp.linear_fc1.weight",
    "mlp.fc1.bias": "mlp.linear_fc1.bias",
    "mlp.fc2.weight": "mlp.linear_fc2.weight",
    "mlp.fc2.bias": "mlp.linear_fc2.bias",
}

# SigLIP2 weights with no equivalent here: both belong to the contrastive pooling
# head, which a VLM does not use.
_SKIP_PREFIXES = ("post_layernorm.", "head.")


@torch.no_grad()
def load_siglip_vision(model: Qwen35Model, snapshot: str | Path) -> None:
    """Transfer a SigLIP2 vision tower into our ViT.

    Two weights need surgery rather than a copy:

    * **patch embedding.** SigLIP2 has a Conv2d over one frame; we have a Linear over
      a pre-extracted `C*T*P*P` patch. Inflate along the temporal axis by repeating
      and dividing by `temporal_patch_size` -- that keeps the response magnitude of a
      static image, where all T frames are identical -- then flatten to the Linear
      layout the state-dict adapter uses.
    * **position embedding.** Bilinearly resampled from SigLIP2's grid to ours.

    The merger and DeepStack mergers are deliberately untouched.
    """
    from transformers import SiglipVisionModel

    encoder = model.vision_encoder
    if encoder is None:
        raise ValueError("model has no vision encoder to load into")

    siglip = SiglipVisionModel.from_pretrained(str(snapshot), local_files_only=True)
    # `SiglipVisionModel.state_dict()` drops the `vision_model.` prefix the checkpoint
    # files carry; strip it defensively so either spelling works.
    src = {k.removeprefix("vision_model."): v for k, v in siglip.state_dict().items()}
    params = dict(model.named_parameters())
    loaded: list[str] = []

    def copy_to(name: str, tensor: torch.Tensor) -> None:
        param = params.get(name)
        if param is None:
            raise KeyError(f"no such parameter: {name}")
        # `param.shape` is the global shape on a DTensor too, so this compares like
        # with like whether or not the model has been sharded already.
        if param.shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {name}: model wants {tuple(param.shape)}, "
                f"SigLIP2 gives {tuple(tensor.shape)}. The towers must agree on "
                "hidden size, patch size and MLP width -- check the SigLIP2 variant."
            )
        source = tensor.to(dtype=param.dtype, device=param.device)
        if isinstance(param, DTensor):
            source = distribute_tensor(source, param.device_mesh, param.placements)
        param.copy_(source)
        loaded.append(name)

    cfg = encoder.config
    # --- patch embedding: Conv2d -> inflated Conv3d -> Linear ------------------
    pe = src["embeddings.patch_embedding.weight"].float()
    if pe.shape[-1] != cfg.patch_size:
        raise ValueError(
            f"patch size differs: SigLIP2 {pe.shape[-1]}, model {cfg.patch_size}"
        )
    t = cfg.temporal_patch_size
    copy_to("vision_encoder.patch_embed.weight",
            (pe.unsqueeze(2).repeat(1, 1, t, 1, 1) / t).reshape(pe.shape[0], -1))
    copy_to("vision_encoder.patch_embed.bias",
            src["embeddings.patch_embedding.bias"])

    # --- position embedding: bilinear resample to our grid ---------------------
    pos = src["embeddings.position_embedding.weight"].float()
    src_n, dim = pos.shape
    tgt_n = params["vision_encoder.pos_embed"].shape[0]
    if src_n != tgt_n:
        src_g, tgt_g = round(src_n**0.5), round(tgt_n**0.5)
        logger.info(f"interpolating SigLIP2 pos_embed {src_g}x{src_g} -> {tgt_g}x{tgt_g}")
        pos = F.interpolate(
            pos.reshape(1, src_g, src_g, dim).permute(0, 3, 1, 2),
            size=(tgt_g, tgt_g), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(tgt_n, dim)
    copy_to("vision_encoder.pos_embed", pos)

    # --- blocks ----------------------------------------------------------------
    n_src = len({int(k.split(".")[2]) for k in src if k.startswith("encoder.layers.")})
    n_dst = len(encoder.layers)
    if n_src != n_dst:
        raise ValueError(f"depth differs: SigLIP2 has {n_src} blocks, model wants {n_dst}")
    for i in range(n_dst):
        for src_suffix, dst_suffix in _BLOCK_MAP.items():
            copy_to(f"vision_encoder.layers.{i}.{dst_suffix}",
                    src[f"encoder.layers.{i}.{src_suffix}"])

    skipped = [k for k in src if k.startswith(_SKIP_PREFIXES)]
    fresh = sorted(n for n in params
                   if n.startswith("vision_encoder.") and n not in set(loaded))
    logger.info(
        f"SigLIP2 -> ViT: loaded {len(loaded)} tensors across {n_dst} blocks, "
        f"skipped {len(skipped)} pooling-head tensors, left {len(fresh)} randomly "
        f"initialised (merger + DeepStack): {fresh[:4]}{' ...' if len(fresh) > 4 else ''}"
    )