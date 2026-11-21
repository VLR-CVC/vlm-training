import re
from typing import Any

import torch

from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

_QWEN_LM = "model.language_model."
_ELLIOT_LM = "language_model.model."
_CHUNK_ORDER = (0, 2, 1, 3)

_VISION = {
    "vision_model.embeddings.position_embedding.weight": "vision_encoder.pos_embed",
    "vision_model.embeddings.patch_embedding.bias": "vision_encoder.patch_embed.bias",
    "vision_model.post_layernorm.weight": "vision_encoder.merger.norm.weight",
    "vision_model.post_layernorm.bias": "vision_encoder.merger.norm.bias",
    "patch_merger.fc1.bias": "vision_encoder.merger.linear_fc1.bias",
    "patch_merger.fc2.weight": "vision_encoder.merger.linear_fc2.weight",
    "patch_merger.fc2.bias": "vision_encoder.merger.linear_fc2.bias",
}
_VISION_LAYER = {
    "layer_norm1": "norm1", "layer_norm2": "norm2",
    "self_attn.q_proj": "attn.wq", "self_attn.k_proj": "attn.wk",
    "self_attn.v_proj": "attn.wv", "self_attn.out_proj": "attn.proj",
    "mlp.fc1": "mlp.linear_fc1", "mlp.fc2": "mlp.linear_fc2",
}
_PATCH_W = ("vision_model.embeddings.patch_embedding.weight", "vision_encoder.patch_embed.weight")
_FC1_W = ("patch_merger.fc1.weight", "vision_encoder.merger.linear_fc1.weight")

def _swap_chunks(w: torch.Tensor) -> torch.Tensor:
    # chunk/cat along the input dim only: FSDP shards dim 0, and DTensor has no
    # sharding rule for advanced indexing
    chunks = w.chunk(4, dim=1)
    return torch.cat([chunks[i] for i in _CHUNK_ORDER], dim=1)

class ElliotStateDictAdapter:
    def __init__(self, model_config, hf_assets_path: str | None = None):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        # the Qwen3.5 adapter with a vision tower uses the multimodal prefix
        self.lm = Qwen35StateDictAdapter(model_config, hf_assets_path)
        self.to_hf_vision = {v: k for k, v in _VISION.items()}

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        lm_sd = {k: v for k, v in state_dict.items() if not k.startswith("vision_encoder.")}
        hf = {k.replace(_QWEN_LM, _ELLIOT_LM, 1): v for k, v in self.lm.to_hf(lm_sd).items()}
        ve = self.model_config.vision_encoder
        for key, value in state_dict.items():
            if not key.startswith("vision_encoder."):
                continue
            if key == _PATCH_W[1]:
                frames = value.reshape(value.shape[0], ve.in_channels, ve.temporal_patch_size,
                                       ve.patch_size, ve.patch_size)
                hf[_PATCH_W[0]] = frames.sum(2)
            elif key == _FC1_W[1]:
                hf[_FC1_W[0]] = _swap_chunks(value)
            elif key in self.to_hf_vision:
                hf[self.to_hf_vision[key]] = value
            else:
                m = re.fullmatch(r"vision_encoder\.layers\.(\d+)\.(.+)\.(weight|bias)", key)
                ours = {v: k for k, v in _VISION_LAYER.items()}
                if m is None or m[2] not in ours:
                    raise KeyError(f"no ElliotVL counterpart for {key}")
                hf[f"vision_model.encoder.layers.{m[1]}.{ours[m[2]]}.{m[3]}"] = value
        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        lm_hf = {k.replace(_ELLIOT_LM, _QWEN_LM, 1): v for k, v in hf_state_dict.items()
                 if k.startswith(_ELLIOT_LM)}
        out = self.lm.from_hf(lm_hf)
        ve = self.model_config.vision_encoder
        for key, value in hf_state_dict.items():
            if key.startswith(_ELLIOT_LM):
                continue
            if key == _PATCH_W[0]:
                t = ve.temporal_patch_size
                frame = value.unsqueeze(2) / t
                out[_PATCH_W[1]] = torch.cat([frame] * t, dim=2).reshape(value.shape[0], -1)
            elif key == _FC1_W[0]:
                out[_FC1_W[1]] = _swap_chunks(value)
            elif key in _VISION:
                out[_VISION[key]] = value
            else:
                m = re.fullmatch(r"vision_model\.encoder\.layers\.(\d+)\.(.+)\.(weight|bias)", key)
                if m is None or m[2] not in _VISION_LAYER:
                    raise KeyError(f"unmapped ElliotVL tensor {key}")
                out[f"vision_encoder.layers.{m[1]}.{_VISION_LAYER[m[2]]}.{m[3]}"] = value
        return out
