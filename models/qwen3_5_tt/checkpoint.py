"""Build Qwen3.5 on the meta device and load an HF snapshot into it.

The load is torchtitan's HF path (``components/checkpointer/dcp.py``,
``from_hf=True``): take the model's own state dict, convert its keys to HF with
the adapter, fill those tensors from the safetensors with DCP's
``HuggingFaceStorageReader``, convert back and ``load_state_dict``. Nothing is
ever materialised as a second full copy on the host.
"""

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader

from .configs import apply_parallelism_config, qwen35_config_from_hf
from .model import Qwen35Model
from .state_dict_adapter import Qwen35StateDictAdapter

def build_meta(
    config_path: str | Path,
    *,
    seq_len: int,
    with_vision: bool = True,
    tp: int | None = None,
    enable_sp: bool = False,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
) -> Qwen35Model:
    """``config_path`` is an HF-format ``config.json`` or a snapshot directory; its
    ``model_type`` picks Qwen3.5 or Qwen3-VL. ``tp`` set (any value, 1 included)
    fills the sharding configs that ``Module.parallelize`` applies; ``None`` builds
    a plain model."""
    path = Path(config_path)
    model_type = json.loads((path / "config.json" if path.is_dir() else path).read_text())["model_type"]
    kwargs = dict(seq_len=seq_len, with_vision=with_vision, attn_backend=attn_backend,
                  decoder_mask=decoder_mask)
    # ponytail: dispatch lives in the Qwen3.5 package because every caller imports
    # it from here; move build_meta/materialize/load_hf to models/common at cutover
    if model_type == "qwen3_5":
        config = qwen35_config_from_hf(config_path, **kwargs)
        parallelism = apply_parallelism_config
    elif model_type == "qwen3_vl":
        from models.qwen3_vl_tt.configs import apply_parallelism_config as parallelism
        from models.qwen3_vl_tt.configs import qwen3_vl_config_from_hf

        config = qwen3_vl_config_from_hf(config_path, **kwargs)
    else:
        raise NotImplementedError(f"no model for model_type {model_type!r}; supported: qwen3_5, qwen3_vl")
    if tp is not None:
        parallelism(config, tp=tp, enable_sp=enable_sp)
    with torch.device("meta"):
        return config.build()

def materialize(model: Qwen35Model, device: torch.device | str) -> None:
    """``to_empty`` then recompute buffers (RoPE caches, ViT ``inv_freq``)
    and restore the weight tie that ``to_empty`` breaks."""
    model.to_empty(device=device)
    model.retie_weights()
    with torch.no_grad():
        for module in model.modules():
            init_buffers = getattr(module, "_init_self_buffers", None)
            if init_buffers is not None:
                init_buffers(buffer_device=torch.device(device))

@torch.no_grad()
def load_hf(model: Qwen35Model, snapshot: str | Path) -> None:
    """Fill every parameter from an HF snapshot; strict about missing keys."""
    adapter = Qwen35StateDictAdapter(model.config, str(snapshot))
    state_dict = model.state_dict()
    hf_state_dict = adapter.to_hf(state_dict)
    dcp.load(hf_state_dict, storage_reader=HuggingFaceStorageReader(str(snapshot)))
    loaded = adapter.from_hf(hf_state_dict)
    missing = set(state_dict) - set(loaded)
    if model.enable_weight_tying:
        missing.discard("tok_embeddings.weight")
    if missing:
        raise KeyError(f"HF snapshot did not provide: {sorted(missing)[:10]} ...")
    model.load_state_dict(loaded, strict=False)
    model.retie_weights()

def from_pretrained(
    snapshot: str | Path,
    *,
    seq_len: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    with_vision: bool = True,
) -> Qwen35Model:
    model = build_meta(snapshot, seq_len=seq_len, with_vision=with_vision)
    materialize(model, device)
    load_hf(model, snapshot)
    return model.to(dtype)
