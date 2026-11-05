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
from .state_dict_adapter import is_local_only_state, Qwen35StateDictAdapter

def build_meta(
    config_path: str | Path,
    *,
    seq_len: int,
    with_vision: bool = True,
    tp: int | None = None,
    enable_sp: bool = False,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
    attention_kind: str = "",
    num_mtp_layers: int = 0,
    moe_load_balance_coeff: float = 1e-3,
    ep: int = 1,
) -> Qwen35Model:
    """``config_path`` is an HF-format ``config.json`` or a snapshot directory; its
    ``model_type`` picks Qwen3.5, Qwen3-VL or text-only Qwen3.5. ``tp`` set (any
    value, 1 included) fills the sharding configs that ``Module.parallelize``
    applies; ``None`` builds a plain model.

    ``attention_kind`` only applies to ``qwen3_5_text`` -- "" or "hybrid" keeps the
    snapshot's schedule, "full_attention"/"softmax" and "linear_attention"/"gdn"
    make every layer that kind. Passing it to another ``model_type`` is an error
    rather than a silent no-op."""
    path = Path(config_path)
    model_type = json.loads((path / "config.json" if path.is_dir() else path).read_text())["model_type"]
    kwargs = dict(seq_len=seq_len, with_vision=with_vision, attn_backend=attn_backend,
                  decoder_mask=decoder_mask)
    if num_mtp_layers and model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(
            f"num_mtp_layers={num_mtp_layers} is only implemented for Qwen3.5; "
            f"this config is {model_type!r}"
        )
    if attention_kind and model_type != "qwen3_5_text":
        raise ValueError(
            f"attention_kind={attention_kind!r} needs model_type 'qwen3_5_text'; "
            f"this config is {model_type!r}"
        )
    # ponytail: dispatch lives in the Qwen3.5 package because every caller imports
    # it from here; move build_meta/materialize/load_hf to models/common at cutover
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        # One builder for both: `qwen35_config_from_hf` branches on `num_experts`
        # in text_config, and the model class is the same (the MoE lives in
        # Qwen35TransformerBlock.moe, which the dense path leaves None).
        config = qwen35_config_from_hf(
            config_path, **kwargs, num_mtp_layers=num_mtp_layers,
            moe_load_balance_coeff=moe_load_balance_coeff,
        )
        parallelism = apply_parallelism_config
    elif model_type == "qwen3_vl":
        from models.qwen3_vl.configs import apply_parallelism_config as parallelism
        from models.qwen3_vl.configs import qwen3_vl_config_from_hf

        config = qwen3_vl_config_from_hf(config_path, **kwargs)
    elif model_type == "qwen3_5_text":
        from models.qwen3_5_text.configs import apply_parallelism_config as parallelism
        from models.qwen3_5_text.configs import qwen35_text_config_from_hf

        # text-only by construction: with_vision is not its caller's to choose
        config = qwen35_text_config_from_hf(
            config_path, **{**kwargs, "with_vision": False},
            attention_kind=attention_kind or "hybrid",
        )
    else:
        raise NotImplementedError(
            f"no model for model_type {model_type!r}; "
            f"supported: qwen3_5, qwen3_vl, qwen3_5_text"
        )
    if ep > 1 and tp is None:
        # `tp=None` skips `parallelize` entirely, so an EP degree would be
        # accepted and then never wired onto anything
        raise ValueError(f"expert_parallel_degree={ep} needs tp set (1 is fine)")
    if tp is not None:
        if ep > 1:
            # only the Qwen3.5 builder takes `ep`; the others have no MoE
            if model_type not in ("qwen3_5", "qwen3_5_moe"):
                raise ValueError(
                    f"expert_parallel_degree={ep} needs a MoE model; "
                    f"this config is {model_type!r}"
                )
            parallelism(config, tp=tp, enable_sp=enable_sp, ep=ep)
        else:
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
    missing = {k for k in set(state_dict) - set(loaded) if not is_local_only_state(k)}
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
