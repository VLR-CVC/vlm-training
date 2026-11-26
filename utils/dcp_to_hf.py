"""DCP checkpoint of a training run -> HF safetensors snapshot.

    python utils/dcp_to_hf.py <checkpoint-step-N dir> <base HF snapshot> <out dir> [--config <run toml>]

Replaces the key-prefix conversion scripts for this model: torchtitan's module
tree splits HF's fused tensors (GDN in_proj_qkv / conv1d, ViT qkv) and renames
modules, so the state-dict adapter does the conversion. The base snapshot gives
the model config and supplies every non-weight file (config, tokenizer, templates).
Single process, CPU.

``--config`` is the run's TOML. Without it the model is rebuilt from the base
snapshot's own config.json, which is only right for a run that did not change
the architecture. With it the model is built the way the trainer built it:
`model.model_config` / `use_model_dir_config`, `attention_kind` and the
`engram*` fields.

Engram runs are exported as a text-only `trust_remote_code` snapshot
(`hf_remote/qwen3_5_engram`): the stock Qwen3.5 decoder weights under
`model.*`, the Engram weights under `model.engram.*`, and a config.json whose
`engram` dict reproduces the training hash layout. The remote-code files and
`models/common/engram_hash.py` are copied next to the weights, so the HF model
hashes with the exact code the trainer did.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.qwen3_5.checkpoint import build_meta, materialize
from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

REMOTE_CODE = REPO / "hf_remote" / "qwen3_5_engram"
REMOTE_FILES = (
    REMOTE_CODE / "configuration_qwen3_5_engram.py",
    REMOTE_CODE / "modeling_qwen3_5_engram.py",
    REPO / "models" / "common" / "engram_hash.py",
)

def _build_from_run_config(run_config: Path, base: Path):
    """The model exactly as `train/train_qwen.py` built it from this TOML."""
    from models.qwen3_5.configs import resolve_model_config
    from train.config import Config
    from train.config_manager import ConfigManager
    from train.utils import build_engram_args, build_lora

    # tyro also reads sys.argv, which here holds this script's own arguments
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        cfg = ConfigManager(Config).parse_args(["--config", str(run_config)])
    finally:
        sys.argv = argv
    margs = cfg.model
    config_path = resolve_model_config(margs.model_config, str(base), margs.use_model_dir_config)
    model = build_meta(
        config_path,
        seq_len=1024,  # only sizes RoPE caches
        attention_kind=margs.attention_kind,
        num_mtp_layers=margs.num_mtp_layers,
        engram=build_engram_args(margs),
    )
    build_lora(margs, model)  # LoRA runs: the DCP holds adapters; merged below
    return model

def _engram_config_json(model, base: Path) -> dict:
    """config.json of the remote-code model: the base text config, the trained
    layer schedule, and the Engram hash layout."""
    raw = json.loads((base / "config.json").read_text())
    text = dict(raw.get("text_config", raw))
    first = next(iter(model.engram.values()))
    layers = [int(k) for k in model.engram.keys()]
    text.update(
        architectures=["Qwen3_5EngramForCausalLM"],
        model_type="qwen3_5_engram",
        auto_map={
            "AutoConfig": "configuration_qwen3_5_engram.Qwen3_5EngramConfig",
            "AutoModelForCausalLM": "modeling_qwen3_5_engram.Qwen3_5EngramForCausalLM",
        },
        tie_word_embeddings=bool(model.enable_weight_tying),
        layer_types=[
            "full_attention" if layer.attention is not None else "linear_attention"
            for layer in model.config.layers
        ],
        engram={
            "layers": layers,
            "orders": list(first.spec.orders),
            "heads_per_order": first.spec.heads_per_order,
            "rows_per_head": first.rows_per_head,
            "dim": first.key_proj.in_features,
            "seed": first.seed,
            "gate_sqrt": first.gate_sqrt,
            "pad_id": first.spec.pad_id,
            "blocked_token_ids": list(first.blocked_token_ids),
        },
    )
    text.pop("full_attention_interval", None)
    return text

def convert(ckpt_dir: Path, base: Path, out: Path, dtype: torch.dtype = torch.bfloat16,
            run_config: Path | None = None) -> None:
    if run_config is not None:
        model = _build_from_run_config(run_config, base)
    else:
        model = build_meta(base, seq_len=1024)  # seq_len only sizes RoPE caches
    materialize(model, "cpu")
    model_sd = model.state_dict()
    # the trainer saves {"model": model, ...}; DCP loads a matching nested dict
    state = {"model": model_sd}
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    model.load_state_dict(state["model"])
    model.retie_weights()
    from models.common.lora import merge_lora

    merged = merge_lora(model)  # LoRA runs export a plain model: W += scale * B A
    if merged:
        print(f"merged LoRA adapters into {merged} linears")

    engram = model.engram is not None
    if engram and model.vision_encoder is not None:
        raise NotImplementedError(
            "the Engram remote-code model is text-only; train with a text model_config "
            "(model_type qwen3_5_text) to export it"
        )
    # an Engram export is a text-only CausalLM snapshot: `model.*` names whatever
    # the base snapshot uses, hence no snapshot path for the adapter
    adapter = Qwen35StateDictAdapter(model.config, None if engram else str(base))
    hf_sd = adapter.to_hf(model.state_dict())
    if engram:
        hf_sd.update(adapter.engram_to_hf(model.state_dict()))
    hf_sd = {k: v.to(dtype).contiguous() for k, v in hf_sd.items()}

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and not f.name.startswith("model.safetensors"):
            shutil.copy2(f, out / f.name)
    if engram:
        (out / "config.json").write_text(json.dumps(_engram_config_json(model, base), indent=2))
        for f in REMOTE_FILES:
            shutil.copy2(f, out / f.name)
    name = "model.safetensors"
    save_file(hf_sd, str(out / name), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in hf_sd.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: name for k in hf_sd}}, indent=2))
    print(f"wrote {len(hf_sd)} tensors, {total / 2**30:.2f} GiB -> {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="the run's TOML: rebuild the model the way the trainer did")
    a = parser.parse_args()
    convert(a.checkpoint, a.base, a.out, run_config=a.config)
