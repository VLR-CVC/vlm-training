"""DCP checkpoint of a training run -> HF safetensors snapshot.

    python utils/dcp_to_hf.py <checkpoint-step-N dir> <base HF snapshot> <out dir>

Replaces the key-prefix conversion scripts for this model: torchtitan's module
tree splits HF's fused tensors (GDN in_proj_qkv / conv1d, ViT qkv) and renames
modules, so the state-dict adapter does the conversion. The base snapshot gives
the model config and supplies every non-weight file (config, tokenizer, templates).
Single process, CPU.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.qwen3_5_tt.checkpoint import build_meta, materialize
from models.qwen3_5_tt.state_dict_adapter import Qwen35StateDictAdapter


def convert(ckpt_dir: Path, base: Path, out: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    model = build_meta(base, seq_len=1024)  # seq_len only sizes RoPE caches
    materialize(model, "cpu")
    model_sd = model.state_dict()
    # the trainer saves {"model": model, ...}; DCP loads a matching nested dict
    state = {"model": model_sd}
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    model.load_state_dict(state["model"])
    model.retie_weights()

    hf_sd = Qwen35StateDictAdapter(model.config, str(base)).to_hf(model.state_dict())
    hf_sd = {k: v.to(dtype).contiguous() for k, v in hf_sd.items()}

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and not f.name.startswith("model.safetensors"):
            shutil.copy2(f, out / f.name)
    name = "model.safetensors"
    save_file(hf_sd, str(out / name), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in hf_sd.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: name for k in hf_sd}}, indent=2))
    print(f"wrote {len(hf_sd)} tensors, {total / 2**30:.2f} GiB -> {out}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
