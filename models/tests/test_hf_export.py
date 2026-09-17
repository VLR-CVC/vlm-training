"""S2 gate (TITAN_MIGRATION_v2.md): a trained DCP checkpoint exports to HF.

    python models/tests/test_titan_hf_export.py <checkpoint-step-N> <base snapshot> <exported snapshot>

The exported snapshot (utils/titan_to_hf.py) must load in `transformers`, differ
from the base weights (it trained), and give the same predictions as our model
loaded straight from the DCP checkpoint.
"""

import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5_tt.checkpoint import build_meta, materialize

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def main(ckpt: Path, base: Path, exported: Path) -> None:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    with safe_open(str(base / "model.safetensors-00001-of-00001.safetensors"), "pt") as f:
        w_base = f.get_tensor(key)
    with safe_open(str(exported / "model.safetensors"), "pt") as f:
        w_new = f.get_tensor(key)
    moved = (w_new.float() - w_base.float()).abs().max().item()
    print(f"[{'PASS' if moved > 0 else 'FAIL'}] exported weights moved from base: max|dw|={moved:.3e}")
    assert moved > 0

    model = build_meta(base, seq_len=1024)
    materialize(model, "cuda")
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(ckpt))
    model.load_state_dict(state["model"])
    model.retie_weights()
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.retie_weights()
    model.eval()

    hf = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(exported), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = AutoTokenizer.from_pretrained(str(base))
    ids = tok("A photo of a horse standing in a field next to", return_tensors="pt").input_ids.cuda()
    pos = torch.arange(ids.shape[1], device="cuda")
    with torch.no_grad():
        ref = hf(input_ids=ids).logits[0].float()
        inputs, _, extra = model.preprocess_inputs({"input": ids[0], "labels": ids[0], "positions": pos})
        ours = model(inputs, **extra).float()
    top1 = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    mean = (ours - ref).abs().mean().item()
    ok = top1 == 1.0 and mean < 0.1
    print(f"[{'PASS' if ok else 'FAIL'}] HF(exported) vs ours(DCP): top1={top1:.4f} mean|dlogit|={mean:.3e}")
    assert ok


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
