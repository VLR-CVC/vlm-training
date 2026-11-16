"""Does in-fusion survive tensor parallelism with sequence parallelism?

    torchrun --nproc_per_node=1 models/tests/test_infusion_tp.py tp1
    torchrun --nproc_per_node=2 models/tests/test_infusion_tp.py tp2sp

The specific doubt. The modality mask built in `Qwen35Model._vision_mask` is one
`(T, 1)` vector over the whole packed row, but under sequence parallelism the
input to the FFN region is sharded over tokens, so each rank holds `T/tp` rows.
Nothing in the mask's construction knows that. Either it breaks loudly on shape,
or -- worse -- it lines up by accident against the wrong tokens.

The attention twins are not in doubt for the same reason: the attention region
gathers tokens before the projections, so a full-length mask is right there. It is
the Eq. 3 branch, inside the sequence-parallel region, that this exists to check.

What each mode proves:

  tp1     the mask selects exactly the image rows, checked row by row. This is the
          ground truth the sharded run has to reproduce.
  tp2sp   the forward and backward run at all, gradients reach the vision-only
          parameters, and the routing is still live (masking changes the result).
          Row-level alignment is not re-checked here because the hidden states come
          back sharded; `tp1` carries that.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG = os.environ.get("INFUSION_CONFIG", "configs/models/qwen3_5_2b_infusion.json")
MODES = {"tp1": (1, False), "tp2": (2, False), "tp2sp": (2, True)}
IMAGE_ID, VIDEO_ID = 151655, 151656  # overwritten from the config below
SEQ = 512
IMAGE_SPAN = slice(64, 192)


def say(*a):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*a, flush=True)


def build(tp: int, sp: bool):
    from models.qwen3_5.checkpoint import build_meta, materialize
    from train.parallel.parallel_dims import ParallelDims
    from train.parallel.parallelize import parallelize_qwen3_5
    from train.parallel.spmd import set_spmd_meshes

    world = torch.distributed.get_world_size()
    pd = ParallelDims(dp_replicate=1, dp_shard=world // tp, cp=1, tp=tp, pp=1, ep=1,
                      world_size=world)
    pd.build_mesh()
    model = build_meta(CONFIG, seq_len=4096, tp=tp, enable_sp=sp, with_vision=True)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    parallelize_qwen3_5(model, pd, mode="fsdp", compile=False,
                        param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    materialize(model, "cuda")
    torch.manual_seed(0)
    model.init_states(buffer_device=torch.device("cuda"))
    set_spmd_meshes(dense_mesh=pd.spmd_dense_mesh(), sparse_mesh=None)
    model._skip_lm_head = True

    # The branches are zero-initialised, which is the point of the design and
    # useless for this test: a zero branch routes nothing and every mask bug hides.
    touched = 0
    for name, p in model.named_parameters():
        if "vision_feed_forward.w2" in name or "vision_wo" in name:
            with torch.no_grad():
                p.data.normal_(mean=0.0, std=0.05)
            touched += 1
    assert touched, "no in-fusion parameters found: wrong config?"
    say(f"[build] tp={tp} sp={sp} world={world}, perturbed {touched} vision tensors")
    return model


def batch():
    tokens = torch.arange(SEQ, device="cuda") % 97 + 1000
    tokens[IMAGE_SPAN] = IMAGE_ID
    positions = torch.arange(SEQ, device="cuda")
    return tokens, positions


def forward(model, tokens, positions, special):
    masks = model.get_attention_masks(positions)
    return model(tokens, attention_masks=masks, positions=positions,
                 special_tokens=special)


def main(mode: str) -> None:
    tp, sp = MODES[mode]
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.autograd.set_multithreading_enabled(False)

    import json
    cfg = json.loads(Path(CONFIG).read_text())
    global IMAGE_ID, VIDEO_ID
    IMAGE_ID = cfg.get("image_token_id", IMAGE_ID)
    VIDEO_ID = cfg.get("video_token_id", VIDEO_ID)
    special = {"image_id": IMAGE_ID, "video_id": VIDEO_ID}

    model = build(tp, sp)
    tokens, positions = batch()

    with torch.no_grad():
        on = forward(model, tokens, positions, special)
        off = forward(model, tokens, positions, None)

    # Routing has to be live, or everything below passes vacuously.
    delta = (on.float() - off.float()).abs().sum()
    if torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(delta)
    assert delta > 0, "masking changed nothing: the branches are not being applied"
    say(f"[routing] |on - off| = {delta.item():.3e}")

    if tp == 1:
        moved = (on.float() - off.float()).abs().sum(-1) > 0
        expect = tokens == IMAGE_ID
        first = int(expect.nonzero()[0])
        # Rows after the first image may legitimately move: causal attention lets
        # them see a vision token that moved. Rows before it may not.
        assert moved[:first].sum() == 0, (
            f"{int(moved[:first].sum())} text rows before the first image moved: "
            "the mask is selecting the wrong tokens"
        )
        assert moved[expect].all(), "some image rows did not move"
        say(f"[align] rows {first}..{int(expect.nonzero()[-1])} move, "
            f"0..{first} do not -- mask lines up with the image span")

    out = forward(model, tokens, positions, special)
    out.float().pow(2).mean().backward()
    checked = 0
    for name, p in model.named_parameters():
        if "vision_feed_forward" in name or "vision_w" in name:
            g = p.grad
            assert g is not None, f"no gradient for {name}"
            local = g.to_local() if hasattr(g, "to_local") else g
            assert torch.isfinite(local).all(), f"non-finite gradient in {name}"
            checked += 1
    say(f"[backward] finite gradients on {checked} in-fusion tensors")
    say(f"PASS {mode}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tp1")
