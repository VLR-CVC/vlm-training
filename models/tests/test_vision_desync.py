"""The deadlock test: one rank has images, the other does not.

This is the MN5 hang, reduced. The vision tower is one FSDP unit and FSDP posts
its collectives from autograd hooks on the parameters. A rank whose micro-batch
contains no images used to skip the tower entirely, so those hooks never fired
while the other rank sat in the matching collective: 100% SM, low power, no
output, at whichever step the first image-free micro-batch landed.

Before the fix this script HANGS. After it, it completes and both ranks report
finite vision gradients. Run it under a timeout so a regression fails instead of
hanging a machine:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5,6 timeout 600 \\
      python -m torch.distributed.run --nproc_per_node=2 models/tests/test_vision_desync.py

Exit 124 from `timeout` means the deadlock is back.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.checkpoint import build_meta, load_hf, materialize  # noqa: E402
from train.parallel.fsdp import apply_data_parallel  # noqa: E402
from train.precompile import synthetic_batch  # noqa: E402
from train.step import forward_backward  # noqa: E402

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
)
SEQ_LEN = 2048
IMAGE_TOKEN_ID = 151655

def _instrument_all_reduce() -> None:
    """SPMD_PROBE=1: report the tensor each all_reduce sees, and force contiguity.

    The failure is `all_reduce input must be contiguous` inside
    `_InvariantToReplicate.backward`, which the C++ autograd engine calls, so the
    forward site never appears in the traceback. Wrapping the collective is the
    only way to see the shape/stride -- and if forcing contiguity makes the run
    complete, contiguity is the whole problem.
    """
    import spmd_types._collectives as C

    inner = C.all_reduce
    seen = [0]

    def wrapped(x, *a, **kw):
        if not x.is_contiguous() and seen[0] < 5:
            seen[0] += 1
            print(
                f"[probe] non-contiguous all_reduce input: shape={tuple(x.shape)} "
                f"stride={x.stride()} dtype={x.dtype}",
                flush=True,
            )
            x = x.contiguous()
        return inner(x, *a, **kw)

    C.all_reduce = wrapped
    import spmd_types._local as L
    L.all_reduce = wrapped

def main() -> None:
    if os.environ.get("SPMD_PROBE") == "1":
        _instrument_all_reduce()
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.autograd.set_multithreading_enabled(False)

    # TP>1 is a different case: tensor-parallel ranks must see IDENTICAL data, so
    # there is no desync to trigger. What it does exercise is the zero-vision path
    # itself running through the sharded tower, which is where spmd_types is
    # strict about layouts.
    tp = int(os.environ.get("TP", "1"))
    pd = None
    if tp > 1:
        from train.parallel.parallel_dims import ParallelDims
        from train.parallel.parallelize import parallelize_qwen3_5
        from train.parallel.spmd import set_spmd_meshes

        pd = ParallelDims(
            dp_replicate=1, dp_shard=world // tp, cp=1, tp=tp, pp=1, ep=1, world_size=world
        )
        pd.build_mesh()
        model = build_meta(SNAPSHOT, seq_len=SEQ_LEN, tp=tp, enable_sp=os.environ.get("SP") == "1")
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)
        parallelize_qwen3_5(
            model, pd, mode="fsdp", compile=False,
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        )
        set_spmd_meshes(dense_mesh=pd.spmd_dense_mesh(), sparse_mesh=None)
        mesh = pd.get_optional_mesh("batch")
    else:
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (world,), mesh_dim_names=("dp",)
        )
        model = build_meta(SNAPSHOT, seq_len=SEQ_LEN, tp=1, enable_sp=False)
        apply_data_parallel(
            model, mesh, mode="fsdp", param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)
    assert model.vision_encoder is not None, "snapshot has no vision tower"

    # TP=1: rank 0 has images, the rest have none (the desync).
    # TP>1: nobody has images -- exercise the zero-vision path under sharding.
    images = 0 if tp > 1 else (2 if rank == 0 else 0)
    patch_dim = model.vision_encoder.patch_embed.weight.shape[1]
    merge_unit = model.vision_encoder.spatial_merge_unit
    batch = synthetic_batch(
        images,
        patches_per_image=merge_unit * 4,
        documents=2,
        seq_len=SEQ_LEN,
        image_token_id=IMAGE_TOKEN_ID,
        vocab_size=151936,
        patch_dim=int(patch_dim),
        spatial_merge_unit=merge_unit,
        device=torch.device("cuda"),
        # CPU generator: synthetic_batch builds on CPU and moves afterwards
        generator=torch.Generator().manual_seed(0 if tp > 1 else rank),
    )
    print(f"[rank {rank}] images={images} keys={'pixel_values' in batch}", flush=True)

    loss, _, _ = forward_backward(
        model, [batch], dp_group=mesh,
        special_tokens={"image_id": IMAGE_TOKEN_ID}, ddp=False, loss_chunks=8,
        # WITHOUT this the spmd mesh context is never entered (`mesh_ctx` becomes
        # a nullcontext in train/step.py) and TP layout errors cannot surface.
        parallel_dims=pd if tp > 1 else None,
    )

    vision = [p for n, p in model.named_parameters() if n.startswith("vision_encoder")]
    missing = sum(p.grad is None for p in vision)
    finite = all(p.grad is None or p.grad.isfinite().all() for p in vision)
    print(
        f"[rank {rank}] OK loss={loss.item():.4f} vision_params={len(vision)} "
        f"grad_none={missing} all_finite={finite}",
        flush=True,
    )
    assert missing == 0, f"rank {rank}: {missing} vision params never got a gradient"
    assert finite, f"rank {rank}: non-finite vision gradient"

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: no desync with mismatched image counts", flush=True)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
