"""MuonSR: an FSDP-sharded (uneven Shard(0)) run must equal the unsharded run,
momentum must be a DTensor (DCP sharding metadata), AdamW groups must still step,
and the param-group split must keep embeddings/lm_head/conv/1-D on AdamW."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.optimizer import MuonSR, adjust_lr_ratio  # noqa: E402
from train.optimizer import build_optimizer_param_groups  # noqa: E402

SHAPES = {"w": (5, 4), "experts": (3, 6, 4), "norm": (7,)}  # 5 rows over 2 ranks: 3 + 2


def _data():
    g = torch.Generator().manual_seed(0)
    init = {k: torch.randn(s, generator=g) for k, s in SHAPES.items()}
    grads = [{k: torch.randn(s, generator=g) for k, s in SHAPES.items()} for _ in range(3)]
    return init, grads


def _run(params, grads, wrap=lambda t: t):
    opt = MuonSR([{"params": [params["w"], params["experts"]], "lr": 1e-2, "weight_decay": 0.1, "use_muon": True},
                  {"params": [params["norm"]], "lr": 1e-2, "weight_decay": 0.0}], betas=(0.9, 0.95))
    for gs in grads:
        for k, p in params.items():
            p.grad = wrap(gs[k].clone())
        opt.step()
    return opt


def _worker(rank, world, port, out):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    init, grads = _data()
    shard = lambda t: distribute_tensor(t, mesh, [Shard(0)])
    params = {k: torch.nn.Parameter(shard(v)) for k, v in init.items()}
    opt = _run(params, grads, shard)
    assert isinstance(opt.state[params["w"]]["momentum_buffer"], DTensor)
    full = {k: p.detach().full_tensor() for k, p in params.items()}
    if rank == 0:
        torch.save(full, out)
    dist.barrier()
    dist.destroy_process_group()


def test_sharded_matches_unsharded(tmp_path):
    out = str(tmp_path / "sharded.pt")
    torch.multiprocessing.spawn(_worker, args=(2, "29781", out), nprocs=2, join=True)
    init, grads = _data()
    params = {k: torch.nn.Parameter(v.clone()) for k, v in init.items()}
    _run(params, grads)
    sharded = torch.load(out)
    for k, p in params.items():
        assert not torch.equal(p.detach(), init[k]), f"{k} did not move"
        torch.testing.assert_close(sharded[k], p.detach(), rtol=0, atol=1e-6, msg=k)


def test_param_group_split():
    named = [(n, torch.nn.Parameter(torch.zeros(s))) for n, s in [
        ("tok_embeddings.weight", (10, 4)), ("lm_head.weight", (10, 4)),
        ("layers.0.attention.wq.weight", (4, 4)), ("layers.0.attention_norm.weight", (4,)),
        ("layers.0.linear_attn.conv_q.weight", (4, 1, 4)), ("vision_encoder.blocks.0.fc1.weight", (8, 4)),
        ("vision_encoder.pos_embed", (16, 4))]]
    groups = build_optimizer_param_groups(named, {"mlp": 1, "vit": 2, "llm": 3}, 0.1, muon=True)
    by_id = {id(p): n for n, p in named}
    muon = sorted(by_id[id(p)] for g in groups if g.get("use_muon") for p in g["params"])
    assert muon == ["layers.0.attention.wq.weight", "vision_encoder.blocks.0.fc1.weight"], muon
    assert all(g["lr"] == 2 for g in groups if any(by_id[id(p)].startswith("vision") for p in g["params"]))


def test_adjust_lr():
    assert adjust_lr_ratio("match_rms_adamw", 2048, 6144) == 0.2 * 6144 ** 0.5
    assert adjust_lr_ratio("original", 4, 16) == 1.0
