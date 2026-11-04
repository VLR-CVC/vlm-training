"""Two-rank check that the EP all-to-all routes every token to the right expert.

    scripts/local_ep_parity.sh            # launches this on one card, 2 ranks

Not a pytest file: it needs a 2-rank process group, so the launcher runs it
under torchrun and it asserts on each rank.

The experts here are deliberately NOT weights -- expert `i` multiplies its
tokens by `i + 1`. That makes the expected output exact arithmetic
(`sum_k score_k * (expert_k + 1) * x`) independent of initialisation, so a
mismatch means the dispatch/permute/combine path put a token on the wrong
expert or returned it to the wrong rank, which is the failure that otherwise
just looks like a slightly worse loss curve.

`models/tests/test_ep_dispatcher.py` covers `_permute` without collectives.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from models.common.token_dispatcher import AllToAllTokenDispatcher
from train.parallel.parallel_dims import ParallelDims
from train.parallel.spmd import set_current_spmd_mesh, set_spmd_meshes

E, K, D, T = 4, 2, 8, 37  # 37 tokens: deliberately not a multiple of anything


def local_expert_apply(
    routed_input_RD: torch.Tensor,
    counts_e: torch.Tensor,
    first_expert: int,
) -> torch.Tensor:
    """Stand-in for GroupedExperts: expert i scales its own rows by (i + 1)."""
    out = torch.empty_like(routed_input_RD)
    start = 0
    for local_idx, n in enumerate(counts_e.tolist()):
        out[start : start + n] = routed_input_RD[start : start + n] * (
            first_expert + local_idx + 1
        )
        start += n
    assert start == routed_input_RD.shape[0], (start, routed_input_RD.shape)
    return out


def reference(x_TD, topk_scores_TK, topk_expert_ids_TK):
    """What the MoE must produce for this rank's tokens, computed locally."""
    out = torch.zeros_like(x_TD)
    for k in range(topk_expert_ids_TK.shape[1]):
        scale = (topk_expert_ids_TK[:, k] + 1).to(x_TD.dtype)
        out += x_TD * (topk_scores_TK[:, k] * scale).unsqueeze(-1)
    return out


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())

    pd = ParallelDims(
        dp_replicate=1, dp_shard=world, cp=1, tp=1, pp=1, ep=world, world_size=world
    )
    pd.build_mesh()
    set_spmd_meshes(dense_mesh=pd.spmd_dense_mesh(), sparse_mesh=pd.spmd_sparse_mesh())

    ep_mesh = pd.get_optional_mesh("ep")
    assert ep_mesh is not None and ep_mesh.size() == world, ep_mesh
    e = E // world  # local experts
    first_expert = rank * e

    d = AllToAllTokenDispatcher.Config(num_experts=E, top_k=K).build()
    d.wire_meshes(ep_mesh=ep_mesh)

    # different tokens and different routing per rank, which is the point: the
    # a2a has to carry uneven splits in both directions
    gen = torch.Generator(device="cuda").manual_seed(100 + rank)
    x_TD = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    topk_scores_TK = torch.rand(T, K, device="cuda", generator=gen).to(torch.bfloat16)
    topk_expert_ids_TK = torch.stack(
        [
            torch.randperm(E, device="cuda", generator=gen)[:K]
            for _ in range(T)
        ]
    )
    routing_map_TE = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(
        -1, topk_expert_ids_TK, True
    )
    num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

    with set_current_spmd_mesh(pd.spmd_dense_mesh()):
        routed_input_RD, counts_e, metadata = d.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        assert counts_e.shape == (e,), (counts_e.shape, e)
        routed_output_RD = local_expert_apply(routed_input_RD, counts_e, first_expert)
        out_TD = d.combine(routed_output_RD, metadata, x_TD)

    want = reference(x_TD, topk_scores_TK, topk_expert_ids_TK)
    err = (out_TD.float() - want.float()).abs().max().item()
    rel = err / want.float().abs().max().item()

    # every rank must have both sent and received tokens, or the test passed
    # without exercising the all-to-all at all
    moved_out = sum(metadata.output_splits) - metadata.output_splits[rank]
    moved_in = sum(metadata.input_splits) - metadata.input_splits[rank]
    assert moved_out > 0 and moved_in > 0, (metadata.input_splits, metadata.output_splits)

    print(
        f"rank {rank}: local experts {first_expert}..{first_expert + e - 1}  "
        f"tokens held {routed_input_RD.shape[0]}  in {moved_in} out {moved_out}  "
        f"max_abs_err {err:.3e} rel {rel:.3e}",
        flush=True,
    )
    assert rel < 2e-2, f"rank {rank}: EP output differs from the local reference ({rel})"
    dist.barrier()
    if rank == 0:
        print("EP dispatch parity OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
