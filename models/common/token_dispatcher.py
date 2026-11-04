# Vendored from torchtitan b21f7d43e: torchtitan/models/common/token_dispatcher.py,
# trimmed to the two dispatchers this port uses: `LocalTokenDispatcher` (EP=1,
# a local argsort) and `AllToAllTokenDispatcher` (EP>1, argsort + variable-size
# all-to-all), plus the `BaseEPTokenDispatcher` the latter needs.
#
# `TorchAOTokenDispatcher` is ported too: it is `AllToAllTokenDispatcher` with a
# padded permute, so a quantized grouped GEMM sees token groups aligned to its
# required multiple. `DeepEPTokenDispatcher` and `HybridEPTokenDispatcher` are
# NOT ported -- they need the DeepEP / HybridEP libraries, which are not
# installed, and HybridEP targets NVLink72 domains this machine does not have.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

from abc import ABC, abstractmethod
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.device_mesh import DeviceMesh

from models.common.configurable import Configurable
from models.common.scatter_add import deterministic_scatter_add
from train.parallel.spmd import maybe_set_sparse_mesh

__all__ = [
    "AllToAllDispatchMetadata",
    "AllToAllTokenDispatcher",
    "BaseEPTokenDispatcher",
    "LocalDispatchMetadata",
    "LocalTokenDispatcher",
    "TorchAOTokenDispatcher",
]

@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815

@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]

class LocalTokenDispatcher(Configurable):
    """Token dispatcher for EP=1. Handles local token reordering only.

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
    ) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override.

        The guard is this port's, not upstream's: `make_routed_experts_config`
        only ever builds this class for EP=1, so a sized `ep_mesh` here means a
        caller wired expert parallelism onto a dispatcher that cannot move a
        token off-rank -- which would silently train every rank on its own
        experts instead of raising.
        """
        if ep_mesh is not None and ep_mesh.size() > 1:
            raise NotImplementedError(
                f"ep_mesh has size {ep_mesh.size()}, but this is the EP=1 "
                "dispatcher; AllToAllTokenDispatcher is the EP>1 one."
            )

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via argsort. Routing scores are applied
        to the expert outputs in ``combine``, after expert computation.

        Args:
            x_TD: ``(T, D)`` input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices

        Returns:
            routed_input_ND: ``(N, D)`` where N = T*K. Tokens in expert-sorted
                order.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
        """
        # Reorder the token indices to match the order of the experts where N = T*K
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices per token
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert

        Returns:
            routed_input_RD: ``[R = sum(num_local_tokens_per_expert_E), input_dim(D)]``.
                Tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        # R = N (no EP all-to-all)
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        out_TD = torch.zeros_like(x_TD)

        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        dim = x_TD.shape[-1]
        out_TD = deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )
        return out_TD

class BaseEPTokenDispatcher(LocalTokenDispatcher, ABC):
    """Base class for EP token dispatchers.

    Owns EP mesh wiring and SP coordinate helpers shared by EP implementations.
    LocalTokenDispatcher intentionally does not know about SP: local dispatch is
    used when EP is off, and expert activations are replicated for expert TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None

    def wire_meshes(self, *, ep_mesh: DeviceMesh | None) -> None:
        """Install the EP mesh used by dispatch / combine."""
        self.ep_mesh = ep_mesh
        self.init_buffer()

    def init_buffer(self) -> None:
        """Initialize backend communication buffers, if any."""

    @abstractmethod
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Dispatch the local token shard to the ranks owning each expert."""
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: object,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Combine expert outputs."""
        raise NotImplementedError

class AllToAllTokenDispatcher(BaseEPTokenDispatcher):
    """Token dispatcher for EP>1: token reorder plus all-to-all dispatch/combine.

    ``ep_mesh`` is wired by ``RoutedExperts.parallelize`` via ``wire_meshes``;
    when it is None (EP=1) both halves fall back to the local path, so this
    class is a drop-in for ``LocalTokenDispatcher`` at any degree.

    DEVIATION from torchtitan b21f7d43e: upstream's `comm_backend` setting picks
    between this and the DeepEP/HybridEP/TorchAO dispatchers. Only this one is
    ported, so there is nothing to pick.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        pass

    def _token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        pg,
        ep_size: int,
    ) -> torch.Tensor:
        """Exchange per-rank expert token counts before the data all-to-all.

        Separate from ``dispatch`` so graph passes can annotate the count
        exchange independently of the token-exchange scheduling markers.
        """
        assert self.ep_mesh is not None
        if torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing():
            return all_to_all_single(
                num_local_tokens_per_expert_E.view(ep_size, -1),
                None,
                None,
                group=self.ep_mesh,
            )

        return spmd.all_to_all(
            num_local_tokens_per_expert_E.view(ep_size, -1),
            pg,
            src=spmd.V,
            dst=spmd.V,
        )

    def _sync_token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        num_global_tokens_per_local_expert_EP_e: torch.Tensor,
        ep_size: int,
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Wait for the counts and materialize the CPU split lists.

        Local input splits can copy to CPU non-blocking; the remote output
        splits must be ready before the variable-size data all-to-all launches,
        so that copy is the one device-to-host sync on this path.
        """
        # explicit wait: the triton kernels downstream do not unwrap an
        # AsyncCollectiveTensor on their own
        num_global_tokens_per_local_expert_EP_e = (
            torch.ops._c10d_functional.wait_tensor(
                num_global_tokens_per_local_expert_EP_e
            )
        )
        num_global_tokens_per_local_expert_E = (
            num_global_tokens_per_local_expert_EP_e.reshape(-1)
        )
        input_splits = (
            num_local_tokens_per_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = (
            num_global_tokens_per_local_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        return (
            num_global_tokens_per_local_expert_E,
            input_splits.tolist(),
            output_splits.tolist(),
        )

    def _dispatch_token_exchange(
        self,
        routed_input_ND: torch.Tensor,
        pg,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        """Launch the dispatch all-to-all that moves routed tokens to experts."""
        assert self.ep_mesh is not None
        if torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing():
            return all_to_all_single(
                routed_input_ND, output_splits, input_splits, self.ep_mesh
            )

        return spmd.all_to_all(
            routed_input_ND,
            pg,
            src=spmd.V,
            dst=spmd.V,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )

    def _combine_token_exchange(
        self,
        routed_output_RD: torch.Tensor,
        pg,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """Launch the combine all-to-all that returns expert outputs to tokens.

        The splits are the dispatch ones swapped: what went out comes back.
        """
        assert self.ep_mesh is not None
        if torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing():
            return all_to_all_single(
                routed_output_RD, input_splits, output_splits, self.ep_mesh
            )

        return spmd.all_to_all(
            routed_output_RD,
            pg,
            src=spmd.V,
            dst=spmd.V,
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        With SP, ``x_TD``/``topk_scores_TK``/``topk_expert_ids_TK`` are already
        the local SP shard provided by the surrounding local SPMD region.

        Returns ``(routed_input_RD, num_global_tokens_per_local_expert_e,
        metadata)``, where R is the number of tokens the local experts own --
        not T*K, which is what the local path returns.
        """
        if self.ep_mesh is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        ep_size = self.ep_mesh.size()
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        if spmd.is_type_checking():  # sparse mesh reinterpret
            spmd.mutate_type(
                num_local_tokens_per_expert_E,
                src=spmd.P,
                dst={"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
            )

        with maybe_set_sparse_mesh():
            pg = "ep"
            if spmd.is_type_checking():
                num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
                    num_local_tokens_per_expert_E, spmd.current_mesh()
                )
                routed_input_ND = spmd.reinterpret_mesh(
                    routed_input_ND, spmd.current_mesh()
                )

            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E, pg, ep_size
                )
                (
                    num_global_tokens_per_local_expert_E,
                    input_splits_list,
                    output_splits_list,
                ) = self._sync_token_count_exchange(
                    num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e,
                    ep_size,
                )

            routed_input_RD = self._dispatch_token_exchange(
                routed_input_ND, pg, output_splits_list, input_splits_list
            )
            # The all-to-all leaves the rows rank-major:
            #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...
            # `_grouped_mm` needs one contiguous run per expert, so `_permute`
            # reshuffles to expert-major:
            #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...
            (
                input_shape,
                routed_input_RD,
                permuted_indices,
                num_global_tokens_per_local_expert_e,
            ) = self._permute(
                routed_input_RD,
                num_global_tokens_per_local_expert_E,
                total_rows=sum(output_splits_list),
            )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
        total_rows: int | None = None,
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Collapses the ``(EP, e)`` count matrix to
        ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing over ranks.

        ``total_rows`` is the number of rows the dispatch all-to-all produced,
        i.e. ``sum(output_splits)``. It must be passed, and must NOT be read back
        off ``routed_input_RD.shape[0]``, because under compile that dim is a
        symbolic expression -- job 2268203 traced it as
        ``256*Max(1, (2048//s56))`` -- and a floor division on a symbolic dim does
        not agree with the integer sum of the counts. `repeat_interleave` checks
        ``output_size == repeats.sum()`` on device and the two disagreed by one in
        either direction, killing the backward (where AC recomputes this) with
        `Repeat.cu:20 Assertion result_size == cumsum_ptr[size - 1] failed`.
        ``output_splits`` is already a materialized CPU list, so this costs no
        extra device-to-host sync. Falls back to the shape only when a caller
        omits it, which is the eager-only path where the two always agree.
        """
        assert self.ep_mesh is not None
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device
        total = routed_input_RD.shape[0] if total_rows is None else total_rows

        # (EP, e) token counts per (rank, local expert)
        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # where each (r, e) segment starts in the rank-major input
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # transpose to expert-major (e, EP) and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens, output_size=total
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        # seg_ids.shape[0] == segment_lens.sum() == total, and reuses the symint
        # repeat_interleave already created rather than introducing another
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            t_mat.sum(0),
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute, all-to-all back, score, scatter_add."""
        if self.ep_mesh is None:
            return LocalTokenDispatcher.combine(self, routed_output_RD, metadata, x_TD)

        assert isinstance(metadata, AllToAllDispatchMetadata)
        with maybe_set_sparse_mesh():
            pg = "ep"
            routed_output_RD = self._unpermute(
                routed_output_RD, metadata.input_shape, metadata.permuted_indices
            )
            # returns an AsyncCollectiveTensor: the a2a runs on the NCCL stream
            # and does not block until the tensor is read
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD, pg, metadata.input_splits, metadata.output_splits
            )

        if spmd.is_type_checking():  # dense mesh reinterpret
            routed_output_RD = spmd.reinterpret_mesh(
                routed_output_RD, spmd.current_mesh()
            )

        out_TD = torch.zeros_like(x_TD)
        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        return deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(
                -1, out_TD.shape[-1]
            ),
            routed_output_RD,
        )

class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """`AllToAllTokenDispatcher` whose permute also PADS each expert's token
    group to a multiple of `pad_multiple`.

    FP8/MXFP8 grouped GEMMs require every group to start on an aligned
    boundary (16 for FP8, 32 for MXFP8); the plain `_permute` packs groups
    back to back and leaves them unaligned. Everything else -- the count
    exchange, both all-to-alls, the combine -- is inherited unchanged.

    IMPORTANT: this alone changes no dtype and sends no fewer bytes. The
    all-to-all still carries bf16 (EP>1 goes through
    `AllToAllTokenDispatcher.dispatch`), and `GroupedExperts._grouped_mm` still
    runs `torch._grouped_mm` in bf16. Padding without a quantized GEMM behind it
    is pure overhead -- it is the PREREQUISITE for one, not the win itself. The
    win needs `_grouped_mm` overridden with e.g.
    `torchao.prototype.moe_training.fp8_grouped_mm._to_fp8_rowwise_then_scaled_grouped_mm`,
    which is not ported.

    EP=1 is supported (`ep_mesh is None`): no all-to-all, local reorder plus the
    padded permute, so the quantized GEMM can be debugged on one GPU.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        # 16 for FP8 rowwise, 32 for MXFP8
        pad_multiple: int = 16

    def __init__(self, config: Config):
        super().__init__(config)
        if config.pad_multiple <= 0:
            raise ValueError(f"pad_multiple must be positive, got {config.pad_multiple}")
        self.pad_multiple = config.pad_multiple

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        if self.ep_mesh is not None:
            # the inherited path already routes through our `_permute` below
            return super().dispatch(
                x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
            )

        # EP=1: no all-to-all, but the groups still need padding. `_permute`
        # reads ep_size=1 here, so `num_local_tokens_per_expert_E` is already
        # the full per-expert count.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_tokens_per_local_expert_padded_e,
        ) = self._permute(routed_input_ND, num_local_tokens_per_expert_E)

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            # no all-to-all to reverse on this path
            input_splits=[],
            output_splits=[],
        )
        return routed_input_RD, num_tokens_per_local_expert_padded_e, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_mesh is not None:
            return super().combine(routed_output_RD, metadata, x_TD)

        assert isinstance(metadata, AllToAllDispatchMetadata)
        # unpad first, then the ordinary local score + scatter_add
        routed_output_RD = self._unpermute(
            routed_output_RD, metadata.input_shape, metadata.permuted_indices
        )
        out_TD = torch.zeros_like(x_TD)
        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)
        return deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(
                -1, x_TD.shape[-1]
            ),
            routed_output_RD,
        )

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
        total_rows: int | None = None,
    ):
        """Expert-major reorder AND pad every group to `pad_multiple`.

        ``total_rows`` is accepted so `dispatch` can call either override the same
        way, and ignored: `permute_and_pad` derives its own output length from the
        counts, so it never had the symbolic-shape problem the base path did.

        `torchao`'s `permute_and_pad` replaces the inherited `cumsum`/
        `repeat_interleave` index arithmetic; it returns one extra sentinel row
        that `_unpermute` drops.
        """
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        ep_size = self.ep_mesh.size() if self.ep_mesh is not None else 1
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_padded_e,
            _group_offsets,
        ) = permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        return input_shape, routed_input_RD, permuted_indices, num_padded_e

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        """Inverse of the padded permute: scatter back, then drop the sentinel
        row `permute_and_pad` appended."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD[:-1]
