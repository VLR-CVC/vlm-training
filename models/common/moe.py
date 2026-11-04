# Vendored from torchtitan b21f7d43e: torchtitan/models/common/moe.py and the MoE
# helpers of torchtitan/models/common/config_utils.py (`make_moe_config`,
# `make_router_config`, `make_token_dispatcher_config`, `make_routed_experts_config`).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
#
# DEVIATIONS from the upstream file, all of them trims:
#
#   1. No remat regions, matching the rest of this port (see the headers of
#      feed_forward.py and vision_encoder.py). Upstream wraps the router's
#      top-k selection in `remat.region(..., recompute=False)` so a replayed
#      forward cannot re-draw a different routing; without remat regions there
#      is no replay to diverge from.
#   2. SwiGLU is hardcoded in `GroupedExperts` instead of going through
#      `ActivationFn`/`SwiGLU` configs. That indirection exists upstream so Kimi
#      can swap in `SiTUGLU`; Qwen3.5 is SwiGLU on every path, and this port's
#      `FeedForward` already hardcodes it the same way.
#   3. No `AuxLoss` / `MicrobatchWiseLoadBalanceLoss`: the router's `aux_loss`
#      field is dropped rather than carried as a permanent `None`. The
#      DeepSeek-V3 balance loss needs `AuxLoss.inject` plus the step's global
#      valid-token count threaded in from the loss, which is not ported.
#   4. Only `LocalTokenDispatcher` (EP=1) and `AllToAllTokenDispatcher` (EP>1)
#      exist; the DeepEP/HybridEP/TorchAO dispatchers are not ported -- see
#      token_dispatcher.py.
#
# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   T = num tokens, D = model dimension,
#   F = hidden (FFN intermediate) dimension, E = num experts,
#   K = top-k, N = routed tokens (T*K),
#   R = routed tokens assigned to local experts,
#   O = expert output features, I = expert input features
#       (roles, not model dims: the _grouped_mm seam takes the expert
#        weight in its stored (E, O, I) orientation, which is (E, F, D)
#        for the up/gate projections and (E, D, F) for the down one)

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from models.common.feed_forward import FeedForward
from models.common.module import Module
from models.common.nn_modules import RouterGateLinear
from train.parallel.parallel_dims import ParallelDims
from train.parallel.spmd import (
    maybe_set_sparse_mesh,
    spmd_mesh_size,
)

from .token_dispatcher import AllToAllTokenDispatcher, LocalTokenDispatcher

__all__ = [
    "GroupedExperts",
    "MoE",
    "RoutedExperts",
    "TokenChoiceTopKRouter",
    "make_moe_config",
    "make_routed_experts_config",
    "make_router_config",
]

class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=self.w1_EFD, offs=offsets_E
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=self.w3_EFD, offs=offsets_E
        )
        h_RF = F.silu(gate_RF) * up_RF
        return self._grouped_mm(A=h_RF, weight_EOI=self.w2_EDF, offs=offsets_E).type_as(
            x_RD
        )

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Grouped matmul of ``A @ weight_EOI.transpose(-2, -1)``.

        ``weight_EOI`` is the grouped expert weight in its stored
        ``(experts, out_features, in_features)`` orientation; the transpose to
        the grouped-GEMM right operand happens here.
        """
        B_t = weight_EOI.bfloat16().transpose(-2, -1)
        return torch._grouped_mm(A, B_t, offs=offs)

class RoutedExperts(Module):
    """Local SPMD region composing token_dispatcher and inner_experts
    as sibling nodes so each can be overridden independently."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: GroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()
        self.token_dispatcher = config.token_dispatcher.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_spmd`` (from ``sharding_config``) establishes
        the local SPMD types for the forward body.
        """
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
        out_TD = self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
        return out_TD

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Parallelize the grouped experts, then wire the EP mesh on the
        dispatcher so dispatch/combine see the right mesh at runtime."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
        )

class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
    routed to top K experts based on the router scores.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: RouterGateLinear.Config
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns expert IDs and scores with shape ``(T, K)``.
        """
        num_tokens = scores_TE.shape[0]
        # Round-robin indices with exact balance
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        return topk_expert_ids_TK, topk_scores_TK

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        return torch.topk(
            scores_for_choice_TE, k=self.top_k, dim=-1, sorted=False
        ).indices

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_TD: Input ``(T, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_TK: Routing scores ``(T, K)``.
            topk_expert_ids_TK: Expert indices ``(T, K)``.
            routing_map_TE: One-hot boolean routing map ``(T, E)``.
        """
        scores_TE = self.gate(x_TD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # RouterGateLinear returns scores_TE in FP32.
        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores_TE = F.softplus(scores_TE).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = self._debug_force_load_balance_routing(
                scores_TE
            )
        else:
            topk_expert_ids_TK = self._select_experts(
                scores_TE, expert_bias_E, **router_kwargs
            )
            # The expert bias is only used for routing. The gating value is
            # still derived from the original scores.
            topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        # Build a one-hot boolean routing map (T, E) marking the experts each
        # token is routed to. Under TP/SP the router outputs are sharded on the
        # token dimension; scatter_ writes along the replicated expert
        # dimension and therefore needs no redistribution.
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )

        return (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        )

class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments.
    2. RoutedExperts.forward() enters a local SPMD region, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
       b. expert computation (GroupedExperts, local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering
          (``scatter_add`` only at EP=1).
    3. Shared experts compute their output.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        routed_experts: RoutedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert_E is accumulated in the model forward pass.
        #       expert_bias_E is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # tokens_per_expert_E will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        """
        Args:
            x_TD: Input ``(T, D)``.

        Returns:
            Output ``(T, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output is redistributed to sp_layout. GroupedExperts operates in a
        local SPMD region.
        """
        # topk scores and expert IDs have shape (T, K); the routing map (T, E)
        # marks the experts each token is routed to (built inside the router).
        (
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self.router(x_TD, self.expert_bias_E, **router_kwargs)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # NOTE: Activation Checkpointing has the side effect of double counting
        #       tokens_per_expert_E -- first in the forward pass, and then in the
        #       backward pass. That has no effect on the expert bias update thanks
        #       to the torch.sign() operator in the hook.
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

        shared_out_TD = (
            self.shared_experts(x_TD) if self.shared_experts is not None else None
        )

        if shared_out_TD is not None:
            out_TD = out_TD + shared_out_TD
        return out_TD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            # Reinitialize MoE counters there when no explicit buffer device is passed.
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.routed_experts.inner_experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(
                    self.routed_experts.inner_experts.num_experts, dtype=torch.float32
                )

def make_router_config(
    *,
    dim: int,
    num_experts: int,
    gate_param_init: dict[str, Callable],
    top_k: int = 1,
    score_func: Literal["sigmoid", "softmax", "sqrtsoftplus"] = "sigmoid",
    route_norm: bool = False,
    route_scale: float = 1.0,
    bias: bool = False,
) -> TokenChoiceTopKRouter.Config:
    """Build a fully-specified TokenChoiceTopKRouter.Config."""
    return TokenChoiceTopKRouter.Config(
        num_experts=num_experts,
        gate=RouterGateLinear.Config(
            in_features=dim,
            out_features=num_experts,
            bias=bias,
            param_init=gate_param_init,
        ),
        top_k=top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=route_scale,
    )

def make_routed_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    param_init: dict[str, Callable],
    ep: int = 1,
) -> RoutedExperts.Config:
    """Build a fully-specified RoutedExperts.Config (inner_experts + token_dispatcher).

    ``ep`` picks the dispatcher: the local argsort at 1, the all-to-all above.
    `AllToAllTokenDispatcher` falls back to the local path when its `ep_mesh` is
    None, so the choice is about intent, not capability -- EP=1 keeps the class
    that cannot silently skip a collective it was supposed to make.

    Upstream's ``comm_backend`` argument is gone: it picks between the all-to-all
    dispatchers, and only the plain one is ported.
    """
    dispatcher = (
        AllToAllTokenDispatcher.Config if ep > 1 else LocalTokenDispatcher.Config
    )
    return RoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            param_init=param_init,
        ),
        token_dispatcher=dispatcher(
            num_experts=num_experts,
            top_k=top_k,
        ),
    )

def make_moe_config(
    *,
    num_experts: int = 8,
    router: TokenChoiceTopKRouter.Config,
    routed_experts: RoutedExperts.Config,
    shared_experts: FeedForward.Config | None = None,
    load_balance_coeff: float | None = 1e-3,
) -> MoE.Config:
    """Build a fully-specified MoE.Config."""
    return MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        router=router,
        routed_experts=routed_experts,
        shared_experts=shared_experts,
    )
