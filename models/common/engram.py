"""Engram: hashed n-gram embedding memory with a context gate.

Conditional memory as in DeepSeek's Engram (arXiv 2601.07372) and the Qwen-style
"PLE" n-gram embedding of HF transformers 5.16.1 (`models/qwen4_exp`), reduced to
one residual stream:

    e_t = concat_{n, k} E[H_{n,k}(x_{t-n+1..t})]                (hashed lookup)
    g_t = < RMSNorm(h_t), RMSNorm(W_K e_t) > / sqrt(d)
    a_t = sigmoid( sign(g_t) * sqrt(|g_t|) )                     (gate_sqrt=True)
    h'_t = h_t + a_t * W_V e_t

The gate is one scalar per token (the reference's per-stream scalar, with one
stream). `gate_sqrt=False` gives the paper's plain `sigmoid(g_t)`.

The module adds nothing at initialisation: `W_V` is zero-initialised by the
builder, so `h' == h` exactly until the first optimiser step.

Value scale (D021). Nothing here bounds ``||a_t W_V e_t||``: scaling W_V up and
the gate down gives the same output, so the gate mean alone says little (T04:
on a frozen backbone it drifted 0.35 -> 0.03 while ||W_V e|| grew 11x). The
DeepSeek reference has the same symmetry -- it does NOT normalise the value
either. Paper (arXiv 2601.07372, eq. 3-5): RMSNorm on query and key only,
``v = W_V e``, ``Y = SiLU(Conv1D(RMSNorm(a v))) + a v``, so the norm only feeds
the conv branch and ``a v`` reaches the residual raw. Demo
(github.com/deepseek-ai/Engram, engram_demo_v1.py): ``value = gates *
value_proj(embeddings)``, no norm. What differs from us: they zero-init the
conv, not W_V (W_V keeps its default random init, so the gate gets a gradient
from step 0), train everything from scratch, decay the projections with the
backbone (0.1), and train the table with Adam at 5x the base LR and no weight
decay (`training.lr_engram_table`, `training.weight_decay_engram`). Watch
``||a W_V e|| / ||h||``, not the gate mean.

Hash ids are NOT computed here. They are a function of the token ids only, so
the model computes them once in `preprocess_inputs` (see
`models/common/engram_hash.py`), where they can be prefetched or annotated for
SPMD like any other input, and passes them in.

Not ported from the references: the short causal convolution of the transformers
module and of the Engram prototype (DeepSeek-V4.1 dropped it), tokenizer
canonicalisation (DeepSeek), multiple residual streams (transformers).
"""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch

from models.common.module import Module
from models.common.nn_modules import Embedding, Linear

from .engram_hash import build_ngram_hash_spec, ngram_hash_ids, NgramHashSpec

__all__ = ["Engram", "EngramTable"]


class EngramTable(Embedding):
    """The packed table of every (order, head) slice of one Engram layer.

    Vocab-parallel under TP exactly like `tok_embeddings` (`Embedding.forward`
    masks the rows another rank owns, the output is a partial sum). Takes the
    ``(T, heads)`` row ids and returns the concatenated ``(T, heads * dim)``
    vectors, zeroed where ``mask`` is false. The mask is applied here, before the
    TP reduction, because this is where the token-id-shaped inputs still live.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        pass  # its own Config so that `build()` makes an EngramTable

    row_lookup = None
    """Set by `train.parallel.engram.apply_engram_row_sharding`: the weight is
    then a ``Shard(0)`` DTensor over the DP ranks and the lookup an all-to-all
    (`models/common/engram_sharded.py`) instead of FSDP's all-gather."""

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore [bad-override]
        if self.row_lookup is not None:
            vectors = self.row_lookup(self.weight, input)
        else:
            vectors = super().forward(input)
        return (vectors * mask[:, None, None].to(vectors.dtype)).flatten(1)


class Engram(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # addressing; see `build_ngram_hash_spec`
        orders: tuple[int, ...] = (2, 3)
        heads_per_order: int = 8
        rows_per_head: int
        vocab_size: int
        pad_id: int
        seed: int = 0
        layer_index: int = 0
        """Index of this module among the model's Engram layers: it picks the
        hash multipliers and primes, so two layers never share an address map."""
        blocked_token_ids: tuple[int, ...] = ()
        """Token ids (vision placeholders) that never enter an n-gram and whose
        own position receives no memory."""
        gate_sqrt: bool = True
        table: EngramTable.Config
        key_proj: Linear.Config
        value_proj: Linear.Config
        key_norm: Module.Config
        query_norm: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.spec: NgramHashSpec = build_ngram_hash_spec(
            orders=config.orders,
            heads_per_order=config.heads_per_order,
            rows_per_head=config.rows_per_head,
            vocab_size=config.vocab_size,
            pad_id=config.pad_id,
            seed=config.seed,
            layer_index=config.layer_index,
        )
        if config.table.num_embeddings != self.spec.num_rows:
            raise ValueError(
                f"table has {config.table.num_embeddings} rows, the hash layout needs "
                f"{self.spec.num_rows}; build the config with `engram_table_rows`"
            )
        # kept for export: with `spec`, they rebuild the same layout elsewhere
        self.rows_per_head = config.rows_per_head
        self.seed = config.seed
        self.blocked_token_ids = tuple(config.blocked_token_ids)
        self.gate_sqrt = config.gate_sqrt
        self.dim = config.query_norm.dim  # pyrefly: ignore [missing-attribute]
        self.table = config.table.build()
        self.key_proj = config.key_proj.build()
        self.value_proj = config.value_proj.build()
        self.key_norm = config.key_norm.build()
        self.query_norm = config.query_norm.build()
        # mean / std of the gate over the last forward's unmasked tokens (this
        # rank's tokens only), for logging; never part of the graph
        self.last_gate_stats: torch.Tensor | None = None

    @property
    def num_heads(self) -> int:
        return self.spec.num_heads

    @torch.no_grad()
    def hash_inputs(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(row ids (T, heads), mask (T,))`` for one packed row. ``mask`` is false
        at blocked (vision) tokens, which receive no memory."""
        blocked = None
        if self.blocked_token_ids:
            blocked = torch.isin(
                tokens, torch.tensor(self.blocked_token_ids, device=tokens.device)
            )
        ids = ngram_hash_ids(tokens, positions, self.spec, blocked=blocked)
        mask = (
            torch.ones_like(tokens, dtype=torch.bool) if blocked is None else ~blocked
        )
        return ids, mask

    def forward(
        self,
        x_TD: torch.Tensor,
        engram_ids_TH: torch.Tensor,
        engram_mask_T: torch.Tensor,
    ) -> torch.Tensor:
        e_TE = self.table(engram_ids_TH, engram_mask_T)
        key_TD = self.key_norm(self.key_proj(e_TE))
        value_TD = self.value_proj(e_TE)
        query_TD = self.query_norm(x_TD)
        gate_T1 = (key_TD.float() * query_TD.float()).sum(dim=-1, keepdim=True)
        gate_T1 = gate_T1 / math.sqrt(self.dim)
        if self.gate_sqrt:
            gate_T1 = gate_T1.abs().clamp_min(1e-6).sqrt() * gate_T1.sign()
        alpha_T1 = torch.sigmoid(gate_T1)
        self._record_gate(alpha_T1, value_TD)
        return x_TD + (alpha_T1 * value_TD.float()).to(x_TD.dtype)

    def _record_gate(self, alpha_T1: torch.Tensor, value_TD: torch.Tensor) -> None:
        # masked positions have e = 0, hence value = 0 and a gate of exactly 0.5
        # that means nothing; leave them out of the statistics
        with spmd.no_typecheck(), torch.no_grad():
            alpha = alpha_T1.detach().float().squeeze(-1)
            used = (value_TD.detach().abs().amax(dim=-1) > 0).float()
            # until W_V moves every value is zero: fall back to all positions.
            # Weighted, not boolean-indexed, so logging never syncs the host.
            weight = torch.where(used.sum() > 0, used, torch.ones_like(used))
            weight = weight / weight.sum()
            mean = (weight * alpha).sum()
            std = (weight * (alpha - mean).square()).sum().sqrt()
            self.last_gate_stats = torch.stack([mean, std])
