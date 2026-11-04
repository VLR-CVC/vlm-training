"""The handful of Megatron symbols the vendored DSA kernels import.

`dsa_min_memory.py` is vendored unmodified except for its import block (see
`scripts/vendor_dsa.py`). It reaches into Megatron for five things; four of them
are either trivial or dead on the path we take:

* `_rotate_half` -- copied verbatim from
  `megatron/core/models/common/embeddings/rope_utils.py`.
* the Yarn quartet -- reached only inside `isinstance(rotary_pos_emb,
  YarnRotaryEmbedding)`. Our indexer hands the kernels a plain `inv_freq`
  carrier, so the sentinel below never matches and the branch never runs.
* `rotate_activation` -- guarded by `use_hadamard`, which every call site in
  `dsa_min_memory.py` passes as the literal `False`. Dead code; it would need
  the `fast_hadamard_transform` package, which we do not install.
* `ensure_params_ready` -- Megatron's wait on an async parameter all-gather
  (DDP `overlap_param_gather`, its own FSDP). We no-op it; see the warning on
  the function.
* `ProcessGroupCollection` -- the kernels touch `pg_collection.tp` only, and
  only to all-reduce the dense teacher across tensor-parallel shards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "ProcessGroupCollection",
    "TRIVIAL_TP",
    "YarnRotaryEmbedding",
    "_rotate_half",
    "_yarn_find_correction_range",
    "_yarn_get_concentration_factor",
    "_yarn_linear_ramp_mask",
    "ensure_params_ready",
    "rotate_activation",
]


def _rotate_half(x: torch.Tensor, rotary_interleaved: bool) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]."""
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


class YarnRotaryEmbedding:
    """Sentinel that nothing is an instance of, so the Yarn branch stays dead.

    Instantiating it is the error, not using it: an actual Yarn indexer would
    need the four helpers below ported, not this placeholder.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("Yarn RoPE is not ported; the DSA indexer uses plain RoPE.")


def _yarn_find_correction_range(*_args, **_kwargs):
    raise NotImplementedError("Yarn RoPE is not ported.")


def _yarn_get_concentration_factor(*_args, **_kwargs):
    raise NotImplementedError("Yarn RoPE is not ported.")


def _yarn_linear_ramp_mask(*_args, **_kwargs):
    raise NotImplementedError("Yarn RoPE is not ported.")


def rotate_activation(*_args, **_kwargs):
    raise NotImplementedError(
        "Hadamard rotation is not ported; every call site passes use_hadamard=False."
    )


def ensure_params_ready(params) -> None:
    """No-op: we have no async parameter all-gather to wait on.

    The kernels read `linear_q.weight` directly instead of calling the module,
    which skips whatever pre-forward hook would have published the gathered
    parameter. Under FSDP2 that weight is a sharded DTensor outside the
    module's own forward, and `F.linear` against it is wrong, not slow -- so the
    indexer linears must stay unsharded, or be read inside their own forward.
    The assert makes the FSDP2 case loud rather than silent.
    """
    for p in params:
        assert not isinstance(p, torch.distributed.tensor.DTensor), (
            "DSA kernels read indexer weights directly; a DTensor here means the "
            "indexer linears got sharded and the local shard would be read as the "
            "full weight. Exclude them from FSDP wrapping."
        )


class _TrivialGroup:
    """Stands in for a process group of one, for single-GPU and tp_size == 1."""

    def size(self) -> int:
        return 1


TRIVIAL_TP = _TrivialGroup()


@dataclass
class ProcessGroupCollection:
    """Only `.tp` is read, and only to all-reduce the dense teacher logits."""

    tp: object = field(default_factory=_TrivialGroup)
