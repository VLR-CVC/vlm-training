# Hashed n-gram addressing for Engram, adapted from the Qwen-style "PLE" n-gram
# embedding of HF transformers 5.16.1 (`models/qwen4_exp/modular_qwen4_exp.py`:
# `_splitmix64`, `_build_layer_multipliers`, `_find_nth_prime_after`,
# `Qwen4ExpTextNGramEmbedding.forward`), with DeepSeek-V4.1-Flash
# `inference/engram.py` (MIT) for the sticky look-back blocking at dead tokens.
#
# DEPENDENCY-FREE ON PURPOSE: this file imports only `math` and `torch`, and is
# copied verbatim next to the `trust_remote_code` modeling file by
# `utils/dcp_to_hf.py`. The trainer and the exported HF model therefore address
# the table with the very same code; a re-implementation that differed in one
# signedness or boundary convention would silently read other rows.
#
# Differences from the transformers reference:
#   * document boundaries come from the packed `positions` (each document
#     restarts at 0), not from EOS tokens in the stream;
#   * look-back also stops at "blocked" tokens (vision placeholders), as
#     DeepSeek's DEAD tokens do;
#   * the n-gram orders are any set, not 2..ngram_size.

import math

import torch

__all__ = ["NgramHashSpec", "build_ngram_hash_spec", "ngram_hash_ids"]

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_LAYER_SEED_STRIDE = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _primes_above(base: int, *, skip: int, count: int) -> list[int]:
    """The (skip+1)-th .. (skip+count)-th primes >= ``base``."""
    primes, candidate = [], base - 1
    while len(primes) < skip + count:
        candidate += 1
        while not _is_prime(candidate):
            candidate += 1
        primes.append(candidate)
    return primes[skip:]


class NgramHashSpec:
    """Everything that fixes which row an n-gram reads. Plain Python ints, so it
    can live on a module without being a buffer (no SPMD placement, nothing in
    the state dict) and be rebuilt bit-identically from the config alone.

    Heads are laid out order-major: ``orders[0]`` heads ``0..H-1``, then
    ``orders[1]`` ..., and head ``i`` owns rows ``[offsets[i], offsets[i] +
    primes[i])`` of one packed table of ``num_rows`` rows.
    """

    __slots__ = ("orders", "heads_per_order", "multipliers", "primes", "offsets",
                 "num_rows", "pad_id")

    def __init__(self, *, orders, heads_per_order, multipliers, primes, offsets,
                 num_rows, pad_id):
        self.orders = tuple(orders)
        self.heads_per_order = heads_per_order
        self.multipliers = tuple(multipliers)
        self.primes = tuple(primes)
        self.offsets = tuple(offsets)
        self.num_rows = num_rows
        self.pad_id = pad_id

    @property
    def num_heads(self) -> int:
        return len(self.orders) * self.heads_per_order


def build_ngram_hash_spec(
    *,
    orders: tuple[int, ...] | list[int],
    heads_per_order: int,
    rows_per_head: int,
    vocab_size: int,
    pad_id: int,
    seed: int = 0,
    layer_index: int = 0,
    row_multiple: int = 128,
) -> NgramHashSpec:
    """Hash constants for one Engram layer.

    * One odd 64-bit multiplier per look-back offset, from SplitMix64 seeded by
      ``seed + 10007 * layer_index``; bounded so ``token_id * multiplier`` never
      overflows int64 for ids below ``vocab_size``.
    * One prime modulus per (order, head): consecutive primes >= ``rows_per_head``,
      continuing across layers (``layer_index``), so no two heads of the model
      share a modulus.
    * ``num_rows`` is the sum of the primes rounded up to ``row_multiple`` --
      tensor parallelism row-shards the table and needs an even split.
    """
    orders = tuple(sorted(set(int(n) for n in orders)))
    if not orders or orders[0] < 2:
        raise ValueError(f"n-gram orders must be >= 2, got {orders}")
    if heads_per_order < 1 or rows_per_head < 2:
        raise ValueError(f"need heads_per_order >= 1 and rows_per_head >= 2, got "
                         f"{heads_per_order} / {rows_per_head}")
    if not 0 <= pad_id < vocab_size:
        raise ValueError(f"pad_id {pad_id} outside the vocabulary [0, {vocab_size})")
    max_order = orders[-1]
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + _LAYER_SEED_STRIDE * layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(max_order)
    ]
    num_heads = len(orders) * heads_per_order
    primes = _primes_above(rows_per_head, skip=layer_index * num_heads, count=num_heads)
    offsets, total = [], 0
    for p in primes:
        offsets.append(total)
        total += p
    num_rows = -(-total // row_multiple) * row_multiple
    return NgramHashSpec(orders=orders, heads_per_order=heads_per_order,
                         multipliers=multipliers, primes=primes, offsets=offsets,
                         num_rows=num_rows, pad_id=pad_id)


def ngram_hash_ids(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    spec: NgramHashSpec,
    *,
    blocked: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row ids ``(T, num_heads)`` of the suffix n-grams ending at each position.

    Args:
        tokens: ``(T,)`` token ids of one packed row.
        positions: ``(T,)`` position inside its document (restarts at 0 at every
            packed-document start). Token ``t-k`` takes part in ``t``'s n-grams
            only if ``positions[t] >= k``: n-grams never cross documents.
        blocked: optional ``(T,)`` bool, true for tokens that must never appear in
            an n-gram (vision placeholders). Look-back stops at the first blocked
            token, and stays stopped further back.

    Null behaviour: a look-back slot without a valid token (document start, or
    past a blocked token) is filled with ``spec.pad_id`` before hashing, so the
    first tokens of a document read "(pad, x)" rows -- the convention of both
    references (DeepSeek ``pad_id``, transformers EOS). Nothing is masked here;
    the caller masks positions whose OWN token is blocked.
    """
    if tokens.ndim != 1 or positions.shape != tokens.shape:
        raise ValueError(f"tokens and positions must be (T,), got {tuple(tokens.shape)} "
                         f"and {tuple(positions.shape)}")
    tokens = tokens.long()
    num_tokens = tokens.shape[0]
    device = tokens.device
    pad = torch.full((), spec.pad_id, dtype=torch.long, device=device)
    multipliers = torch.tensor(spec.multipliers, dtype=torch.long, device=device)
    primes = torch.tensor(spec.primes, dtype=torch.long, device=device)
    offsets = torch.tensor(spec.offsets, dtype=torch.long, device=device)

    valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    mixed = tokens * multipliers[0]
    ids, head = [], 0
    for shift in range(1, spec.orders[-1]):
        source = torch.cat([tokens.new_full((shift,), spec.pad_id), tokens])[:num_tokens]
        valid = valid & (positions >= shift)
        if blocked is not None:
            source_blocked = torch.cat([blocked.new_ones(shift), blocked])[:num_tokens]
            valid = valid & ~source_blocked
        mixed = torch.bitwise_xor(mixed, torch.where(valid, source, pad) * multipliers[shift])
        if shift + 1 in spec.orders:
            heads = slice(head, head + spec.heads_per_order)
            ids.append(torch.remainder(mixed.unsqueeze(-1), primes[heads]) + offsets[heads])
            head += spec.heads_per_order
    return torch.cat(ids, dim=-1)
