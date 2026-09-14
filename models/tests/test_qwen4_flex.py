"""QSA fast paths: the flex-attention kernel probe and the indexer's chunking.

Both cover bugs that were silent in the sense that mattered -- the model kept
producing correct numbers while running a path orders of magnitude more
expensive than the one it was supposed to take.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from models.qwen4 import model as m

REAL_CONFIG = "/data/151-2/users/tockier/models/qwen4/config.json"

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def clean_flex_state():
    """`_FLEX_ATTENTION` caches the resolved kernel options process-wide."""
    saved = dict(m._FLEX_ATTENTION)
    m._FLEX_ATTENTION.pop("kernel_options", None)
    m._FLEX_ATTENTION.pop("compile_failed", None)
    yield
    m._FLEX_ATTENTION.clear()
    m._FLEX_ATTENTION.update(saved)


def _qkv(total, heads, head_dim, device, requires_grad=True):
    g = torch.Generator(device=device).manual_seed(0)
    mk = lambda n: torch.randn(  # noqa: E731
        1, n, total, head_dim, device=device, dtype=torch.bfloat16,
        generator=g, requires_grad=requires_grad,
    )
    return mk(heads), mk(2), mk(2)


def _causal_mask(total, device):
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        lambda b, h, q_idx, kv_idx: kv_idx <= q_idx,
        B=None, H=None, Q_LEN=total, KV_LEN=total, device=device,
    )


@cuda_only
@pytest.mark.parametrize("head_dim", [128, 256])
def test_kernel_options_resolve_and_compile(clean_flex_state, head_dim):
    """A backward-capable config is found for both head dims we ship."""
    total = 512
    device = "cuda"
    q, k, v = _qkv(total, 8, head_dim, device)
    out = m.run_flex_attention(q, k, v, _causal_mask(total, device), head_dim ** -0.5)
    out.sum().backward()

    assert m._FLEX_ATTENTION.get("compile_failed") is None
    assert "kernel_options" in m._FLEX_ATTENTION
    assert m._FLEX_ATTENTION["kernel_options"] in m._FLEX_BWD_BLOCKS
    assert q.grad is not None and torch.isfinite(q.grad).all()


@cuda_only
def test_probe_survives_no_grad(clean_flex_state):
    """The first call can land in a `no_grad` region, and must still probe.

    `PipelineStage` runs a shape-inference forward under `no_grad` before
    training starts, so under PP this is always the first call. A probe that
    calls `backward()` without re-enabling grad raises "element 0 of tensors
    does not require grad", which then gets cached as a compile failure and
    drops every QSA layer onto the eager path for the whole run.
    """
    total, head_dim = 512, 256
    device = "cuda"
    q, k, v = _qkv(total, 8, head_dim, device)
    block_mask = _causal_mask(total, device)

    with torch.no_grad():
        m.run_flex_attention(q, k, v, block_mask, head_dim ** -0.5)

    assert m._FLEX_ATTENTION.get("compile_failed") is None, (
        "probing under no_grad marked flex-attention as uncompilable"
    )
    assert m._FLEX_ATTENTION["kernel_options"] in m._FLEX_BWD_BLOCKS


@cuda_only
def test_probe_leaves_no_grads_on_caller_tensors(clean_flex_state):
    """The probe runs its own backward; it must not touch the real tensors."""
    total, head_dim = 512, 256
    device = "cuda"
    q, k, v = _qkv(total, 8, head_dim, device)
    with torch.no_grad():
        m.run_flex_attention(q, k, v, _causal_mask(total, device), head_dim ** -0.5)
    assert q.grad is None and k.grad is None and v.grad is None


@pytest.mark.parametrize(
    "total,num_blocks",
    [(4096, 1024), (4096, 1), (32768, 8192), (128, 4), (65536, 65536)],
)
def test_query_chunk_bounds(total, num_blocks):
    chunk = m._qsa_query_chunk(total, num_blocks)
    assert chunk >= min(1024, m._QSA_CHUNK_BUDGET)   # never below the old fixed size
    assert chunk <= max(total, 1024)                 # never more rows than exist
    assert isinstance(chunk, int) and chunk > 0


def test_query_chunk_shrinks_with_block_count():
    """A wider block grid must not grow the transient allocation."""
    wide = m._qsa_query_chunk(65536, 65536)
    narrow = m._qsa_query_chunk(65536, 64)
    assert narrow > wide


@cuda_only
@pytest.mark.parametrize("lens", [[512] * 8, [4096], [1, 7, 100, 555, 1000, 2433]])
def test_selection_independent_of_chunking(lens):
    """Chunking is a memory knob; it must not change which blocks are picked."""
    from models.qwen4.config import Qwen4TextConfig

    cfg = Qwen4TextConfig.from_dict(
        json.load(open(REAL_CONFIG))["text_config"]
    )
    device = "cuda"
    total = sum(lens)
    torch.manual_seed(0)
    indexer = m.QSAIndexer(cfg).to(device, torch.bfloat16)

    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), device=device, dtype=torch.int32
    )
    seg = torch.repeat_interleave(
        torch.arange(len(lens), device=device), torch.tensor(lens, device=device)
    )
    g = torch.Generator(device=device).manual_seed(1)
    rot = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    x = torch.randn(
        1, total, cfg.hidden_size, device=device, dtype=torch.bfloat16, generator=g
    )
    cos, sin = (
        torch.randn(1, total, rot, device=device, dtype=torch.bfloat16, generator=g)
        for _ in range(2)
    )

    ref = indexer(x, cos, sin, cu, seg, query_chunk=None)
    for chunk in (1, 97, 1024, total):
        got = indexer(x, cos, sin, cu, seg, query_chunk=chunk)
        for a, b in zip(ref, got):
            assert torch.equal(a, b), f"chunk={chunk} changed the selection"


@cuda_only
def test_block_mask_matches_uncompiled(clean_flex_state):
    """Compiling `create_block_mask` must not change a single block index."""
    from torch.nn.attention.flex_attention import create_block_mask

    total, device = 1024, "cuda"
    g = torch.Generator(device=device).manual_seed(0)
    num_blocks = total // 4
    selected = torch.rand(total, num_blocks, device=device, generator=g) < 0.1
    block_of_token = (torch.arange(total, device=device) // 4).clamp(max=num_blocks - 1)
    tail_start = (torch.arange(total, device=device) // 4) * 4
    seg_id = torch.arange(total, device=device) // 256

    def mask_mod(b, h, q_idx, kv_idx):
        return (
            (seg_id[q_idx] == seg_id[kv_idx])
            & (kv_idx <= q_idx)
            & (selected[q_idx, block_of_token[kv_idx]] | (kv_idx >= tail_start[q_idx]))
        )

    fast = m._create_block_mask()(
        mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device
    )
    slow = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device
    )
    assert fast.sparsity() == slow.sparsity()
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices"):
        a, b = getattr(fast, name), getattr(slow, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name
