"""The (B, S, H, D) attention rewrite must not move a single bit.

`Qwen3VLTextAttention` used to hop to (B, H, S, D) for the norms and rope and
back again; the round trip is not a view, so it copied q, k and v on every
layer. The layout change is supposed to be pure bookkeeping -- these tests hold
it to that against the shape dance it replaced.
"""

import pytest
import torch

from models.qwen3_vl.model import (
    Qwen3VLTextAttention,
    Qwen3VLTextConfig,
    apply_rope,
    apply_rope_shd,
    dispatch_varlen_attention,
)


def _cfg(**kw):
    base = dict(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=4, head_dim=16, rms_norm_eps=1e-6,
        rope_theta=10000.0, mrope_section=[3, 3, 2],
        tie_word_embeddings=False, max_position_embeddings=512,
    )
    base.update(kw)
    return Qwen3VLTextConfig(**base)


def _ref_forward(attn, x, cos, sin, cu_seqlens, max_seqlen):
    """The pre-rewrite body, verbatim."""
    total = x.shape[1]
    q = attn.q_proj(x).view(1, total, attn.num_heads, attn.head_dim)
    k = attn.k_proj(x).view(1, total, attn.num_kv_heads, attn.head_dim)
    v = attn.v_proj(x).view(1, total, attn.num_kv_heads, attn.head_dim)

    q = attn.q_norm(q).transpose(1, 2)
    k = attn.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    q = q.transpose(1, 2).reshape(total, attn.num_heads, attn.head_dim).contiguous()
    k = k.transpose(1, 2).reshape(total, attn.num_kv_heads, attn.head_dim).contiguous()
    v = v.transpose(1, 2).reshape(total, attn.num_kv_heads, attn.head_dim).contiguous()

    out = dispatch_varlen_attention(q, k, v, cu_seqlens, max_seqlen)
    out = out.reshape(1, total, attn.num_heads * attn.head_dim)
    return attn.o_proj(out)


def _inputs(cfg, lens, device, dtype):
    total = sum(lens)
    torch.manual_seed(0)
    x = torch.randn(1, total, cfg.hidden_size, device=device, dtype=dtype)
    cos = torch.randn(1, total, cfg.head_dim, device=device, dtype=dtype)
    sin = torch.randn(1, total, cfg.head_dim, device=device, dtype=dtype)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                      device=device, dtype=torch.int32)
    return x, cos, sin, cu, max(lens)


def test_rope_layouts_agree():
    torch.manual_seed(0)
    B, S, H, D = 1, 32, 8, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H // 2, D)
    cos = torch.randn(S, D)
    sin = torch.randn(S, D)
    qs, ks = apply_rope_shd(q, k, cos, sin)
    qh, kh = apply_rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
    assert torch.equal(qs, qh.transpose(1, 2))
    assert torch.equal(ks, kh.transpose(1, 2))


@pytest.mark.cuda_only
@pytest.mark.parametrize("lens", [[32], [16, 48, 8]])
def test_attention_matches_previous_layout(lens):
    if not torch.cuda.is_available():
        pytest.skip("varlen_attn is CUDA-only")
    device, dtype = "cuda", torch.bfloat16
    cfg = _cfg()
    torch.manual_seed(0)
    attn = Qwen3VLTextAttention(cfg).to(device=device, dtype=dtype)
    x, cos, sin, cu, mx = _inputs(cfg, lens, device, dtype)

    got = attn(x, cos, sin, cu, mx)
    want = _ref_forward(attn, x, cos, sin, cu, mx)
    assert torch.equal(got, want)


@pytest.mark.cuda_only
def test_attention_grads_match_previous_layout():
    if not torch.cuda.is_available():
        pytest.skip("varlen_attn is CUDA-only")
    device, dtype = "cuda", torch.bfloat16
    cfg = _cfg()
    torch.manual_seed(0)
    attn = Qwen3VLTextAttention(cfg).to(device=device, dtype=dtype)
    x, cos, sin, cu, mx = _inputs(cfg, [16, 48], device, dtype)

    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    attn(a, cos, sin, cu, mx).sum().backward()
    grads_new = [p.grad.clone() for p in attn.parameters()]
    for p in attn.parameters():
        p.grad = None
    _ref_forward(attn, b, cos, sin, cu, mx).sum().backward()

    assert torch.equal(a.grad, b.grad)
    for g_new, p in zip(grads_new, attn.parameters()):
        assert torch.equal(g_new, p.grad)
