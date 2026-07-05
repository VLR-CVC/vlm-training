from __future__ import annotations

import typing

import torch

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
    from fla.modules.fused_norm_gate import rms_norm_gated as _fla_rms_norm_gated
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
    from fla.ops.gated_delta_rule.chunk import (
        l2norm_fwd as _l2norm_fwd,
        l2norm_bwd as _l2norm_bwd,
        prepare_chunk_indices as _prepare_chunk_indices,
        chunk_gated_delta_rule_fwd as _gdr_fwd,
        chunk_gated_delta_rule_bwd as _gdr_bwd_kernel,
    )
    # low-level split of causal_conv1d and rms_norm_gated (fla)
    from causal_conv1d.causal_conv1d_interface import (
        causal_conv1d_fwd_function as _conv_fwd_fn,
        causal_conv1d_bwd_function as _conv_bwd_fn,
    )
    from fla.modules.fused_norm_gate import (
        layer_norm_gated_fwd as _lng_fwd,
        layer_norm_gated_bwd as _lng_bwd,
    )
except Exception:
    pass

# TODO: deal with this
_GDR_CHUNK_SIZE = 64

@torch.library.custom_op("qwen3_5::gated_delta_rule", mutates_args=())
def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor,
                     cu_seqlens: torch.Tensor) -> torch.Tensor:
    out, _ = _fla_chunk_gated_delta_rule(
        q, k, v, g, beta,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.to(torch.int64),
    )
    return out


@gated_delta_rule.register_fake
def _(q, k, v, g, beta, cu_seqlens):
    return torch.empty_like(v)  # output is the value stream (B, L, Hv, Dv)


@torch.library.custom_op("qwen3_5::gated_delta_rule_bwd", mutates_args=())
def _gated_delta_rule_bwd(grad_out: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                          cu_seqlens: torch.Tensor) -> typing.List[torch.Tensor]:
    with torch.enable_grad():
        qd, kd, vd, gd, bd = (t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
        out, _ = _fla_chunk_gated_delta_rule(
            qd, kd, vd, gd, bd,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens.to(torch.int64),
        )
        grads = torch.autograd.grad(out, (qd, kd, vd, gd, bd), grad_out)
    return list(grads)


@_gated_delta_rule_bwd.register_fake
def _(grad_out, q, k, v, g, beta, cu_seqlens):
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
            torch.empty_like(g), torch.empty_like(beta)]


def _gdr_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _gdr_backward(ctx, grad_out):
    gq, gk, gv, gg, gb = _gated_delta_rule_bwd(grad_out, *ctx.saved_tensors)
    return gq, gk, gv, gg, gb, None  # None -> cu_seqlens


gated_delta_rule.register_autograd(_gdr_backward, setup_context=_gdr_setup)

@torch.library.custom_op("qwen3_5::gated_delta_rule_native", mutates_args=())
def _gdr_native_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    g: torch.Tensor, beta: torch.Tensor, cu_seqlens: torch.Tensor
                    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # the low-level fwd/bwd kernels lack fla's @input_guard, which makes inputs
    # contiguous; under compile the inputs can be non-contiguous, so guard here.
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = q.shape[-1] ** -0.5
    q_n, q_rstd = _l2norm_fwd(q)
    k_n, k_rstd = _l2norm_fwd(k)
    chunk_indices = _prepare_chunk_indices(cu_seqlens.to(torch.int64), _GDR_CHUNK_SIZE)
    g_out, o, A, _, _ = _gdr_fwd(
        q=q_n, k=k_n, v=v, g=g, beta=beta, scale=scale,
        initial_state=None, output_final_state=False,
        cu_seqlens=cu_seqlens.to(torch.int64), cp_context=None,
        chunk_indices=chunk_indices, transpose_state_layout=False,
    )
    return o.to(q.dtype), q_n, q_rstd, k_n, k_rstd, g_out, A


@_gdr_native_fwd.register_fake
def _(q, k, v, g, beta, cu_seqlens):
    B, L, H, _ = q.shape
    o = torch.empty_like(v)
    q_n = torch.empty_like(q)
    k_n = torch.empty_like(k)
    q_rstd = q.new_empty((B, L, H), dtype=torch.float32)
    k_rstd = q.new_empty((B, L, H), dtype=torch.float32)
    g_out = torch.empty_like(g)  # (B, L, H) float32
    A = q.new_empty((B, L, H, _GDR_CHUNK_SIZE), dtype=q.dtype)
    return o, q_n, q_rstd, k_n, k_rstd, g_out, A


@torch.library.custom_op("qwen3_5::gated_delta_rule_native_bwd", mutates_args=())
def _gdr_native_bwd(grad_o: torch.Tensor, q_n: torch.Tensor, q_rstd: torch.Tensor,
                    k_n: torch.Tensor, k_rstd: torch.Tensor, v: torch.Tensor,
                    g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor,
                    cu_seqlens: torch.Tensor, scale: float) -> typing.List[torch.Tensor]:
    cu = cu_seqlens.to(torch.int64)
    grad_o = grad_o.contiguous()
    q_n, k_n, v, g_out, beta, A = (t.contiguous() for t in (q_n, k_n, v, g_out, beta, A))
    chunk_indices = _prepare_chunk_indices(cu, _GDR_CHUNK_SIZE)
    dq, dk, dv, db, dg, _ = _gdr_bwd_kernel(
        q=q_n, k=k_n, v=v, g=g_out, beta=beta, A=A, scale=scale,
        initial_state=None, do=grad_o, dht=None, cu_seqlens=cu,
        cp_context=None, chunk_indices=chunk_indices, transpose_state_layout=False,
    )
    dq = _l2norm_bwd(q_n, q_rstd, dq)
    dk = _l2norm_bwd(k_n, k_rstd, dk)
    return [dq, dk, dv, dg, db]  # mapped to inputs (q, k, v, g, beta)


@_gdr_native_bwd.register_fake
def _(grad_o, q_n, q_rstd, k_n, k_rstd, v, g_out, beta, A, cu_seqlens, scale):
    return [torch.empty_like(q_n), torch.empty_like(k_n), torch.empty_like(v),
            torch.empty_like(g_out), torch.empty_like(beta)]


def _gdr_native_setup(ctx, inputs, output):
    q, k, v, g, beta, cu_seqlens = inputs
    o, q_n, q_rstd, k_n, k_rstd, g_out, A = output
    ctx.scale = q.shape[-1] ** -0.5
    ctx.save_for_backward(q_n, q_rstd, k_n, k_rstd, v, g_out, beta, A, cu_seqlens)


def _gdr_native_backward(ctx, grad_o, *rest):
    # rest = grads for the saved-activation outputs; unused (they don't feed loss).
    q_n, q_rstd, k_n, k_rstd, v, g_out, beta, A, cu_seqlens = ctx.saved_tensors
    dq, dk, dv, dg, db = _gdr_native_bwd(
        grad_o, q_n, q_rstd, k_n, k_rstd, v, g_out, beta, A, cu_seqlens, ctx.scale
    )
    return dq, dk, dv, dg, db, None  # None -> cu_seqlens


_gdr_native_fwd.register_autograd(_gdr_native_backward, setup_context=_gdr_native_setup)


def gated_delta_rule_native(q, k, v, g, beta, cu_seqlens):
    """Native-backward gated delta rule. Returns just `o` (drops the saved
    activations the custom op exposes for its own backward)."""
    return _gdr_native_fwd(q, k, v, g, beta, cu_seqlens)[0]


@torch.library.custom_op("qwen3_5::causal_conv1d", mutates_args=())
def causal_conv1d(x: torch.Tensor, weight: torch.Tensor,
                  bias: typing.Optional[torch.Tensor],
                  seq_idx: torch.Tensor) -> torch.Tensor:
    return _causal_conv1d_fn(x=x, weight=weight, bias=bias, seq_idx=seq_idx, activation="silu")


@causal_conv1d.register_fake
def _(x, weight, bias, seq_idx):
    return torch.empty_like(x)  # (B, C, L) preserved


@torch.library.custom_op("qwen3_5::causal_conv1d_bwd", mutates_args=())
def _causal_conv1d_bwd(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                       bias: typing.Optional[torch.Tensor],
                       seq_idx: torch.Tensor) -> typing.List[torch.Tensor]:
    # always returns [grad_x, grad_weight, grad_bias]; grad_bias is a 0-size
    # sentinel when there is no bias (the autograd wrapper maps it back to None).
    with torch.enable_grad():
        xd = x.detach().requires_grad_(True)
        wd = weight.detach().requires_grad_(True)
        if bias is not None:
            bd = bias.detach().requires_grad_(True)
            out = _causal_conv1d_fn(x=xd, weight=wd, bias=bd, seq_idx=seq_idx, activation="silu")
            gx, gw, gb = torch.autograd.grad(out, (xd, wd, bd), grad_out)
        else:
            out = _causal_conv1d_fn(x=xd, weight=wd, bias=None, seq_idx=seq_idx, activation="silu")
            gx, gw = torch.autograd.grad(out, (xd, wd), grad_out)
            gb = x.new_empty(0)
    return [gx, gw, gb]


@_causal_conv1d_bwd.register_fake
def _(grad_out, x, weight, bias, seq_idx):
    gb = torch.empty_like(bias) if bias is not None else x.new_empty(0)
    return [torch.empty_like(x), torch.empty_like(weight), gb]


def _conv_setup(ctx, inputs, output):
    x, weight, bias, seq_idx = inputs
    ctx.bias_is_none = bias is None
    ctx.save_for_backward(x, weight, bias, seq_idx)


def _conv_backward(ctx, grad_out):
    x, weight, bias, seq_idx = ctx.saved_tensors
    gx, gw, gb = _causal_conv1d_bwd(grad_out, x, weight, bias, seq_idx)
    return gx, gw, (None if ctx.bias_is_none else gb), None  # None -> seq_idx


causal_conv1d.register_autograd(_conv_backward, setup_context=_conv_setup)


@torch.library.custom_op("qwen3_5::rms_norm_gated", mutates_args=())
def rms_norm_gated(hs: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    return _fla_rms_norm_gated(
        hs, gate, weight, None, "swish",
        residual=None, eps=eps, prenorm=False, residual_in_fp32=False,
    )


@rms_norm_gated.register_fake
def _(hs, gate, weight, eps):
    return torch.empty_like(hs)


@torch.library.custom_op("qwen3_5::rms_norm_gated_bwd", mutates_args=())
def _rms_norm_gated_bwd(grad_out: torch.Tensor, hs: torch.Tensor, gate: torch.Tensor,
                        weight: torch.Tensor, eps: float) -> typing.List[torch.Tensor]:
    with torch.enable_grad():
        hd, gd, wd = (t.detach().requires_grad_(True) for t in (hs, gate, weight))
        out = _fla_rms_norm_gated(
            hd, gd, wd, None, "swish",
            residual=None, eps=eps, prenorm=False, residual_in_fp32=False,
        )
        grads = torch.autograd.grad(out, (hd, gd, wd), grad_out)
    return list(grads)


@_rms_norm_gated_bwd.register_fake
def _(grad_out, hs, gate, weight, eps):
    return [torch.empty_like(hs), torch.empty_like(gate), torch.empty_like(weight)]


def _rms_setup(ctx, inputs, output):
    hs, gate, weight, eps = inputs
    ctx.eps = eps
    ctx.save_for_backward(hs, gate, weight)


def _rms_backward(ctx, grad_out):
    hs, gate, weight = ctx.saved_tensors
    ghs, ggate, gweight = _rms_norm_gated_bwd(grad_out, hs, gate, weight, ctx.eps)
    return ghs, ggate, gweight, None  # None -> eps (non-tensor)


rms_norm_gated.register_autograd(_rms_backward, setup_context=_rms_setup)


def _to_channel_last(x):
    # causal_conv1d requires channel-last (stride(1)==1) when seq_idx is used.
    return x if x.stride(1) == 1 else x.movedim(1, -1).contiguous().movedim(-1, 1)


@torch.library.custom_op("qwen3_5::causal_conv1d_native", mutates_args=())
def _conv_native_fwd(x: torch.Tensor, weight: torch.Tensor,
                     bias: typing.Optional[torch.Tensor],
                     seq_idx: torch.Tensor) -> torch.Tensor:
    x = _to_channel_last(x)
    return _conv_fwd_fn(x, weight.contiguous(), bias, seq_idx.contiguous(), None, None, True)


@_conv_native_fwd.register_fake
def _(x, weight, bias, seq_idx):
    return torch.empty_like(_to_channel_last(x))


@torch.library.custom_op("qwen3_5::causal_conv1d_native_bwd", mutates_args=())
def _conv_native_bwd(dout: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                     bias: typing.Optional[torch.Tensor],
                     seq_idx: torch.Tensor) -> typing.List[torch.Tensor]:
    x = _to_channel_last(x)
    dout = _to_channel_last(dout)
    dx, dw, db, _ = _conv_bwd_fn(x, weight.contiguous(), bias, dout, seq_idx.contiguous(),
                                 None, None, None, False, True)
    if db is None:
        db = x.new_empty(0)
    return [dx, dw, db]


@_conv_native_bwd.register_fake
def _(dout, x, weight, bias, seq_idx):
    gb = torch.empty_like(bias) if bias is not None else x.new_empty(0)
    return [torch.empty_like(x), torch.empty_like(weight), gb]


def _conv_native_setup(ctx, inputs, output):
    x, weight, bias, seq_idx = inputs
    ctx.bias_is_none = bias is None
    ctx.save_for_backward(x, weight, bias, seq_idx)


def _conv_native_backward(ctx, grad_out):
    x, weight, bias, seq_idx = ctx.saved_tensors
    dx, dw, db = _conv_native_bwd(grad_out, x, weight, bias, seq_idx)
    return dx, dw, (None if ctx.bias_is_none else db), None  # None -> seq_idx


_conv_native_fwd.register_autograd(_conv_native_backward, setup_context=_conv_native_setup)


def causal_conv1d_native(x, weight, bias, seq_idx):
    return _conv_native_fwd(x, weight, bias, seq_idx)


@torch.library.custom_op("qwen3_5::rms_norm_gated_native", mutates_args=())
def _rms_native_fwd(hs: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                    eps: float) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    hs, gate, weight = hs.contiguous(), gate.contiguous(), weight.contiguous()
    y, _mean, rstd, _res = _lng_fwd(
        hs, gate, weight, None, "swish", eps,
        residual=None, residual_dtype=None, is_rms_norm=True,
    )
    return y, rstd


@_rms_native_fwd.register_fake
def _(hs, gate, weight, eps):
    return torch.empty_like(hs), hs.new_empty((hs.shape[0],), dtype=torch.float32)


@torch.library.custom_op("qwen3_5::rms_norm_gated_native_bwd", mutates_args=())
def _rms_native_bwd(dy: torch.Tensor, hs: torch.Tensor, gate: torch.Tensor,
                    weight: torch.Tensor, rstd: torch.Tensor,
                    eps: float) -> typing.List[torch.Tensor]:
    dy, hs, gate, weight = (t.contiguous() for t in (dy, hs, gate, weight))
    dx, dg, dw, _db, _dres = _lng_bwd(
        dy, hs, gate, weight, None, "swish", eps,
        None, rstd, None, False, True, hs.dtype,
    )
    return [dx, dg, dw]


@_rms_native_bwd.register_fake
def _(dy, hs, gate, weight, rstd, eps):
    return [torch.empty_like(hs), torch.empty_like(gate), torch.empty_like(weight)]


def _rms_native_setup(ctx, inputs, output):
    hs, gate, weight, eps = inputs
    _y, rstd = output
    ctx.eps = eps
    ctx.save_for_backward(hs, gate, weight, rstd)


def _rms_native_backward(ctx, grad_y, grad_rstd):
    hs, gate, weight, rstd = ctx.saved_tensors
    dx, dg, dw = _rms_native_bwd(grad_y, hs, gate, weight, rstd, ctx.eps)
    return dx, dg, dw, None  # None -> eps


_rms_native_fwd.register_autograd(_rms_native_backward, setup_context=_rms_native_setup)


def rms_norm_gated_native(hs, gate, weight, eps):
    return _rms_native_fwd(hs, gate, weight, eps)[0]
