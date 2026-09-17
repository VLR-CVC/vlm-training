# Chunked loss vendored from torchtitan b21f7d43e: torchtitan/components/loss.py
# (ChunkedLossWrapper, GradAccumulator, _DecoderOutputGradientBackProp,
# _LossParallelCrossEntropy, cross_entropy_loss), single output, no metrics.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""Token-summed cross-entropy for the torchtitan-port path.

The trainer divides the sum by the number of valid tokens in the whole step --
every micro-batch on every DP rank, counted and all-reduced before the first
forward (torchtitan `trainer.py:872`). Each supervised token then carries the same
weight whatever the accumulation pattern or DP size.

`chunked_loss` never materialises the full [T, V] logits. Measured on Qwen3.5-2B,
T=8192, one GPU, compiled blocks: logits in bf16, their fp32 copy, the fp32
log-softmax and the fp32 logit gradient added ~23 GiB to the step's peak
(50.6 GiB against ~28 GiB after the forward) -- more than all 24 decoder layers'
activations.
"""

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate_with_fsdp import ReplicateModule

from train.parallel.spmd import current_spmd_mesh, spmd_mesh_size

IGNORE_INDEX = -100


def cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of token losses over ``logits[T, V]`` and already-shifted ``labels[T]``."""
    return F.cross_entropy(logits.float(), labels, reduction="sum", ignore_index=IGNORE_INDEX)


class _LossParallelCrossEntropy(torch.autograd.Function):
    """
    Vocab-parallel cross-entropy on local ``[T, V_local]`` logits.

    Replaces ``torch.distributed.tensor.parallel.loss_parallel()`` with an
    explicit autograd Function so that SPMD code can operate on local tensors
    and process groups directly, without the DTensor-based context manager.

    Supports uneven vocab sharding (last TP rank may hold fewer classes) and
    ``IGNORE_INDEX`` labels.  Forward uses three TP all-reduces (max, sumexp,
    gather) to aggregate intermediate results in distributed softmax;
    backward is fused (NLL + log-softmax) with zero collectives.

    All inputs and outputs are plain ``torch.Tensor`` (not DTensor).
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> None:
        """
        SPMD type: logits S(-1)@TP, labels I@TP -> loss I@TP.
        Non-TP axes are passed through from logits to the output.
        """
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        spmd.assert_local_type_like(
            result,
            logits,
            {tp_group: spmd.I},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute exact CE from local vocab shards via TP all-reduces.

        ``reduction="sum"`` returns the scalar summed loss (SFT/CE).
        ``reduction="none"`` returns the per-token NLL ``[T]``, which GRPO
        negates to get per-token logprobs without all-gathering the vocab.
        """
        logits_dtype = logits.dtype
        logits = logits.float()

        # Compute this rank's vocab shard bounds for the local logits.
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
        vocab_start = min(global_vocab_size, chunk_size * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk_size)
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "_LossParallelCrossEntropy does not support empty vocab shards."
            )

        torch._assert_async(
            torch.all(
                (labels == IGNORE_INDEX)
                | ((labels >= 0) & (labels < global_vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {global_vocab_size})",
        )

        # All-reduce max for numerically stable distributed log-softmax.
        local_max = torch.amax(logits, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # All-reduce sum over shifted logits for the global softmax denominator.
        shifted = logits - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(
            shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        log_probs = shifted - torch.log(shifted_sumexp)

        # Mask labels outside this vocab shard; the TP all-reduce below selects
        # the owner rank's log probability for each target token.
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
        out_of_range = (safe_labels < vocab_start) | (
            safe_labels >= vocab_start + local_vocab_size
        )
        local_labels = safe_labels - vocab_start
        local_labels[out_of_range] = 0

        local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        local_result[out_of_range.unsqueeze(-1)] = 0
        local_result = funcol.all_reduce(
            local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )

        # Per-token NLL, dropping ignored labels (logprob 0 for ignored).
        result = -local_result.squeeze(-1)
        result = torch.where(labels != IGNORE_INDEX, result, 0)

        # Save local-shard log probabilities for the fused CE backward.
        ctx.save_for_backward(log_probs, labels)
        ctx.logits_dtype = logits_dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        ctx.reduction = reduction
        if reduction == "none":
            return result
        return result.sum()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        log_probs, labels = ctx.saved_tensors
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
        out_of_range = (safe_labels < ctx.vocab_start) | (
            safe_labels >= ctx.vocab_start + ctx.local_vocab_size
        )
        local_labels = safe_labels - ctx.vocab_start
        local_labels[out_of_range] = 0

        grad_input = torch.zeros_like(log_probs)
        row_idx = torch.arange(local_labels.shape[0], device=local_labels.device)
        grad_update = out_of_range.to(grad_input.dtype) - 1.0
        grad_input[row_idx, local_labels] = grad_update

        # reduction="none" gives a per-token ``[T]`` upstream grad; unsqueeze to
        # ``[T, 1]`` to broadcast over the local vocab. "sum" gives the scalar
        # loss grad, which broadcasts as-is.
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where(
            (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        grad_logits = grad_logits.to(ctx.logits_dtype)
        return grad_logits, None, None, None, None


def vocab_parallel_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, global_vocab_size: int
) -> torch.Tensor:
    """torchtitan's ``cross_entropy_loss`` under TP: logits are this rank's vocab
    shard (``lm_head`` output ``Shard(-1)``), never gathered."""
    return _LossParallelCrossEntropy.apply(
        logits.float(), labels, current_spmd_mesh().get_group("tp"), global_vocab_size
    )


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Returns the accumulated hidden-state gradient as this node's backward, so
    ``loss.backward()`` continues through the decoder in a single pass."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, grad: torch.Tensor, total_loss: torch.Tensor):
        ctx.save_for_backward(grad)
        return total_loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad, None, None


# torchtitan compiles its loss function (BaseLoss._maybe_compile). Eager F.cross_entropy
# over a [T/8, 248320] chunk ran cunn_SoftMaxForward/Backward at 85 ms/step against
# 18 ms for the fused triton kernel (Qwen3.5-2B, 1 GPU, 8192 x 2).
_compiled_cross_entropy_sum = torch.compile(cross_entropy_sum)
# Same under TP: eager _LossParallelCrossEntropy ran its exp/add/max as separate
# kernels, ~60 ms/step more than torchtitan's compiled loss (2B, TP=2+SP, 8192 x 2).
_compiled_vocab_parallel_cross_entropy_sum = torch.compile(vocab_parallel_cross_entropy_sum)


def chunked_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    denom: torch.Tensor,
    num_chunks: int = 8,
    compile_loss: bool = False,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """``sum_t CE(lm_head(hidden_t), labels_t) / denom``, computed chunk by chunk.

    Each chunk: lm_head -> fp32 CE -> backward into a detached leaf, its gradient
    copied into a preallocated fp32 buffer. The returned loss's backward hands that
    buffer to the decoder. ``hidden`` comes from the model with ``_skip_lm_head``.

    Under FSDP lm_head stays unsharded across chunks and its gradient is reduced
    once, on the last chunk. Under replicate the gradient is reduced per chunk:
    torch 2.14's replicate breaks when gradient sync is disabled
    (see train/titan_step.py).
    """
    seq_len = hidden.shape[0]
    if seq_len % num_chunks:
        raise ValueError(f"sequence length {seq_len} not divisible by {num_chunks} loss chunks")
    chunk_len = seq_len // num_chunks
    h_chunks = [c.detach().requires_grad_(True) for c in torch.split(hidden, chunk_len)]
    l_chunks = torch.split(labels, chunk_len)
    grad_buffer = torch.zeros_like(hidden, dtype=torch.float32)
    total = hidden.new_zeros((), dtype=torch.float32)

    sharded = isinstance(lm_head, FSDPModule) and not isinstance(lm_head, ReplicateModule)
    if sharded:
        lm_head.set_reshard_after_forward(False)
        lm_head.set_reshard_after_backward(False)
        lm_head.set_requires_gradient_sync(False, recurse=False)
        lm_head.unshard()

    # the backward carries TP collectives outside any typechecked region
    for i, (h, lab) in enumerate(zip(h_chunks, l_chunks)):
        if sharded and i == num_chunks - 1:
            lm_head.set_requires_gradient_sync(True, recurse=False)
        if spmd_mesh_size("tp") > 1:
            vp_ce = (_compiled_vocab_parallel_cross_entropy_sum if compile_loss
                     else vocab_parallel_cross_entropy_sum)
            chunk_loss = vp_ce(lm_head(h), lab, global_vocab_size)
        else:
            ce = _compiled_cross_entropy_sum if compile_loss else cross_entropy_sum
            chunk_loss = ce(lm_head(h), lab)
        loss = chunk_loss / denom
        total = total + loss.detach()
        with spmd.no_typecheck():
            loss.backward()
        grad_buffer[i * chunk_len : (i + 1) * chunk_len] = h.grad
        h.grad = None

    if sharded:
        lm_head.set_reshard_after_forward(True)
        lm_head.set_reshard_after_backward(True)
        lm_head.reshard()

    return _DecoderOutputGradientBackProp.apply(hidden, grad_buffer.to(hidden.dtype), total)
