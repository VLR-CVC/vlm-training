# Chunked loss vendored from torchtitan b21f7d43e: torchtitan/components/loss.py
# (ChunkedLossWrapper, GradAccumulator, _DecoderOutputGradientBackProp,
# _LossParallelCrossEntropy, cross_entropy_loss), single output, no metrics.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
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
    """Vocab-parallel cross-entropy on local ``[T, V_local]`` logits. SPMD friendly"""

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
        negates to get per-token logprobs without all-gathering the vocab (not used in practice).
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

class _MultiOutputGradientBackProp(torch.autograd.Function):
    """Variadic ``_DecoderOutputGradientBackProp``: returns one accumulated
    hidden-state gradient per model output, so ``loss.backward()`` continues
    through the decoder AND every MTP depth in a single pass.

    MTP depth k's output feeds depth k+1, so the graph reconverges: autograd sums
    the injected gradient at each output with whatever arrives from the depth
    above it. That is why the gradients are injected per output rather than
    summed into the main hidden state.
    """

    @staticmethod
    def forward(ctx, total_loss, num_outputs, *hiddens_and_grads):
        ctx.num_outputs = num_outputs
        ctx.save_for_backward(*hiddens_and_grads[num_outputs:])
        return total_loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        grads = ctx.saved_tensors
        # (total_loss, num_outputs, *hiddens, *grads)
        return (None, None, *grads, *([None] * ctx.num_outputs))

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

# always worth compiling, 85ms/step vs 18ms/step (Qwen3.5-2B, 1 GPU, 8192 x 2)
_compiled_cross_entropy_sum = torch.compile(cross_entropy_sum)
_compiled_vocab_parallel_cross_entropy_sum = torch.compile(vocab_parallel_cross_entropy_sum)

def chunked_loss(
    hidden: torch.Tensor | tuple[torch.Tensor, ...],
    labels: torch.Tensor | tuple[torch.Tensor, ...],
    lm_head: nn.Module,
    global_tokens: torch.Tensor,
    num_chunks: int = 8,
    compile_loss: bool = False,
    global_vocab_size: int | None = None,
    sync_grads: bool = True,
    mtp_scale: float = 0.3,
    metrics: dict | None = None,
) -> torch.Tensor:
    """``sum_t CE(lm_head(hidden_t), labels_t) / global_tokens``, computed chunk by chunk"""
    hiddens = (hidden,) if isinstance(hidden, torch.Tensor) else tuple(hidden)
    label_rows = (labels,) if isinstance(labels, torch.Tensor) else tuple(labels)
    if len(hiddens) != len(label_rows):
        raise ValueError(
            f"chunked_loss needs one labels row per output, got {len(hiddens)} "
            f"outputs and {len(label_rows)} label rows"
        )
    num_outputs = len(hiddens)
    num_depths = num_outputs - 1
    # main path weight 1, then mtp_scale spread over the depths. `num_depths`
    # is 0 on the ordinary single-output path, which must not divide by zero.
    weights = [1.0] + [mtp_scale / num_depths] * num_depths if num_depths else [1.0]

    seq_len = hiddens[0].shape[0]
    if seq_len % num_chunks:
        raise ValueError(f"sequence length {seq_len} not divisible by {num_chunks} loss chunks")
    chunk_len = seq_len // num_chunks
    chunked = [
        [c.detach().requires_grad_(True) for c in torch.split(h, chunk_len)]
        for h in hiddens
    ]
    chunked_labels = [torch.split(lab, chunk_len) for lab in label_rows]
    grad_buffers = [torch.zeros_like(h, dtype=torch.float32) for h in hiddens]
    total = hiddens[0].new_zeros((), dtype=torch.float32)
    per_output = [hiddens[0].new_zeros((), dtype=torch.float32) for _ in hiddens]

    sharded = isinstance(lm_head, FSDPModule) and not isinstance(lm_head, ReplicateModule)
    if sharded:
        lm_head.set_reshard_after_forward(False)
        lm_head.set_reshard_after_backward(False)
        lm_head.set_requires_gradient_sync(False, recurse=False)
        lm_head.unshard()

    # the backward carries TP collectives outside any typechecked region
    for i in range(num_chunks):
        for out_idx in range(num_outputs):
            h = chunked[out_idx][i]
            lab = chunked_labels[out_idx][i]
            # the gradient is reduced once, after the very last (chunk, output)
            if sharded and i == num_chunks - 1 and out_idx == num_outputs - 1:
                lm_head.set_requires_gradient_sync(sync_grads, recurse=False)
            if spmd_mesh_size("tp") > 1:
                vp_ce = (_compiled_vocab_parallel_cross_entropy_sum if compile_loss
                         else vocab_parallel_cross_entropy_sum)
                chunk_loss = vp_ce(lm_head(h), lab, global_vocab_size)
            else:
                ce = _compiled_cross_entropy_sum if compile_loss else cross_entropy_sum
                chunk_loss = ce(lm_head(h), lab)
            # the weight is applied BEFORE backward so the depth's gradient
            # carries it; weighting the reported loss afterwards would leave the
            # gradients unscaled and the auxiliary objective at full strength
            loss = chunk_loss * weights[out_idx] / global_tokens
            total = total + loss.detach()
            per_output[out_idx] = per_output[out_idx] + loss.detach()
            with spmd.no_typecheck():
                loss.backward()
            grad_buffers[out_idx][i * chunk_len : (i + 1) * chunk_len] = h.grad
            h.grad = None

    if sharded:
        lm_head.set_reshard_after_forward(True)
        lm_head.set_reshard_after_backward(True)
        lm_head.reshard()

    # single output: `loss_main` would just repeat the reported loss, and the
    # caller labels this dict "mtp:" -- nothing to break down
    if metrics is not None and num_outputs > 1:
        metrics["loss_main"] = per_output[0]
        for depth in range(1, num_outputs):
            # already weighted; divide back out to report the raw depth loss
            metrics[f"loss_mtp_{depth}"] = per_output[depth] / weights[depth]

    grads = [buf.to(h.dtype) for buf, h in zip(grad_buffers, hiddens, strict=True)]
    if num_outputs == 1:
        return _DecoderOutputGradientBackProp.apply(hiddens[0], grads[0], total)
    return _MultiOutputGradientBackProp.apply(total, num_outputs, *hiddens, *grads)
