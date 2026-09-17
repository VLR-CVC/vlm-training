"""Forward/backward of one optimizer step for `models/qwen3_5_tt` and `models/qwen3_vl_tt`.

Shared by `train_qwen.py` and `models/tests/test_dp_parity.py`, so the test
exercises the code the trainer runs.
"""

import contextlib

import torch
from torch.profiler import record_function

from train.parallel.spmd import set_current_spmd_mesh, set_spmd_meshes

from train.loss import chunked_loss, cross_entropy_sum


def forward_backward(
    model,
    batches,
    *,
    dp_group,
    special_tokens: dict,
    ddp: bool = False,
    loss_chunks: int = 8,
    compile_loss: bool = False,
    parallel_dims=None,
):
    """Run every micro-batch of a step; return (accumulated_loss, local_tokens, denom).

    torchtitan's normalisation (`trainer.py:872`): count valid tokens in every
    micro-batch on every DP rank before the first forward, then weight each token
    by 1/global_tokens. Gradients are reduced as a SUM (FSDP division is off), so
    the result is independent of DP size and accumulation.

    ``accumulated_loss`` is local_loss_sum / global_tokens: its SUM over DP ranks is
    the exact global per-token mean.

    Gradients are reduced on every micro-batch, also under replicate. Skipping the
    sync on all but the last (``set_requires_gradient_sync(False)``) is what
    torchtitan does for HSDP, but FSDP2 ``replicate`` in torch 2.14 then fails in
    backward with "'FSDPParam' object has no attribute '_unsharded_param'"
    (models/tests/test_dp_parity.py, both 1 and 2 ranks). With SUM reduction
    reducing every micro-batch is still exact, it only costs the extra all-reduces.
    ponytail: revisit ``ddp`` no-sync on a torch where replicate supports it.
    """
    local_tokens = sum(b.pop("num_valid_tokens") for b in batches)
    global_tokens = torch.tensor(local_tokens, dtype=torch.int64, device="cuda")
    if dp_group is not None and dp_group.size() > 1:
        torch.distributed.all_reduce(global_tokens, group=dp_group.get_group())
    # an all-masked step would divide by zero; its gradient is zero anyway
    denom = global_tokens.clamp(min=1).to(torch.float32)

    accumulated = torch.zeros((), dtype=torch.float32, device="cuda")
    # torchtitan's get_spmd_context: module-boundary redistributions and the
    # vocab-parallel loss look up the TP group from the current dense mesh
    if parallel_dims is not None:
        set_spmd_meshes(dense_mesh=parallel_dims.spmd_dense_mesh(), sparse_mesh=None)
        mesh_ctx = lambda: set_current_spmd_mesh(parallel_dims.spmd_dense_mesh())  # noqa: E731
    else:
        mesh_ctx = contextlib.nullcontext
    for i, batch in enumerate(batches):
        with mesh_ctx(), record_function("forward_pass"):
            inputs, labels, extra = model.preprocess_inputs(batch, parallel_dims=parallel_dims)
            if loss_chunks > 1:
                # lm_head is applied per chunk inside the loss (torchtitan's
                # ChunkedLossWrapper); the model returns hidden states
                model._skip_lm_head = True
                hidden = model(inputs, special_tokens=special_tokens, **extra)
                loss = chunked_loss(
                    hidden, labels, model.lm_head, denom, loss_chunks,
                    compile_loss=compile_loss, global_vocab_size=model.config.vocab_size,
                )
            else:
                if parallel_dims is not None and parallel_dims.tp > 1:
                    raise ValueError("unchunked loss under TP would read vocab-sharded logits; use loss_chunks > 1")
                model._skip_lm_head = False
                logits = model(inputs, special_tokens=special_tokens, **extra)
                loss = cross_entropy_sum(logits, labels) / denom
        with mesh_ctx(), record_function("backward_pass"):
            loss.backward()
        accumulated += loss.detach()
        del loss
    return accumulated, local_tokens, denom
