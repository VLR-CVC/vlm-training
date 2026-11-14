import contextlib

import torch
from torch.profiler import record_function

from train.parallel.spmd import set_current_spmd_mesh, set_spmd_meshes

from train.loss import chunked_loss, cross_entropy_sum, cross_entropy_tokens

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
    mtp_scale: float = 0.3,
    loss_metrics: dict | None = None,
):
    """Run every micro-batch of a step; return (accumulated_loss, local_tokens, global_tokens).

    ``accumulated_loss`` is local_loss_sum / global_tokens: its SUM over DP ranks is
    the exact global per-token mean.

    Gradients are reduced once per window, on the last micro-batch; the others run
    with ``set_requires_gradient_sync(False)`` and accumulate locally.
    """
    local_tokens = sum(b.pop("num_valid_tokens") for b in batches)
    weighted = "loss_weights" in batches[0]  # data.loss_unit / loss_sqrt
    if weighted:
        # objective normaliser and logged-loss normaliser, reduced in one collective
        local_norms = [sum(b.pop(k) for b in batches) for k in ("loss_norm", "log_norm")]
        norms = torch.tensor(local_norms, dtype=torch.float64, device="cuda")
        if dp_group is not None and dp_group.size() > 1:
            torch.distributed.all_reduce(norms, group=dp_group.get_group())
        global_tokens, log_norm = norms.clamp(min=1e-12).to(torch.float32)
        # returned for the per-rank mean: accumulated * log_norm / local log weight
        local_tokens = local_norms[1]
    else:
        global_tokens = torch.tensor(local_tokens, dtype=torch.int64, device="cuda")
        if dp_group is not None and dp_group.size() > 1:
            torch.distributed.all_reduce(global_tokens, group=dp_group.get_group())
        # an all-masked step would divide by zero; its gradient is zero anyway
        global_tokens = global_tokens.clamp(min=1).to(torch.float32)

    accumulated = torch.zeros((), dtype=torch.float32, device="cuda")
    # torchtitan's get_spmd_context: module-boundary redistributions and the
    # vocab-parallel loss look up the TP group from the current dense mesh
    if parallel_dims is not None:
        set_spmd_meshes(
            dense_mesh=parallel_dims.spmd_dense_mesh(),
            sparse_mesh=parallel_dims.spmd_sparse_mesh(),
        )
        mesh_ctx = lambda: set_current_spmd_mesh(parallel_dims.spmd_dense_mesh())  # noqa: E731
    else:
        mesh_ctx = contextlib.nullcontext
    # `replicate` cannot do this (see the docstring); a model with no FSDP at all --
    # the TP-only parity test -- has no such method to call.
    defer_sync = not ddp and len(batches) > 1 and hasattr(model, "set_requires_gradient_sync")

    # for gradient accumulation, sequential by design
    for i, batch in enumerate(batches):
        last = i == len(batches) - 1
        if defer_sync:
            model.set_requires_gradient_sync(last)
        weights = (batch.pop("loss_weights"), batch.pop("log_weights"), log_norm) if weighted else None
        logged = None
        with mesh_ctx(), record_function("forward_pass"):
            inputs, labels, extra = model.preprocess_inputs(batch, parallel_dims=parallel_dims)
            if loss_chunks > 1:
                # lm_head is applied per chunk inside the loss (torchtitan's
                # ChunkedLossWrapper); the model returns hidden states
                model._skip_lm_head = True
                hidden = model(inputs, special_tokens=special_tokens, **extra)
                loss = chunked_loss(
                    hidden, labels, model.lm_head, global_tokens, loss_chunks,
                    compile_loss=compile_loss, global_vocab_size=model.config.vocab_size,
                    # lm_head is its own FSDP unit, so the root's setting does not
                    # reach it -- the loss re-enables its sync on the last chunk
                    sync_grads=not defer_sync or last,
                    # a no-op unless `hidden`/`labels` came back as MTP tuples
                    mtp_scale=mtp_scale,
                    metrics=loss_metrics,
                    weights=weights,
                )
            else:
                if parallel_dims is not None and parallel_dims.tp > 1:
                    raise ValueError("unchunked loss under TP would read vocab-sharded logits; use loss_chunks > 1")
                if not isinstance(labels, torch.Tensor):
                    raise ValueError(
                        "MTP returns one prediction per depth, which only the "
                        "chunked loss combines; set loss_chunks > 1"
                    )
                model._skip_lm_head = False
                logits = model(inputs, special_tokens=special_tokens, **extra)
                if weighted:
                    ce_tok = cross_entropy_tokens(logits, labels)
                    loss = (ce_tok * weights[0]).sum() / global_tokens
                    logged = (ce_tok.detach() * weights[1]).sum() / log_norm
                else:
                    loss = cross_entropy_sum(logits, labels) / global_tokens
        with mesh_ctx(), record_function("backward_pass"):
            loss.backward()
        # weighted: chunked_loss already returns the logged value
        accumulated += loss.detach() if logged is None else logged
        del loss
    return accumulated, local_tokens, log_norm if weighted else global_tokens
