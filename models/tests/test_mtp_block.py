"""Qwen3.5 MTP block and decoder path.

    CUDA_VISIBLE_DEVICES=1 pytest models/tests/test_mtp_block.py -q

`test_mtp_roll.py` covers the shift kernel. This covers what is built on it:

  * `Qwen35MTPBlock` -- the fusion
    `eh_proj(cat(enorm(shifted_embed), hnorm(prev_hidden)))` followed by an
    ordinary Qwen3.5 block body, then `mtp_norm`.
  * `Qwen35Model` with `mtp_layers` -- aligned input/label tuples out of
    `preprocess_inputs`, one prediction per depth out of `forward`, and the
    previous depth's PRE-norm hidden state chained into the next.

The properties worth pinning are the ones that produce a plausible-looking
number when wrong rather than an error: which hidden state a depth consumes,
whether the validity mask actually zeroes anything, and whether a model built
with no MTP layers still behaves exactly like the old single-tensor model.
"""

from __future__ import annotations

import pytest
import torch

from models.qwen3_5.configs import apply_parallelism_config, qwen35_config_from_hf

CONFIG = "configs/models/qwen3_5_moe_debug.json"
SEQ_LEN = 256

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GatedDeltaNet and grouped_mm need CUDA"
)


def _model(num_mtp_layers: int, device="cuda", dtype=torch.bfloat16):
    from models.qwen3_5.checkpoint import materialize

    config = qwen35_config_from_hf(
        CONFIG, seq_len=SEQ_LEN, num_mtp_layers=num_mtp_layers
    )
    apply_parallelism_config(config, tp=1, enable_sp=False)
    with torch.device("meta"):
        model = config.build()
    materialize(model, device)
    for p in model.parameters():
        p.data = p.data.to(dtype)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device(device))
    return model.train()


def _two_document_batch(device="cuda", seq_len=64, split=40):
    """A packed row: two documents, so the shift has a boundary to respect."""
    tokens = torch.randint(0, 2000, (seq_len,), device=device)
    positions = torch.cat(
        [torch.arange(split), torch.arange(seq_len - split)]
    ).to(device)
    return {"input": tokens, "labels": tokens.clone(), "positions": positions}


# --------------------------------------------------------------------- config


@pytest.mark.parametrize("depths", [0, 1, 3])
def test_depths_are_built_from_the_last_main_layer(depths):
    config = qwen35_config_from_hf(CONFIG, seq_len=SEQ_LEN, num_mtp_layers=depths)
    assert len(config.mtp_layers) == depths

    last = config.layers[-1]
    for block in config.mtp_layers:
        # the stock every-4th schedule ends on a full-attention layer, and an
        # MTP depth feeds the LM head directly, so it should not be a GDN block
        assert block.attention is not None
        assert block.delta_net is None
        assert (block.moe is None) == (last.moe is None)
        # fusion takes the concatenation, hence 2 * dim in
        assert block.eh_proj.in_features == 2 * config.dim
        assert block.eh_proj.out_features == config.dim
        assert block.eh_proj.bias is False


def test_mtp_is_opt_in_not_read_from_the_json():
    """Every Qwen3.5 config.json carries `mtp_num_hidden_layers: 1`, but the
    released checkpoints have no MTP weights. Honouring the field would bolt
    randomly-initialised depths onto every fine-tune of a real snapshot."""
    import json
    from pathlib import Path

    raw = json.loads(Path(CONFIG).read_text())
    assert raw["text_config"]["mtp_num_hidden_layers"] == 1, "premise of this test"
    assert qwen35_config_from_hf(CONFIG, seq_len=SEQ_LEN).mtp_layers == []


def test_dedicated_embeddings_are_refused():
    """Only the shared-embedding variant is implemented: the shifted rows go
    through `tok_embeddings`. Silently ignoring the flag would train a model
    that cannot be loaded back into HF."""
    import json
    import tempfile
    from pathlib import Path

    raw = json.loads(Path(CONFIG).read_text())
    raw["text_config"]["mtp_use_dedicated_embeddings"] = True
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(NotImplementedError, match="dedicated_embeddings"):
            qwen35_config_from_hf(p, seq_len=SEQ_LEN, num_mtp_layers=1)
        # ... and only when MTP was actually asked for
        assert qwen35_config_from_hf(p, seq_len=SEQ_LEN).mtp_layers == []


# ------------------------------------------------------------------ the block


@requires_cuda
def test_block_fuses_embedding_and_previous_hidden():
    """Reproduce the documented formula by hand and compare. A block that
    dropped `hnorm`, swapped the concat order, or skipped `mtp_norm` would still
    return the right shape."""
    model = _model(1)
    block = model.mtp_layers[0]
    dim = model.config.dim

    torch.manual_seed(0)
    embed = torch.randn(32, dim, device="cuda", dtype=torch.bfloat16)
    prev = torch.randn(32, dim, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(32, dtype=torch.bool, device="cuda")

    positions = torch.arange(32, device="cuda")
    masks = model.get_attention_masks(positions)

    with torch.no_grad():
        got = block(embed, prev, valid, masks, positions)

        fused = block.eh_proj(
            torch.cat([block.enorm(embed), block.hnorm(prev)], dim=-1)
        )
        # the body is the ordinary Qwen3.5 block, inherited not re-implemented
        body = super(type(block), block).forward(fused, masks, positions)
        want = block.mtp_norm(body)

    assert got.shape == (32, dim)
    assert torch.equal(got, want)


@requires_cuda
def test_the_validity_mask_zeroes_the_previous_hidden_state():
    """At an invalid position the previous hidden state must not reach the
    fusion, or one document's state leaks into the next one's prediction."""
    model = _model(1)
    block = model.mtp_layers[0]
    dim = model.config.dim

    torch.manual_seed(1)
    embed = torch.randn(16, dim, device="cuda", dtype=torch.bfloat16)
    prev = torch.randn(16, dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(16, device="cuda")
    masks = model.get_attention_masks(positions)

    valid = torch.ones(16, dtype=torch.bool, device="cuda")
    valid[7] = False

    with torch.no_grad():
        masked = block(embed, prev, valid, masks, positions)
        # same call with the previous state already zeroed at 7 by hand
        prev_zeroed = prev.clone()
        prev_zeroed[7] = 0
        by_hand = block(embed, prev_zeroed, valid, masks, positions)
        # and with the mask claiming 7 is valid: must differ
        unmasked = block(embed, prev, torch.ones_like(valid), masks, positions)

    assert torch.equal(masked, by_hand), "mask must zero prev_embed, nothing else"
    assert not torch.equal(masked[7], unmasked[7]), "mask had no effect at all"


# ---------------------------------------------------------------- the decoder


@requires_cuda
def test_preprocess_returns_aligned_input_and_label_tuples():
    model = _model(2)
    batch = _two_document_batch()
    inputs, labels, rest = model.preprocess_inputs(dict(batch))

    assert isinstance(inputs, tuple) and isinstance(labels, tuple)
    assert len(inputs) == len(labels) == 3, "main path plus one row per depth"
    assert len(rest["mtp_input_valid_masks"]) == 2
    # main path untouched
    assert torch.equal(inputs[0], batch["input"])
    assert torch.equal(labels[0], batch["labels"])
    # depth k is the row shifted by k, so each loses one more valid target
    valid_counts = [m.sum().item() for m in rest["mtp_input_valid_masks"]]
    assert valid_counts[0] > valid_counts[1], valid_counts


@requires_cuda
def test_forward_returns_one_prediction_per_depth():
    model = _model(2)
    inputs, labels, rest = model.preprocess_inputs(_two_document_batch())
    with torch.no_grad():
        out = model(inputs, **rest)

    assert len(out) == 3 == len(labels)
    vocab = model.config.vocab_size
    for pred, label in zip(out, labels, strict=True):
        assert pred.shape == (label.shape[0], vocab)
        assert torch.isfinite(pred).all()


@requires_cuda
def test_depths_consume_the_pre_norm_hidden_state():
    """`norm` belongs to the main head. Feeding a depth the post-norm state
    would leave the main prediction identical and change only the depths --
    exactly the kind of bug that never raises."""
    model = _model(1)
    inputs, _labels, rest = model.preprocess_inputs(_two_document_batch())

    captured = {}
    real_forward = model.mtp_layers[0].forward

    def spy(embed, prev, valid, masks, positions=None):
        captured["prev"] = prev.detach().clone()
        return real_forward(embed, prev, valid, masks, positions)

    model.mtp_layers[0].forward = spy
    with torch.no_grad():
        out = model(inputs, **rest)
    model.mtp_layers[0].forward = real_forward

    # the main prediction is lm_head(norm(h)); the depth saw h itself
    with torch.no_grad():
        pre_norm = captured["prev"]
        assert torch.equal(out[0], model.lm_head(model.norm(pre_norm)))
        assert not torch.allclose(
            model.norm(pre_norm).float(), pre_norm.float(), atol=1e-2
        ), "norm is near-identity here, so this test cannot tell the two apart"


@requires_cuda
def test_depths_chain_rather_than_sharing_one_hidden_state():
    """Depth 2 must consume depth 1's output, not the main decoder's. Sharing
    one state is the 'only shift the labels' design, not paper-style MTP."""
    model = _model(2)
    inputs, _labels, rest = model.preprocess_inputs(_two_document_batch())

    seen = []
    originals = [layer.forward for layer in model.mtp_layers]

    def make_spy(i, real):
        def spy(embed, prev, valid, masks, positions=None):
            seen.append((i, prev.detach().clone()))
            out = real(embed, prev, valid, masks, positions)
            seen.append((f"{i}_out", out.detach().clone()))
            return out

        return spy

    for i, (layer, real) in enumerate(zip(model.mtp_layers, originals, strict=True)):
        layer.forward = make_spy(i, real)
    with torch.no_grad():
        model(inputs, **rest)
    for layer, real in zip(model.mtp_layers, originals, strict=True):
        layer.forward = real

    by_key = dict(seen)
    assert torch.equal(by_key[1], by_key["0_out"]), (
        "depth 2's previous state must be depth 1's output"
    )
    assert not torch.equal(by_key[1], by_key[0]), "depths are sharing one state"


@requires_cuda
def test_backward_reaches_every_mtp_parameter():
    model = _model(2)
    inputs, _labels, rest = model.preprocess_inputs(_two_document_batch())
    out = model(inputs, **rest)
    sum(o.float().pow(2).mean() for o in out).backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing[:5]
    # specifically the fusion modules, which only the MTP path touches
    for name in ("enorm", "hnorm", "eh_proj", "mtp_norm"):
        p = getattr(model.mtp_layers[0], name)
        grads = [x.grad for x in p.parameters()]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), name


@requires_cuda
def test_no_mtp_layers_is_the_old_model_exactly():
    """Everything above must stay off the default path: a model with no MTP
    layers takes one tensor and returns one tensor."""
    model = _model(0)
    assert model.mtp_layers is None

    batch = _two_document_batch()
    inputs, labels, rest = model.preprocess_inputs(dict(batch))
    assert isinstance(inputs, torch.Tensor) and isinstance(labels, torch.Tensor)
    assert "mtp_input_valid_masks" not in rest

    with torch.no_grad():
        out = model(inputs, **rest)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch["input"].shape[0], model.config.vocab_size)


@requires_cuda
def test_mtp_needs_positions():
    """Without positions a shift reads across packed-document boundaries, so
    this is refused rather than silently producing wrong targets."""
    model = _model(1)
    batch = _two_document_batch()
    del batch["positions"]
    with pytest.raises(ValueError, match="positions"):
        model.preprocess_inputs(batch)


@requires_cuda
def test_forward_rejects_a_mismatched_tuple():
    model = _model(2)
    batch = _two_document_batch()
    inputs, _labels, rest = model.preprocess_inputs(dict(batch))

    with pytest.raises(ValueError, match="one row per depth"):
        model(inputs[:2], **rest)
    with pytest.raises(ValueError, match="one validity mask per depth"):
        model(inputs, **{**rest, "mtp_input_valid_masks": rest["mtp_input_valid_masks"][:1]})
    with pytest.raises(ValueError, match="one token tensor"):
        _model(0)(inputs)


def test_mtp_weights_do_not_break_an_hf_load():
    """MTP depths are ordinary trainable weights, but no released Qwen3.5
    checkpoint has them, and `load_hf` raises on anything in `state_dict()` that
    the snapshot did not provide. Without this they are 15 phantom 'missing'
    tensors per depth."""
    from models.qwen3_5.state_dict_adapter import (
        is_local_only_state,
        Qwen35StateDictAdapter,
    )

    config = qwen35_config_from_hf(CONFIG, seq_len=SEQ_LEN, num_mtp_layers=2)
    with torch.device("meta"):
        model = config.build()
    state = model.state_dict()

    mtp_keys = [k for k in state if k.startswith("mtp_layers.")]
    assert mtp_keys, "premise of this test"
    assert all(is_local_only_state(k) for k in mtp_keys)
    # and nothing on the main path got swept up with them
    assert not is_local_only_state("layers.0.attn.wq.weight")
    assert not is_local_only_state("norm.weight")

    adapter = Qwen35StateDictAdapter(config, None)
    recovered = adapter.from_hf(adapter.to_hf(state))
    missing = {
        k for k in set(state) - set(recovered) if not is_local_only_state(k)
    }
    assert not missing, sorted(missing)[:5]
    # they must also not be exported as if HF understood them
    assert not [k for k in adapter.to_hf(state) if "mtp" in k]
