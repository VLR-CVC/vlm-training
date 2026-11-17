"""Engram addressing and the module on its own: runs on CPU.

    pytest models/tests/test_engram_hash.py -q

`test_engram_model.py` covers the module inside Qwen3.5 (needs CUDA). This file
pins what can be checked without a decoder:

  * the hash: deterministic, in range for every head's slice, causal, reset at
    packed-document boundaries, stopped at vision placeholders;
  * the module: zero-initialised W_V adds exactly nothing, W_V is the only
    parameter with a gradient at step 0, the gate is the documented formula;
  * the trainer plumbing: freeze switches and the optimizer groups.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.common.engram_hash import build_ngram_hash_spec, ngram_hash_ids

VOCAB = 248_320
PAD = 248_044


def _spec(**kw):
    args = dict(orders=(2, 3), heads_per_order=8, rows_per_head=1009, vocab_size=VOCAB, pad_id=PAD)
    args.update(kw)
    return build_ngram_hash_spec(**args)


def _row(seed=0, lengths=(40, 24)):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (sum(lengths),), generator=g)
    positions = torch.cat([torch.arange(n) for n in lengths])
    return tokens, positions


# ------------------------------------------------------------------- the hash


def test_spec_layout_is_disjoint_and_prime():
    spec = _spec()
    assert spec.num_heads == 16
    assert len(set(spec.primes)) == 16
    assert all(p >= 1009 for p in spec.primes)
    for i in range(1, 16):
        assert spec.offsets[i] == spec.offsets[i - 1] + spec.primes[i - 1]
    assert spec.num_rows >= spec.offsets[-1] + spec.primes[-1]
    assert spec.num_rows % 128 == 0
    # odd multipliers that cannot overflow int64 against any token id
    assert all(m % 2 == 1 and m * VOCAB < 2**63 for m in spec.multipliers)


def test_layers_never_share_a_modulus_or_multipliers():
    a, b = _spec(layer_index=0), _spec(layer_index=1)
    assert not set(a.primes) & set(b.primes)
    assert a.multipliers != b.multipliers


def test_hash_is_deterministic_and_in_range():
    spec = _spec()
    tokens, positions = _row()
    ids = ngram_hash_ids(tokens, positions, spec)
    again = ngram_hash_ids(tokens.clone(), positions.clone(), _spec())
    assert ids.dtype == torch.long and ids.shape == (tokens.shape[0], 16)
    assert torch.equal(ids, again)
    for head in range(16):
        lo, hi = spec.offsets[head], spec.offsets[head] + spec.primes[head]
        assert bool(((ids[:, head] >= lo) & (ids[:, head] < hi)).all()), head


def test_hash_matches_the_formula():
    """((x_t m0) xor (x_{t-1} m1) [xor (x_{t-2} m2)]) mod p + offset, by hand."""
    spec = _spec()
    tokens = torch.tensor([11, 22, 33, 44])
    ids = ngram_hash_ids(tokens, torch.arange(4), spec)
    m = spec.multipliers
    t = 3
    bigram = (44 * m[0]) ^ (33 * m[1])
    trigram = bigram ^ (22 * m[2])
    for k in range(8):
        assert ids[t, k] == bigram % spec.primes[k] + spec.offsets[k]
        assert ids[t, 8 + k] == trigram % spec.primes[8 + k] + spec.offsets[8 + k]
    # position 0 has no history: both look-back slots read the pad id
    first = (11 * m[0]) ^ (PAD * m[1])
    assert ids[0, 0] == first % spec.primes[0] + spec.offsets[0]
    assert ids[0, 8] == (first ^ (PAD * m[2])) % spec.primes[8] + spec.offsets[8]


def test_hash_is_causal():
    """Changing token t+1 never changes the ids at positions <= t."""
    spec = _spec()
    tokens, positions = _row(lengths=(64,))
    ids = ngram_hash_ids(tokens, positions, spec)
    for t in (0, 5, 31, 62):
        changed = tokens.clone()
        changed[t + 1] = (changed[t + 1] + 1) % VOCAB
        other = ngram_hash_ids(changed, positions, spec)
        assert torch.equal(other[: t + 1], ids[: t + 1])
        assert not torch.equal(other[t + 1], ids[t + 1])


def test_ngrams_never_cross_a_packed_document_boundary():
    """The second document hashes exactly as if it were alone in the row, so the
    first document's tail cannot reach it."""
    spec = _spec()
    tokens, positions = _row(lengths=(40, 24))
    packed = ngram_hash_ids(tokens, positions, spec)
    alone = ngram_hash_ids(tokens[40:], positions[40:], spec)
    assert torch.equal(packed[40:], alone)
    # ... and changing the first document's last tokens leaves it untouched
    changed = tokens.clone()
    changed[38:40] = 7
    assert torch.equal(ngram_hash_ids(changed, positions, spec)[40:], packed[40:])


def test_lookback_stops_at_a_blocked_token():
    spec = _spec()
    tokens, positions = _row(lengths=(32,))
    blocked = torch.zeros(32, dtype=torch.bool)
    blocked[10] = True
    ids = ngram_hash_ids(tokens, positions, spec, blocked=blocked)
    # after the image token, the context behind it is invisible: change it freely
    changed = tokens.clone()
    changed[:11] = 3
    again = ngram_hash_ids(changed, positions, spec, blocked=blocked)
    assert torch.equal(again[11:], ids[11:])
    # position 11 sees nothing behind it (pad, pad); position 12 sees one token
    restart = ngram_hash_ids(tokens[11:], torch.arange(21), spec)
    assert torch.equal(ids[11:], restart)


# ----------------------------------------------------------------- the module


def _module(dim=64, heads=4, rows=257, gate_sqrt=True, seed=0):
    from models.qwen3_5.engram import EngramArgs, qwen35_engram_configs

    (config,) = qwen35_engram_configs(
        EngramArgs(layers=(0,), heads_per_order=heads, rows_per_head=rows, dim=dim,
                   gate_sqrt=gate_sqrt),
        dim=dim, vocab_size=VOCAB, n_layers=1,
        raw_config={"text_config": {"eos_token_id": PAD}, "image_token_id": 9},
    )
    torch.manual_seed(seed)
    module = config.build()
    module.init_states()
    return module


def _inputs(module, n=48):
    tokens, positions = _row(lengths=(n - 16, 16))
    tokens[5] = 9  # an image placeholder
    ids, mask = module.hash_inputs(tokens, positions)
    return ids, mask


def test_zero_init_adds_exactly_nothing():
    module = _module()
    assert torch.count_nonzero(module.value_proj.weight) == 0
    ids, mask = _inputs(module)
    x = torch.randn(ids.shape[0], 64)
    assert torch.equal(module(x, ids, mask), x)


def test_at_init_only_value_proj_gets_a_gradient():
    """W_V = 0 blocks every gradient upstream of it at step 0: the table, W_K and
    the norms see nothing, W_V gets alpha * e. One step of W_V unblocks the rest."""
    module = _module()
    ids, mask = _inputs(module)
    x = torch.randn(ids.shape[0], 64, requires_grad=True)
    module(x, ids, mask).square().sum().backward()
    grads = {n: p.grad for n, p in module.named_parameters()}
    assert grads["value_proj.weight"] is not None
    assert grads["value_proj.weight"].abs().sum() > 0
    for name in ("table.weight", "key_proj.weight", "key_norm.weight", "query_norm.weight"):
        assert grads[name] is None or torch.count_nonzero(grads[name]) == 0, name


def test_gate_formula_and_masking():
    module = _module(gate_sqrt=True)
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
    ids, mask = _inputs(module)
    x = torch.randn(ids.shape[0], 64)
    with torch.no_grad():
        got = module(x, ids, mask)
        e = module.table(ids, mask)
        g = (module.key_norm(module.key_proj(e)) * module.query_norm(x)).sum(-1, keepdim=True)
        g = g / math.sqrt(64)
        alpha = torch.sigmoid(g.sign() * g.abs().clamp_min(1e-6).sqrt())
        want = x + alpha * module.value_proj(e)
    torch.testing.assert_close(got, want)
    # the image placeholder gets no memory at all
    assert not mask[5]
    assert torch.equal(got[5], x[5])
    assert module.last_gate_stats is not None and module.last_gate_stats.shape == (2,)


def test_plain_sigmoid_gate():
    module = _module(gate_sqrt=False)
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
    ids, mask = _inputs(module)
    x = torch.randn(ids.shape[0], 64)
    with torch.no_grad():
        e = module.table(ids, mask)
        g = (module.key_norm(module.key_proj(e)) * module.query_norm(x)).sum(-1, keepdim=True)
        want = x + torch.sigmoid(g / 8.0) * module.value_proj(e)
        torch.testing.assert_close(module(x, ids, mask), want)


def test_dim_must_split_over_heads():
    from models.qwen3_5.engram import EngramArgs, qwen35_engram_configs

    with pytest.raises(ValueError, match="divisible"):
        qwen35_engram_configs(EngramArgs(dim=100), dim=64, vocab_size=VOCAB, n_layers=4,
                              raw_config={"eos_token_id": PAD})
    with pytest.raises(ValueError, match="outside"):
        qwen35_engram_configs(EngramArgs(layers=(4,)), dim=64, vocab_size=VOCAB, n_layers=4,
                              raw_config={"eos_token_id": PAD})


# ------------------------------------------------------------ config + trainer


def test_engram_is_off_by_default_and_on_for_every_qwen35_type():
    from models.qwen3_5.checkpoint import build_meta
    from models.qwen3_5.engram import EngramArgs
    from train.config import Model
    from train.utils import build_engram_args

    assert build_engram_args(Model()) is None
    assert build_meta("configs/models/qwen3_5_0_8b.json", seq_len=256).engram is None
    args = EngramArgs(rows_per_head=1009)
    for cfg in ("configs/models/qwen3_5_0_8b.json", "configs/models/qwen3_5_2b_text.json",
                "configs/models/qwen3_5_moe_debug.json"):
        model = build_meta(cfg, seq_len=256, tp=1, engram=args)
        assert list(model.engram.keys()) == ["2"], cfg
    with pytest.raises(ValueError, match="only implemented for Qwen3.5"):
        build_meta("configs/models/qwen3_vl_2b.json", seq_len=256, engram=args)


def test_vision_placeholders_are_blocked_on_the_vlm():
    import json

    from models.qwen3_5.checkpoint import build_meta
    from models.qwen3_5.engram import EngramArgs

    raw = json.loads(open("configs/models/qwen3_5_0_8b.json").read())
    model = build_meta("configs/models/qwen3_5_0_8b.json", seq_len=256,
                       engram=EngramArgs(rows_per_head=1009))
    blocked = set(model.engram["2"].blocked_token_ids)
    assert {raw["image_token_id"], raw["video_token_id"]} <= blocked
    assert model.engram["2"].spec.pad_id == raw["text_config"]["eos_token_id"]


def test_train_engram_alone_and_its_param_group():
    from models.qwen3_5.checkpoint import build_meta
    from train.config import Model
    from train.utils import build_engram_args, build_optimizer_param_groups, set_trainable_parts

    margs = Model(engram=True, engram_rows_per_head=1009, train_llm=False, train_engram=True)
    model = build_meta("configs/models/qwen3_5_2b_text.json", seq_len=256,
                       engram=build_engram_args(margs))
    set_trainable_parts(margs, model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("engram.") for n in trainable)

    groups = build_optimizer_param_groups(
        model.named_parameters(), {"mlp": 1.0, "vit": 2.0, "llm": 3.0, "engram": 7.0}, 0.1
    )
    assert all(g["lr"] == 7.0 for g in groups)
    table = model.engram["2"].table.weight
    by_id = {id(p): g for g in groups for p in g["params"]}
    assert by_id[id(table)]["weight_decay"] == 0.0
    assert by_id[id(model.engram["2"].value_proj.weight)]["weight_decay"] == 0.1
    assert by_id[id(model.engram["2"].key_norm.weight)]["weight_decay"] == 0.0


def test_local_only_and_engram_export_names():
    from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter, is_local_only_state

    assert is_local_only_state("engram.2.table.weight")
    sd = {"engram.2.table.weight": 1, "engram.2.value_proj.weight": 2, "norm.weight": 3}
    assert Qwen35StateDictAdapter.engram_to_hf(sd) == {
        "model.engram.2.table.weight": 1, "model.engram.2.value_proj.weight": 2,
    }
