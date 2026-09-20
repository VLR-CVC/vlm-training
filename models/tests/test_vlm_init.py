"""VLM creation: a text-only decoder + a fresh SigLIP2 tower -> a Qwen3-VL-shaped model.

The name-mapping checks need no weights and always run. The transfer checks need two
local snapshots and skip without them:

    QWEN3_TEXT_SNAPSHOT=.../qwen3_1_7b_base SIGLIP2_SNAPSHOT=.../siglip2_large \
        python -m pytest models/tests/test_vlm_init.py

Reading reference tensors: `safe_open` hands back memory-mapped tensors, so every
reference is read and compared inside the `with` block that opened its shard. A helper
that returns the tensor after the file closes gives silently wrong data -- it produced
a full set of false mismatches while this was being written.
"""

import glob
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.qwen3_5.checkpoint import build_meta, materialize
from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from models.vlm_init import _BLOCK_MAP, load_siglip_vision, load_text_weights

CONFIG = "configs/models/qwen3_vl_2b.json"
TEXT = os.environ.get("QWEN3_TEXT_SNAPSHOT", "")
SIGLIP = os.environ.get("SIGLIP2_SNAPSHOT", "")
needs_snapshots = pytest.mark.skipif(
    not (TEXT and SIGLIP and Path(TEXT).is_dir() and Path(SIGLIP).is_dir()),
    reason="set QWEN3_TEXT_SNAPSHOT and SIGLIP2_SNAPSHOT to local snapshot dirs",
)


def _meta_model():
    return build_meta(CONFIG, seq_len=1024)


def test_text_adapter_addresses_the_decoder_as_text_only():
    """A config with no vision encoder is the text-only view of the same decoder, so
    the adapter emits `model.*` instead of `model.language_model.*` -- which is what
    lets a Qwen3 checkpoint load with no hand-written name map."""
    model = _meta_model()
    decoder = {k: v for k, v in model.state_dict().items()
               if not k.startswith("vision_encoder.")}
    hf_keys = Qwen35StateDictAdapter(replace(model.config, vision_encoder=None)).to_hf(decoder)

    assert hf_keys
    assert not [k for k in hf_keys if "language_model" in k]
    assert all(k.startswith(("model.", "lm_head.")) for k in hf_keys)


def test_siglip_map_covers_the_tower_and_leaves_the_projector():
    """Everything in the ViT comes from SigLIP2 except the merger and the DeepStack
    mergers, which have no source and are what the first training stage learns."""
    model = _meta_model()
    names = {n for n, _ in model.named_parameters() if n.startswith("vision_encoder.")}

    covered = {"vision_encoder.patch_embed.weight", "vision_encoder.patch_embed.bias",
               "vision_encoder.pos_embed"}
    covered |= {f"vision_encoder.layers.{i}.{s}"
                for i in range(len(model.vision_encoder.layers))
                for s in _BLOCK_MAP.values()}

    assert covered <= names, sorted(covered - names)[:3]
    fresh = sorted(names - covered)
    assert fresh, "the projector must not be covered by the SigLIP2 map"
    assert all(".merger." in n or ".deepstack_mergers." in n for n in fresh), fresh[:5]


def _reference_tensors(wanted: dict[str, tuple[str, str]]) -> dict[str, torch.Tensor]:
    """{our name: (snapshot dir, checkpoint key)} -> {our name: tensor}, read and
    copied inside the `with` that opens each shard (see the module docstring)."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for directory in {d for d, _ in wanted.values()}:
        for shard in sorted(glob.glob(f"{directory}/*.safetensors")):
            with safe_open(shard, "pt") as handle:
                keys = set(handle.keys())
                for ours, (d, key) in wanted.items():
                    if d == directory and key in keys:
                        out[ours] = handle.get_tensor(key).float().clone()
    missing = set(wanted) - set(out)
    assert not missing, f"reference keys not in the snapshots: {sorted(missing)}"
    return out


@needs_snapshots
def test_assembled_vlm_matches_both_sources():
    wanted = {
        "tok_embeddings.weight": (TEXT, "model.embed_tokens.weight"),
        "layers.0.attn.wq.weight": (TEXT, "model.layers.0.self_attn.q_proj.weight"),
        "layers.13.attn.k_norm.weight": (TEXT, "model.layers.13.self_attn.k_norm.weight"),
        "layers.27.feed_forward.w2.weight": (TEXT, "model.layers.27.mlp.down_proj.weight"),
        "norm.weight": (TEXT, "model.norm.weight"),
        "vision_encoder.layers.0.norm1.weight":
            (SIGLIP, "vision_model.encoder.layers.0.layer_norm1.weight"),
        "vision_encoder.layers.5.attn.wk.weight":
            (SIGLIP, "vision_model.encoder.layers.5.self_attn.k_proj.weight"),
        "vision_encoder.layers.23.mlp.linear_fc2.bias":
            (SIGLIP, "vision_model.encoder.layers.23.mlp.fc2.bias"),
    }
    reference = _reference_tensors(wanted)

    model = _meta_model()
    materialize(model, "cpu")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    untouched = {k: v.clone() for k, v in model.state_dict().items()}

    load_text_weights(model, TEXT)
    load_siglip_vision(model, SIGLIP)
    state = model.state_dict()

    for name, expected in reference.items():
        assert torch.equal(state[name].float(), expected), name

    # the patch embedding is the one weight that is transformed, not copied: SigLIP2's
    # Conv2d inflated over the temporal axis, then flattened to our Linear layout
    pe = _reference_tensors(
        {"pe": (SIGLIP, "vision_model.embeddings.patch_embedding.weight")})["pe"]
    t = model.vision_encoder.config.temporal_patch_size
    inflated = (pe.unsqueeze(2).repeat(1, 1, t, 1, 1) / t).reshape(pe.shape[0], -1)
    assert torch.equal(state["vision_encoder.patch_embed.weight"].float(), inflated)

    # and nothing else survives the two loads except the projector
    survivors = sorted(k for k in untouched if torch.equal(untouched[k], state[k]))
    assert survivors, "init_states produced tensors identical to the checkpoints?"
    assert all(".merger." in k or ".deepstack_mergers." in k for k in survivors), survivors[:5]


if __name__ == "__main__":
    test_text_adapter_addresses_the_decoder_as_text_only()
    test_siglip_map_covers_the_tower_and_leaves_the_projector()
    if TEXT and SIGLIP:
        test_assembled_vlm_matches_both_sources()
        print("vlm_init: 3 checks passed")
    else:
        print("vlm_init: 2 mapping checks passed (set the snapshot env vars for the rest)")
