"""Building a VLM on a Qwen3.5 decoder: the keys have to resolve.

Qwen3.5 ships no text-only variant -- every size is a
`Qwen3_5ForConditionalGeneration` -- so a decoder taken from one is nested at
`model.language_model.*`, while Qwen3-1.7B keeps it at `model.*`.
`Qwen35StateDictAdapter` switches on `config.vision_encoder`, so handing it the
wrong config makes every key miss and `load_text_weights` raises "did not
provide". This checks the names against the real snapshot index, which needs no
weights loaded and no GPU.
"""

import json
from pathlib import Path

import pytest

from models.vlm_init import _snapshot_is_multimodal

REPO = Path(__file__).resolve().parents[2]

# whichever box this runs on -- only config.json and the safetensors index are
# read, never the weights, so this needs no GPU and costs nothing
def _first_dir(*candidates):
    return next((p for p in map(Path, candidates) if p.is_dir()), None)


VLM_SNAPSHOT = _first_dir(
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
    "/gpfs/scratch/ehpc543/tockier/qwen_models/qwen3_5_2b",
)
TEXT_SNAPSHOT = _first_dir(
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_1_7b_base",
    "/gpfs/scratch/ehpc543/tockier/qwen_models/qwen3_1_7b_base",
)


def index_keys(snapshot):
    idx = snapshot / "model.safetensors.index.json"
    if not idx.is_file():
        pytest.skip(f"no index at {idx}")
    return set(json.loads(idx.read_text())["weight_map"])


def test_detects_a_qwen35_snapshot_as_multimodal():
    if VLM_SNAPSHOT is None:
        pytest.skip("no Qwen3.5 snapshot on this box")
    assert _snapshot_is_multimodal(VLM_SNAPSHOT)
    keys = index_keys(VLM_SNAPSHOT)
    # the layout the adapter has to be told about
    assert any(k.startswith("model.language_model.") for k in keys)
    assert not any(k.startswith("model.layers.") for k in keys)


def test_detects_a_text_only_snapshot():
    if TEXT_SNAPSHOT is None:
        pytest.skip("no text-only snapshot on this box")
    assert not _snapshot_is_multimodal(TEXT_SNAPSHOT)


SIGLIP_SNAPSHOT = _first_dir(
    "/data/151-1/users/tockier/qwen_finetune/cache/siglip2_large",
    "/gpfs/scratch/ehpc543/tockier/qwen_models/siglip2_large",
)


@pytest.mark.parametrize(
    "model_config", ["configs/models/qwen3_5_2b.json", "configs/models/qwen3_vl_2b.json"]
)
def test_siglip2_matches_the_vision_tower_it_replaces(model_config):
    """`load_siglip_vision` raises on any mismatch in width, depth or patch size,
    and it raises *after* the model is built and sharded. Both create configs
    depend on this holding, so check it from the JSON instead."""
    if SIGLIP_SNAPSHOT is None:
        pytest.skip("no SigLIP2 snapshot on this box")
    siglip = json.loads((SIGLIP_SNAPSHOT / "config.json").read_text())["vision_config"]
    ours = json.loads((REPO / model_config).read_text())["vision_config"]

    assert ours["hidden_size"] == siglip["hidden_size"]
    assert ours["intermediate_size"] == siglip["intermediate_size"]
    assert ours["num_heads"] == siglip["num_attention_heads"]
    assert ours["depth"] == siglip["num_hidden_layers"]
    # siglip's patch size is the default 16 when the config omits it
    assert ours["patch_size"] == siglip.get("patch_size", 16)
    # positions are allowed to differ -- that one is interpolated, not asserted
    assert ours["num_position_embeddings"] != (siglip["image_size"] // ours["patch_size"]) ** 2


def test_missing_config_is_an_error(tmp_path):
    """Guessing the prefix from the directory name would silently load nothing."""
    with pytest.raises(FileNotFoundError):
        _snapshot_is_multimodal(tmp_path)


def test_decoder_keys_resolve_against_the_vlm_snapshot():
    """The real assertion: every name the adapter asks for exists in the
    checkpoint. This is what fails, as an empty intersection, with the config the
    adapter was handed before."""
    if VLM_SNAPSHOT is None:
        pytest.skip("no Qwen3.5 snapshot on this box")
    from dataclasses import replace

    from models.qwen3_5.configs import qwen35_config_from_hf
    from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

    config = qwen35_config_from_hf(REPO / "configs/models/qwen3_5_2b.json", seq_len=1024)
    available = index_keys(VLM_SNAPSHOT)

    nested = Qwen35StateDictAdapter(config, str(VLM_SNAPSHOT)).from_hf_map
    wanted = {k for k, v in nested.items() if v is not None and "{}" not in k}
    assert wanted & available, "adapter asks for nothing the snapshot has"
    assert not {k for k in wanted if k.startswith("model.language_model.")} - available

    # and the text-only spelling, which is what the old code used, matches nothing
    flat = Qwen35StateDictAdapter(replace(config, vision_encoder=None), str(VLM_SNAPSHOT))
    flat_wanted = {
        k for k, v in flat.from_hf_map.items()
        if v is not None and "{}" not in k and k.startswith("model.")
    }
    assert not (flat_wanted & available), sorted(flat_wanted & available)


def test_vision_loader_dispatches_on_the_snapshot():
    """`vision_model_dir` alone decides which tower is loaded, so a config
    cannot ask for SigLIP2 and be handed the native one."""
    import models.vlm_init as vi

    seen = {}
    orig = vi.load_siglip_vision, vi.load_native_vision
    vi.load_siglip_vision = lambda m, s: seen.update(which="siglip")
    vi.load_native_vision = lambda m, s: seen.update(which="native")
    try:
        for snapshot, expect in ((SIGLIP_SNAPSHOT, "siglip"), (VLM_SNAPSHOT, "native")):
            if snapshot is None:
                continue
            seen.clear()
            vi.load_vision_weights(None, snapshot)
            assert seen.get("which") == expect, (snapshot, seen)
    finally:
        vi.load_siglip_vision, vi.load_native_vision = orig


def test_native_vision_skips_exactly_the_projector():
    """Against the real parameter names, not a restatement of the filter: on a
    meta-built Qwen3-VL every `vision_encoder.*` name carrying "merger" is a
    projector tensor and every one that does not is tower."""
    from models.qwen3_5.checkpoint import build_meta

    model = build_meta(REPO / "configs/models/qwen3_vl_2b.json", seq_len=1024)
    vision = [n for n, _ in model.named_parameters() if n.startswith("vision_encoder.")]
    assert vision, "no vision parameters built"

    skipped = [n for n in vision if "merger" in n]
    kept = [n for n in vision if "merger" not in n]
    # the projector is the merger plus the three DeepStack mergers, nothing else
    assert skipped, "filter would keep the whole tower"
    assert kept, "filter would drop the whole tower"
    assert all(n.startswith(("vision_encoder.merger.",
                             "vision_encoder.deepstack_mergers.")) for n in skipped), skipped
    assert not any(".merger" in n or "deepstack" in n for n in kept), kept
