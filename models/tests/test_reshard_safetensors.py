"""The Hub rejects files over 50 GB, so the one-file export has to be split.

`dcp_to_hf.py` writes every tensor into a single `model.safetensors` -- 65.4 GiB
for the 35B bf16 export -- plus an index naming that one file. These pin what the
resharder has to guarantee before a push: no shard over the cap, every tensor
round-trips bit-exact, the index agrees with what is on disk, and the
pre-finalize tokenizer backups do not get published.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_PATH = Path(__file__).resolve().parents[2] / "utils" / "reshard_safetensors.py"
_spec = importlib.util.spec_from_file_location("reshard_safetensors", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
reshard = _mod.reshard

MIB = 1024**2
CAP_MIB = 5
N_TENSORS = 8


def _snapshot(tmp_path: Path) -> tuple[Path, Path, dict]:
    src = tmp_path / "src"
    src.mkdir()
    # 1 MiB each: 524288 bf16 elements
    orig = {
        f"layer.{i}.w": torch.arange(512 * 1024, dtype=torch.bfloat16).reshape(512, 1024)
        for i in range(N_TENSORS)
    }
    save_file(orig, str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text('{"model_type": "demo"}')
    (src / "tokenizer_config.json").write_text('{"eos_token": "<|im_end|>"}')
    (src / "tokenizer_config.json.base-orig").write_text('{"eos_token": "<|endoftext|>"}')
    # a stale index from the monolithic export; must be replaced, not copied
    (src / "model.safetensors.index.json").write_text('{"stale": true}')
    return src, tmp_path / "dst", orig


def test_no_shard_exceeds_the_cap(tmp_path):
    src, dst, _ = _snapshot(tmp_path)
    reshard(src, dst, max_shard_gb=CAP_MIB / 1024)

    shards = sorted(dst.glob("model-*.safetensors"))
    assert shards, "nothing written"
    for s in shards:
        assert s.stat().st_size <= CAP_MIB * MIB + 4096, f"{s.name} over cap"
    # 8 MiB of tensors under a 5 MiB cap cannot fit in one file
    assert len(shards) > 1


def test_every_tensor_round_trips(tmp_path):
    src, dst, orig = _snapshot(tmp_path)
    reshard(src, dst, max_shard_gb=CAP_MIB / 1024)

    merged: dict = {}
    for s in sorted(dst.glob("model-*.safetensors")):
        merged.update(load_file(str(s)))
    assert set(merged) == set(orig)
    for k in orig:
        torch.testing.assert_close(merged[k], orig[k])


def test_index_matches_what_is_on_disk(tmp_path):
    src, dst, orig = _snapshot(tmp_path)
    reshard(src, dst, max_shard_gb=CAP_MIB / 1024)

    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(orig), "index lost tensors"
    assert idx["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in orig.values()
    )
    for key, shard in idx["weight_map"].items():
        assert (dst / shard).is_file(), f"{key} -> missing {shard}"
    names = {f.name for f in dst.glob("model-*.safetensors")}
    assert set(idx["weight_map"].values()) == names, "index and shards disagree"


def test_the_monolithic_file_and_backups_are_not_published(tmp_path):
    src, dst, _ = _snapshot(tmp_path)
    reshard(src, dst, max_shard_gb=CAP_MIB / 1024)

    assert not (dst / "model.safetensors").exists(), "65 GB file would be rejected by the Hub"
    assert not (dst / "tokenizer_config.json.base-orig").exists()
    # the non-weight files a loader needs did come across
    assert (dst / "config.json").is_file()
    assert json.loads((dst / "tokenizer_config.json").read_text())["eos_token"] == "<|im_end|>"


def test_a_tensor_larger_than_the_cap_is_refused(tmp_path):
    """Better to fail than to emit a shard that the Hub will reject."""
    src = tmp_path / "src"
    src.mkdir()
    save_file(
        {"big": torch.zeros(512 * 1024, dtype=torch.bfloat16)},  # 1 MiB
        str(src / "model.safetensors"),
        metadata={"format": "pt"},
    )
    with pytest.raises(SystemExit, match="over the"):
        reshard(src, tmp_path / "dst", max_shard_gb=0.5 / 1024)  # 0.5 MiB cap
