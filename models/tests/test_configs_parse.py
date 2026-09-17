"""Every config under `configs/` must load.

`ConfigManager._dict_to_dataclass` rejects unknown keys, so renaming a field in
`train/config.py` breaks every config that still uses the old name -- at job
startup, after the allocation is granted. Six configs had rotted this way before
this test existed: `ac_mode`, `random_init_mlp`, `bfloat16`, `text_cache_dir`,
`resume`, `model_impl`, `start_idx`/`end_idx` and a `pp_size` for a parallelism
that was never implemented.

Renaming a field is fine. Renaming it without updating the configs is what this
catches.

    pytest models/tests/test_configs_parse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train.config import Config           # noqa: E402
from train.config_manager import ConfigManager  # noqa: E402

# Configs without `model.model_config` target the deprecated model definitions
# (`models/qwen3_5`, `models/qwen3_vl`, `models/qwen3`) and no longer load.
CONFIGS = sorted(p for p in REPO.glob("configs/**/*.toml") if "model_config = " in p.read_text())


def test_configs_exist():
    """A glob that silently matches nothing would make every test below pass."""
    assert CONFIGS, f"no configs found under {REPO / 'configs'}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_config_parses(path, monkeypatch):
    # tyro reads sys.argv when `args` leaves anything unconsumed
    monkeypatch.setattr(sys, "argv", ["pytest"])
    ConfigManager(Config).parse_args(["--config", str(path)])
