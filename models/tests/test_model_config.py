"""`model.model_config` / `use_model_dir_config` resolution and the module-type
knobs of the titan builder.  python -m models.tests.test_model_config"""

import re
import tempfile
from pathlib import Path

from models.qwen3_5_tt.configs import qwen35_config_from_hf, resolve_model_config

REPO = Path(__file__).resolve().parents[2]


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def main():
    json_9b = REPO / "configs/models/qwen3_5_9b.json"

    # model_config wins by default; model_dir's own config.json is ignored
    assert resolve_model_config(str(json_9b), "/some/snapshot", False) == json_9b
    assert resolve_model_config("NULL", "/some/snapshot", True) == Path("/some/snapshot/config.json")
    assert raises(lambda: resolve_model_config("NULL", "/some/snapshot", False))
    assert raises(lambda: resolve_model_config(str(json_9b), "/some/snapshot", True))

    # a JSON file and a snapshot directory holding the same file build the same config
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json_9b.read_text())
        from_file = qwen35_config_from_hf(json_9b, seq_len=1024).to_dict()
        from_dir = qwen35_config_from_hf(d, seq_len=1024).to_dict()
    # init callables serialise as reprs with object addresses
    strip = lambda c: re.sub(r" at 0x[0-9a-f]+", "", str(c))  # noqa: E731
    assert strip(from_file) == strip(from_dir)

    assert raises(lambda: qwen35_config_from_hf(json_9b, seq_len=1024, attn_backend="flex"))
    assert raises(lambda: qwen35_config_from_hf(json_9b, seq_len=1024, decoder_mask="bidir"))
    print("model_config ok")


if __name__ == "__main__":
    main()
