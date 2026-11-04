"""`QWEN_FINALIZE_INSTRUCT=1` must fix what dcp_to_hf inherits from a Base snapshot.

A run finetuned off `*-Base` exports with `eos_token = <|endoftext|>` (248044)
while the SFT chat template closes turns with `<|im_end|>` (248046), and with no
`generation_config.json`, so `generate` runs to `max_new_tokens`. These pin the
two edits that fix it, and that the template lands in BOTH places a loader reads.
"""

import importlib.util
import json
from pathlib import Path

import pytest

# `scripts/` is not a package, and an unrelated installed `scripts` module would
# shadow it anyway, so load the file by path.
_PATH = Path(__file__).resolve().parents[2] / "scripts" / "finalize_instruct_tokenizer.py"
_spec = importlib.util.spec_from_file_location("finalize_instruct_tokenizer", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
finalize = _mod.finalize

ENDOFTEXT, IM_END = 248044, 248046
BASE_TEMPLATE = "{{- 'base template' }}"
SFT_TEMPLATE = "{{- 'sft template, closes with <|im_end|>' }}"


def _snapshot(tmp_path: Path, *, with_turn_end: bool = True) -> tuple[Path, Path]:
    """A Base-style export: Base template, Base eos, no generation_config."""
    out = tmp_path / "step-2600"
    out.mkdir()
    decoder = {str(ENDOFTEXT): {"content": "<|endoftext|>"}}
    if with_turn_end:
        decoder[str(IM_END)] = {"content": "<|im_end|>"}
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": decoder,
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "chat_template": BASE_TEMPLATE,
                "model_max_length": 16384,
            },
            indent=4,
        )
    )
    (out / "chat_template.jinja").write_text(BASE_TEMPLATE)
    tpl = tmp_path / "chat_template_sft.jinja"
    tpl.write_text(SFT_TEMPLATE)
    return out, tpl


def test_eos_and_template_are_rewritten(tmp_path):
    out, tpl = _snapshot(tmp_path)
    finalize(out, tpl)

    cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert cfg["eos_token"] == "<|im_end|>"
    assert cfg["pad_token"] == "<|endoftext|>", "pad must stay distinct from the stop token"
    assert cfg["chat_template"] == SFT_TEMPLATE
    assert cfg["model_max_length"] == 16384, "unrelated keys must survive"


def test_generation_config_is_written_when_there_is_no_reference(tmp_path):
    out, tpl = _snapshot(tmp_path)
    finalize(out, tpl)

    gen = json.loads((out / "generation_config.json").read_text())
    # turn terminator first, then the document separator, as the releases order them
    assert gen["eos_token_id"] == [IM_END, ENDOFTEXT]
    assert gen["pad_token_id"] == ENDOFTEXT
    assert "temperature" not in gen, "we must not invent sampling defaults"


def test_generation_config_is_copied_from_a_reference(tmp_path):
    out, tpl = _snapshot(tmp_path)
    ref = tmp_path / "official"
    ref.mkdir()
    payload = {
        "eos_token_id": [IM_END, ENDOFTEXT],
        "pad_token_id": ENDOFTEXT,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }
    (ref / "generation_config.json").write_text(json.dumps(payload, indent=4))

    finalize(out, tpl, str(ref))

    assert json.loads((out / "generation_config.json").read_text()) == payload


def test_a_tokenizer_without_the_turn_terminator_is_refused(tmp_path):
    """Better to fail than to set a stop token that is not in the vocab."""
    out, tpl = _snapshot(tmp_path, with_turn_end=False)
    with pytest.raises(SystemExit, match="refusing to set eos_token"):
        finalize(out, tpl)

    cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert cfg["eos_token"] == "<|endoftext|>", "must not half-apply"
    assert not (out / "generation_config.json").exists()
