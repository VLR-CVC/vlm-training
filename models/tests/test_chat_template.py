"""The reasoning-dropping chat template, pinned.

Qwen's shipped template keeps `<think>` spans only for turns after the last user
query. That is right for generation and wrong for SFT, and it silently cut every
qwen3.5 plotqa measurement in `PERFORMANCE.md` to roughly a quarter of its text
(§17.6). These tests exist so the same template cannot come back unnoticed.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.train_qwen import _apply_chat_template  # noqa: E402

SFT_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "chat_template_sft.jinja",
)

# The guard that makes the shipped template inference-only: it renders the
# reasoning block only for turns past the last user query. `ns.last_query_index`
# itself survives in the fixed template -- tool-call handling still uses it -- so
# the comparison is what identifies an inference template, not the name.
INFERENCE_GUARD = "{%- if loop.index0 > ns.last_query_index %}"
SFT_GUARD = "{%- if reasoning_content %}"


def test_sft_template_renders_reasoning_unconditionally():
    with open(SFT_TEMPLATE) as f:
        template = f.read()
    assert SFT_GUARD in template
    assert "> ns.last_query_index" not in template, (
        "assets/chat_template_sft.jinja has been overwritten with an inference "
        "template -- every <think> span but the last turn's will be dropped"
    )


def test_override_replaces_the_processors_template():
    proc = types.SimpleNamespace(chat_template="original")
    _apply_chat_template(proc, SFT_TEMPLATE)
    assert SFT_GUARD in proc.chat_template
    assert proc.chat_template != "original"


@pytest.mark.parametrize("path", ["NULL", "", None])
def test_null_path_keeps_the_shipped_template(path):
    proc = types.SimpleNamespace(chat_template="original")
    _apply_chat_template(proc, path)
    assert proc.chat_template == "original"


def test_warns_when_the_shipped_template_drops_reasoning(caplog):
    proc = types.SimpleNamespace(chat_template=INFERENCE_GUARD)
    _apply_chat_template(proc, "NULL")
    assert "drops" in caplog.text


def test_silent_when_the_shipped_template_is_already_fine(caplog):
    proc = types.SimpleNamespace(chat_template=SFT_GUARD)
    _apply_chat_template(proc, "NULL")
    assert "drops" not in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
