"""The SFT chat template, pinned on both of its invariants.

Qwen's shipped template keeps `<think>` spans only for turns after the last user
query. That is right for generation and wrong for SFT, and it silently cut every
qwen3.5 plotqa measurement in `PERFORMANCE.md` to roughly a quarter of its text
(§17.6). These tests exist so the same template cannot come back unnoticed.

The second invariant is the empty block. Every assistant turn renders a `<think>`
block, empty when the turn does not reason, so that both generation prefills stay
prefixes of what training saw -- the fusion of thinking and non-thinking data into
one model rests on exactly that. A rendered answer that does not start with the
prefill teaches the model never to close the block, and vLLM's `qwen3` reasoning
parser then returns the whole answer as `reasoning_content` with `content` empty.
"""

import os
import sys
import types

import jinja2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.task_encoder_factory import apply_chat_template  # noqa: E402

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
# Any marker that only the fixed template carries. The block is no longer guarded
# at all, so this is the unconditional emit rather than an `{%- if %}`.
SFT_GUARD = "'\\n<think>\\n' + reasoning_content"


def _render(messages, **kwargs):
    with open(SFT_TEMPLATE) as f:
        template = jinja2.Environment().from_string(f.read())
    template.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    kwargs.setdefault("add_generation_prompt", False)
    return template.render(messages=messages, **kwargs)


USER = [{"role": "user", "content": "q"}]
THINKING_TURN = USER + [
    {"role": "assistant", "content": "<think>\nreason\n</think>\n\nanswer"}
]
PLAIN_TURN = USER + [{"role": "assistant", "content": "answer"}]


def test_sft_template_renders_reasoning_unconditionally():
    with open(SFT_TEMPLATE) as f:
        template = f.read()
    assert SFT_GUARD in template
    assert "> ns.last_query_index" not in template, (
        "assets/chat_template_sft.jinja has been overwritten with an inference "
        "template -- every <think> span but the last turn's will be dropped"
    )
    assert "reason" in _render(THINKING_TURN)


def test_a_non_reasoning_turn_still_renders_an_empty_think_block():
    assert "<think>\n\n</think>\n\n" in _render(PLAIN_TURN), (
        "a non-reasoning answer renders without the empty block: it starts right "
        "after `assistant\\n`, which no generation prefill reproduces"
    )


@pytest.mark.parametrize("thinking", [True, False])
def test_both_generation_prefills_are_prefixes_of_a_plain_turn(thinking):
    kwargs = {} if thinking else {"enable_thinking": False}
    prefill = _render(USER, add_generation_prompt=True, **kwargs)
    assert _render(PLAIN_TURN).startswith(prefill), (
        "training text and the generation prompt have diverged, so the model is "
        "served a prefix it never saw"
    )


def test_override_replaces_the_processors_template():
    proc = types.SimpleNamespace(chat_template="original")
    apply_chat_template(proc, SFT_TEMPLATE)
    assert SFT_GUARD in proc.chat_template
    assert proc.chat_template != "original"


@pytest.mark.parametrize("path", ["NULL", "", None])
def test_null_path_keeps_the_shipped_template(path):
    proc = types.SimpleNamespace(chat_template="original")
    apply_chat_template(proc, path)
    assert proc.chat_template == "original"


def test_warns_when_the_shipped_template_drops_reasoning(caplog):
    proc = types.SimpleNamespace(chat_template=INFERENCE_GUARD)
    apply_chat_template(proc, "NULL")
    assert "drops" in caplog.text


def test_silent_when_the_shipped_template_is_already_fine(caplog):
    proc = types.SimpleNamespace(chat_template=SFT_GUARD)
    apply_chat_template(proc, "NULL")
    assert "drops" not in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
