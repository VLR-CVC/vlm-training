"""Empty think blocks are prompt format, not supervised answer tokens.

`assets/chat_template_sft_nothink.jinja` renders reasoning-free answers as
`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<answer>` (Qwen3.5's non-thinking
format). The labels must start at <answer>, and the old template's labels must not
change.

    pytest models/tests/test_empty_think_labels.py
"""

import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.energon_dataloader import PackedBatchEncoder  # noqa: E402

TOKENIZER = "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
MESSAGES = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "<think>\nadd\n</think>\n\n6"},
]


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(TOKENIZER):
        pytest.skip(f"{TOKENIZER} not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def _labelled_text(tokenizer, template):
    tokenizer.chat_template = (REPO / "assets" / template).read_text()
    text = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    enc = types.SimpleNamespace(
        assistant_prefix=tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        empty_think=tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False),
        turn_end_token=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    labels = PackedBatchEncoder.assistant_labels(enc, ids)
    spans, cur = [], []
    for i, lab in zip(ids, labels.tolist()):
        if lab != -100:
            cur.append(i)
        elif cur:
            spans.append(tokenizer.decode(cur))
            cur = []
    if cur:
        spans.append(tokenizer.decode(cur))
    return text, spans


def test_nothink_template_skips_the_empty_block(tokenizer):
    text, spans = _labelled_text(tokenizer, "chat_template_sft_nothink.jinja")
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>" in text
    assert spans[0] == "4<|im_end|>\n"
    # a real reasoning span is still supervised
    assert spans[1].startswith("<think>\nadd\n</think>\n\n6")


def test_original_template_labels_unchanged(tokenizer):
    text, spans = _labelled_text(tokenizer, "chat_template_sft.jinja")
    assert "<|im_start|>assistant\n4<|im_end|>" in text
    assert spans[0] == "4<|im_end|>\n"
    assert spans[1].startswith("<think>\nadd\n</think>\n\n6")


def test_generation_prompt_is_non_thinking_by_default(tokenizer):
    tokenizer.chat_template = (REPO / "assets" / "chat_template_sft_nothink.jinja").read_text()
    prompt = tokenizer.apply_chat_template(MESSAGES[:1], tokenize=False, add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    thinking = tokenizer.apply_chat_template(
        MESSAGES[:1], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")
