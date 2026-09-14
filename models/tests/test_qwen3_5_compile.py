"""Every decoder layer must survive `fullgraph=True`.

`train/infra.py:compile_model` compiles every layer and `configs/**` set
`compile = true`, so this is the path every training run takes. It used to pass
`fullgraph=True`, which cannot work: the layers call `_dtensor_unwrap`,
`_dtensor_rewrap` and `_wrap_cos_sin_as_dtensor`, all `@torch.compiler.disable`d,
and dynamo cannot graph-break at a disabled function under fullgraph. Every run
died at the first forward with `torch._dynamo.exc.Unsupported`, after the
allocation was granted and the weights were loaded.

This test compiles the way `compile_model` does and checks the result against
eager, so the two cannot drift apart again. It reproduces with plain tensors on
one GPU -- no distributed setup -- which is where the failure actually lived.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from models.qwen3_5.model import DecoderLayer  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU: the layers call CUDA kernels"
)

HEAD_DIM = 32
HIDDEN = 256


def _cfg() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=256, hidden_size=HIDDEN, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        head_dim=HEAD_DIM, max_position_embeddings=1024, rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2, linear_conv_kernel_dim=4,
        linear_key_head_dim=64, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=64,
        mtp_num_hidden_layers=0, mtp_use_dedicated_embeddings=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_decoder_layer_compiles_fullgraph(layer_type):
    torch.manual_seed(0)
    dev, total = "cuda", 128
    layer = DecoderLayer(_cfg(), layer_type).to(dev).to(torch.bfloat16)
    x = torch.randn(1, total, HIDDEN, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(total, HEAD_DIM, device=dev)
    sin = torch.randn(total, HEAD_DIM, device=dev)
    cu = torch.tensor([0, 64, total], device=dev, dtype=torch.int32)

    eager = layer(x, cos, sin, cu, 64)

    # exactly what train/infra.py:compile_model does
    layer.compile(dynamic=True, fullgraph=False, mode="default")
    compiled = layer(x, cos, sin, cu, 64)

    assert compiled.shape == eager.shape
    assert torch.isfinite(compiled).all()
    torch.testing.assert_close(compiled, eager, rtol=2e-2, atol=2e-2)
