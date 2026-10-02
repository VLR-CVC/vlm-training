"""Layer-selective activation checkpointing (`train/parallel/ac.py`).

AC is pure recompute: it must change memory and nothing else. The two ways it
silently goes wrong are both checked here --

1. `checkpoint_wrapper` inserts a `_checkpoint_wrapped_module` level. If its
   prefix-stripping hooks ever stop firing, every state-dict key gains a level
   and the HF adapter stops matching -- which looks like a loading bug, not an
   AC bug.
2. Wrapping changes gradients. It must not: the backward recomputes the same
   forward, so grads are bit-identical, not merely close.
"""

import torch
import torch.nn as nn

from train.parallel.ac import apply_ac

DIM = 8

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))

class Tower(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): Block() for i in range(n)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers.values():
            x = block(x)
        return x

class Model(nn.Module):
    """Same shape as the real thing: `.layers` plus `.vision_encoder.layers`."""

    def __init__(self, n_text: int = 4, n_vision: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): Block() for i in range(n_text)})
        self.vision_encoder = Tower(n_vision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vision_encoder(x)
        for block in self.layers.values():
            x = block(x)
        return x

def _is_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "_checkpoint_wrapped_module")

def _wrapped_flags(model: Model) -> list[bool]:
    return [_is_wrapped(m) for m in model.layers.values()] + [
        _is_wrapped(m) for m in model.vision_encoder.layers.values()
    ]

def test_freq_zero_is_a_noop():
    model = Model()
    apply_ac(model, 0, include_vision=True)
    assert _wrapped_flags(model) == [False] * 6

def test_freq_two_wraps_every_second_block_across_both_towers():
    model = Model(n_text=4, n_vision=2)
    apply_ac(model, 2, include_vision=True)
    # counter runs decoder blocks 1..4 then vision 5..6, wrapping the even ones
    assert _wrapped_flags(model) == [False, True, False, True, False, True]

def test_vision_is_included_by_default():
    """The ViT holds most of the activations on image batches, so it must be
    wrapped. This only works because `VisionAttention.attn_scale` keeps the
    softmax scale a float constant; `include_vision=False` is the off switch if
    that regresses."""
    model = Model(n_text=4, n_vision=2)
    apply_ac(model, 2)
    assert _wrapped_flags(model) == [False, True, False, True, False, True]

def test_vision_can_be_excluded():
    model = Model(n_text=4, n_vision=2)
    apply_ac(model, 2, include_vision=False)
    assert _wrapped_flags(model) == [False, True, False, True, False, False]

def test_freq_one_wraps_everything():
    model = Model()
    apply_ac(model, 1, include_vision=True)
    assert all(_wrapped_flags(model))

def test_state_dict_keys_survive_wrapping():
    plain, wrapped = Model(), Model()
    apply_ac(wrapped, 2, include_vision=True)
    assert list(plain.state_dict()) == list(wrapped.state_dict())
    assert not any("_checkpoint_wrapped_module" in k for k in wrapped.state_dict())
    # and the wrapped model still loads a plain checkpoint
    wrapped.load_state_dict(plain.state_dict())

def test_op_level_wraps_every_block_and_ignores_freq():
    """op-level SAC recomputes inside a block, so it applies to all of them --
    `freq` only gates whole-block recompute."""
    model = Model(n_text=4, n_vision=2)
    apply_ac(model, 2, op_level=True)
    assert all(_wrapped_flags(model))

def test_op_level_gradients_are_unchanged():
    torch.manual_seed(0)
    plain, wrapped = Model(), Model()
    wrapped.load_state_dict(plain.state_dict())
    apply_ac(wrapped, 1, op_level=True)

    x = torch.randn(3, DIM)
    for model in (plain, wrapped):
        model(x.clone().requires_grad_(True)).sum().backward()
    for (name, a), (_, b) in zip(
        plain.named_parameters(), wrapped.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

def test_gradients_are_unchanged():
    torch.manual_seed(0)
    plain = Model()
    wrapped = Model()
    wrapped.load_state_dict(plain.state_dict())
    apply_ac(wrapped, 2, include_vision=True)

    x = torch.randn(3, DIM)
    for model in (plain, wrapped):
        model(x.clone().requires_grad_(True)).sum().backward()

    for (name, a), (_, b) in zip(
        plain.named_parameters(), wrapped.named_parameters(), strict=True
    ):
        assert a.grad is not None and b.grad is not None, name
        # recompute, not approximation: exact equality
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
            print(f"{fn.__name__} ok")
