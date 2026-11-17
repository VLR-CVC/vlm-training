"""Qwen3.5 text model + Engram n-gram memory, for `trust_remote_code` loading.

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    model.generate(...)

A thin layer over transformers' `Qwen3_5ForCausalLM`: the decoder is the stock
one, and each Engram module is attached as a forward hook on the decoder layer
it follows (its output is the input of `h + sigmoid(gate) * W_V e`), the same
place `models/qwen3_5/model.py` applies it in training.

N-gram context under generation: with a KV cache, `forward` only sees the new
tokens. `prepare_inputs_for_generation` therefore also hands over the full
`input_ids` / `attention_mask` (as `engram_input_ids` / `engram_attention_mask`)
and the hashes are taken over the whole history; only the last positions are
used. Hashing is integer work on token ids, cheap next to the decoder.

Left padding: a padded slot is "blocked" -- n-grams never reach into it, it gets
no memory -- and each row's positions count from its first real token, which is
what the packed-document positions are in training.
"""

import math

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5RMSNorm

from .configuration_qwen3_5_engram import Qwen3_5EngramConfig
from .engram_hash import build_ngram_hash_spec, ngram_hash_ids


class Qwen3_5EngramModule(nn.Module):
    """Mirror of `models/common/engram.py:Engram` (same parameter names)."""

    def __init__(self, config, *, layer_index: int):
        super().__init__()
        e = config.engram
        self.spec = build_ngram_hash_spec(
            orders=e["orders"],
            heads_per_order=e["heads_per_order"],
            rows_per_head=e["rows_per_head"],
            vocab_size=config.vocab_size,
            pad_id=e["pad_id"],
            seed=e["seed"],
            layer_index=layer_index,
        )
        width = e["dim"]
        dim = config.hidden_size
        self.dim = dim
        self.gate_sqrt = bool(e["gate_sqrt"])
        self.blocked_token_ids = tuple(e.get("blocked_token_ids", ()))
        self.table = nn.Embedding(self.spec.num_rows, width // self.spec.num_heads)
        self.key_proj = nn.Linear(width, dim, bias=False)
        self.value_proj = nn.Linear(width, dim, bias=False)
        self.key_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)
        self.query_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)

    @torch.no_grad()
    def hash_rows(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        """``(ids (B, L, heads), mask (B, L))`` over the full history."""
        batch, length = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        real = attention_mask.bool()
        blocked_ids = torch.tensor(self.blocked_token_ids, dtype=torch.long, device=device)
        ids, masks = [], []
        for row in range(batch):
            tokens = input_ids[row].long()
            positions = (real[row].long().cumsum(0) - 1).clamp_min(0)
            blocked = ~real[row]
            if self.blocked_token_ids:
                blocked = blocked | torch.isin(tokens, blocked_ids)
            ids.append(ngram_hash_ids(tokens, positions, self.spec, blocked=blocked))
            masks.append(~blocked)
        return torch.stack(ids), torch.stack(masks)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.table(ids)
        e = (e * mask[..., None, None].to(e.dtype)).flatten(-2)
        key = self.key_norm(self.key_proj(e))
        value = self.value_proj(e)
        query = self.query_norm(x)
        gate = (key.float() * query.float()).sum(dim=-1, keepdim=True) / math.sqrt(self.dim)
        if self.gate_sqrt:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        return x + (torch.sigmoid(gate) * value.float()).to(x.dtype)


class Qwen3_5EngramForCausalLM(Qwen3_5ForCausalLM):
    config_class = Qwen3_5EngramConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        layers = sorted(set(config.engram["layers"]))
        self.model.engram = nn.ModuleDict(
            {str(layer): Qwen3_5EngramModule(config, layer_index=i) for i, layer in enumerate(layers)}
        )
        self._engram_inputs = None
        for name in self.model.engram:
            self.model.layers[int(name)].register_forward_hook(self._engram_hook(name))
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        # only reached for weights missing from the checkpoint: keep them inert
        if isinstance(module, Qwen3_5EngramModule):
            nn.init.zeros_(module.value_proj.weight)

    def _engram_hook(self, name: str):
        def hook(layer, args, output):
            if self._engram_inputs is None:
                raise RuntimeError("Engram inputs missing: call the model through forward()")
            ids, mask = self._engram_inputs[name]
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = self.model.engram[name](hidden, ids, mask)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        return hook

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, **kwargs)
        inputs["engram_input_ids"] = input_ids
        inputs["engram_attention_mask"] = attention_mask
        return inputs

    def forward(self, input_ids=None, attention_mask=None, *args, engram_input_ids=None,
                engram_attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("Engram addresses its memory with token ids: pass input_ids")
        history = input_ids if engram_input_ids is None else engram_input_ids
        history_mask = attention_mask if engram_input_ids is None else engram_attention_mask
        length = input_ids.shape[1]
        inputs = {}
        for name, module in self.model.engram.items():
            ids, mask = module.hash_rows(history, history_mask)
            inputs[name] = (ids[:, -length:], mask[:, -length:])
        self._engram_inputs = inputs
        try:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        finally:
            self._engram_inputs = None


__all__ = ["Qwen3_5EngramForCausalLM"]
