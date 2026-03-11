from typing import Optional

import torch
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor import DTensor

from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
)

def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, None]:

    query = query.to_local() if isinstance(query, DTensor) else query
    key = key.to_local() if isinstance(key, DTensor) else key
    value = value.to_local() if isinstance(value, DTensor) else value

    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    if isinstance(query, DTensor):
        attn_output = DTensor.from_local(
            attn_output, 
            device_mesh=query.device_mesh, 
            placements=query.placements,
        )

    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    mesh = query_states.device_mesh
    cos = distribute_tensor(cos, mesh, [Replicate()])
    sin = distribute_tensor(sin, mesh, [Replicate()])

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output, attn_weights = flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def replace_attention_qwenvl(training_args):
    import transformers


    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = (
        return_mask
    )    
    ## qwen2_5_vl
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )

    ## qwen3vl
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )

    ## qwen3vl moe
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
        return_mask
    )

    # NOTE: Verify the exact import path for your transformers version

def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )