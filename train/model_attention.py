from typing import Optional

import time

import torch
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor import DTensor

from torch.nn.attention.varlen import varlen_attn

#from flash_attn.flash_attn_interface import flash_attn_varlen_func
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

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as apply_rotary_pos_emb_qwen_3_5

original_compute_3d = Qwen3VLModel.compute_3d_position_ids

def patched_compute_3d_position_ids(
    self,
    input_ids=None,
    inputs_embeds=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    past_key_values=None,
    mm_token_type_ids=None,
):
    dense_mask = attention_mask
    
    if attention_mask is not None and attention_mask.dim() == 1 and attention_mask[0] == 0:
        total_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        valid_len = attention_mask[-1].item()
        
        dense_mask = torch.zeros((1, total_len), device=attention_mask.device, dtype=torch.long)
        dense_mask[:, :valid_len] = 1
        mm_token_type_ids.unsqueeze_(0)
        # everything as (1, seq_len)

    return original_compute_3d(
        self,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=dense_mask,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
    )

Qwen3VLModel.compute_3d_position_ids = patched_compute_3d_position_ids

original_get_rope_index = Qwen3_5Model.get_rope_index

def patched_get_rope_index(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
):
    dense_mask = attention_mask
    if attention_mask is not None and attention_mask.dim() == 1 and attention_mask[0] == 0:
        total_len = input_ids.shape[1] if input_ids is not None else mm_token_type_ids.shape[-1]
        valid_len = attention_mask[-1].item() # Note: still a CPU-GPU sync bottleneck
        
        dense_mask = torch.zeros((1, total_len), device=attention_mask.device, dtype=torch.long)
        dense_mask[:, :valid_len] = 1
        
        if mm_token_type_ids.dim() == 1:
            mm_token_type_ids = mm_token_type_ids.unsqueeze(0)

    return original_get_rope_index(
        self,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=dense_mask,
    )

Qwen3_5Model.get_rope_index = patched_get_rope_index

def varlen_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, None]:

    query = query.to_local() if isinstance(query, DTensor) else query
    key = key.to_local() if isinstance(key, DTensor) else key
    value = value.to_local() if isinstance(value, DTensor) else value

    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    # batch, seqlen, head, dim

    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    cu_seqlens = attention_mask.squeeze()

    with torch.no_grad():
        max_seqlen = torch.diff(cu_seqlens).max().item()

    attn_output = varlen_attn(
        query,
        key,
        value,
        cu_seq_q=cu_seqlens,
        cu_seq_k=cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
        window_size=(-1, 0), # causal
    )

    if isinstance(query, DTensor):
        attn_output = DTensor.from_local(
            attn_output, 
            device_mesh=query.device_mesh, 
            placements=query.placements,
        )

    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


def qwen3_5_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    if isinstance(gate, DTensor):
        gate = gate.to_local()

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if isinstance(query_states, DTensor):
        mesh = query_states.device_mesh
        cos = distribute_tensor(cos, mesh, [Replicate()])
        sin = distribute_tensor(sin, mesh, [Replicate()])
    query_states, key_states = apply_rotary_pos_emb_qwen_3_5(query_states, key_states, cos, sin)

    attn_output, attn_weights = varlen_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen3_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if isinstance(query_states, DTensor):
        mesh = query_states.device_mesh
        cos = distribute_tensor(cos, mesh, [Replicate()])
        sin = distribute_tensor(sin, mesh, [Replicate()])

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output, attn_weights = varlen_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if isinstance(query_states, DTensor):
        mesh = query_states.device_mesh
        cos = distribute_tensor(cos, mesh, [Replicate()])
        sin = distribute_tensor(sin, mesh, [Replicate()])

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output, attn_weights = varlen_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    attention_mask,
    **kwargs,
):
    """
    HF should not use this function since we directly provide the attention mask
    """
    return attention_mask.unsqueeze_(0)


def replace_attention_qwenvl():
    import transformers

    ## qwen3-text
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3.modeling_qwen3.create_causal_mask = (
        return_mask
    )

    ## qwen3vl
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        #torch.compile(qwen3vl_forward, mode="max-autotune-no-cudagraphs")
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

    ## qwen3.5
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention.forward = (
        qwen3_5_forward
    )
    transformers.models.qwen3_5.modeling_qwen3_5.create_causal_mask = (
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