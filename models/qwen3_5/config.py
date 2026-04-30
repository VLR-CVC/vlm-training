import json
from dataclasses import dataclass

@dataclass
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool

    # moe
    num_experts: int
    num_experts_per_tok: int
    router_aux_loss_coef: float
    shared_expert_intermediate_size: int

    # linear attention
    layer_types: list[str]
    full_attention_interval: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int

    # multi token prediction
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool

    rope_parameters: dict

@dataclass
class Qwen3_5VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    hidden_act: str
    deepstack_visual_indexes: list[int]

@dataclass
class Qwen3VLConfig:
    text: Qwen3_5TextConfig
    vision: Qwen3_5VisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path: str) -> "Qwen3VLConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        tc = raw["text_config"]
        rs = tc.get("rope_scaling") or {}
        text = Qwen3_5TextConfig(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            max_position_embeddings=tc["max_position_embeddings"],
            rms_norm_eps=tc["rms_norm_eps"],
            layer_types=tc['layer_types'],
            full_attention_interval=tc['full_attention_interval'],
            linear_conv_kernel_dim=tc['linear_conv_kernel_dim'],
            linear_key_head_dim=tc['linear_key_head_dim'],
            linear_num_key_heads=tc['linear_num_key_heads'],
            linear_num_value_heads=tc['linear_num_value_heads'],
            linear_value_head_dim=tc['linear_value_head_dim'],
            mtp_num_hidden_layers=tc['mtp_num_hidden_layers'],
            mtp_use_dedicated_embeddings=tc['mtp_use_dedicated_embeddings'],
            tie_word_embeddings=tc.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            rope_parameters=tc['rope_parameters'],
            num_experts=tc['num_experts'],
            num_experts_per_tok=tc['num_experts_per_tok'],
            router_aux_loss_coef=tc['router_aux_loss_coef'],
            shared_expert_intermediate_size=tc['shared_expert_intermediate_size'],
        )
        vc = raw["vision_config"]
        vision = Qwen3_5VisionConfig(
            depth=vc["depth"],
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_heads=vc["num_heads"],
            in_channels=vc["in_channels"],
            patch_size=vc["patch_size"],
            temporal_patch_size=vc["temporal_patch_size"],
            spatial_merge_size=vc["spatial_merge_size"],
            num_position_embeddings=vc["num_position_embeddings"],
            out_hidden_size=vc["out_hidden_size"],
            hidden_act=vc["hidden_act"],
            deepstack_visual_indexes=vc["deepstack_visual_indexes"],
        )
        return cls(
            text=text,
            vision=vision,
            image_token_id=raw["image_token_id"],
            video_token_id=raw["video_token_id"],
            vision_start_token_id=raw["vision_start_token_id"],
            vision_end_token_id=raw["vision_end_token_id"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            torch_dtype=raw.get("torch_dtype") or tc.get("dtype", "bfloat16"),
        )

