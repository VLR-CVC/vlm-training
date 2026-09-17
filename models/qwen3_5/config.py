# DEPRECATED (Qwen3.5 model definition; its config dataclass still feeds train/flops_estimation.py): the old model definitions and DTensor parallelism are no longer used by train/train_qwen.py. Training runs on models/qwen3_5_tt and models/qwen3_vl_tt with train/parallel/ (TITAN_MIGRATION_v2.md). Kept for reference until they are deleted.
import json
from dataclasses import dataclass

@dataclass
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool

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

    # Qwen Sparse Attention. Absent from a stock Qwen3.5 config, in which case
    # `use_qsa` is False and the full-attention layers run the dense varlen
    # kernel exactly as before -- this is additive, not a behaviour change.
    #
    # `indexer_budget` is the number of tokens a query may attend outside its
    # always-visible tail; `indexer_compress_ratio` is the block size the keys
    # are mean-pooled into, so the top-k is over `budget // ratio` blocks.
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None

    @property
    def use_qsa(self) -> bool:
        return self.indexer_n_heads is not None

    def validate_qsa(self) -> None:
        """All-or-nothing, and the invariants the indexer relies on."""
        fields = (
            self.indexer_n_heads, self.indexer_kv_heads, self.indexer_head_dim,
            self.indexer_budget, self.indexer_compress_ratio,
        )
        if all(f is None for f in fields):
            return
        if any(f is None for f in fields):
            raise ValueError("indexer_* must be set all together or not at all")
        if any(f <= 0 for f in fields):
            raise ValueError("every indexer_* value must be positive")
        if self.indexer_kv_heads != 1:
            raise ValueError("the indexer pools one key stream; indexer_kv_heads must be 1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        rotary_dim = self.rotary_dim
        if self.indexer_head_dim < rotary_dim:
            raise ValueError(
                f"indexer_head_dim ({self.indexer_head_dim}) must be at least the "
                f"rotary width ({rotary_dim}): `apply_rope` takes rotary_dim from "
                "cos.shape[-1] and would slice past the end of the indexer head"
            )

    @property
    def rotary_dim(self) -> int:
        """Width RoPE actually rotates, which may be a fraction of `head_dim`."""
        factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0)
        return int(self.head_dim * factor)

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
class Qwen3_5Config:
    text: Qwen3_5TextConfig
    vision: Qwen3_5VisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path: str) -> "Qwen3_5Config":
        with open(path, "r") as f:
            raw = json.load(f)
        tc = raw["text_config"]
        rs = tc.get("rope_scaling") or {}
        text = Qwen3_5TextConfig(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
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
            indexer_n_heads=tc.get('indexer_n_heads'),
            indexer_kv_heads=tc.get('indexer_kv_heads'),
            indexer_head_dim=tc.get('indexer_head_dim'),
            indexer_budget=tc.get('indexer_budget'),
            indexer_compress_ratio=tc.get('indexer_compress_ratio'),
        )
        text.validate_qsa()
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

