import json
from dataclasses import dataclass, field

@dataclass
class Qwen4TextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    hidden_act: str

    # hybrid attention schedule: "linear_attention" | "qwen_sparse_attention"
    layer_types: list[str]
    full_attention_interval: int

    # linear attention (GatedDeltaNet)
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    output_gate_type: str

    # mixture of experts
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    router_aux_loss_coef: float

    # hyper-connections
    hc_count: int
    hc_lowrank: int

    # qwen sparse attention (QSA) indexer
    indexer_n_heads: int | None
    indexer_kv_heads: int | None
    indexer_head_dim: int | None
    indexer_budget: int | None
    indexer_compress_ratio: int | None

    # per-layer embeddings (PLE)
    ple_layer_ids: list[int]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    seed: int
    split_ngram_parts: int

    eos_token_id: int

    rope_parameters: dict

    # PLE tensor-parallel head shard. Not a checkpoint field: the trainer sets
    # it before the model is built, because the n-gram table is far too large
    # to materialize whole and split afterwards the way every other weight is.
    ple_tp_size: int = 1
    ple_tp_rank: int = 0

    @property
    def use_qsa(self) -> bool:
        return self.indexer_n_heads is not None

    @classmethod
    def from_dict(cls, tc: dict, tie_fallback: bool = False) -> "Qwen4TextConfig":
        """Build from an HF ``text_config`` dict (also accepts a flat text-only config)."""
        interval = tc.get("full_attention_interval", 4)
        layer_types = tc.get("layer_types")
        if layer_types is None:
            layer_types = [
                "linear_attention" if (i + 1) % interval else "qwen_sparse_attention"
                for i in range(tc["num_hidden_layers"])
            ]
        else:
            # released checkpoints label the indexed layers "full_attention"
            layer_types = [
                "qwen_sparse_attention" if t == "full_attention" else t for t in layer_types
            ]

        eos = tc.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if eos else None

        out = cls(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            max_position_embeddings=tc["max_position_embeddings"],
            rms_norm_eps=tc["rms_norm_eps"],
            tie_word_embeddings=tc.get("tie_word_embeddings", tie_fallback),
            hidden_act=tc.get("hidden_act", "silu"),
            layer_types=layer_types,
            full_attention_interval=interval,
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_num_value_heads=tc["linear_num_value_heads"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            output_gate_type=tc.get("output_gate_type") or tc.get("hidden_act", "silu"),
            moe_intermediate_size=tc["moe_intermediate_size"],
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            norm_topk_prob=tc.get("norm_topk_prob", True),
            router_aux_loss_coef=tc.get("router_aux_loss_coef", 0.001),
            hc_count=tc["hc_count"],
            hc_lowrank=tc["hc_lowrank"],
            indexer_n_heads=tc.get("indexer_n_heads"),
            indexer_kv_heads=tc.get("indexer_kv_heads"),
            indexer_head_dim=tc.get("indexer_head_dim"),
            indexer_budget=tc.get("indexer_budget"),
            indexer_compress_ratio=tc.get("indexer_compress_ratio"),
            ple_layer_ids=sorted(set(tc.get("ple_layer_ids") or [])),
            ple_embed_dim=tc.get("ple_embed_dim") or tc["hidden_size"],
            ple_conv_kernel_size=tc.get("ple_conv_kernel_size", 4),
            ngram_size=tc.get("ngram_size", 3),
            heads_per_ngram=tc.get("heads_per_ngram", 8),
            ngram_vocab_size_base=tc.get("ngram_vocab_size_base", 20_000_000),
            make_ngram_vocab_size_divisible_by=tc.get("make_ngram_vocab_size_divisible_by", 128),
            seed=tc.get("seed", 1234),
            split_ngram_parts=tc.get("split_ngram_parts", 512),
            eos_token_id=eos,
            rope_parameters=tc["rope_parameters"],
        )
        out.validate()
        return out

    def validate(self) -> None:
        """Port of HF ``Qwen4ExpTextConfig.validate_architecture``."""
        bad = sorted(set(self.layer_types) - {"linear_attention", "qwen_sparse_attention"})
        if bad:
            raise ValueError(f"Unsupported Qwen4 layer types: {bad}.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}."
            )
        if self.output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError(f"Unsupported output gate activation: {self.output_gate_type}.")
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4 requires hc_count > 1, got {self.hc_count}.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                f"num_experts_per_tok must be in [1, {self.num_experts}], got {self.num_experts_per_tok}."
            )
        if self.moe_intermediate_size <= 0 or self.shared_expert_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size and shared_expert_intermediate_size must be > 0.")

        qsa_fields = (
            "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim",
            "indexer_budget", "indexer_compress_ratio",
        )
        qsa = {n: getattr(self, n) for n in qsa_fields}
        if any(v is not None for v in qsa.values()):
            missing = [n for n, v in qsa.items() if v is None]
            if missing:
                raise ValueError(f"QSA config is missing required fields: {missing}.")
            if any(v <= 0 for v in qsa.values()):
                raise ValueError(f"QSA config values must be positive: {qsa}.")
            if self.indexer_kv_heads != 1:
                raise ValueError("Qwen4 QSA requires indexer_kv_heads=1.")
            if self.indexer_budget % self.indexer_compress_ratio != 0:
                raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
            partial = self.rope_parameters.get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial)
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    f"attention RoPE dims must fit the QSA index head: rotary_dim={rotary_dim}, "
                    f"indexer_head_dim={self.indexer_head_dim}."
                )

        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            if ngram_heads <= 0 or self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads != 0:
                raise ValueError(
                    f"ple_embed_dim must be divisible by the n-gram head count: "
                    f"{self.ple_embed_dim} % {ngram_heads} != 0."
                )
            bad_ids = [i for i in self.ple_layer_ids if i < 1 or i > self.num_hidden_layers]
            if bad_ids:
                raise ValueError(
                    f"ple_layer_ids must be one-indexed ids in [1, {self.num_hidden_layers}], got {bad_ids}."
                )
            non_linear = [i for i in self.ple_layer_ids if self.layer_types[i - 1] != "linear_attention"]
            if non_linear:
                raise ValueError(f"PLE is only supported on linear_attention layers, got {non_linear}.")
            if self.eos_token_id is None:
                raise ValueError("eos_token_id must be set when PLE layers are enabled.")
            if self.ple_tp_size < 1 or not (0 <= self.ple_tp_rank < self.ple_tp_size):
                raise ValueError(
                    f"ple_tp_rank must be in [0, ple_tp_size): got rank={self.ple_tp_rank}, "
                    f"size={self.ple_tp_size}."
                )
            if ngram_heads % self.ple_tp_size != 0:
                raise ValueError(
                    f"the n-gram table is sharded by head, so the head count must be "
                    f"divisible by the TP size: {ngram_heads} % {self.ple_tp_size} != 0."
                )

@dataclass
class Qwen4VisionConfig:
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

@dataclass
class Qwen4Config:
    text: Qwen4TextConfig
    vision: Qwen4VisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path: str) -> "Qwen4Config":
        with open(path, "r") as f:
            raw = json.load(f)
        tc = raw["text_config"]
        text = Qwen4TextConfig.from_dict(
            tc, tie_fallback=raw.get("tie_word_embeddings", False)
        )

        vc = raw["vision_config"]
        vision = Qwen4VisionConfig(
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
