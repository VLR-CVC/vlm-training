"""Generate a scaled-down Qwen4-Exp checkpoint.

The trainer takes its architecture from ``model_dir/config.json`` and always
loads weights before re-initializing them (``random_init`` runs *after*
``from_pretrained``), so a shrunk model needs a real directory: a config, a
safetensors file, and the tokenizer/processor files the dataloader reads.

    python -m utils.make_qwen4 --size 9b --out /path/to/qwen4_9b

Sizes are named by parameter count (``615m``, ``9b``, ``27b``). Everything
structural is kept at every size -- the linear/sparse layer cycle,
hyper-connections at ``hc_count=4``, the MoE router and shared expert, the QSA
indexer, and a PLE layer whose n-gram table is written as ``split_ngram_parts``
shards so the concatenating branch of ``load_safetensors_into`` is exercised.
Only the widths, depths and expert counts move.

``--count-only`` builds the model on the meta device and prints the parameter
count without writing weights, which is how the sizes below were tuned.

Every size is TP=2-friendly: ``hidden_size``, ``moe_intermediate_size``,
``shared_expert_intermediate_size``, the attention/GDN head counts,
``indexer_n_heads``, ``hc_lowrank`` and the vision ``num_heads`` /
``intermediate_size`` are all even.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.qwen4.config import Qwen4Config
from models.qwen4.model import Qwen4ForCausalLM

# Files copied (symlinked) from the real snapshot: the tokenizer and the
# image/video processors that `train_qwen.py` builds from `model_dir`.
AUX_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

# `vocab_size` stays at the real 248320 at every size: the tokenizer emits ids
# up to that range and a narrower embedding would index out of bounds on real
# data. Embeddings are untied, so it costs 2 * 248320 * hidden_size -- 381M of
# the 615M model, but only 4% of the 27B one.
COMMON_TEXT = {
    "model_type": "qwen4_exp",
    "vocab_size": 248320,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_cache": False,
    "initializer_range": 0.02,
    "bos_token_id": 248043,
    "eos_token_id": 248044,
    "pad_token_id": 248042,

    # same 4-layer period as the real model
    "full_attention_interval": 4,

    # GatedDeltaNet. `linear_key_head_dim` stays 128 because FlashQLA's
    # `kkt_solve` asserts it; head *counts* scale instead, keeping the real
    # 1:3 key:value ratio.
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "output_gate_type": "sigmoid",

    "num_experts_per_tok": 4,
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.001,
    "output_router_logits": False,

    "hc_count": 4,

    # QSA. `indexer_budget // indexer_compress_ratio` blocks kept per query.
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_compress_ratio": 4,

    # PLE on layer 2 (one-indexed, and it must be a linear_attention layer).
    # The real `ngram_vocab_size_base` of 20e6 gives a 95 GiB table on its own,
    # so it is the field that scales least with the rest.
    "ple_layer_ids": [2],
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 4,
    "make_ngram_vocab_size_divisible_by": 128,
    "seed": 1234,
    "split_ngram_parts": 8,

    # head_dim is 128 at every size (the real model uses 256), so
    # partial_rotary_factor 0.25 leaves 32 rotary dims and mrope_section sums
    # to 16 (= rotary_dim / 2), which fits indexer_head_dim.
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [6, 5, 5],
        "mrope_interleaved": True,
    },
    "dtype": "bfloat16",
}

# text overrides per size; `layer_types` is derived from `num_hidden_layers`
SIZES: dict[str, dict] = {
    # ~615M. Fits on a single 40 GB card. Measured peak on one L40S, forward +
    # backward + AdamW step with no activation checkpointing: 13.6 GiB at
    # seq_len 2048, 24.9 GiB at 4096.
    "615m": {
        "text": {
            "hidden_size": 768,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 32768,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 12,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "num_experts": 32,
            "hc_lowrank": 128,
            "indexer_n_heads": 4,
            "indexer_budget": 512,
            "ple_embed_dim": 768,
            "ngram_vocab_size_base": 16384,
        },
        "vision": {
            "depth": 6,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_heads": 6,
        },
    },
    # ~9.0B.
    "9b": {
        "text": {
            "hidden_size": 1536,
            "num_hidden_layers": 24,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 24,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 128,
            "hc_lowrank": 192,
            "indexer_n_heads": 4,
            "indexer_budget": 2048,
            "ple_embed_dim": 1536,
            "ngram_vocab_size_base": 131072,
        },
        "vision": {
            "depth": 16,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 8,
        },
    },
    # ~27B.
    "27b": {
        "text": {
            "hidden_size": 2048,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "linear_num_key_heads": 12,
            "linear_num_value_heads": 36,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "num_experts": 190,
            "hc_lowrank": 256,
            "indexer_n_heads": 4,
            "indexer_budget": 2048,
            "ple_embed_dim": 2048,
            "ngram_vocab_size_base": 131072,
        },
        # the real vision tower, unchanged
        "vision": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
        },
    },
}


# The real Qwen4-Exp text stack, layer count cut from 48 to 12 so it fits on
# one card. Everything that shapes a kernel is the production value -- above
# all `head_dim = 256`, which every other preset here sets to 128 and which is
# the one that decides whether flex-attention's backward fits in shared memory.
# Use this to measure QSA, not `9b`/`27b`.

def _real_shape(layers: int) -> dict:
    return {
        "text": {
            "hidden_size": 2560,
            "num_hidden_layers": layers,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "max_position_embeddings": 32768,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            # 512 experts is the production number; at 12 layers it is also
            # most of the parameters, so cut it to keep the model on one card.
            # `num_experts_per_tok` is what expert FLOPs scale with, so the
            # compute mix is unchanged.
            "num_experts": 32,
            "num_experts_per_tok": 10,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_budget": 2048,
            "ple_embed_dim": 2560,
            "ngram_vocab_size_base": 131072,
            # head_dim 256 puts rotary_dim at 64, so mrope_section sums to 32
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
        "vision": {
            "depth": 6,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_heads": 6,
        },
    }


def _wide(hidden: int) -> dict:
    """A deliberately wide, shallow shape, for the float8 crossover sweep.

    fp8 only pays once a GEMM is big enough that the quantize/scale work is
    amortized. Every Qwen4 preset above is *narrow* -- the released checkpoint
    is only `hidden_size = 2560` -- so their dense projections sit below that
    crossover and `training.float8 = true` costs throughput instead of buying
    it (see the README). These presets move `hidden_size` and hold everything
    else fixed, so a sweep over them isolates width as the variable.

    Four layers, and only 8 experts: the point is the width of the dense
    projections, and depth would only buy OOM. Note that expert FLOPs depend on
    `num_experts_per_tok`, not `num_experts`, so cutting the expert count saves
    memory without changing the compute mix.
    """
    return {
        "text": {
            "hidden_size": hidden,
            "num_hidden_layers": 4,
            "num_attention_heads": hidden // 128,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 32768,
            "linear_num_key_heads": hidden // 256,
            "linear_num_value_heads": 3 * (hidden // 256),
            "moe_intermediate_size": hidden // 4,
            "shared_expert_intermediate_size": hidden // 4,
            "num_experts": 8,
            "hc_lowrank": 256,
            "indexer_n_heads": 4,
            "indexer_budget": 2048,
            "ple_embed_dim": hidden,
            "ngram_vocab_size_base": 131072,
        },
        # held constant across the sweep so only the text width moves
        "vision": {
            "depth": 4,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_heads": 6,
        },
    }


# 27b with the expert count cut back, for fitting a 4-GPU box. The optimizer
# is what binds: fp32 master + fp32 grads + two fp32 AdamW moments is 16 bytes
# per parameter, so 4 x 95 GiB caps a job at ~25.5B parameters before a single
# activation. `num_experts` is the cheapest dial -- expert FLOPs scale with
# `num_experts_per_tok`, so cutting the count moves memory without moving the
# compute mix. See `configs/cvc/qwen4/pp4.toml`.
def _experts_variant(num_experts: int) -> dict:
    import copy as _copy
    spec = _copy.deepcopy(SIZES["27b"])
    spec["text"]["num_experts"] = num_experts
    return spec


for _name, _e in (("22b", 150), ("21b", 142), ("18b", 118)):
    SIZES[_name] = _experts_variant(_e)

SIZES["hd256"] = _real_shape(12)

# wide2k .. wide8k
for _hidden in (2048, 3072, 4096, 6144, 8192):
    SIZES[f"wide{_hidden // 1024}k"] = _wide(_hidden)


def build_config(size: str) -> dict:
    spec = SIZES[size]
    text = dict(COMMON_TEXT)
    text.update(spec["text"])

    interval = text["full_attention_interval"]
    layers = text["num_hidden_layers"]
    if layers % interval:
        raise ValueError(f"{size}: num_hidden_layers must be a multiple of {interval}")
    text["layer_types"] = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(layers)
    ]

    vision = {
        "model_type": "qwen4_exp",
        "in_channels": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
        "deepstack_visual_indexes": [],
        "out_hidden_size": text["hidden_size"],
        **spec["vision"],
    }

    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "language_model_only": False,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": text,
        "vision_config": vision,
    }


def init_weights(model: torch.nn.Module, num_layers: int, seed: int = 42) -> None:
    """The `train.utils.init_qwen4` recipe, minus its distributed broadcasts."""
    std = 0.02
    scaled_std = std / math.sqrt(2 * num_layers)

    def apply(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        elif "Norm" in m.__class__.__name__:
            if getattr(m, "weight", None) is not None:
                torch.nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

    torch.manual_seed(seed)
    model.apply(apply)

    with torch.no_grad():
        for name, param in model.named_parameters():
            # stacked expert weights are bare Parameters, so `apply` misses them
            if name.endswith("experts.gate_up_proj"):
                torch.nn.init.normal_(param, mean=0.0, std=std)
            elif name.endswith("experts.down_proj"):
                torch.nn.init.normal_(param, mean=0.0, std=scaled_std)
            elif "o_proj.weight" in name or "down_proj.weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=scaled_std)


def shard_ple_tables(
    state: dict[str, torch.Tensor],
    parts: int,
    model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Split each n-gram table into the `shard_{i}.weight` keys the loader reads.

    Writing the table whole would work -- `load_safetensors_into` falls back to
    a 1:1 key match -- but then the shard-concatenation branch never runs, and
    that branch is the one piece of loading logic Qwen4 does not share with
    Qwen3.5.

    The runtime table pads every n-gram head to a common row count so it can be
    head-sharded across TP ranks; the checkpoint packs the heads back to back
    at their true prime sizes. Pass `model` to convert on the way out -- the
    row layouts are otherwise not the same tensor.
    """
    tables = {}
    if model is not None:
        for name, module in model.named_modules():
            if hasattr(module, "packed_table"):
                tables[f"{name}.ngram_embedding.weight"] = module.packed_table()

    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not key.endswith(".ngram_embedding.weight"):
            out[key] = tensor
            continue
        tensor = tables.get(key, tensor)
        prefix = key[: -len(".weight")]
        rows = tensor.shape[0]
        height = rows // parts
        for i in range(parts):
            start = i * height
            stop = rows if i == parts - 1 else start + height
            out[f"{prefix}.shard_{i}.weight"] = tensor[start:stop].clone()
    return out


def report(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total / 1e9:.3f}B")
    print(f"  bfloat16 weights: {total * 2 / 2**30:.1f} GiB")
    print(f"  fp32 weights + grads + AdamW moments: {total * 16 / 2**30:.1f} GiB")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="615m", choices=sorted(SIZES), help="which preset to build")
    ap.add_argument("--out", help="directory to create the snapshot in")
    ap.add_argument(
        "--source",
        default="/data/151-2/users/tockier/models/qwen4",
        help="real snapshot to link the tokenizer and processor files from",
    )
    ap.add_argument(
        "--config-only",
        action="store_true",
        help="write config.json and the aux links, skip generating weights",
    )
    ap.add_argument(
        "--count-only",
        action="store_true",
        help="build on the meta device, print the parameter count, write nothing",
    )
    args = ap.parse_args()

    config = build_config(args.size)

    if args.count_only:
        # `Qwen4Config.from_json` is the only parser, so round-trip through a
        # temporary file rather than duplicating it.
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        cfg = Qwen4Config.from_json(path)
        os.unlink(path)
        with torch.device("meta"):
            model = Qwen4ForCausalLM(cfg)
        report(model)
        return

    if not args.out:
        ap.error("--out is required unless --count-only is given")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"wrote {config_path}")

    source = Path(args.source)
    for name in AUX_FILES:
        src, dst = source / name, out / name
        if not src.exists():
            print(f"  skip {name} (not in {source})")
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    print(f"linked tokenizer and processor files from {source}")

    # Parsing here rather than trusting the literal: `validate()` is the same
    # check the trainer runs, and a bad config should fail now, not at launch.
    cfg = Qwen4Config.from_json(str(config_path))
    if args.config_only:
        return

    model = Qwen4ForCausalLM(cfg)
    init_weights(model, num_layers=cfg.text.num_hidden_layers)
    report(model)

    # Integer buffers (the PLE hashing constants) must keep their dtype.
    state = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    state = shard_ple_tables(state, cfg.text.split_ngram_parts, model)
    state = {k: v.contiguous() for k, v in state.items()}

    weights_path = out / "model.safetensors"
    save_file(state, str(weights_path), metadata={"format": "pt"})

    # A single-file snapshot loads without an index, but then the PLE shard
    # keys are matched 1:1 and never concatenated. The index is what routes
    # them through the concatenating path.
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state.values())},
        "weight_map": {k: weights_path.name for k in state},
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    size = weights_path.stat().st_size / 2**30
    print(f"wrote {weights_path} ({size:.2f} GiB, {len(state)} tensors)")


if __name__ == "__main__":
    main()
