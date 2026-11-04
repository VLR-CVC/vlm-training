# Copyright (c) 2026. Benchmark-only config functions for TorchTitan's qwen3_5.
#
# Lives outside the pinned checkout on purpose: TORCHTITAN_BENCHMARK.md fixes
# torchtitan at b21f7d43e, and a reference you have edited is not a reference.
# Reached with `--module ttbench`, which ConfigManager._load_config resolves as a
# fully-qualified module path; put this directory on PYTHONPATH.
#
# Only two things here cannot be a CLI override:
#   seq_len   -- config_fn() is called with no arguments (config/manager.py:163)
#                and model_registry(flavor, seq_len=...) feeds it into
#                get_config() itself, not just max_context_length. Overriding
#                --training.max-context-length builds a 262144 model and runs
#                10240 rows through it, which is a different measurement.
#   dataset   -- MM_DATASETS entries carry Suppress-ed callables, so tyro will
#                not construct one from the command line.
#
# Everything else in TORCHTITAN_BENCHMARK.md section 3 is a flag:
#   --parallelism.tensor-parallel-degree 4
#   --parallelism.fsdp-reshard-after-forward never
#   --activation-checkpoint none        (tyro subcommand; "none" disables AC)
#   --compile.enable
#   --training.disable-cuda-graphs      (mandatory -- see below)
#   --training.num-tokens-per-microbatch-per-dp-rank 10240
#   --training.steps 20

import torch._dynamo

from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS
from torchtitan.models.qwen3_5.config_registry import (
    _multimodal_collator_config,
    qwen35_9b,
    qwen35_35b_a3b,
    qwen35_122b_a10b,
    qwen35_397b_a17b,
)
from torchtitan.trainer import Trainer

BENCH_SEQ_LEN = 10240

# Packed multimodal rows carry a different number of images each step, so the
# pixel tensor changes shape and dynamo recompiles. At the default limit of 8 the
# run dies at step 14 with
#   torch._dynamo.exc.Unsupported: Dynamo recompile limit exceeded
# inside spmd_redistribute_per_axis (job 1810834). 32 is what our own
# configs/jupiter/scaling/*.toml set (`dynamo_recompile_limit`), so this matches
# the arms rather than favouring one. Torchtitan does the same thing for gpt_oss
# in models/gpt_oss/parallelize.py:40.
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

# NOT set: automatic_dynamic_shapes. It was tried (job 1811333) against the
# attn_gym gdn_chunk_fwd stride assert and did not help -- that bug is in the
# fake kernel's layout, not in the symbolic graph, and the static run produced
# the same failure plus heavy recompile churn (most steps 200-300 tok/s/GPU
# against ~4,000 with dynamic shapes). Fixed properly by
# patches/attn_gym_gdn_fake_strides.py instead.


def qwen35_9b_local(seq_len: int = BENCH_SEQ_LEN) -> Trainer.Config:
    """`qwen35_9b` at our seq_len, reading tars from disk instead of the hub.

    Stock `qwen35_9b` streams `pixparse/cc12m-wds` from HuggingFace. JUPITER's
    compute nodes have no outbound network, so that config cannot start here.
    `cc12m-test` is the same processor over a local directory of WebDataset
    tars -- 3 MB, shipped in the checkout. Enough for a step-time measurement,
    not enough for phase 2, which needs our own shards.
    """
    config = qwen35_9b(seq_len=seq_len)
    config.dataloader.dataset = MM_DATASETS["cc12m-test"]
    config.dataloader.collator = _multimodal_collator_config(MM_DATASETS["cc12m-test"])
    return config


# ---------------------------------------------------------------------------
# Phase 2: our own shards.
#
# `utils/prepare_nemotron_energon.py` writes WebDataset tars of `{key}.png` +
# `{key}.json`, which is the same shape `cc12m-test` reads. The work is not the
# plumbing, it is matching what the trainer actually feeds the model. Four
# things, and all four move token counts:
#
#   1. the chat template   -- assets/chat_template_sft.jinja, the SFT one.
#      PERFORMANCE.md 17.6: the inference template drops <think> spans from all
#      but the last turn and cuts the median plotqa sample from 4677 to 1686
#      tokens. Benchmarking against the wrong template compares row occupancies.
#   2. the message rebuild -- `data/cookers.nemotron_messages`.
#   3. max_pixels          -- 1048576 (train_qwen.py:308). Their default is
#      16777216, 16x larger, which would inflate the vision tower's work.
#   4. cap_image_size      -- long side <= 1024 before the processor sees it
#      (energon_dataloader.py:15, applied on both encoder paths).
#
# 2 and 4 are duplicated below rather than imported: `data/energon_dataloader`
# imports megatron.energon at module level, and installing energon into the
# torchtitan venv to reuse twenty lines trades a small duplication for a
# dependency-resolution risk in the environment the benchmark runs in. The
# duplication is checked, not assumed -- see `verify_token_counts` at the end.
#
# What deliberately does NOT match: labels. `_process_mm_sample` trains on every
# non-vision token; we mask to assistant spans. Chunked CE runs over all
# positions either way, so step time is unaffected -- but the losses are not
# comparable between the two frameworks and the write-up must not put them in
# the same table.

import io
import json
import os
from typing import Any

import torch
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    _process_mm_sample,
    HuggingFaceStreamingSource,
    MMSamplePackingConfig,
    MultiModalProcessor,
    SingleDatasetConfig,
)

MAX_IMAGE_SIZE = 1024  # energon_dataloader.py:13
MAX_PIXELS = 1048576  # train_qwen.py:308

# Deployed next to this module, not resolved against the cwd: grain worker
# processes do not promise to keep it.
CHAT_TEMPLATE = os.path.join(os.path.dirname(__file__), "chat_template_sft.jinja")

# The exact string the template emits per image. Split on it to hand
# `_process_mm_sample` a None slot, which it refills with the right number of
# image tokens once it knows the patch count -- re-inserting it ourselves would
# double the markers.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

_env = None


def _template(path: str):
    """Compile the jinja chat template once per worker."""
    global _env
    if _env is None:
        from jinja2 import Environment
        from jinja2.exceptions import TemplateError

        def raise_exception(message):
            raise TemplateError(message)

        # Same jinja settings transformers compiles chat templates with
        # (`_compile_jinja_template`). With trim_blocks off, every `%}` leaves
        # its trailing newline in the output and the median plotqa sample gains
        # ~200 tokens against the trainer's rendering of the same template.
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        with open(path) as f:
            _env = env.from_string(f.read())
    return _env


def nemotron_messages(turns: list, add_system_prompt: bool = True) -> list[dict]:
    """Copy of `data/cookers.nemotron_messages`. Keep in sync."""
    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for turn in turns:
        content = []
        for part in turn["content"]:
            # v3 writes text parts as bare strings, v2 as {"type": "text", ...}
            if isinstance(part, str):
                if part:
                    content.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                content.append({"type": "image"})
            elif part.get("text"):
                content.append({"type": "text", "text": part["text"]})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": turn["role"], "content": content})

    return messages


def cap_image_size(image, max_size: int = MAX_IMAGE_SIZE):
    """Copy of `data/energon_dataloader.cap_image_size`. Keep in sync."""
    from PIL import Image

    w, h = image.size
    longest = max(w, h)
    if longest > max_size:
        scale = max_size / longest
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = image.resize(new_size, Image.BICUBIC)

    # smart_resize raises above 200:1, and the ratio survives the downscale
    w, h = image.size
    if max(w, h) > 200 * min(w, h):
        target = (max(w, -(-h // 200), 1), max(h, -(-w // 200), 1))
        out = Image.new(image.mode, target)
        out.paste(image, ((target[0] - w) // 2, (target[1] - h) // 2))
        image = out
    return image


def _process_nemotron_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    **kwargs,
) -> dict[str, Any] | None:
    """Our nemotron shards, rendered exactly as `train.train_qwen` renders them."""
    record = sample.get("json")
    image = sample.get("png")
    if record is None or image is None:
        return None
    if isinstance(record, (bytes, bytearray, str)):
        record = json.loads(record)

    rendered = _template(CHAT_TEMPLATE).render(
        messages=nemotron_messages(record["messages"]),
        add_generation_prompt=False,
    )

    # One image per sample is a guarantee of the packer
    # (`cooker_nemotron`), so a single split is the whole interleave.
    chunks = rendered.split(VISION_BLOCK)
    if len(chunks) != 2:
        return None

    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    image = cap_image_size(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    return _process_mm_sample(
        texts=[chunks[0], None, chunks[1]],
        images=[None, buf.getvalue(), None],
        tokenizer=tokenizer,
        **kwargs,
    )


def nemotron_dataset(path: str, max_doc_tokens: int | None = None) -> SingleDatasetConfig:
    """A local directory of our WebDataset tars, read the way the trainer reads it."""
    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path=path,
            split="train",
            load_dataset_kwargs={"data_files": {"train": "*.tar"}},
        ),
        processor=MultiModalProcessor.Config(
            sample_processor=_process_nemotron_sample,
            max_pixels=MAX_PIXELS,
        ),
        post_filters=(
            (lambda sample: sample is not None,)
            if max_doc_tokens is None
            # our energon encoder raises SkipSample for a document longer than the
            # row; torchtitan's packer instead fails with "Inputs to PackedBatch must
            # be truncated to max length". Drop them here the same way.
            else (lambda sample: sample is not None and len(sample["input_ids"]) <= max_doc_tokens,)
        ),
    )


PLOTQA = os.environ.get("TTBENCH_PLOTQA", "/e/scratch/open-sci-mm/ockier1/vlm_datasets/plotqa_cot")


def qwen35_9b_plotqa(seq_len: int = BENCH_SEQ_LEN) -> Trainer.Config:
    """`qwen35_9b` on plotqa_cot, packed -- the phase 3 arm.

    Packing is not optional. Unpacked, every row is one document padded to
    max_context_length; our runs pack to 99% occupancy, and comparing a full row
    against a padded one measures padding, not implementations. No config in
    their `config_registry.py` enables `MMSamplePackingConfig`, so this is the
    first one that does.
    """
    config = qwen35_9b(seq_len=seq_len)
    dataset = nemotron_dataset(PLOTQA)
    config.dataloader.dataset = MMSamplePackingConfig(
        dataset=dataset,
        num_packing_bins=8,
    )
    config.dataloader.collator = _multimodal_collator_config(dataset)
    return config


# ---------------------------------------------------------------------------
# Scaling studies. Only seq_len needs a function per value (config_fn() takes no
# arguments); parallelism, token budget and gradient accumulation are all CLI:
#
#   strong scaling -- fixed global batch, DDP, accumulation absorbs the scale
#     --parallelism.data-parallel-replicate-degree N
#     --parallelism.data-parallel-shard-degree 1
#     --training.num-tokens-per-microbatch-per-dp-rank <fits one GPU>
#     --training.num-tokens-per-train-step <fixed across all N>
#
#   weak scaling -- FSDP, no accumulation, global batch grows with the machine
#     --parallelism.data-parallel-shard-degree -1
#     --training.num-tokens-per-microbatch-per-dp-rank <max that fits at this N>
#
# There is no separate DDP path in qwen3_5/parallelize.py -- it only ever calls
# apply_fsdp_to_*. `shard_degree=1` with `replicate_degree=N` is DDP in all but
# name: parameters replicated, gradients all-reduced, nothing sharded. Phase 1
# ran that mesh already (dp_shard=1, tp=4).

SEQ_16384 = 16384


def qwen35_9b_plotqa_16384(seq_len: int = SEQ_16384) -> Trainer.Config:
    """`qwen35_9b_plotqa` at 16384 rather than 10240.

    Attention is per-document under varlen masking, so a longer row costs
    memory but no extra attention work -- what it buys is packing. PERFORMANCE.md
    17.6a measures the plotqa skip rate at **11.8% at 10240 and 0.1% at 16384**:
    at 10240 the trainer silently discards one document in eight, and always the
    longest ones. For a throughput study that is a biased workload, not just a
    smaller one.
    """
    return qwen35_9b_plotqa(seq_len=seq_len)


# Weak scaling with seq_len grown to spend the memory FSDP frees as dp rises.
# Calibrated from the four constant-batch runs (jobs 1814540/42/44/45), which all
# ran 32768 tokens/rank:
#     peak GiB      4n 85.29   8n 82.89   16n 81.14   32n 79.77
#     => model+opt  4n 15.3    8n 12.9    16n 11.1    32n  9.8
#     => activations 70 GiB at 32768 tokens/rank = 2.14 MiB/token
# Targeting ~88 GiB peak (cards report 95 GiB usable) with 2 rows per rank.
# The whole saving between 4 and 32 nodes is 5.5 GiB, so seq_len moves only
# 16896 -> 18176 (+7.6%): that is the entire budget FSDP hands back, and it is
# far smaller than the intuition "more nodes, much more room" suggests.

def qwen35_9b_plotqa_16896() -> Trainer.Config:
    """Weak-scaling point for 4 nodes (shard=4, model+optimizer ~15.3 GiB)."""
    return qwen35_9b_plotqa(seq_len=16896)


def qwen35_9b_plotqa_17408() -> Trainer.Config:
    """Weak-scaling point for 8 nodes (shard=8, model+optimizer ~12.9 GiB)."""
    return qwen35_9b_plotqa(seq_len=17408)


def qwen35_9b_plotqa_17920() -> Trainer.Config:
    """Weak-scaling point for 16 nodes (shard=16, model+optimizer ~11.1 GiB)."""
    return qwen35_9b_plotqa(seq_len=17920)


def qwen35_9b_plotqa_18176() -> Trainer.Config:
    """Weak-scaling point for 32 nodes (shard=32, model+optimizer ~9.8 GiB)."""
    return qwen35_9b_plotqa(seq_len=18176)


# Refill round (jobs 1817033-36 as calibration). Peak = model+opt + 2.16 MiB/token,
# model+opt fitted as 8.93 + 25.5/nodes GiB (4n 15.3, 32n 9.73 measured). 8n and
# 16n peaked ~1.5 GiB under target, so both move up; 16n reuses 18176. Target
# 87.1 GiB, the 4n peak that ran clean. seq_len rounded down to a multiple of 128.

def qwen35_9b_plotqa_17728() -> Trainer.Config:
    """Refilled weak-scaling point for 8 nodes (predicted peak ~87.1 GiB)."""
    return qwen35_9b_plotqa(seq_len=17728)


def qwen35_9b_plotqa_18432() -> Trainer.Config:
    """Weak-scaling point for 64 nodes (model+optimizer ~9.3 GiB)."""
    return qwen35_9b_plotqa(seq_len=18432)


def qwen35_9b_plotqa_18496() -> Trainer.Config:
    """Weak-scaling point for 128 nodes (model+optimizer ~9.1 GiB)."""
    return qwen35_9b_plotqa(seq_len=18496)


# ---------------------------------------------------------------------------
# Local A/B against the port (TITAN_MIGRATION_v2.md): Qwen3.5-2B, one RTX PRO 6000.
#
#   TTBENCH_PLOTQA=/data/151-1/datasets/plotqa_cot TTBENCH_HF_ASSETS=<2B snapshot>
#   --module ttbench --config qwen35_2b_plotqa_local
#
# Two changes from upstream `qwen35_2b`, both to make the model the one the port
# trains, not to tune torchtitan:
#   - attn_backend "varlen" (upstream default "flex"); the port only has varlen.
#   - enable_weight_tying: the 2B checkpoint ties lm_head to the embedding, and
#     upstream `_2b` does not, which would add a 509M-parameter lm_head.
# Everything else -- fp32 master weights, fused torch AdamW with fp32 moments,
# chunked loss, FSDP mixed precision -- stays at torchtitan's defaults.

def _blackwell_shims() -> None:
    """torchtitan b21f7d43e refuses to start on compute capability >= 10.0 unless
    torch has `matmul.fp32_precision = "bfx9"` (pytorch/pytorch#195301, nightly
    only; torch 2.14.0 raises "Unknown precision: bfx9"). That setting emulates
    fp32 matmuls with bf16; bf16 compute is untouched. For this local benchmark the
    fp32 matmuls simply run as fp32. Module-level caller is `distributed/utils.py`
    (`init_distributed`), which resolves the name in its own module globals."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0):
        import torchtitan.distributed.utils as dist_utils

        dist_utils.enable_fp32_matmul_emulation_with_bf16x9 = lambda: None

        # VarlenInnerAttention activates FA4 on capability >= 10.0, which needs the
        # external `flash_attn` (cute) package. The port runs torch's built-in varlen
        # flash kernel, so keep torchtitan on the same one. The lookup is a
        # function-local import, so patching the module attribute takes effect.
        import torchtitan.tools.utils as tt_utils

        tt_utils.get_cuda_flash_attention_impl = lambda: None


def qwen35_2b_plotqa_local(seq_len: int = 8192) -> Trainer.Config:
    _blackwell_shims()
    from torchtitan.models.qwen3_5.config_registry import model_registry, qwen35_2b

    config = qwen35_2b(seq_len=seq_len)
    spec = model_registry("2B", seq_len=seq_len, attn_backend="varlen")
    spec.model.enable_weight_tying = True
    config.model_spec = spec
    config.hf_assets_path = os.environ["TTBENCH_HF_ASSETS"]
    dataset = nemotron_dataset(PLOTQA)
    config.dataloader.dataset = MMSamplePackingConfig(dataset=dataset, num_packing_bins=8)
    config.dataloader.collator = _multimodal_collator_config(dataset)
    return config


def qwen35_9b_plotqa_local_8192() -> Trainer.Config:
    """9B at 8192 tokens per row: fp32 master weights do not leave room for 16384
    on 96 GiB cards at TP=2 (OOM at 93.75 GiB after step 1)."""
    return qwen35_9b_plotqa_local(seq_len=8192)


def qwen35_9b_plotqa_local(seq_len: int = 16384) -> Trainer.Config:
    """S3 A/B (TITAN_MIGRATION_v2.md): upstream `qwen35_9b` on local plotqa, varlen
    attention (the port has no flex decoder path), same Blackwell shims as the 2B.
    The 9B checkpoint is untied, so no weight-tying change."""
    _blackwell_shims()
    from torchtitan.models.qwen3_5.config_registry import model_registry

    config = qwen35_9b(seq_len=seq_len)
    config.model_spec = model_registry("9B", seq_len=seq_len, attn_backend="varlen")
    config.hf_assets_path = os.environ["TTBENCH_HF_ASSETS"]
    dataset = nemotron_dataset(PLOTQA, max_doc_tokens=seq_len)
    config.dataloader.dataset = MMSamplePackingConfig(dataset=dataset, num_packing_bins=8)
    config.dataloader.collator = _multimodal_collator_config(dataset)
    return config


# ---------------------------------------------------------------------------
# The MoE arms: 35B-A3B, 122B-A10B, 397B-A17B. Same two reasons the 9B needed a
# function here -- their dataloader is `MM_DATASETS["cc12m"]`, a hub stream that
# cannot start on a compute node, and `seq_len` is not a CLI field. Nothing else
# is touched, so each arm keeps torchtitan's own recipe for its size.
#
# Assets are in place: ./assets/hf/Qwen3.5-{35B-A3B,122B-A10B,397B-A17B},
# tokenizer + config only (meta-device init needs no safetensors), which is the
# path each stock config already hardcodes.
#
# Parallelism is deliberately NOT overridden here. The stock configs bake in a
# mesh -- 35B ep=4/tp=1, 122B ep=8/tp=4, 397B ep=16/tp=8 -- and every one of
# those is a CLI flag. Set it on the submit line so the mesh that produced a
# number is visible in the job, not buried in this file. Two JUPITER-specific
# notes when you do:
#   * tp > 4 crosses the node boundary onto InfiniBand (4 GH200 per node,
#     NVLink stops there). The 397B default of tp=8 is an AWS-shaped choice.
#   * ep constraints are in job_config.py: with cp=1 and etp=tp,
#     tp <= ep <= dp_shard and dp_shard % ep == 0.


# "flex" is torchtitan's default and what the stock configs use. It does not
# survive inductor here: the vision tower's flex_attention has a dynamic patch
# count, and lowering its SM_SCALE emits a sympy FloatTrueDiv into the generated
# kernel's inductor_meta with no import for the name --
#   NameError: name 'FloatTrueDiv' is not defined
#   SM_SCALE: FloatTrueDiv(1.0, OpaqueUnaryFn_sqrt(ToFloat((1152//((1152//s56))))))
# a torch 2.14 codegen bug, not a config error. "varlen" is also the backend the
# port implements, so it is the like-for-like arm anyway (TITAN_MIGRATION_v2.md
# S2 addendum used it for the 2B A/B for exactly that reason).
ATTN_BACKEND = os.environ.get("TTBENCH_ATTN_BACKEND", "varlen")

MOE_FLAVORS = {
    "qwen35_35b_a3b": "35B-A3B",
    "qwen35_122b_a10b": "122B-A10B",
    "qwen35_397b_a17b": "397B-A17B",
}


def _moe_plotqa(config_fn, seq_len: int) -> Trainer.Config:
    """One of the stock MoE configs, repointed at packed local plotqa_cot."""
    config = config_fn(seq_len=seq_len)

    if ATTN_BACKEND != "flex":
        from torchtitan.models.qwen3_5 import model_registry

        config.model_spec = model_registry(
            MOE_FLAVORS[config_fn.__name__],
            seq_len=seq_len,
            attn_backend=ATTN_BACKEND,
            moe_comm_backend="standard",
        )
    # max_doc_tokens: their packer raises on a document longer than the row,
    # where our encoder drops it. Matching the drop keeps the workload the same
    # and stops a 0.1% tail from killing an hour-long job.
    dataset = nemotron_dataset(PLOTQA, max_doc_tokens=seq_len)
    config.dataloader.dataset = MMSamplePackingConfig(dataset=dataset, num_packing_bins=8)
    config.dataloader.collator = _multimodal_collator_config(dataset)
    return config


def qwen35_35b_a3b_plotqa(seq_len: int = SEQ_16384) -> Trainer.Config:
    """35B total / 3B active, 256 experts top-8, 40 layers, dim 2048."""
    return _moe_plotqa(qwen35_35b_a3b, seq_len)


def qwen35_122b_a10b_plotqa(seq_len: int = SEQ_16384) -> Trainer.Config:
    """122B total / 10B active."""
    return _moe_plotqa(qwen35_122b_a10b, seq_len)


def qwen35_397b_a17b_plotqa(seq_len: int = SEQ_16384) -> Trainer.Config:
    """397B total / 17B active."""
    return _moe_plotqa(qwen35_397b_a17b, seq_len)
