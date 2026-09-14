from dataclasses import dataclass, field
from enum import Enum, auto

class ModelType(Enum):
    Qwen3_5 = auto()
    Qwen3_vl = auto()
    Qwen3_text = auto()
    Qwen4 = auto()

@dataclass
class Model:
    # this defines the CLASS to initilize the model
    model_name: str = "NULL"
    """
    Supported:
    - Qwen3-VL
    - Qwen3.5
    """

    # freeze model parts, its used by `utils.set_model`
    train_llm: bool = True
    train_mlp: bool = True
    train_vit: bool = False

@dataclass
class Wandb:
    run_name: str = "default"
    project_name: str = "test_151_qwen_vl"
    entity_name: str = "bsc_runs"

    # per-rank Top-K performance logging
    log_topk: bool = True
    top_k: int = 4

@dataclass
class Training:

    # ALWAYS CHANGE
    model_dir: str = "NULL"
    """
    This defines the model to be used. We perform `.from_pretrained`
    from this directory. The `AutoProcessor` is also defined with this.
    """

    # where to checkpoint
    output_dir: str = "checkpoints"

    # whether or not to load the text model
    load_text_model: bool = False
    text_model_dir: str = "NULL"

    # whether or not to load a pre-trained vision encoder (e.g. SigLIP2)
    load_vision_model: bool = False
    vision_model_dir: str = "NULL"

    # whether to resume from previous checkpoints or not
    resume_checkpoint: bool = False

    # "will checkpoint each `save_steps`"
    save_steps: int = 1000

    # execute with mixed precision
    bfloat16: bool = True

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    # init of the projecter and deepstack layers
    random_init: bool = False

    # gradient accumulation
    tpi_multiplier: float = 1.0

    # more training args
    eps: float =  1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # SCHEDULER -----
    # "wsd" or "cosine"
    scheduler_type: str = "wsd"

    # the run will end
    # it defines the lenght of the scheduler
    total_steps: int = 1_000
    warmup_steps: int = 50

    # percentage of final decay steps, only for WSD
    wsd_decay_ratio: float = 0.1

    # percentage of minumum lr to decay, only for COSINE
    min_lr_ratio: float = 0.1
    # ---------------

    data_parallel: str = "ddp" # fsdp, ddp
    tp_size: int = 1 # 1 means disabled
    """
    Use `fsdp` when you want to decrease usage to increase seq_len/batch_size.
    """

    pp_size: int = 1
    """
    Pipeline parallelism. 1 disables it. `world_size` must be divisible by
    `pp_size * tp_size`; whatever is left over becomes the data-parallel dim.

    Unlike TP and FSDP, PP splits the model *before* it reaches the GPU: each
    rank drops the layers it does not own while the model is still on CPU, so
    the peak of `train_qwen.py`'s build-then-upcast path scales with the stage,
    not the whole model. That is the axis that makes models too big to
    materialize on one card reachable at all.
    """

    pp_num_layers_first: int = 0
    pp_num_layers_last: int = 0
    """
    Decoder layers pinned to the first and last stages. 0 means "even split".
    Rank 0 also carries the vision tower and `embed_tokens`, and the last rank
    the hyper-connection mixer and `lm_head`, so those stages are heavier than
    their layer count suggests -- give them fewer layers when the pipeline is
    imbalanced.
    """

    pp_schedule: str = "gpipe"
    """
    "gpipe" or "1f1b". Single-stage-per-rank schedules only.
    """

    pp_microbatches: int = 1
    """
    Microbatches per optimizer step. 1F1B needs >= pp_size to pipeline at all.
    The dataloader emits one packed (1, total) row per step, so anything above
    1 currently tiles the *same* row -- useful to benchmark the schedule, not
    to train. See `_train_step_pp`.
    """

    adamw_impl: str = "foreach"
    """
    Which AdamW implementation to use: "foreach", "fused", "forloop", "fp8",
    "8bit" or "4bit".

    Not just a speed knob -- it decides how big a model fits. `foreach` batches
    the update through `torch._foreach_*`, and those allocate temporaries the
    size of the whole parameter set: a 21B model over 4 pipeline stages dies
    inside `torch._foreach_sqrt` *after* both AdamW moments have already been
    allocated successfully. `fused` runs the same arithmetic in one kernel with
    no such temporary, and is what makes the largest PP runs fit at all.
    `forloop` allocates least and is by far the slowest.

    `fused` is **not compatible with `tp_size > 1`**: TP leaves some parameters
    as DTensors and some as plain tensors, and `aten._fused_adamw_` rejects the
    mix ("got mixed torch.Tensor and DTensor"). That is why the default is
    `foreach` even though `fused` is both faster and smaller -- use it on the
    pure-PP configs, where every parameter is a plain tensor. See
    `configs/cvc/qwen4/pp4.toml`.

    "fp8" is torchao's `AdamWFp8`, which is a different axis: it keeps the fp32
    master weights and quantizes only the two AdamW moments, to fp8 with a
    scale per 256-element block. That takes the per-parameter cost from 16
    bytes (4 param + 4 grad + 4 + 4) to 10 (4 + 4 + 1 + 1), which is what
    raises the 4-GPU ceiling past 25.5B parameters, and is what makes the 27B
    model run on this box at all.

    "8bit" and "4bit" are torchao's `AdamW8bit` / `AdamW4bit`, the same idea at
    other widths. All three keep fp32 master weights; see `master_dtype` for
    the other axis.
    """

    adamw_stochastic_round: bool = False
    """
    Round the parameter update stochastically instead of to nearest. Only has
    an effect on bf16 parameters (`master_dtype = "bfloat16"`) and only with
    the torchao implementations.

    This is the standard fix for the thing that makes bf16 master weights fail:
    bf16 carries 8 mantissa bits, so an update smaller than about 2^-9 of the
    weight it is applied to rounds away to nothing, and round-to-nearest makes
    that loss systematic -- the same small update is discarded every step
    forever. Stochastic rounding keeps it unbiased in expectation instead.
    """

    master_dtype: str = "float32"
    """
    Dtype of the master weights the optimizer updates: "float32" or "bfloat16".

    Compute is bf16 either way (`torch.autocast`, plus FSDP's
    `MixedPrecisionPolicy`); this is only about the copy the optimizer owns.
    fp32 master costs 4 bytes per parameter for the weights and another 4 for
    the gradients; bf16 halves both. Pair it with `adamw_stochastic_round`.
    """

    float8: bool = False
    """
    Swap the model's `nn.Linear` layers for torchao's `Float8Linear`, so their
    GEMMs run in fp8 with dynamic scaling. Weights, gradients and the optimizer
    stay high precision -- only the matmul operands are quantized.

    Needs `compile = true` to be worth anything: the quantize/scale ops are
    separate kernels in eager, and on a 4x4096 linear stack they cost more than
    the fp8 GEMM saves (measured on SM120: bf16 12.2 ms, eager fp8 22.8 ms,
    compiled fp8 7.7 ms). The MoE experts are not `nn.Linear` -- they are 3D
    parameters behind `torch._grouped_mm` -- so they keep running in bf16; see
    `apply_float8`.

    Needs SM89 or newer.
    """

    float8_recipe: str = "tensorwise"
    """
    "tensorwise" (one scale per tensor), "rowwise" (one per output row, more
    accurate, more scaling work) or "rowwise_with_gw_hp".

    Pick by hardware. On SM120 (RTX PRO 6000) rowwise is *slower than bf16*
    compiled (14.3 ms vs 12.2 ms on the microbenchmark above) because those
    cards have no rowwise-scaled tensor-core path; on SM90/SM100 it is the
    usual default. Tensorwise is the fast recipe here.
    """

    float8_moe: bool = False
    """
    Put the stacked MoE experts in fp8 too. They are 3D parameters behind
    `torch._grouped_mm`, which `float8` cannot reach -- and on a MoE model they
    hold most of the FLOPs, so this is where the win actually is.

    Needs SM90 (H100, GH200) or SM100 (B100/B200, GB200). The gate in
    `torch._scaled_grouped_mm` is an exact set of architectures, not a floor:
    SM120 (RTX PRO 6000) is *not* in it despite being newer than both. Turning
    this on elsewhere raises at startup rather than failing mid-step.

    Independent of `float8`: the two cover disjoint parts of the model and can
    be enabled separately.
    """

    float8_moe_recipe: str = "fp8_rowwise"
    """
    "fp8_rowwise" (SM90 or SM100), "mxfp8" or "mxfp8_wgrad_with_hp" (SM100
    only). Separate from `float8_recipe` because the expert path and the linear
    path do not offer the same choices -- torchao's MoE handler has no
    tensorwise option at all.
    """

    # compiler flag for TP (goes faster)
    async_tp: bool = True

    # torch dynamo compiler
    compile: bool = True
    """
    Always on by default, unless you have an error.
    """

    ac_memory_budget: float = 1.0
    """
    When set, uses ``torch._functorch.config.activation_memory_budget`` instead
    of checkpoint_wrapper-based AC. Requires ``compile = true``.
    Range 0.0–1.0: 0.0 = recompute everything, 1.0 = save everything.
    """

    clear_cache_vram: int = 100
    """
    Each optimizer steps to call `torch.cuda.empty_cache()` to clear GPU VRAM.
    Degrates performance. Set to 0 to disable.
    """

    debug_batch_stats: bool = False

@dataclass
class Data:
    # must be an energon dataset. currently only CrudeWebdatasets are expected
    data_path: str = "NULL"

    shuffle_buffer_size: int = 100
    max_samples_per_sequence: int = 100
    packing_buffer_size: int = 0

    batch_size: int = 4
    """
    this currently determines if we use online datapacking or not. Default = sequence packing (4 batch size).
    given a non-zero integer, the energon task encoder builds the sequences with that number of samples.
    flash attention varlen with cu_seqlens is used either way, with a single sequence batch.

    Dispatch:
        data.text_dataset == True                 -> QwenTextEncoder
        data.text_dataset == False, batch_size>0  -> SingleBatchEncoder
        data.text_dataset == False, batch_size==0 -> PackedBatchEncoder (online datapacking)

    DO NOT forget to define `packing_buffer_size` if using online datapacking.
    """

    text_dataset: bool = False
    """
    when true, uses the text-ony task encoder (QwenTextEncoder)
    when false, dispatch according to everything above
    """

    repeat: bool = False
    """
    passed to energon `get_train_dataset(repeat=...)`. when False the loader
    raises StopIteration once the dataset is exhausted (finite epoch); when True
    it loops the dataset forever (needed for step-based training that runs more
    steps than there are samples).
    """

    save_dataloader_state: bool = True
    """
    when true `energon` saves and loads the dataloader state like with the train state
    """

    seq_len: float = 4096
    """
    maximum sequence lenght used when building the batches. with a large batch size, the sequence may
    exceed this number. tune both parameters when using batch_size.

    you always want to have a fixed sized input into the decoder, as it helps with compilation.
    """

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    wandb: Wandb = field(default_factory=Wandb)

    config: str = '/home/tockier/vlm-training/configs/cvc_config.toml'
