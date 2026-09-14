from dataclasses import dataclass, field
from enum import Enum, auto

class ModelType(Enum):
    Qwen3_5 = auto()
    Qwen3_vl = auto()
    Qwen3_text = auto()

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
    topk_interval: int = 50
    """Steps between per-rank performance gathers. `_gather_perf` is an
    `all_gather` over WORLD, so at 512 ranks doing it every step puts a
    world-scale collective on the critical path of every step -- measured as a
    tail that grew 0.665 -> 0.706 s across the 4/8/16-node sweep while
    everything else was held constant. The cross-rank spread does not change
    step to step, so sampling it is not a loss of information. Values <= 1 mean
    every step."""

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

    # directory to load the resume checkpoint from. "NULL" (default) -> load from
    # output_dir. set this to resume a run whose checkpoints live elsewhere while
    # writing new checkpoints into output_dir.
    load_dir: str = "NULL"

    # which checkpoint step to resume from when resume_checkpoint is set.
    # 0 (default) -> resume from the latest checkpoint in the load dir.
    start_step: int = 0

    # "will checkpoint each `save_steps`"
    save_steps: int = 1000

    # execute with mixed precision
    bf16_compute: bool = True

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
    skip_nonfinite_grads: bool = True
    """
    Zero the gradients for any step whose global grad norm is not finite,
    instead of letting it reach the weights.

    Without this a single `nan` gradient ends the run: it is all-reduced to
    every rank, `clip_grads_with_norm_` scales everything by `nan`, and the
    optimizer writes `nan` into the parameters and the moments. Every measured
    run above 16 nodes died this way.

    Needs `max_grad_norm > 0`, which is where the global norm comes from. Turn
    it off only to reproduce the failure on purpose.
    """

    max_grad_norm: float = 1.0

    # SCHEDULER -----
    # "wsd" or "cosine"
    scheduler_type: str = "wsd"

    # the run will end
    # it defines the lenght of the scheduler
    total_steps: int = 1_000
    warmup_steps: int = 50

    # length of the final decay ("cooldown") phase, only for WSD.
    # give it either as a raw number of steps (wsd_decay_steps > 0 wins) or as a
    # fraction of total_steps (wsd_decay_ratio). exactly one is used per run.
    wsd_decay_steps: int = 0
    wsd_decay_ratio: float = 0.1
    """
    Use `0.0` to disable the decay.
    """

    # percentage of minumum lr to decay, only for COSINE
    min_lr_ratio: float = 0.1
    # ---------------

    data_parallel: str = "ddp" # fsdp, ddp
    tp_size: int = 1 # 1 means disabled

    reshard_after_forward: str = "never"
    """
    FSDP only. "never" keeps the all-gathered parameters resident for the whole
    step -- fastest, but each rank pays the full model size no matter how large
    `dp` is. "always" frees them after the forward and re-gathers in the
    backward: one extra all-gather per block, and the parameter memory drops by
    roughly `1 - 1/dp`.

    At 9B on 16 nodes "never" OOM'd on the third step with 75.23 GiB allocated.
    Use "always" whenever the model is large relative to the GPU.
    """
    """
    Use `fsdp` when you want to decrease usage to increase seq_len/batch_size.
    """

    loss_chunk_mb: int = 0
    """
    Cap the fp32 working set inside the cross-entropy, in MiB. 0 (default) runs
    `F.cross_entropy` over the whole packed row at once.
    """

    adamw_impl: str = "torchao"
    """
    Which AdamW implementation to use: "torchao", "foreach", "fused", "forloop",
    "fp8", "8bit" or "4bit".

    "torchao" is torchao's unquantized `_AdamW`. It is the default because it is
    the only implementation that honours `adamw_stochastic_round`, which is what
    makes `master_dtype = "bfloat16"` safe. Its step is a per-parameter
    `torch.compile(single_param_adam)` rather than a `_foreach_*` batch, so it
    trades a few hundred kernel launches per step for halving the optimizer
    state. Switch to "foreach" (and accept round-to-nearest) if that shows up.

    `fused` is **not compatible with `tp_size > 1`**: TP leaves some parameters
    as DTensors and some as plain tensors, and the fused kernel does not support them

    "fp8" is torchao's `AdamWFp8`,  quantizes only the two AdamW moments,

    "8bit" and "4bit" are torchao's `AdamW8bit` / `AdamW4bit`
    """

    adamw_stochastic_round: bool = True
    """
    Round the parameter update stochastically instead of to nearest. Only has
    an effect on bf16 parameters (`master_dtype = "bfloat16"`) and only with
    the torchao implementations; with any other `adamw_impl` it is ignored with
    a warning.
    """

    master_dtype: str = "bfloat16"
    """
    Dtype of the master weights the optimizer updates: "float32" or "bfloat16".
    Storage only, and parameters only, see `bf16_compute` also.

    "bfloat16" is the default and needs `adamw_stochastic_round` to stay true,
    which needs a torchao `adamw_impl`. It takes the optimizer's per-parameter
    footprint from 16 bytes (4 param + 4 grad + 8 moments) to 8, i.e. 35.1 -> 17.5
    GiB per GPU for 9B at tp=4, dp=1. FSDP keeps this: `MixedPrecisionPolicy`
    reduces gradients in fp32 but casts the sharded gradient back to the parameter
    dtype (`_fsdp_collectives.py`, `_to_dtype_if_needed(reduce_output, orig_dtype)`),
    so the fp32 buffer is transient, per bucket.
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

    native_kernels: bool = False
    """
    Qwen3.5 only. Call the fla / causal_conv1d entry points directly inside
    `torch.compiler.disable`, so each keeps the hand-written backward its own
    autograd.Function ships. The default instead routes through
    `torch.library.custom_op` wrappers whose backward re-runs the forward under
    `enable_grad` and differentiates through it -- and 24 of the 9B's 32 layers
    are linear attention, so that is a second gated-delta-rule forward on three
    quarters of the model, every step.

    Same math. The tradeoff is one graph break per kernel call, which is what the
    code cost before the custom ops existed. Which side wins is a measurement.
    """

    compile_vision: str = "dynamic"
    """
    How to compile the vision blocks, independently of the decoder blocks.

    "dynamic" -- compile with `dynamic=True`. The right default: the patch count
    is the leading dimension and it changes almost every step (544 to 19132 in a
    single 40-step run), so a static trace recompiles until it evicts.
    "static"  -- follow `compile_dynamic`, the old behaviour.
    "off"     -- leave the tower eager. Worth measuring: the blocks are ~28% of
                 the step, but a symbolic trace of them may be worth less than
                 an eager one, and it removes the compile-time cost entirely.
    """

    dynamo_recompile_limit: int = 0
    """
    `torch._dynamo.config.recompile_limit`, per code object. 0 leaves torch's
    default of 8. Raise it only as a safety net: a run that needs a high limit
    is usually specialising on something that should have been marked dynamic,
    and `TORCH_TRACE` + `tlparse` will say what.
    """

    compile_dynamic: bool = False
    """
    `dynamic=` for every `torch.compile` call. False (static shapes) is the
    default because `dynamic=True` under TP builds a chain of ~1274 dependent
    SymInt proxies and dies with `RecursionError` in `proxy_tensor.py`; a plain
    SwiGLU MLP under Colwise/RowwiseParallel reproduces it. Static shapes mean a
    recompile whenever the packed length changes, which the packer avoids.
    """

    compile_gdn: str = "auto"
    """
    Whether to compile the linear-attention (GatedDeltaNet) blocks:
    "auto" -> only when `data_parallel = 'fsdp'`, "on", "off".

    With TP alone and no FSDP they fail in the DTensor backward with
    `AttributeError: 'Tensor' object has no attribute '_local_tensor'`. Adding
    FSDP makes the same layers compile and run, hence "auto".
    """

    compile_block_mode: str = "default"
    """
    `torch.compile` mode for the decoder and vision blocks. "default",
    "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune".
    """

    compile_head_mode: str = "max-autotune-no-cudagraphs"
    """
    `torch.compile` mode for the three separately-compiled modules
    (`language_model.norm`, `lm_head`, `visual.merger`), or "off" to leave them
    eager. Autotuning `lm_head` means benchmarking every Triton candidate for a
    [T, 4096] x [4096, 248320] GEMM at startup; "default" uses cuBLAS, "off" is
    what the pre-regression code did (lm_head sat outside the compiled module).
    """

    grad_reduce_dtype: str = "float32"
    """
    `MixedPrecisionPolicy.reduce_dtype` -- the dtype FSDP reduce-scatters
    gradients in. "float32" halves the rounding error of the dp reduction at twice
    the wire bytes; "bfloat16" halves the bytes. The *stored* sharded gradient is
    cast back to the parameter dtype either way, so this costs no standing memory.
    Only applies with `data_parallel = 'fsdp'` and `bf16_compute = true`.
    """

    log_graph_code: bool = False
    """
    `torch._logging.set_logs(graph_code=True)` -- dump every traced FX graph's
    source, on every rank. 5 MB per rank at 16 nodes, 42 MB at 256, written during
    the dynamo tracing that already dominates startup.
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

    restore_dataloader_state: bool = True
    """
    one-shot escape hatch for resume. when false, a resumed run skips restoring the
    energon dataloader state (the data stream starts from scratch) but still saves
    its own dataloader state on subsequent checkpoints. use it when the saved state
    is structurally incompatible with the new run, e.g. resuming a `repeat=false`
    checkpoint with `repeat=true` (which inserts a `RepeatDataset` node and makes
    the positional state-tree restore fail). only consulted when
    `save_dataloader_state` is true.
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
