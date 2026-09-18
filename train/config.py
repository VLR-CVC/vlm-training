from dataclasses import dataclass, field

@dataclass
class Model:
    model_name: str = "NULL"
    """
    Run label only. The model is chosen by `model_type` in the `model_config` JSON:
    "qwen3_5" (`models/qwen3_5_tt`) or "qwen3_vl" (`models/qwen3_vl_tt`).
    """

    # freeze model parts, see `utils.set_trainable_parts`
    train_llm: bool = True
    train_mlp: bool = True
    train_vit: bool = False

    # the architecture and module types
    model_config: str = "NULL"
    """
    HF-format `config.json` that defines the architecture (sizes, layer schedule,
    vision tower), e.g. `configs/models/qwen3_5_9b.json`. `training.model_dir`
    then only provides weights (unless `random_init`) and the processor, and its
    own `config.json` is ignored.
    """
    use_model_dir_config: bool = False
    """
    Take the architecture from `training.model_dir`/config.json instead. Mutually
    exclusive with `model_config`.
    """
    attn_backend: str = "varlen"
    """
    Kernel of the decoder's full-attention layers. "varlen" -- flash `varlen_attn`.
    """
    decoder_mask: str = "causal_doc"
    """
    Decoder attention mask. "causal_doc" -- causal inside each packed document,
    nothing across documents (full attention and GatedDeltaNet alike).
    """

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

    chat_template: str = "NULL"
    """
    Path to a jinja chat template that overrides the one in `model_dir`.

    Qwen ships an *inference* template: it keeps `<think>` spans only for turns
    after the last user query and strips them everywhere else, which is right for
    generation and wrong for SFT on reasoning data. On plotqa_cot it cuts the
    median sample from 4677 to 1686 tokens and on clevr_1 from 12770 to 1528 --
    training on roughly a quarter of the reasoning it was given (`PERFORMANCE.md`
    §17.6). Point this at `assets/chat_template_sft.jinja` for any CoT dataset.

    "NULL" keeps whatever `model_dir` ships, and the trainer warns at startup if
    that template is the reasoning-dropping one.
    """

    # where to checkpoint
    output_dir: str = "checkpoints"

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

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    # init of the projecter and deepstack layers
    random_init: bool = False

    # gradient accumulation
    tpi_multiplier: float = 1.0

    # more training args
    eps: float =  1e-8
    deterministic: bool = True
    """
    `torch.use_deterministic_algorithms(True)` plus cuDNN/cuBLAS determinism. It is
    not free: profiled on Qwen3.5-2B (models/qwen3_5_tt, 1 GPU, 8192 x 2) it selects
    the deterministic flash-attention backward (122 vs 33 ms/step), fills every
    `torch.empty` (5,867 fill kernels, 101 ms/step) and sorts inside `index_put` --
    ~210 ms of a 1.65 s step. torchtitan leaves it off by default.
    """
    adam_betas: tuple[float, float] = (0.9, 0.999)
    """
    AdamW (beta1, beta2). torchtitan's `default_adamw` uses (0.9, 0.95). `eps` and
    this are passed to "foreach_sr" and the torch.optim implementations; torchao's
    ignore them.
    """
    weight_decay: float = 0.01

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

    sequence_parallel: bool = True
    """
    With `tp_size > 1`: shard the residual stream over TP
    between blocks (torchtitan's default). Without it torchtitan b21f7d43e
    double-counted `attention_norm` gradients; fixed in `models/qwen3_5_tt/sharding.py`.
    """

    loss_chunks: int = 8
    """
    Split the micro-batch into this many chunks for
    lm_head + cross-entropy (torchtitan's ChunkedLossWrapper), so full [T, V]
    logits never exist. `seq_len` must be divisible by it; 1 disables chunking.
    Measured on Qwen3.5-2B at 8192, one GPU: peak 50.6 -> 29.9 GiB (text row),
    63.5 -> 42.7 GiB (16k-patch image row).
    """

    adamw_impl: str = "torchao"
    """
    Which AdamW implementation to use: "foreach_sr", "torchao", "foreach",
    "fused", "forloop", "fp8", "8bit" or "4bit". `train/utils.py:ADAMW_IMPLS`
    is the authority.

    "foreach_sr" (`train/adamw_sr.py`) is ours and is what production runs: the
    same update as torchao's `_AdamW` including stochastic rounding, but batched
    through `_foreach_*` instead of a per-parameter compiled step. Measured
    1.790 -> 1.267 s per step single-node on the 9B, +41% throughput.

    "torchao" is torchao's unquantized `_AdamW`. It also honours
    `adamw_stochastic_round`, which is what makes `master_dtype = "bfloat16"`
    safe. Its step is a per-parameter
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
    Storage only, and parameters only: compute is bf16 (FSDP `param_dtype`).

    "bfloat16" is the default and needs `adamw_stochastic_round` to stay true,
    which needs a torchao `adamw_impl`. It takes the optimizer's per-parameter
    footprint from 16 bytes (4 param + 4 grad + 8 moments) to 8, i.e. 35.1 -> 17.5
    GiB per GPU for 9B at tp=4, dp=1. FSDP keeps this: `MixedPrecisionPolicy`
    reduces gradients in fp32 but casts the sharded gradient back to the parameter
    dtype (`_fsdp_collectives.py`, `_to_dtype_if_needed(reduce_output, orig_dtype)`),
    so the fp32 buffer is transient, per bucket.
    """

    # torch dynamo compiler
    compile: bool = True
    """
    Always on by default, unless you have an error.
    """

    dynamo_recompile_limit: int = 0
    """
    `torch._dynamo.config.recompile_limit`, per code object. 0 leaves torch's
    default of 8. Raise it only as a safety net: a run that needs a high limit
    is usually specialising on something that should have been marked dynamic,
    and `TORCH_TRACE` + `tlparse` will say what.
    """

    grad_reduce_dtype: str = "float32"
    """
    `MixedPrecisionPolicy.reduce_dtype` -- the dtype FSDP reduce-scatters
    gradients in. "float32" halves the rounding error of the dp reduction at twice
    the wire bytes; "bfloat16" halves the bytes. The *stored* sharded gradient is
    cast back to the parameter dtype either way, so this costs no standing memory.
    Only applies with `data_parallel = 'fsdp'`.
    """

    clear_cache_vram: int = 100
    """
    Each optimizer steps to call `torch.cuda.empty_cache()` to clear GPU VRAM.
    Degrates performance. Set to 0 to disable.
    """

@dataclass
class Data:
    # must be an energon dataset. currently only CrudeWebdatasets are expected
    data_path: str = "NULL"

    shuffle_buffer_size: int = 100
    max_samples_per_sequence: int = 100
    packing_buffer_size: int = 100
    """Samples held for first-fit-decreasing packing into rows."""

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

    seq_len: int = 4096
    """
    Row length: the longest document kept (longer ones are skipped). Every row is
    packed and padded to exactly this many tokens. torchtitan's `seq_len`.
    """

    tokens_per_microbatch: int = 0
    """
    Tokens per micro-batch per DP rank, a multiple of `seq_len`: the micro-batch is
    `tokens_per_microbatch // seq_len` rows, joined into one varlen sequence.
    0 means one row (`seq_len`). torchtitan's `num_tokens_per_microbatch_per_dp_rank`;
    TORCHTITAN_BENCHMARK.md 9.1 is `seq_len = 16384`, `tokens_per_microbatch = 32768`.
    """

    @property
    def microbatch_tokens(self) -> int:
        tokens = self.tokens_per_microbatch or self.seq_len
        if tokens % self.seq_len:
            raise ValueError(f"tokens_per_microbatch {tokens} is not a multiple of seq_len {self.seq_len}")
        return tokens

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    wandb: Wandb = field(default_factory=Wandb)

    config: str = 'configs/local/qwen3_5_2b.toml'
