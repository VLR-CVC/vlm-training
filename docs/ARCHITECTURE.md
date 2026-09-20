# Architecture

How this codebase defines models and how it applies parallelism. High level
only description.

The components under `models/common/` and `models/qwen3_5/` are vendored from
torchtitan `b21f7d43e` and trimmed; the file headers say what was dropped.

---

## How models are defined

### The shape of it

A model is **a config tree that builds a module tree**. Nothing is constructed
by calling `__init__` with hyperparameters; you build a `Config` dataclass and
call `.build()` on it. The config tree mirrors the module tree one-to-one, so
the FQN of a config (`layers.3.attention.wq`) is the FQN of the module it
produces.

Two base classes carry this:

| Class | File | Role |
|---|---|---|
| `Configurable` | `models/common/configurable.py` | nested `Config` dataclass, auto-wired `build()` |
| `Module` | `models/common/module.py:31` | `Configurable` + `nn.Module`, plus init and parallelization |

`Configurable.__init_subclass__` enforces that every nested `Config` is a
`@dataclass(kw_only=True, slots=True)` and points its `_owner` back at the
enclosing class. That is what makes `config.build()` work with no boilerplate
per module.

```python
class OffsetRMSNorm(Module):                    # models/qwen3_5/model.py
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.dim))
```

Note `torch.empty`: **modules never initialize their own parameters.**

### Initialization is data, not code

`Module.Config` carries a `param_init: dict[str, Callable]` mapping parameter
name to initializer. `init_states()` (`models/common/module.py:63`) walks the
tree depth-first, and for each module either applies that dict or falls back to
the module's own `reset_parameters()`. A module with parameters and neither one
raises.

This is why the init rules live in the *config builder*, not in the modules —
`models/qwen3_5/configs.py` holds `_LINEAR_INIT`, `_depth_init(layer_id)` (the
`1/sqrt(2L)` depth scaling), `_a_log_init`, and so on. Changing an init rule is
a change to one dict in one file, and it applies to a meta-device model that has
no storage yet.

Buffers are separate: `_init_self_buffers(buffer_device=...)` is overridden by
the modules that have them (RoPE caches, ViT `inv_freq`) and is re-run after
`to_empty()`, because materialization destroys buffer contents.

### Where a config comes from

**The HF `config.json` is the source of truth for architecture.** There are no
`_2b()` / `_9b()` size functions restating hyperparameters — those were deleted
in favour of reading the checkpoint's own config.

```
config.json ──> qwen35_config_from_hf()  ──> Qwen35Model.Config ──> .build() ──> model
               (models/qwen3_5/configs.py:361)
```

`build_meta()` (`models/qwen3_5/checkpoint.py:21`) is the entry point, and it
dispatches on the JSON's `model_type` field: `qwen3_5` or `qwen3_vl`. It builds
under `torch.device("meta")`, so no rank ever holds a materialized full model.

The three-step lifecycle used by the trainer:

```python
model = build_meta(config_path, seq_len=..., tp=..., enable_sp=...)  # meta, no storage
...                                                                  # parallelize here
materialize(model, device)                                           # to_empty + re-init buffers
load_hf(model, model_dir)        # or model.init_states() for random init
```

`load_hf` goes through `Qwen35StateDictAdapter` and DCP's
`HuggingFaceStorageReader`: our state dict's keys are converted to HF names, the
tensors are filled from the safetensors *in place* (already sharded), then
converted back. A missing key is an error. No second full copy on the host.

### The module inventory

```
models/
  common/          shared, model-agnostic
    module.py            Module base: init_states, parallelize
    configurable.py      Config protocol, build() wiring
    sharding.py          ShardingConfig dataclass
    decoder.py           Decoder base: tok_embeddings, layers, norm, lm_head
    attention.py         BaseAttention, varlen metadata
    nn_modules.py        Linear, Embedding, Conv1d, LayerNorm, ... (Module-ized)
    rope.py              MRoPE
    vision_encoder.py    ViT blocks
    feed_forward.py      FFN builder
    multimodal.py        get_vision_positions, scatter_vision_embeds
    *_sharding.py        TP placement helpers (see section 2)
  qwen3_5/         hybrid GatedDeltaNet + gated full attention, + ViT
  qwen3_vl/        dense GQA decoder + the same ViT + DeepStack
```

`qwen3_vl` is the template for adding a model: it defines only what differs
(`RMSNorm` instead of `OffsetRMSNorm`, no output gate, DeepStack merging) and
imports the vision tower, the init rules and the sharding rules from
`qwen3_5`. `models/qwen3_vl/configs.py:125` is a good short read — its
`apply_parallelism_config` calls Qwen3.5's and then patches the one extra input.

### Inputs

Batches are **packed and varlen**, not padded rectangles. There is no batch
dimension: `input` is `(total_tokens,)` and document boundaries are carried in
`positions` (each document restarts at 0). `get_attention_masks` turns those
restarts into `cu_seqlens` for both the varlen attention kernel and the
GatedDeltaNet.

`preprocess_inputs` (`models/qwen3_5/model.py`) is the boundary: it builds the
masks, picks 3-D `mrope_positions` over 1-D `positions` when the batch is
multimodal, applies the SPMD annotations, and splits out `input`/`labels`.

**Read next:** `models/common/module.py`, then `models/qwen3_5/configs.py`
bottom-up from `qwen35_config_from_hf`.

---

## How parallelism is applied

### Three layers, applied in order

```
parallelize_qwen3_5()            train/parallel/parallelize.py
  1. model.parallelize(dims)     TP   — shards params, wraps forwards
  2. apply_compile(...)          per-block torch.compile(fullgraph=True)
  3. apply_data_parallel(...)    FSDP2 (or replicate) over the DP axes
```

All three run **on the meta model**, before `materialize`. That ordering is not
cosmetic: FSDP must see TP-sharded parameters so it shards what is left, and
compile must wrap the TP-wrapped forward.

Supported: TP, SP, FSDP, HSDP, DDP. **Not supported:** CP (GatedDeltaNet needs
the full sequence), PP, EP (the mesh axes exist,
nothing drives them). PP and EP will be implemented in the future.

### The mesh

`ParallelDims` (`train/parallel/parallel_dims.py:62`) holds the degrees and
`build_mesh()` (`:119`) unflattens the world into several **named views over the
same devices**:

| View | Axes | Used for |
|---|---|---|
| `dataloading` | `pp, batch, cp, tp` | which shard of data a rank reads |
| `loss` | flatten of `batch, cp` | the loss all-reduce |
| `dense` | `pp, dp_replicate, dp_shard, cp, tp` | passed to `fully_shard` |
| `spmd_dense_for_fwdbwd` | `pp, dp, cp, tp` | fwd/bwd typechecking (`dp` folds replicate × shard) |

Degrees come from the trainer (`train/train_qwen.py:117`): `tp_size` and
`dp_shard_size` from the config, `dp_replicate` derived as
`world_size // (tp * shard)`. So `dp_shard_size` is the nob to use either full FSDP, HSDP or DDP.

Size-1 axes are kept alive with a `fake` backend rather than dropped, so code
can name an axis unconditionally without checking whether it exists.

### TP is declarative

There are no `parallelize_module(...)` / `ColwiseParallel()` call sites in the
model code. Instead each `Module.Config` carries a `ShardingConfig`
(`models/common/sharding.py`) with four things:

- `state_shardings` — per-parameter/buffer layout, e.g. `{"weight": Shard(0)}`
- `in_src_shardings` / `in_dst_shardings` — activation layout entering the
  module, and what to redistribute it to
- `out_src_shardings` / `out_dst_shardings` — same for the output
- `local_spmd` — run the body on plain local tensors, skipping typechecking

Redistribution is always an explicit **(source, destination) pair**, because
local SPMD types are erased at runtime and there is nothing to infer from.

`Module.parallelize(parallel_dims)` (`models/common/module.py:210`) walks the
tree and, for every module that has a `ShardingConfig`:

1. shards its own parameters and buffers onto the resolved mesh
   (`_distribute_states`, `:330`), rejecting layouts that would shard unevenly;
2. replaces `self.forward` with
   `redistribute inputs -> forward -> redistribute outputs`.

The collectives are therefore a consequence of the declared layouts. `Shard(0)`
weight with `Shard(-1)` output is column-parallel; `Shard(1)` weight with a
`Partial` output redistributed to `Replicate` is row-parallel with an
all-reduce. `models/common/decoder_sharding.py:122` has both as
`colwise_config()` / `rowwise_config()`.

**Sequence parallelism** is one flag through the same machinery: `enable_sp`
switches the residual-stream layout from replicated to `Shard(0)` over tokens,
which turns the row-parallel all-reduce into a reduce-scatter.

Who fills these in: `apply_parallelism_config(config, tp=, enable_sp=)`
(`models/qwen3_5/configs.py:453`) validates that TP divides every head count,
then `set_qwen35_sharding_config` (`models/qwen3_5/sharding.py:105`) walks the
config tree setting `sharding_config` everywhere. **It runs on the config,
before `build()`.** Skip it and `Module.parallelize` is a no-op.

With `tp=1` the sharding configs are still installed and every redistribution is
over a size-1 axis, no collectives are used.

### FSDP

`apply_data_parallel` (`train/parallel/fsdp.py:85`) wraps in torchtitan's units:

- the whole vision encoder as **one** unit (one all-gather for the tower)
- `tok_embeddings`; `[norm, lm_head]` together (all three when weights are tied)
- every decoder block
- the root

`mode="fsdp"` calls `fully_shard`, `mode="ddp"` calls FSDP2's `replicate`, same mixed-precision policy, so the two share one code path.

Two details worth knowing before touching the loss:

- **Gradient division is off** (`disable_fsdp_gradient_division`). The loss is
  already divided by the *global* valid-token count, so FSDP's default mean over
  DP would divide twice. Gradients reduce as a SUM.
- Because of that, gradient accumulation is exact with the reduction deferred to
  the last micro-batch — which is what `train/step.py` does with
  `set_requires_gradient_sync(False)`.

### Compile

`apply_compile` (`train/parallel/compile.py`) compiles **each decoder block and
each ViT block separately** with `fullgraph=True`. Per-block rather than
whole-model keeps compile time bounded and makes a graph break an error instead
of a silent slowdown.

With `async_tp = true` it also enables Inductor's `_micro_pipeline_tp`, which
decomposes `all_gather → mm` and `mm → reduce_scatter` so the collective
overlaps the matmul. It needs SP to be on, without it, those layers emit an
all-reduce, which the pass cannot decompose. It rewrites collectives, so it is
off by default and the TP parity tests are the gate.

**Recompiles are a distributed problem.** A step stalls whenever *any* rank
recompiles, because every other rank blocks at the next collective. Each rank
only compiles ~9 graphs but reaches them on its own data shard's schedule, so at
32 nodes the union had not settled by step 400. `precompile = true`
(`train/precompile.py`) runs a fixed list of synthetic batches through
forward+backward before the loop so every rank compiles the same shapes at the
same time. Measured at 32 nodes: warmup 416 steps → 1, wall clock 23:01 → 17:40.

### The config knobs

`train/config.py`, all documented inline:

| Knob | Effect |
|---|---|
| `data_parallel` | `"fsdp"` or `"ddp"` |
| `tp_size` | TP degree; 1 disables |
| `dp_shard_size` | FSDP shard degree; the rest becomes `dp_replicate` (HSDP) |
| `sequence_parallel` | shard the residual stream over TP |
| `compile` | per-block `fullgraph` compile |
| `async_tp` | Inductor micro-pipelined TP; needs `compile` + TP + SP |
| `precompile` | synthetic warmup batches before step 1 |
| `reshard_after_forward` | FSDP memory/bandwidth trade |
| `grad_reduce_dtype` | dtype of the FSDP gradient reduce-scatter |

**Read next:** `train/parallel/parallelize.py` (25 lines, the whole order),
then `models/common/module.py:210` for what TP actually does, then
`models/common/decoder_sharding.py` for the placement vocabulary.

---

## Verifying a parallelism change

Parallelism bugs are silent, the loss curve may look fine but the gradients are
wrong by a constant. The parity tests exist for exactly this:

| Test | Checks |
|---|---|
| `models/tests/test_tp_parity.py` | TP=2 vs TP=1, per-parameter gradients |
| `models/tests/test_dp_parity.py` | DP sharding + accumulation vs single rank |
| `models/tests/test_qwen3_5_parity.py` | our model vs `transformers`, needs a snapshot |
| `models/tests/test_qwen3_vl_parity.py` | same for Qwen3-VL |
| `models/tests/test_fsdp_nosync.py` | the deferred-reduction path |

`test_tp_parity.py` is the one that caught a real bug: norm weights need
`Replicate` at TP only when SP makes each rank's gradient partial. Without SP
every rank already holds the full gradient, and declaring `R` made
`attention_norm.weight` come out exactly 2.000× the TP=1 value. The deviation
from upstream is documented at `models/qwen3_5/sharding.py:87`.
