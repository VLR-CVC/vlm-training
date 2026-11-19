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

### Engram (n-gram memory)

Optional, Qwen3.5 only, off unless `model.engram = true` (flags:
[USAGE.md](USAGE.md#engram-n-gram-memory)). After each decoder layer in
`engram_layers`, the hidden state gets a gated read from a hashed n-gram table:

```
e_t  = concat over (order n, head k) of  table[ H_{n,k}(x_{t-n+1..t}) ]
g_t  = < RMSNorm(h_t), RMSNorm(W_K e_t) > / sqrt(dim)
h'_t = h_t + sigmoid(sign(g_t) sqrt|g_t|) * W_V e_t        (W_V zero-initialised)
```

| Piece | File | Notes |
|---|---|---|
| hashing | `models/common/engram_hash.py` | integer only, no torch modules: shared verbatim with the HF export |
| table + gate | `models/common/engram.py` | `EngramTable` is a vocab-parallel `Embedding` |
| Qwen3.5 configs | `models/qwen3_5/engram.py` | init rules, `EngramArgs`, EOS as fill, vision ids blocked |
| wiring | `models/qwen3_5/model.py` | ids in `preprocess_inputs`, injection in both layer loops |

- **Addressing.** Per position, the token ids of the suffix n-grams are mixed as
  `(x_t·m_0) xor (x_{t-1}·m_1) xor …` with fixed odd 64-bit multipliers
  (SplitMix64 from `engram_seed`). Each (order, head) pair reduces that modulo its
  own prime and adds its offset into **one packed table**. Every head and every
  Engram layer has a distinct prime.
- **Ids are inputs.** They depend on token ids only, so `preprocess_inputs` builds
  them once (`engram_ids`, `engram_mask`) from the 1-D document `positions`,
  before those are swapped for MRoPE positions. They are SPMD-annotated like
  `input`, which also makes them available for prefetching in the future.
- **Boundaries.** Token `t-k` enters `t`'s n-grams only if `positions[t] >= k`,
  so nothing crosses a packed document. Look-back also stops at image/video
  placeholders. A missing slot is filled with the EOS id, so a document's first
  tokens read "(EOS, x)" rows. Placeholders themselves get no memory: their
  vectors are masked to zero.
- **Init.** `W_V = 0`, so a model loaded from a snapshot is exactly the snapshot
  until the first update. `load_hf` cannot fill `engram.*` (local-only), so the
  trainer calls `init_engram` after it.
- **TP.** The table is row-sharded and vocab-parallel like `tok_embeddings`:
  replicated ids in, a partial sum out, reduced (or reduce-scattered under SP)
  straight into the residual layout. The rest is token-local, so the
  projections are laid out like the norms (`I` without SP, `R` with it).
- **FSDP.** One unit per Engram module. Without it the table would sit in the
  root unit and stay gathered for the whole step. With
  `engram_sharding = "rows"` the table leaves FSDP instead (next section).
- **Not ported** from the references (DeepSeek Engram / V4.1, Qwen-style "PLE"
  in transformers): tokenizer canonicalisation, the short causal convolution,
  multiple residual streams, host-offloaded tables.

### Engram training at scale

Measured in T20 / EXP-015 (n-gram hub) with `train/bench.py`,
`configs/bench/engram_0_8b.toml`, FineWeb-Edu dev shard, seq_len 4096, L40S 46 GB
(cudahpc51, PCIe, `NCCL_P2P_DISABLE=1`), 40 steps, the first 10 left out,
**tok/s from the median step** (a data-loader refill stalls one step per run).
Frozen-backbone MFU uses the 6N convention and overstates the real FLOPs.

**Where the time goes with a dense FSDP table** (`engram_sharding = "fsdp"`, the
default). Engram's own kernels -- hashing, gather, scatter, the two projections
and the gate -- cost < 1% of a step at any size. Its cost is the table as a
dense FSDP parameter: a bf16 all-gather of the *whole* table every
micro-batch, a full-size bf16 gradient, an fp32 reduce-scatter, and AdamW over
every row. None of it shrinks with more GPUs. Profile (2×A40, 0.8B frozen, 1M
rows/head, 16k tokens/rank): 35% of the step the GPU idles on the all-gather
(487 ms) and reduce-scatter (322 ms), and AdamW takes 120 ms.

**Row-sharded table** (`engram_sharding = "rows"`,
`models/common/engram_sharded.py`, `train/parallel/engram.py`). The table is a
`Shard(0)` DTensor over every rank, outside FSDP (`ignored_params`); each
micro-batch deduplicates its ids, sends them to their owners (all-to-all; one
host sync for the split sizes), gets the rows back, and in backward the owners
sum the row gradients into their shard. Traffic is O(unique ids × row width)
-- 5-8 ms per step in the profile -- and per-rank memory is `16 B × params /
world`. Loss and gradients match the dense table (parity test, 500-step curve
within 1e-4); DCP checkpoints are interchangeable between the two modes.
`engram_table_optim = "lazy_adamw"` additionally updates only the rows the step
read (opt-in: untouched rows keep their moments, which changes the dynamics).

0.8B frozen, 24576 tokens/rank, table size sweep (tok/s, peak reserved GiB):

| rows/head (table params) | 4 GPUs dense FSDP | 4 GPUs rows | 4 GPUs rows + lazy |
|---|---|---|---|
| no Engram | 90,033 (32.0) | | |
| 262k (0.27B) | 83,524 (34.1) | 87,109 (33.4) | 87,758 (33.4) |
| 1M (1.07B) | 69,849 (40.0) | 84,710 (36.6) | 86,960 (36.6) |
| 2M (2.15B) | 26,157 (43.6, thrashing) | 82,122 (38.7) | |
| 4M (4.29B) | OOM | 77,022 (43.5) | 79,503 (43.5) |

Engram overhead against no Engram at equal batch, 4 GPUs (tok/s, change):

| model, backbone, tokens/rank | no Engram | 262k dense / rows | 1M dense / rows |
|---|---|---|---|
| 0.8B frozen, 24576 | 90,033 | −7% / −3% | −22% / −6% (lazy −3%) |
| 0.8B full FT, 12288 | 51,403 | −6% / −2% | −24% / −5% |
| 2B frozen, 12288 | 47,706 | −13% / −4% | −38% / −9% (lazy −5%) |
| 2B full FT, 8192 | 23,607 | −10% / −3% | −61% / −7% |

Weak scaling 1 → 2 → 4 GPUs (0.8B frozen, 262k, tok/s per GPU): no Engram
24.7k / 22.8k / 22.5k; dense 23.1k / 20.7k / 20.9k; rows 23.4k / 21.7k / 21.8k
(93%). With 1M rows/head dense FSDP gets 8.2k / 17.5k per GPU on 2 / 4 GPUs
(memory-bound; OOM on one) against rows 20.4k / 21.2k.

Largest table that trains per 46 GB card, 0.8B frozen at 24576 tokens/rank:
1 GPU 262k rows/head both ways (rows does not help on one GPU: the shard is the
whole table); 2 GPUs 1M dense (at the memory ceiling, 2.5× slower) vs 2M rows
(2.15B params); 4 GPUs 2M dense (thrashing) vs 4M rows (4.29B params). 2B frozen,
4 GPUs: 1M dense vs 2M rows (4.29B params).

**Multi-node.** Shard the table over *every* rank of every node (that is what
`rows` does): a replica per node would need a dense cross-node all-reduce of the
table gradient each step, while the all-to-all moves only the rows a
micro-batch reads, independent of the table size. A local 2-"node" run
(`--nnodes 2`, c10d rendezvous on localhost, sockets) matches single-node 4 GPUs
step for step within run-to-run noise (≤ 4e-4). Leonardo:
`configs/leonardo/qwen3_5_2b_engram.toml` with `scripts/multinode_leo.sh`.

**Leonardo launch checklist** (Booster, 4× A100 64 GB per node; untested
there -- prepared and checked locally).

1. On Leonardo, a checkout with this branch; `scripts/leo_env.sh` as on
   `hpc/leonardo`. In `configs/leonardo/qwen3_5_2b_engram.toml` set `output_dir`
   to your `$SCRATCH/checkpoints/...` and `data_path` to an energon-prepared
   raw-text FineWeb-Edu copy on `$FAST` (the config's path is a placeholder).
2. Throughput first (no checkpoint, one `BENCH_RESULT` json line at the end):
   ```
   CONFIG_FILE=configs/leonardo/qwen3_5_2b_engram.toml TRAIN_MODULE=train.bench \
     TRAIN_ARGS="--training.total-steps 60" sbatch --nodes=2 scripts/multinode_leo.sh
   ```
   then the real job: `CONFIG_FILE=configs/leonardo/qwen3_5_2b_engram.toml sbatch --nodes=2 scripts/multinode_leo.sh`.
3. Expect, from the 4×L40S numbers (2B full FT + 1M rows/head, rows: 5.5k
   tok/s/GPU at 8192 tokens, 7% below no Engram): **~20-28k tok/s per node**
   (5-7k per A100), i.e. 40-55k for 2 nodes if the inter-node FSDP traffic of
   the backbone hides behind compute. Memory per A100 at 16384 tokens/rank on
   8 ranks: ~45-52 GiB allocated (table 2.15B × 16 B / 8 = 4.3 GB, backbone
   state 4 GB, the rest activations). Above ~58 GiB reserved, drop to 12288.
4. In the first 20 steps:
   - the startup line `Engram table row-sharded over 8 ranks: ~2.1M of ~16.8M
     rows x 128 per rank`;
   - step 1: `engram: gate ~0.50 +- 0.20`, `table gnorm 0` for the first
     steps (W_V = 0, so step 1 is the base model), non-zero from step ~3;
   - tok/s flat from step ~5 (compile done); no `recompile` warnings;
   - `mem` well below 60 GiB and stable (growing = leak, flat near 63 = thrash);
   - no NCCL timeouts; the all-to-all is small, so a hang points at
     rendezvous/IB setup, not at Engram.
5. Then a resume from the step-100 checkpoint (`--training.resume-checkpoint`)
   to confirm DCP on Lustre with the sharded table (checked locally: save at
   step 20 on 2 ranks, resume, continue).

**Host-memory tables** (`training.engram_table_device = "host"`,
`train/engram_host.py`). This is what DeepSeek do for inference (arXiv
2601.07372 sec. 2.5: a 100B table in host DRAM, < 3% overhead), extended to
training. Each rank's row shard and both AdamW moments (12 B/param) live in
host RAM; the GPU keeps a row cache.
- **Prefetch.** The row ids depend only on the token ids, so the data
  iterator is wrapped to run `engram_host_lookahead` micro-batches ahead. A
  single FIFO worker thread then, for each new batch:
  1. hashes it on the CPU;
  2. all-to-alls the unique ids to their owners over a gloo group of its own;
  3. gathers the rows from host RAM into pinned buffers;
  4. copies them to the GPU on a side stream.
- **Update.** The forward routes ids exactly as `rows` does and reads the
  cache. After the backward, the touched rows' gradients join the global clip
  norm. Lazy AdamW then updates them in the cache, and they are copied back to
  host memory by the same FIFO worker.
- **Consistency.** Every host read queues behind every earlier write-back. A
  prefetch that predates a step's write-back is merged after that step's
  update, with the cached copy winning. Rows are evicted only at a step
  boundary, keeping those of the batches still ahead. The result is the same
  lazy AdamW as on the GPU: `models/tests/test_engram_host.py` checks it
  against a dense reference, including clip, accumulation, misses and
  eviction.
- **Fallback.** A row that is not cached (lookahead < accumulation, eval,
  precompile) is fetched synchronously: correct, slower, and counted in the
  closing log line.
- **Checkpoints.** The layout is unchanged (`Shard(0)` DTensor on a CPU mesh,
  moments under the usual optimizer keys), so checkpoints move between `fsdp`,
  `rows` and `host` (`models/tests/test_engram_rows_dcp.py`). The batches the
  lookahead pulled early are saved with the dataloader state.
- **Measured** (0.8B frozen, 24k tokens/rank, 2 GPUs, 40 steps, 0 misses throughout):

  | table | GPUs | `rows` + `lazy_adamw` | `host` |
  |---|---|---|---|
  | 1M rows/head (1.07B params) | 2× RTX PRO 6000 | 73.5k tok/s, 36.7 GiB | 73.3k, 31.6 GiB |
  | 1M rows/head | 2× L40S | 43.5k, 36.7 GiB | 42.0k, 31.9 GiB |
  | 8M rows/head (8.6B params, 52 GB host per rank) | 2× L40S | OOM | 42.5k, 31.7 GiB |

  No Engram on 2× L40S: 45.5k. Losses vs `rows`: step 1 identical; mean
  |diff| 0.0003–0.0004 over 40 steps (summation order; batch noise 0.086).
- **Host-side traps, all handled in `train/engram_host.py`:**
  - **4 KB pages.** Every random row is a TLB miss. The shards are allocated
    on transparent huge pages (`host_zeros`, `madvise`): scattering 250k rows
    into 3 × 17 GB takes 8 ms instead of 96 ms. Where THP is "never" this
    falls back to normal pages.
  - **First touch.** Untouched moment buffers fault in on the first
    write-backs: ~13 s per rank, spread over the first steps of the 8M run.
    They are zeroed (faulted in) at allocation instead.
  - **Deterministic mode.** `training.deterministic` (on by default; off in the
    bench config) sends every CPU row scatter to a single-threaded sorted path:
    40 ms instead of 3 ms per 250k × 64 rows. The write-back scatters disjoint
    row chunks on 8 threads (exact, still deterministic).

  The 8M run spends 0.03 s per step in the optimizer hook (0.29 s before these
  fixes) and runs at 42.3k tok/s. The closing log line breaks the host time
  down per phase, as wall and CPU ms per step.
- **Resume.** Through `train.train_qwen`, a run that saves at step 10 and resumes
  matches the continuous run on steps 11–20 within 0.0015 loss, the same as the
  GPU `rows` baseline (0.0016, run-to-run noise). This needs
  `data.save_dataloader_state`. The lookahead's pending batches are saved per
  rank next to it.

**Not done.** Sparse table gradients for `rows` (the shard gradient is still
dense, 4 B/param); TP with `rows`; a GPU-resident hot-row cache across steps
(DeepSeek's Zipfian tiering) for `host`.

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

Degrees come from the trainer (`train/train_qwen.py:95`): `tp_size` and
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

`test_tp_parity.py` is the one that caught a real bug: norm weights need
`Replicate` at TP only when SP makes each rank's gradient partial. Without SP
every rank already holds the full gradient, and declaring `R` made
`attention_norm.weight` come out exactly 2.000× the TP=1 value. The deviation
from upstream is documented at `models/qwen3_5/sharding.py:87`.
