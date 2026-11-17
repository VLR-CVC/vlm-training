# Usage Guide

Configuring and launching a training run. For what the code *is*, read
[ARCHITECTURE.md](ARCHITECTURE.md); for what the throughput numbers mean, read
[METRICS.md](METRICS.md).

## Read this first

**`train/config.py` is the documentation for every config field.** It is a set of
dataclasses with a docstring on each non-obvious field explaining what it does and,
where it was measured, what it cost. This guide covers the shape of a run; that
file covers the fields.

Unknown keys are a hard error at startup — `ConfigManager._dict_to_dataclass`
rejects them. That is deliberate (a typo'd field silently doing nothing is worse),
but it means a renamed field breaks every config still using the old name, after
the allocation is granted. `models/tests/test_configs_parse.py` catches it before
you queue:

```bash
python -m pytest models/tests/test_configs_parse.py
```

## Installation

See [INSTALL.md](../INSTALL.md). Summary: `torch==2.14.0`, `transformers==5.16.1`,
python 3.13, plus `flash-linear-attention` and `causal-conv1d` for Qwen3.5's
linear-attention layers.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

On ARM (JUPITER), the CUDA extensions need workarounds that INSTALL.md spells out.
They must be rebuilt on every torch upgrade.

## Datasets

Datasets are **NVIDIA Energon CrudeWebdatasets**, cooked with Energon's cookers
([docs](https://nvidia.github.io/Megatron-Energon/advanced/crude_datasets.html)).
Crude datasets are what make metadatasets (several sources in one run) easy.
Tokenization happens on the fly, asynchronously; no measured throughput impact.

### Packing

Rows are **packed**, not padded-per-sample. Documents are first-fit-decreasing
packed into rows of exactly `data.seq_len` tokens, joined into one varlen sequence,
and attention runs causally *inside* each document with nothing crossing the
boundaries (`data/energon_dataloader.py:47`).

Two knobs:

| Field | Meaning |
|---|---|
| `data.seq_len` | row length. Documents longer than this are **skipped**, not truncated |
| `data.tokens_per_microbatch` | tokens per micro-batch per DP rank, a multiple of `seq_len`. `0` means one row |

Packing quality degrades when the sample buffer runs low. Raise
`data.packing_buffer_size` if `batch utilisation` in the logs drops.

### DocMaestro

[DocMaestro](https://github.com/VLR-CVC/DocMaestro) task shards (`type_dataset: docmaestro`,
`cooker_docmaestro`) carry no pixels. Each sample's `assets.json` *references* page
renders and figure crops, and each of the 13 tasks has a contract for how many of each
it gets (`data/docmaestro.py: CONTRACTS`, copied from DocMaestro's `training/contract.py`).
Three tasks get no page on purpose. For `caption_prediction` and `figure_selection` the
answer is printed on the page, so they get figure crops only. `doc_reconstruction` gets
no image at all. A sample whose assets break its contract raises `AssetArityError`, and
a missing file raises `MissingAsset`. Neither is skipped: both mean the shards are wrong.

The cooker needs two roots, plus a prompt set:

| subflavor | env var (fallback) | what |
|---|---|---|
| `docmaestro_pages_dir` | `DOCMAESTRO_PAGES_DIR` | page store from DocMaestro's `render_pages.py`: `<doc_id>/pNNN.jpg` |
| `docmaestro_crops_root` | `DOCMAESTRO_CROPS_ROOT` | extraction tree with MinerU's crops: `<doc_id>/[hybrid_auto/]images/` |
| `docmaestro_prompts` | `DOCMAESTRO_PROMPTS` | prompt templates; default `assets/docmaestro_doctasks.yaml`, `fixed` for `DOCMAESTRO_DESC` |
| `docmaestro_prompt_seed` | `DOCMAESTRO_PROMPT_SEED` | template choice seed (default 0) |

```yaml
# metadataset blend entry
- weight: 1
  path: /data/.../docmaestro/train
  subflavors:
    type_dataset: docmaestro
    docmaestro_pages_dir: /data/.../training/pages
    docmaestro_crops_root: /data/.../processed
```

What the model sees:
- **Images.** Images come first, in `assets.json` order (pages, then figures). When a
  sample has several images, each one is preceded by a label, `Page i:` or
  `Figure i:`, which the target's indices refer to. In `figure_ordering`, `i` is the
  real page index.
- **Page numbers.** `page_sorting` pages have their `<page-number>` boxes blacked out
  first; otherwise the task is solved by reading them.
- **Prompt.** The prompt is one of the task's four DocMaestro templates (version
  `doctasks-1`), filled from the sample's input. The template is chosen by
  `sha1(task|key|seed)`, the same rule as DocMaestro's `PromptRegistry.select` with
  `epoch = seed`. The choice is stable across workers and restarts. It does not change
  between epochs, because a stateless cooker has no epoch.
- **Prompt version.** The version used is recorded on `EnergonSample.prompt_version`.
  Re-copy the YAML from DocMaestro when it bumps the version; a test pins the copy.

Shards written before DocMaestro had `assets.json` keep the old behaviour: they get the
page `json.page_idx`, `DOCMAESTRO_DESC[task]` and the raw input.
`models/tests/test_cooker_docmaestro.py` covers each task's contract. Its real-shard
half needs a fixture built by `utils/build_docmaestro_fixture.py`, pointed to by
`DOCMAESTRO_FIXTURE`.

## Model weights

The architecture and the weights are **separate inputs**.

```toml
[model]
model_config = "configs/models/qwen3_vl_2b.json"   # architecture (HF-format config.json)

[training]
model_dir = "/shared/cache/qwen3_vl_2b"            # weights + processor + tokenizer
```

`model_config` is what actually selects the model: its `model_type` field picks
`qwen3_5` or `qwen3_vl`. `model.model_name` is a **run label only** — it does not
choose anything.

Set `model.use_model_dir_config = true` to read the architecture from
`model_dir/config.json` instead. The two are mutually exclusive.

Download weights on a login node — compute nodes have no internet:

```bash
python utils/down.py      # edit the script for repo ids and destinations
```

Loading is direct-to-shard: our state-dict keys are converted to HF names, DCP
fills the tensors from the safetensors in place, then they are converted back. No
rank ever materializes a full model, and no network access is needed at train time.

### Supported architectures

- **Qwen3.5** (`models/qwen3_5`) hybrid: GatedDeltaNet linear attention on 3 of
  every 4 layers, gated full attention on the rest, plus a ViT tower.
- **Qwen3-VL** (`models/qwen3_vl`) dense GQA decoder, same ViT, DeepStack merging.

Anything else raises in `build_meta`. The old HF-backed path is gone.

## Configuration

```
configs/
├── models/     architecture JSONs (HF-format config.json)
├── local/      local / CVC cluster runs
├── mn5/        MareNostrum 5
└── jupiter/    JUPITER
```

### The fields you always set

```toml
[model]
model_name   = "Qwen/Qwen3-VL-2B"                   # label only
model_config = "configs/models/qwen3_vl_2b.json"    # architecture
train_llm = true
train_mlp = true
train_vit = true                                     # freeze switches

[training]
model_dir  = "/shared/cache/qwen3_vl_2b"
output_dir = "/scratch/checkpoints/qwen3_vl_2b"
total_steps = 10000
save_steps  = 1000

[data]
data_path = "/scratch/datasets/plotqa_cot"
seq_len   = 32768

[wandb]
run_name = "qwen3_vl_2b"
project_name = "qwen3_vl_jupiter"
entity_name  = "bsc_runs"
```

Learning rates are per part: `lr_llm`, `lr_mlp`, `lr_vit`.

### Parallelism

```toml
[training]
tp_size           = 2          # tensor parallel; 1 disables
data_parallel     = "fsdp"     # "fsdp" or "ddp"
dp_shard_size     = 0          # 0 = shard over every rank; other = HSDP
sequence_parallel = true       # shard the residual stream over TP
```

`world_size = tp_size × dp_shard_size × dp_replicate`, and `dp_replicate` is
derived. So `dp_shard_size` alone picks the regime:

| `dp_shard_size` | Regime |
|---|---|
| `0` | plain FSDP — parameters sharded over every rank |
| `n > 0` | HSDP — FSDP inside groups of `n`, gradient all-reduce between them |

**`tp_size × dp_shard_size = 4` is the setting that makes multi-node fast on
GH200.** It keeps both sharding collectives inside a node (NVLink) and leaves only
the gradient all-reduce on the interconnect. That mesh is what the 90%
strong-scaling result in the README was measured with.

**Expert parallelism** (MoE models only): `training.ep_size = n` splits the routed
experts over `n` ranks, `num_experts / n` each, and exchanges tokens with an
all-to-all. `n` must divide `num_experts` and `dp_shard × tp`; what is left of
`dp_shard × tp / n` shards the expert weights with FSDP, so `ep_size = dp_shard`
means the experts are sharded by EP alone. Dense parameters keep the dense mesh.
The 35B-A3B runs with `tp_size = 2, ep_size = 4`
(`configs/jupiter/qwen3_5_35b_a3b.toml`).

Other MoE / MTP options are in `train/config.py`: `model.moe_load_balance_coeff`
(0 turns load balancing off), `model.num_mtp_layers` and `model.mtp_scale`
(Multi-Token Prediction, needs `loss_chunks > 1`).

Not supported: context parallel (raises — GatedDeltaNet needs the full sequence),
pipeline parallel.

### Compile

```toml
[training]
compile    = true      # per-block fullgraph compile; on by default
precompile = true      # synthetic warmup batches before step 1
async_tp   = false     # Inductor micro-pipelined TP
```

**`precompile` is the one to know about at scale.** A step stalls whenever *any*
rank recompiles, because every other rank blocks at the next collective. Each rank
compiles ~9 graphs but reaches them on its own data shard's schedule, so the union
takes hundreds of steps to settle. At 32 nodes:

| | warmup ends | wall clock, 500 steps | stalled steps |
|---|---|---|---|
| off | step 416 | 23:01 | 21.4% |
| on | step 1 | 17:40 | 2.2% |

It costs roughly the ~73 s a rank already spends compiling, moved to before the
loop. `WARMUP_PLAN.md` has the full measurements, including what did *not* work.

`async_tp` needs `compile` and `tp_size > 1`, and only pays off with
`sequence_parallel = true`. It rewrites collectives and is **not yet numerically
validated** — run `test_tp_parity.py` with it on before trusting it.

### Memory

```toml
[training]
reshard_after_forward = "never"   # or "always"
loss_chunks = 8                   # chunked lm_head + CE
master_dtype = "float32"
grad_reduce_dtype = "float32"
```

`reshard_after_forward = "never"` keeps all-gathered parameters resident for the
whole step — fastest, but every rank pays the full model size. At 9B on 16 nodes
it OOM'd on the third step at 75.23 GiB. Use `"always"` when the model is large
relative to the GPU: one extra all-gather per block, parameter memory drops by
roughly `1 - 1/dp`.

`loss_chunks` splits the micro-batch for `lm_head` + cross-entropy so full
`[tokens, vocab]` logits never exist. On Qwen3.5-2B at 8192 on one GPU: peak
50.6 → 29.9 GiB. `seq_len` must be divisible by it; `1` disables.

### Chat templates

Qwen ships an **inference** template that strips `<think>` spans from all but the
last turn. For SFT on reasoning data that throws away most of the target: on
plotqa_cot it cuts the median sample from 4677 to 1686 tokens. Point
`training.chat_template` at `assets/chat_template_sft.jinja` for any CoT dataset.
The trainer warns at startup if the active template is the reasoning-dropping one.

## Running

```bash
./scripts/finetune.sh      --config configs/local/qwen3_5_2b.toml
./scripts/mn5_finetune.sh  --config configs/mn5/qwen3_5_9b.toml
./scripts/jup_finetune.sh  --config configs/jupiter/qwen3_vl_2b.toml
```

`finetune.sh` auto-detects GPUs from `CUDA_VISIBLE_DEVICES` (or `nvidia-smi`),
picks a free master port and sets `OMP_NUM_THREADS`.

Multi-node:

```bash
CONFIG_FILE=configs/mn5/qwen3_5_9b.toml       sbatch scripts/multinode_mn5.sh
CONFIG_FILE=configs/jupiter/qwen3_vl_2b.toml  sbatch scripts/multinode_jup.sh
```

The batch scripts take the config from `CONFIG_FILE` (arguments after the script name
are not read); without it they fall back to `configs/mn5/qwen3_vl_2b.toml` and
`configs/jupiter/qwen3_5_35b_a3b.toml`.

### Overrides

Every field is overridable on the command line, dotted by section:

```bash
./scripts/finetune.sh \
    --config configs/local/qwen3_5_2b.toml \
    --data.seq_len 8192 \
    --training.tp_size 2
```

### Resuming

```toml
[training]
resume_checkpoint = true
load_dir   = "NULL"   # "NULL" -> resume from output_dir
start_step = 0        # 0 -> latest checkpoint in the load dir
```

Energon's dataloader state is saved and restored with the model state. If a
resume fails inside the dataloader state tree — most often because `repeat`
changed between runs, which inserts a node and breaks the positional restore —
set `data.restore_dataloader_state = false` once to skip it.

## Monitoring

Runs log to Weights & Biases (`WANDB_MODE=offline` on the HPC systems; sync later
with `scripts/sync_wandb.sh`). Logged: loss, LR schedule, throughput, gradient
norms, memory, batch utilisation, document statistics, and per-rank top-K
performance.

`log_topk` gathers per-rank performance with an `all_gather` over WORLD. At 512
ranks every step that is a world-scale collective on the critical path — measured
as a tail growing 0.665 → 0.706 s. `topk_interval` (default 50) samples it
instead; the cross-rank spread does not change step to step.

Read [METRICS.md](METRICS.md) before quoting any throughput number. In particular:
**a short benchmark of this workload measures warmup and nothing else** — at 32
nodes the same config reads 3,704 tok/s/GPU over 40 steps and 12,778 over 500.

## Troubleshooting

### Out of memory

In the order worth trying:

1. `reshard_after_forward = "always"`, the big one for large models.
2. `data_parallel = "fsdp"` if still on `ddp`.
3. Lower `data.tokens_per_microbatch` (raise `tpi_multiplier` to keep the global
   batch fixed by accumulating instead).
4. `loss_chunks` higher, if the vocab logits are the peak.
5. `tp_size` higher — splits the model itself, needs TP to divide every head count.
6. `grad_reduce_dtype = "bfloat16"` — halves the reduce-scatter bytes.

### A slow first step kills the job

Sub-meshes from `ParallelDims.build_mesh()` run on torch's 600 s NCCL default;
`QWEN_NCCL_TIMEOUT_S` only reaches `init_process_group`. Compiling against a
shared filesystem can exceed that. Use a node-local inductor cache, or
`precompile = true`.

### Warmup never ends

Expected without `precompile` at scale (see above). If it persists with
`precompile = true`, a shape is escaping the synthetic spec list: run with
`TORCH_LOGS=recompiles` and check the guard reasons, then add a spec to
`DEFAULT_SPECS` in `train/precompile.py`.

### The loss is fine but the model is wrong

Run the parity tests. TP gradient bugs do not show up in the loss curve — a
misdeclared norm sharding made `attention_norm.weight` come out exactly 2.000×
the TP=1 value while training looked healthy.
