# Production branch — Qwen3-VL 2B on MareNostrum 5

This branch is what MN5 runs. JUPITER is for benchmarking; the numbers that
justify each setting below were measured there and are written up in
`PERFORMANCE.md`, `OPTIMIZATION_PLAN.md` and `ATTENTION_BENCHMARK.md`.

**Target:** maximise throughput of Qwen3-VL-2B-Instruct on MN5 `acc`
(4 × H100 per node, `ehpc543` / `acc_ehpc`), without changing any training
recipe.

---

## 1. What MN5 was missing

The production checkout sat at `5c355a9`, **six commits behind**. It never had:

| | |
|---|---|
| `5600205` | bf16 master weights + the low-precision AdamW variants |
| `de38c80` | chunked cross-entropy for qwen3-vl |
| `24dc505` | the bf16-master / torchao optimizer merge |
| `3f6b7b5` | env standardisation, the `enable_gqa` shim, the `HAS_FLASH` fix |

Plus everything since: the non-finite gradient guard, the qwen3-vl forward-path
migration (six optimisations ported from qwen3.5), `round_max_seqlen`, and the
per-node inductor cache.

## 2. Config changes

Applied to every `configs/mn5/**` config. **Only performance knobs changed** —
learning rates, schedules, step counts, dataset paths and which towers train are
untouched in each one.

### No numerics change

```toml
loss_chunk_mb = 512        # chunked cross-entropy
compile_vision = "dynamic" # the tower's patch count moves nearly every step
compile_head_mode = "off"  # skip autotuning the vocab-wide lm_head GEMM
skip_nonfinite_grads = true
```

`loss_chunk_mb` matters more here than on the 9B: at Qwen3-VL-2B's 151,669-token
vocabulary a 6144-token row upcasts to ~3.8 GiB inside the loss without it.

`skip_nonfinite_grads` zeroes a step whose gradient norm is non-finite instead of
letting one transient poison the weights — about 1 step in 300 at 64 ranks on
JUPITER, no measurable cost. Cheap insurance on a 50,000-step run.

### Numerics change — gated, see §3

```toml
adamw_impl = "foreach_sr"
adamw_stochastic_round = true
master_dtype = "bfloat16"
```

8 B/param instead of 16, and **+41% throughput single-node on the 9B**
(1.790 → 1.267 s/step). Stochastic rounding is not optional: plain
round-to-nearest does not train at these learning rates with a bf16 master.

## 3. The gate before a long run

**`foreach_sr` has never been A/B'd against the current optimizer on real MN5
data.** The mechanism is right and covered by unit tests, and it is validated on
synthetic data — but `PERFORMANCE.md`'s own open-questions list says the
synthetic set cannot produce a discriminating loss curve. Committing 50,000–
100,000 steps of budget to an unvalidated optimizer is not a good trade.

Run ~500 steps each way on a real dataset and compare loss curves before any
long run:

```bash
# A: current production numerics
--training.adamw-impl torchao --training.master-dtype float32
# B: this branch's default
--training.adamw-impl foreach_sr --training.master-dtype bfloat16
```

If the curves diverge, fall back to `adamw_impl = "torchao"` with
`master_dtype = "bfloat16"`, which keeps the memory halving and stochastic
rounding and gives up only the `_foreach_*` batching.

## 4. Environment on MN5

**Nothing new is required.** Verified on `torch 2.11.0+cu126` in
`/gpfs/projects/ehpc543/envs/torch11_cuda12_6`:

```
qwen3_vl import + gqa shim     PASS
AdamW-SR bf16 step             PASS
chunked CE entry point         PASS
non-finite gradient guard      PASS
foreach_sr registered          PASS
```

Present: torch 2.11.0+cu126, transformers 5.3.0, torchao 0.17.0, energon 7.3.2
(has `get_savable_loader`, so `save_dataloader_state` works), triton 3.6.0.

Absent: `flash_attn`, `fla`, `causal_conv1d`, `tilelang`, `flash_qla`, `einops`,
`accelerate`, `pytest`.

**None of those block Qwen3-VL.** They are the GatedDeltaNet stack, and model
imports are lazy (`train/utils.py:261`) — a Qwen3-VL run never imports
`models/qwen3_5/model.py`, so it never reaches them. They are only needed to run
Qwen3.5 on MN5.

### Worth adding, offline

MN5 has no outbound internet, so these need a wheel carried in or a source build.

| package | why | notes |
|---|---|---|
| `flash_attn` | MN5 currently takes the `torch.nn.attention.varlen` fallback. Gain on H100 is **unmeasured** — benchmark before assuming it is worth the build. | needs cp312 / torch 2.11 / cu126 / sm90. Source build wants `nvcc` and takes 30–60 min. |
| `pytest` | run the test suite on MN5 | pure python, trivial wheel |

The `HAS_FLASH` bug fixed in `3f6b7b5` (it was assigned `False` in *both*
branches, so the flash kernel was never called while the log claimed it was)
**changes nothing on MN5**, because `flash_attn` is not installed and the
ImportError branch was already correct. It matters on JUPITER.

## 5. The biggest remaining lever

**At 2B the vision tower is 47% of forward — larger than the entire language
model.**

```
Qwen3-VL first-forward decomposition
  8B, 10240   layers 355.1ms 70%   visual  93.6ms 19%   lm_head 26.4ms 5%   loss 19.1ms 4%
  2B,  4096   visual  85.1ms 47%   layers  79.0ms 43%   lm_head  6.9ms 4%   loss  7.8ms 4%
```

The 2B pairs a 24-layer tower with a 28-layer text model, so the tower barely
shrinks while the text side does. "Is the vision tower efficient" has been open
since `PERFORMANCE.md` revision 5, and it is a **2B question, not an 8B one** —
which makes it this model's question.

Nothing here optimises it, because nothing has measured its achieved TFLOP/s.
Start with `QWEN_SECTION_TIMING=1` on a real MN5 run.

One idea worth testing, not implemented: most stages run `train_vit = false`, so
the tower is forward-only and its output is a pure function of the image. With
`repeat = true` over multiple epochs the same images are re-encoded every epoch.
Caching vision embeddings could remove most of that 47%. Correctness depends on
the image preprocessing being deterministic — check before building it.

## 6. Deliberately unchanged

| | why |
|---|---|
| `seq_len` (4096 / 6144 / 8192) | Longer was better on JUPITER, but that was FSDP at 16+ nodes where it amortises communication. MN5 runs DDP at `tp_size = 1` on 4 nodes; the tradeoff is different and unmeasured. |
| `data_parallel = 'ddp'`, `tp_size = 1` | Right for 2B. TP costs collectives a 2B model does not need, and DDP was faster than FSDP wherever it fit on JUPITER. |
| `ac_memory_budget` | Already tuned per stage (0.5 for init/midtraining, 0.0 for instruct). It only bites inside compiled regions, and `compile = true` here, so it is live. |

## 7. Measuring on MN5

```bash
QWEN_SECTION_TIMING=1   # per-section forward breakdown: visual / layers / lm_head / loss
```

The step line reports `mfu`, `tflops`, `time`, `fwd`, `mem`, `nsamples` and
`batch_util`. `batch_util` below ~95% means packing is wasting sequence budget,
which is a `seq_len` / `packing_buffer_size` problem rather than a kernel one.

**One caveat on MFU:** `flops_per_token` is computed once from config and
`seq_len`, and `vision_flops` sizes the tower from `seq_len` rather than the
actual image count — so MFU is not comparable across datasets of differing image
density. Step time is.
