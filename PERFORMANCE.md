# Qwen3.5-9B training performance: findings, theories, and a path back to 10240

Investigation date: 2026-09-13. Cluster: JUPITER (GH200 96 GiB, 4 GPUs/node).
Checkout: `/e/project1/open-sci-mm/ockier1/vlm-training` @ `24dc505` + local changes.
Environment: `torch_main` — torch 2.14.0+cu130, triton 3.8.0, fla-core 0.5.2,
torchao 0.18.0, causal_conv1d 1.7.0, flash_attn 2.8.3.post1.

Every number below is either **measured** on JUPITER today (marked as such, with
the wandb run id) or **derived arithmetic** from the model config. Theories are
labelled as theories and each carries the experiment that would confirm or kill
it. Nothing here is a guess dressed as a measurement.

**Revision 2, same day.** Section 5's tail was attributed to per-step collectives
and host syncs. All of them have since been removed, and run `1779125` shows the
tail unchanged. Sections 0, 4, 5, 9 and 11 are updated; the correction is set out
in section 5.

**Revision 9, same day.** **The `nan` is fixed and characterised**: one
non-finite gradient every ~300 steps at 64 ranks, now skipped rather than fatal,
at no measurable cost. The optimizer and the dataset are both exonerated by
controls. Also: the qwen3-vl migration, a real training dataset, and the answer
to why Qwen3-VL-8B costs 5x what Qwen3.5-9B does on the same hardware -- which is
the tokenizer, not the model. Section 17.

**Revision 8, 2026-09-14.** The sweep to 256 nodes, and it found a bug that
matters more than any timing in this document: **runs go `nan` and never
recover, with a probability that scales with rank count.** 64 nodes at step 25,
256 nodes at step 13. Revision 5's "60 steps clean" was true only at the scales
it was measured at. Also: DDP's failure above 4 nodes is a **memory** ceiling,
not a communication one, and it is not a ceiling at 8192 -- where DDP is 17%
faster than FSDP. Section 16.

**Revision 7, same day.** **`reshard_after_forward = "never"` is worth ~28% at
every scale**, and at 4 nodes gives the best result in this document: **10240,
1.497 s, 9.4% MFU, 92.9 TFLOP/s** -- beating DDP while still checkpointing. It
shifts the scaling curve down and does **not** change its slope; an earlier draft
of this line claimed otherwise, from comparing two different configs (15.3). Also
a correction: compile
and communication are **not** independent items -- compile alone bought nothing
at 16 nodes, because an exposed all-gather was hiding it. Section 15.

**Revision 6, same day.** `torch.compile` turned on and its recompilation
diagnosed with `TORCH_TRACE` + `tlparse`. Two structural causes, both fixed:
the vision tower was being compiled with the decoder blocks' settings despite a
leading dimension that changes every step, and `DecoderLayer.forward` was one
code object shared by two layer types. Compile also turns out to be a **13.5 GiB
memory saving**, which no earlier revision predicted. Section 14.

**Revision 5, same day.** **`seq_len = 10240` runs**, at 4, 8 and 16 nodes, 60
steps each, clean. The title of this document is out of date in the good
direction. It took three things and none of them was activation checkpointing:
`foreach_sr`, multi-node parameter sharding, and the chunked cross-entropy port.
Section 13 has the sweeps. The bottleneck has moved: FSDP communication now costs
0.42 s/step at 4 nodes and grows with node count, and it is the only thing
measured so far that gets *worse* as hardware is added.

**Revision 4, same day.** Revision 3 read the optimizer A/B as evidence that
`adamw_impl = "foreach"` was safe. It is not: at `lr_llm = 2e-5` with
`master_dtype = "bfloat16"`, round-to-nearest cannot represent the update and the
weights do not move at all. Section 4.1 has the arithmetic and the measurement.
The fix is `adamw_impl = "foreach_sr"` (`train/adamw_sr.py`), which keeps the
speed and the stochastic rounding: **step 1.790 s -> 1.267 s, +41% throughput,
MFU 4.6% -> 6.4%** (job `1781056`).

**Revision 3, same day.** A per-module microbenchmark (`models/tests/bench_layers.py`,
jobs `1779321` / `1779604`) and CUDA-event section timers in the forward (job
`1779630`) replaced most of section 6's theories with measurements, and the
2000-step optimizer A/B (`1779186` / `1779187`) settled section 4. **The decoder
layers run at 42-45% of peak; the kernels were never the problem.** The new
decomposition is section 6.5, and it is the section to read if you read only one.
T3 and T6 are dead. A third of the step is the vision tower, which no earlier
revision mentions at all. The execution plan is `OPTIMIZATION_PLAN.md`, also at
revision 3.

---

## 0. Executive summary

The headline is that Qwen3.5-9B trains at **3.8% MFU** (37 TFLOP/s/GPU against
989.4 peak bf16) in the current 4-node configuration, and the historical
"+500 TFLOP/s/GPU" in `SCALABILITY.md` was never a comparable number.

Four independent factors, each worth 1.4-2x, multiply into the gap:

Superseded by the measured decomposition in section 6.5. Kept for the record of
what each revision believed:

| factor | cost | status |
|---|---|---|
| `adamw_impl = "torchao"` per-parameter optimizer | **0.53 s/step** | **measured**, A/B `1779186`/`1779187` |
| ~~per-step logging collectives + grad clip~~ | ~~0.20 s/step~~ | **falsified at 1 node** (§5); worth ~0.15 s once torchao is gone |
| `compile = false` | ~0.2 s/step | **measured** |
| ~~custom-op backward re-runs the GDN forward (T1)~~ | the whole GDN kernel is 29 ms | **irrelevant** (§6.5) |
| ~~non-GEMM work inside fwd/bwd~~ | the layers run at 42-45% of peak | **falsified** (§6.5) |
| **vision tower** | **~0.38 s/step, 28%** | **measured**, never previously examined |
| **DTensor dispatch + TP collectives around the layers** | **~0.38 s/step** | **measured as a 2.4x multiplier** (§6.5) |

Delivered so far: `adamw_impl = "foreach_sr"` (a new optimizer, section 4.2)
plus the sync removal took the single-node 6144 step from 1.790 s to **1.267 s**
-- 29% off the step, **+41% throughput**, MFU 4.6% -> 6.4% -- with the tail down
from 0.596 s to 0.155 s. Note this is *not* the plain `foreach` route that
revision 3 recommended: that one is faster still and does not train (section
4.1). The next
config-only lever is `train_vit = false` (see 6.5), worth an estimated further
0.24 s if training the vision tower at `lr_vit = 1e-6` is not deliberate.

Separately: `seq_len = 10240` is blocked by the loss, not by the model. The
vocab-parallel `lm_head` is all-gathered to the full 248320 columns on every TP
rank and then upcast to fp32, which costs three vocab-wide fp32 tensors. Porting
the chunked cross-entropy that already exists in `models/qwen3_vl/model.py`
removes ~19-28 GiB at 10240 and is the single highest-value change in this
document.

Finally, on the comparison to Megatron-LM: **40% MFU is the wrong target for this
architecture.** Those figures are for dense transformers. Qwen3.5 is 3/4
gated-delta-net, a chunked recurrence that is memory-bound by construction.
15-25% is the realistic ceiling. That is still 4-6x above where we are, and the
missing pieces are all identifiable.

---

## 1. The baseline is not what it looks like

The regression hunt started from `5c521e0` (2026-04-23), whose `SCALABILITY.md`
records:

```
## Qwen3.5-9B @ JUPITER
- scaling test from 16 to 256 nodes
- +500 TFLOPS/s/gpu
```

That commit changes `SCALABILITY.md` and nothing else. Reading the tree it points
at explains the number.

### 1.1 The FLOPs were counted 4x

At `5c521e0`, `train/utils.py:526`:

```python
num_flops_per_token = 6 * nparams
```

with no division by the TP group size. Commit `6a800a6 [fix] flops divided by TP
group size` lands *after*. With `tp_size = 4`, every TFLOP/s figure from that era
is 4x the current convention.

Cross-check that the two estimators otherwise agree: `6 * 9,409,813,744` =
56.5 GFLOP/token; today's Megatron-style estimator yields ~52.9 GFLOP/token
before the `/tp_size`. Within 7%. So the whole reported drop from that term is
the divide.

**500 TFLOP/s/GPU in the old convention is ~125 TFLOP/s/GPU in today's.**

### 1.2 FSDP was silently a no-op

`configs/jupiter/qwen3_5_9b.toml` @ `5c521e0` declares `data_parallel = 'fsdp'`.
But `apply_fsdp` at that commit reads:

```python
def apply_fsdp(model_type, model, **kwargs):
    if model_type == ModelType.Qwen3_text:
        apply_fsdp_qwen3(model, **kwargs)
    elif model_type == ModelType.Qwen3_vl:
        apply_fsdp_qwen3_vl(model, **kwargs)
```

No `Qwen3_5` branch, no `else`. `train_qwen.py:114` sets
`model_type = ModelType.Qwen3_5` for any model whose name contains `"Qwen3.5"`,
so the call fell off the end and returned. The trainer still logged that
sharding had been applied.

Consequences for that 16-to-256-node "scaling test":

- No parameter all-gather, no gradient reduce-scatter, **no cross-node
  communication of any kind**. Adding nodes was free by construction, which is
  why the curve was flat.
- Data-parallel ranks never synchronised gradients, so it was not one training
  run — it was N independent replicas.
- Per-GPU memory at 256 nodes was identical to per-GPU memory at 1 node.

That last point corrects an earlier explanation of the "10240 on 256 nodes but
not 6144 on one node" puzzle. It was never dp-dimension sharding. Both runs held
the same per-GPU state. The difference was 8 bytes/param then versus 16 now.

### 1.3 No activation checkpointing, and bf16 everywhere

The old config set `ac_mode = 'off'` — no recompute at all. And
`train_qwen.py:157` did `self.model = self.model.to(torch.bfloat16)`, so
parameters, gradients and Adam moments were all bf16: **8 bytes/param**, sharded
4x by TP, ~17.5 GiB per GPU. That is how `seq_len = 10240` fit with AC off.

The current default is fp32 master weights (`master_dtype = "float32"`), which is
16 bytes/param, ~35.1 GiB. That change was made for a real reason — bf16 loading
produced a bad loss curve — and it is not something to revert. `master_dtype =
"bfloat16"` with stochastic rounding recovers the old footprint exactly while
keeping a genuine master copy; see §6.

### 1.4 What this means

The 125 TFLOP/s figure describes a run with no data-parallel communication, no
activation recompute, bf16 everything, and `seq_len = 10240`. It is not a target
to recover. It is a measurement of a different, and incorrect, configuration.

The real question is not "why are we slower than 500" but "why are we at 3.8%
MFU". Those are different investigations, and only the second one matters.

---

## 2. Measurement method

`train/train_qwen.py` times forward and backward as one block:

```python
s_model = time.perf_counter()
with record_function("forward_pass"):   ...      # line 597
with record_function("backward_pass"):  ...      # line 604
self.fwd_bwd_time = time.perf_counter() - s_model  # line 609
```

`optimizer.step()` is at line 614, **outside** that window. So

```
tail = perf/step_time - perf/fwd_bwd_time
```

isolates everything that is not model compute: grad clipping, the optimizer
step, the logging reductions, and the wandb call. This decomposition is the
backbone of everything below.

`perf/step_time` is `time_delta / current_accum_target`; accumulation is 1 in all
runs here (`tpi_multiplier = 1.0`), so step time and wall time per step agree.

---

## 3. Measured: 12 runs, 2026-09-13

All Qwen3.5-9B, `bsc_runs/scaling_9b`. `cmp` = compile, `md` = master dtype,
`tail` = step − fwd/bwd, in seconds.

```
id           seq tp cmp     opt   md    step fwdbwd   tail tail%   TF/s   tok/s
es0dler6    4096  4   F foreach floa  1.496  1.305  0.192   13%   35.6    2738
tvuxaa5b    8192  4   F foreach bflo  2.679  2.464  0.215    8%   40.9    3058
6fkapml5    8192  4   F foreach floa  3.072  2.775  0.297   10%   35.7    2667
94g0h5g5    1024  2   T torchao bflo  1.615  1.011  0.604   37%   16.1     634
aq18n4e3    1024  2   F torchao bflo  1.671  1.082  0.589   35%   15.6     613
ysbgnnoy    4096  2   T torchao bflo  1.796  1.188  0.608   34%   59.3    2281
a7iwwqj2    6144  4   T torchao bflo  1.547  0.951  0.596   39%   52.4    3972
vdkl86xt    6144  4   T torchao bflo  1.587  1.007  0.580   37%   51.0    3871
nn0xkqv9    8192  4   T torchao bflo 11.440 10.843  0.596    5%    9.6     716
d2t3esvy    6144  4   F torchao bflo  2.387  1.692  0.695   29%   33.9    2574   sweep A 4n
uloz9luh    6144  4   F torchao bflo  2.924  2.220  0.704   24%   27.7    2101   sweep A 8n
kjnowkz0    6144  4   F torchao bflo  2.214  1.534  0.680   31%   36.6    2775   sweep A 16n
```

`nn0xkqv9` is the 8192 OOM; its fwd/bwd number is allocator thrashing before the
crash, not model compute. Excluded from all reasoning below except §5.

Also measured, from `perf_topk/mem_gib_*` (peak allocated per log window; the
counter is reset after every log):

```
sweep A   lowest rank   highest rank   spread
  4n        43.85          60.76        16.9 GiB
  8n        43.82          60.16        16.3 GiB
 16n        51.27          62.27        11.0 GiB
```

---

## 4. Finding 1 (measured): the torchao optimizer costs a flat 0.40 s/step

Group the tail column by `adamw_impl`:

```
foreach   0.192  0.215  0.297                                  -> 0.19-0.30 s
torchao   0.604  0.589  0.608  0.596  0.580  0.695  0.704  0.680 -> 0.58-0.70 s
```

The torchao tail is invariant across:

- `seq_len` 1024 -> 8192 (8x the work)
- `tp_size` 2 and 4
- `compile` true and false
- `dp` 1, 2, 4, 8, 16

A cost that ignores tensor size and world size is a **per-parameter Python
loop**. `torchao/optim/adam.py:_AdamBase.step` is exactly that:

```python
with torch._dynamo.utils.disable_cache_limit():
    for group in self.param_groups:
        for p in group["params"]:
            ...
            torch.compile(single_param_adam, fullgraph=True, dynamic=False)(
                p.detach(), grad, state["step"], state["exp_avg"], ...
            )
```

One `torch.compile` call per parameter tensor, on the *same* function object,
with the dynamo cache limit disabled. Qwen3.5-9B has on the order of 600
parameter tensors. Every call walks a guard chain that has grown one entry per
distinct parameter signature.

**Cost: ~0.40 s/step above `foreach`.** At `seq_len = 6144` that is 18% of the
step. At 1024 it is 37%.

**Update (run `1779125`).** With every per-step host sync and four of the six
collectives removed, the tail at 6144 / tp=4 / 1 node is 0.592 s median, of which
logging accounts for 0.0007 s. Whatever else was once in that window is gone, and
torchao's step is what remains: **~0.59 s of a 0.592 s tail.** The figure above
is the delta against `foreach`; this is the absolute. Both stand.

**Settled (jobs `1779186` / `1779187`, 2000 steps each, matched seeds).**

Both arms completed all 2000 steps.

```
              step_time   fwd_bwd    tail    tok/s/GPU
torchao         1.815 s   1.194 s   0.621 s     3385
foreach         1.342 s   1.249 s   0.094 s     4578
```

**Step time -26%, throughput +35%.** (An earlier draft labelled the throughput
column "+26%", which was the step-time reduction against the wrong denominator.)

Loss, mean per 250-step block:

```
   1- 250   torchao 0.05137   foreach 0.04050   -21.2%
 251- 500           0.03466           0.03229    -6.8%
 501- 750           0.03410           0.03468    +1.7%
 751-1000           0.03283           0.03302    +0.6%
1001-1250           0.03611           0.03157   -12.6%
1251-1500           0.03162           0.03072    -2.8%
1501-1750           0.03321           0.03228    -2.8%
1751-2000           0.03018           0.02967    -1.7%
```

foreach is lower in six of eight blocks and never worse than +1.7%. The reason to
read the *blocks* rather than the mean: a round-to-nearest stall would appear as
foreach's curve separating progressively upward from torchao's as updates shrink.
It does not -- the last four blocks are the tightest of the run.

Step 1 is bit-identical in both, as it must be, being computed before any update.
Divergence after that is expected and is evidence of nothing: stochastic rounding
is stochastic, so two torchao runs would not match each other either.

Two caveats on reading this as permission to switch permanently:

- The dataset is `vlm_datasets/synth/fv` and the loss is already at 0.01-0.03.
  A curve that matches on data this easy is weaker evidence than it looks.
- 1500 steps at lr 2e-5 with no warmup keeps updates large. The failure mode that
  motivated stochastic rounding -- round-to-nearest silently discarding updates
  smaller than ~2^-9 of the weight -- is a *late* training failure and is not
  exercised here.

Good evidence for benchmarking. **Not valid for training** -- see 4.1, which
supersedes this reading.

### 4.1 Why the A/B could not have detected the real problem

bf16 keeps **7 explicit mantissa bits**. At the model's typical weight scale --
`1/sqrt(4096) = 2^-6`, the initialisation std of a hidden-size weight -- one ULP
is `2^-13 = 1.22e-4`, so **half a ULP is 6.1e-5**.

`lr_llm = 0.00002`. An Adam update has magnitude `lr * m/sqrt(v)`, which for a
consistently-signed gradient approaches `lr`. **2e-5 is three times below the
half-ULP threshold**, so under round-to-nearest the update lands back on the bf16
value it started from.

Measured, not argued:

```
200 consecutive same-direction updates on a bf16 tensor at w = 0.015625

lr 2e-5 :    0/4096 elements moved,   drift 0        (ideal 0.004)
lr 1e-4 : 4096/4096 elements moved,   drift 0.0211   (ideal 0.02)
lr 1e-3 : 4096/4096 elements moved,   drift 0.196    (ideal 0.2)
```

The effect is scale-dependent, which makes it worse than a clean failure: weights
near 0.0156 freeze completely while smaller ones keep moving. It is a silent,
biased, partial freeze of the largest weights.

**And the A/B in section 4 could not have seen it.** Both arms sat flat at
0.030-0.035 from step 250 to step 2000 (see the block table above -- torchao
0.03466 -> 0.03018, foreach 0.03229 -> 0.02967 over 1750 steps). On
`vlm_datasets/synth/fv` a model that stopped learning at step 250 and one that
kept training are indistinguishable. The matching curves are close to vacuous for
the numerics question; they remain valid for the timing question, which is what
the A/B was built to answer.

Stochastic rounding is therefore not a refinement at these learning rates. It is
the difference between training and not training.

### 4.2 `AdamWSR`: the speed without the risk

`train/adamw_sr.py`. Same update as `torchao.optim._AdamW` term for term, same
fp32 intermediates, same bf16 moments, same stochastic rounding -- so the same
8 B/param footprint. What changes is the loop.

torchao iterates parameters in Python and calls a separately `torch.compile`d
function per parameter: ~500 guard-chain evaluations and launches per step, none
of which get cheaper with more work per tensor. `AdamWSR` buckets parameter
*slices* and drives each bucket through flat fp32 scratch -- one
`_foreach_copy_` to gather, plain tensor ops on one contiguous buffer, one
`_foreach_copy_` to scatter. **~18 launches per bucket regardless of tensor
count.**

Flat buffers rather than `_foreach_*` throughout for a specific reason: there is
no `torch._foreach_bitwise_and_` in torch 2.14, and stochastic rounding is a
bit-level operation. On a flat tensor it is three kernels; per tensor it would be
four launches times the parameter count, which is the cost being escaped.

Slices rather than whole tensors because `lm_head` and `embed_tokens` are
254M-element shards at tp=4 -- a bucket built around one of those needs ~5 GB of
fp32 scratch. Splitting is safe because every operation in the step is
elementwise.

```
20-step smoke, 6144 / tp=4 / 1 node, bf16 master   step    fwd_bwd   tail    tok/s   MFU
torchao        + SR   (1779631)                   1.790 s  1.194 s  0.596 s   3400   4.6%
foreach        no SR  (1779187)                   1.342 s  1.249 s  0.094 s   4578   5.6%
AdamWSR        + SR   (1781056)                   1.267 s  1.112 s  0.155 s   4772   6.4%
```

Stochastic rounding costs ~0.06 s/step over plain foreach -- the `randint` plus
two integer kernels on the flat buffer, ~4.5% of the step. Scratch is 20 bytes per
bf16 parameter in the largest bucket; measured cost 0.6 GiB at `bucket_mb = 64`,
and the knob trades it directly.

On going further: `torch._fused_adamw_` is a real multi-tensor CUDA kernel but has
no stochastic-rounding hook and cannot be extended from Python. `torch.compile`
over the step body would fold the flat-tensor chain into fewer kernels (one
decorator); a Triton multi-tensor kernel would reach one launch per bucket. At
0.06 s/step neither is worth doing while the vision tower is 0.38 s.

Tests: `models/tests/test_adamw_sr.py`, 9 checks, CPU-only. Unbiasedness for both
signs over 200k samples, exactness of the bf16 cast, parity with `_AdamW` to 1e-7
with rounding off, bucket-size invariance, per-group learning rates, state_dict
round-trip of the step counter, and the one that matters -- **bf16 weights move at
lr 2e-5 with rounding on and do not without it**.

One caveat recorded honestly: the first seven of those tests passed while the
optimizer could not run on the real model at all. They used small equal-sized
parameters, so neither the oversized-parameter path nor the variable-bucket path
was exercised; job `1781029` died with `scratch too small: 21233664 < 32219136`.
Two regression tests now cover it.

### Why it is currently switched on

`adamw_impl = "torchao"` is the only implementation that honours
`adamw_stochastic_round`, and stochastic rounding is what makes
`master_dtype = "bfloat16"` safe: bf16 has 8 mantissa bits, so around a weight of
1.0 the smallest representable step is 2^-8, and `torch.optim.AdamW` rounds to
nearest and therefore discards any update below half of that. With
`lr_llm = 2e-5` decaying to 0.1x, most of a long run sits in that regime.

This is pinned by a test — `models/tests/test_adamw_stochastic_round.py`,
4 passed. At lr 1e-6 on a bf16 parameter, 50 steps of `foreach` leave the tensor
**bit-identical**; `torchao` with stochastic rounding moves it.

### The trade to make explicit

18% of throughput is a real price for stochastic rounding. Three options:

1. **Keep torchao.** Correct, slow.
2. **`foreach` + bf16 master, round-to-nearest.** Measure whether loss actually
   diverges over a few thousand steps. The failure mode is silent stalling late
   in training, so a 40-step probe will not show it — this needs a real run.
3. **`foreach` + fp32 master.** No rounding question at all, but 16 bytes/param
   instead of 8, which costs ~17.5 GiB per GPU at tp=4 and takes `seq_len` back
   down. Only viable if the loss work in §6 lands first.

### Theory worth testing separately

If the guard chain is the mechanism, the cost should scale roughly with the
*square* of the parameter count (n calls each scanning an O(n) chain). A 2B
probe with ~1/3 the tensors should show a tail well below 1/3 of 0.40 s. If it
shows exactly 1/3, the cost is launch overhead instead and the fix is different
(batching the loop rather than avoiding compile).

---

## 5. Finding 2 (revised): the residual tail was **not** the collectives

Revision 1 of this section attributed the 0.19-0.30 s that `foreach` pays outside
fwd/bwd to per-step collectives and host syncs, and predicted that removing them
would recover ~0.15 s. **That was wrong, and the experiment that was supposed to
confirm it falsified it instead.**

### What the window contained

Per step, before the change:

- `clip_grad_norm_mixed` over every parameter, with its own all-reduce and, on
  the multi-mesh path, a `float(n.item())` per group
- `dist_mean(loss)`, `dist_max(loss)`, three `dist_sum` calls — five collectives,
  each typed `-> float` and so each ending in `.item()`
- two more `.item()` calls in `batch_generator`, on the batch token counts
- `_all_ranks_have_batch`, an all-reduce over WORLD plus `.item()`, per batch
- `_gather_perf`'s `dist_all_gather` over WORLD for `perf_topk`, every step
- `topk_metrics` calling `.item()` 96 times on a CUDA tensor
- `optimizer.step()`, `zero_grad()`, `wandb.log`
- `torch.cuda.empty_cache()` every `clear_cache_vram = 100` steps

Eight host syncs and six-plus collectives. The reasoning was that each `.item()`
drains the stream and costs the CPU its run-ahead, which on a launch-bound model
should show up directly as step time.

### What happened when they were removed

All of it went: five reductions became two, every `.item()` moved behind a pinned
buffer read a step later, the WORLD gather went behind an interval, the flag moved
to its own stream, `topk_metrics` does one `.cpu()`. Run `1779125`, 20 steps,
6144 / tp=4 / 1 node, torchao:

```
steps>2   step_time med 1.737s   fwd_bwd med 1.154s   tail med 0.592s
          log_wait  med 0.012 ms  (max 0.019)     <- host waiting for staged values
          log_emit  med 0.698 ms  (max 4.6)       <- formatting + wandb.log
```

The deferral itself works perfectly: `log_wait` is 12 microseconds, so the host
never waits. And the tail is **0.592 s against the 0.596 s it was meant to
replace.** Nothing was recovered.

### Why the reasoning failed

The run-ahead argument assumes the CPU is the thing being held back. At `dp=1`
the five reductions reduce over a group of one, and the queue a `.item()` drains
is short. More fundamentally, if the GPU is already saturated for the whole step,
stalling the CPU costs nothing — it had slack. That is the likelier reading, and
it is consistent with `log_emit` being 0.04% of a step: this part of the loop was
never on the critical path at one node.

### What is still open

The one piece of evidence that pointed at collectives has not been re-tested: the
tail grew 0.665 -> 0.681 -> 0.706 s across sweep A's 4n -> 8n -> 16n points under
otherwise identical configuration. That growth is real and it is world-size
dependent, which local work is not. The changes target exactly it, and whether
they flatten it is **unmeasured** — it needs a multi-node run to compare against
those three numbers.

(Superseded in part by "Revision 3" at the end of this section: they did pay,
once torchao stopped hiding them.)

So: keep the changes, because they are correct and they remove work that provably
scales with node count. Do not credit them with any single-node gain. And note
that `foreach`'s 0.19-0.30 s tail is now itself in question — job `1779187` will
say whether it drops with the syncs gone or stays put, and "stays put" would mean
the remaining tail is optimizer work in both implementations.

TorchTitan still reduces logging metrics every N steps rather than every step, and
that is still the right design.

### Revision 3: they did pay, once torchao was out of the way

Job `1779187` answers the question this section left open. With `foreach` and the
syncs removed, the tail is **0.094 s** -- against the 0.19-0.30 s `foreach` paid
before the change. So the sync removal was worth roughly 0.15 s after all. It was
invisible in the torchao measurement because torchao's own per-parameter Python
loop stalls the CPU regardless; removing one stall while another remains buys
nothing.

The two changes compose: 0.622 s of tail with torchao and syncs, 0.094 s with
neither. Revision 2's correction ("nothing was recovered") was accurate about the
measurement in front of it and wrong to generalise from it.

A second round of sync removal -- the five inside the model forward, section 9.6 --
produced no further single-node gain (`1779631` vs `1779125`, step 1.79 vs 1.74 s
median, within noise). Same reasoning: at tp=4 the GPU stays ahead of the CPU, so
the remaining syncs cost nothing here. They stay removed regardless; keeping them
out is free.

---

## 6. Finding 3 (measured residual, theorised cause): fwd/bwd is ~6x its GEMM floor

### 6.1 The FLOPs denominator is honest

Before trusting a 3.8% MFU number, the estimator has to be checked. Counting the
architecture by hand from the checkpoint config — 32 layers, hidden 4096,
intermediate 12288, vocab 248320, 8 full-attention layers (every 4th) with
head_dim 256 and 16 query / 4 KV heads, 24 gated-delta-net layers with 16 key
heads x 128 and 32 value heads x 128 — and using 2*in*out per linear, x3 for
forward plus both backward passes:

| component | per token |
|---|---|
| MLP, 32 layers x (3 x 4096x12288) | 28.99 GFLOP |
| GDN projections, 24 layers x (qkv 8192 + ba 64 + z 4096 + out 4096) | 9.70 GFLOP |
| GDN recurrence, chunk 64, 32 heads x 128x128 | 0.23 GFLOP |
| full-attention projections, 8 layers x (q+gate 8192, k 1024, v 1024, o 4096) | 2.82 GFLOP |
| full attention itself, causal, avg context 3072 | 1.21 GFLOP |
| lm_head 4096 x 248320 | 6.10 GFLOP |
| **total** | **49.05 GFLOP** |

The estimator reports ~52.9 GFLOP/token pre-division. Within 8%, on the
conservative side. **3.8% MFU is a real number**, not an artefact of the
denominator.

### 6.2 The GEMM floor

Pure GEMM work is 47.6 of those 49.1 GFLOP/token. Per GPU at tp=4, at
`seq_len = 6144`:

```
47.61 GFLOP/token / 4 * 6144 tokens = 73.1 TFLOP per step
```

A [6144, 4096] x [4096, 3072] matmul is large enough to run near peak on this
hardware. At a conservative 400-500 TFLOP/s achieved, those GEMMs take
**0.15-0.18 s**.

Measured compiled fwd/bwd at that shape: **0.951 s** (`a7iwwqj2`).

So the accounting for the best compiled step available today is:

```
1.547 s total
  0.15 s   GEMM                     10%
  0.80 s   non-GEMM in fwd/bwd      52%
  0.60 s   tail (optimizer+logging) 38%
```

**Roughly 84% of forward and backward is not matmul.** That is the gap, and it is
not one hotspot — it is dispersed.

### 6.3 Theories for the 0.80 s, ranked

Each of these is visible in the code. None is yet individually measured; a
`torch.profiler` trace would rank them properly, and `train/training_debug.py`
already has `build_debug_profiler` for exactly this.

**Verdict (r3): irrelevant.** The whole gated-delta-rule kernel, forward and backward, is 1.1-1.2 ms per layer, 29 ms across all 24. Whatever it does internally cannot matter at that size.

**T1. The custom-op backward re-runs the forward.** `models/qwen3_5/compile_ops.py`
wraps the three fused kernels in `torch.library.custom_op`, and every backward is
implemented as:

```python
def _gated_delta_rule_bwd(grad_out, q, k, v, g, beta, cu_seqlens):
    with torch.enable_grad():
        qd, kd, vd, gd, bd = (t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
        out, _ = _fla_chunk_gated_delta_rule(qd, kd, vd, gd, bd, ...)   # full forward, again
        grads = torch.autograd.grad(out, (qd, kd, vd, gd, bd), grad_out)
    return list(grads)
```

Same pattern for `causal_conv1d` and `rms_norm_gated`. 24 of 32 layers are GDN,
so every step runs an extra gated-delta-rule forward, an extra causal conv, and
an extra gated RMSNorm on three quarters of the model. Before the custom-op
rewrite this code called fla directly under `@torch.compiler.disable`, using
fla's own hand-written backward, which saves `A`, `g_input` and the l2norm rstds
in its forward and reads them back.

*Status:* the `native_kernels` config flag now routes around it (see §9.2).
*Experiment:* one flag, single node, fixed seq_len. This is the cheapest
double-digit win available.

**Verdict (r3): not the bottleneck.** A whole linear-attention layer, Triton kernels included, runs at 411 TFLOP/s -- 42% of peak. Fusion across the kernel boundaries would help, but there is no large pool of time here to recover.

**T2. Three Triton kernels are opaque to inductor.** fla's chunked delta rule,
`causal_conv1d_fn`, and fla's gated RMSNorm are all custom-op or
`compiler.disable` boundaries. Inductor cannot fuse the elementwise work that
surrounds them — the `softplus`, the `exp`, the `sigmoid`, the rope application,
the residual adds — into those kernels. Each boundary forces a full HBM
round-trip of the activations.

This is the structural difference from Megatron-LM, which ships hand-fused CUDA
for its equivalents, and from TorchTitan, whose blocks are pure PyTorch and
therefore fuse end to end.

*Experiment:* profile and sum the time in kernels that are neither GEMM nor the
three named Triton kernels. If that bucket is large, T2 is confirmed.

**Verdict (r3): dead, by arithmetic.** The transposes, the `repeat_interleave` and the per-layer `seq_idx` together move ~13 GB per forward. At ~3 TB/s that is 4.4 ms of a ~420 ms forward, ~1%. The `seq_idx` hoist is still worth ten lines for the ~140 kernel launches it removes; the rest is not worth touching.

**T3. Gratuitous materialisation in the hot path.** Three concrete instances:

- `GatedDeltaNet.forward` does `k = k.repeat_interleave(repeat, dim=2)` to expand
  16 key heads to 32 value heads. That is a full 2x copy of `k`, 24 times per
  step, in both directions.
- The same function recomputes `seq_idx` inside every layer:
  ```python
  seq_idx = torch.bucketize(
      torch.arange(L, device=qkv.device), cu_seqlens[1:-1], right=True
  ).to(torch.int32).unsqueeze(0).expand(B, -1).contiguous()
  ```
  It depends only on `cu_seqlens`, so all 24 layers compute an identical tensor.
  The tensor is small, but that is 24 x 4 redundant kernel launches per step.
- `SelfAttention.forward` makes three separate `.contiguous()` copies of q, k
  and v after a transpose-and-reshape.

*Experiment:* hoist `seq_idx` to `LanguageModel.forward` and pass it down;
replace `repeat_interleave` with an expanded view where the kernel tolerates it.
Both are small, local diffs.

**Verdict (r3): real, and worse than stated.** `_apply_tp_to_decoder_qwen3_5` has no sequence parallelism: norms are `NoParallel()` and both `out_proj` and `mlp.down_proj` use `output_layouts=Replicate()`, which emits an *all-reduce*. `_micro_pipeline_tp` only matches `all_gather -> mm` and `mm -> reduce_scatter`, so **`async_tp = true` is a no-op on this plan** -- it cannot decompose an all-reduce. TP+SP is a precondition for any overlap here, and separately stops every norm and residual being computed and stored 4x. Part of the 2.4x dispatch/collective multiplier in 6.5.

**T4. Un-overlapped collectives.** `async_tp = false`, so tensor-parallel
all-reduces do not overlap the GEMMs they bracket. `reshard_after_forward =
"default"` means parameters are re-gathered in backward — with 32 decoder blocks
plus the vision tower, that is on the order of 100 FSDP collectives per step.

Supporting evidence: the tp=2 runs (`94g0h5g5` at 1024 -> 1.011 s fwd/bwd,
`ysbgnnoy` at 4096 -> 1.188 s) show 4x the tokens costing 17% more time. Fitting
a line gives ~0.95 s of **sequence-independent** cost inside fwd/bwd. Parameter
traffic is sequence-independent; activation compute is not. The dp=1 runs
(`es0dler6`, `6fkapml5`, both tp=4 single-node, where FSDP shards nothing) fit to
approximately **zero** fixed cost, which is what you would expect if the fixed
term is FSDP.

*Caveat:* the tp=2 pair differs from the tp=4 pair in more than dp, so this fit
is suggestive rather than conclusive.
*Experiment:* run the same seq_len at dp=1 and dp=2 with everything else held,
and compare fwd/bwd.

**Verdict (r3): a memory problem, not a throughput one.** Measured at 25.0 ms fwd+bwd, and `lm_head` at 16.2 ms: together 5% of the step. The 7.58 GiB fp32 logits tensor is still exactly why 8192 OOMs, so section 7 stands unchanged. Revision 2's suggestion that this might be the largest throughput item was wrong.

**T5. The loss.** `apply_tp` hardcodes `loss_parallel = False`
(`train/infra.py`, `_tp_decoder(outer.model, tp_mesh, False, enable_tp_async)`),
so `lm_head` is:

```python
"lm_head": ColwiseParallel(
    input_layouts=Replicate(),
    output_layouts=Shard(-1) if loss_parallel else Replicate(),
    use_local_output=not loss_parallel,
),
```

Every rank all-gathers to the full 248320-column logits rather than keeping its
62080-column shard. Then `models/qwen3_5/utils.py` upcasts the lot:

```python
shift_logits = logits[..., :-1, :].contiguous().float()
```

At 6144 tokens that is a 3.05 GB gather, a 6.1 GB fp32 write, a saved fp32
log-softmax, and a 6.1 GB gradient. Roughly 40 GB of HBM traffic, ~20-30 ms.

*Assessment:* ~1.5% of the step. **Not a throughput problem.** It is, however,
the entire memory problem — see §7.

**Verdict (r3): dead.** Swept 64 / 128 / 256 on the real shapes: 1.07, 1.06, 1.07 ms. No effect. The serial-scan reasoning was sound and the kernel is simply not spending its time there.

**T6. Chunk size.** fla's `chunk_gated_delta_rule` defaults to `chunk_size = 64`,
and `compile_ops.py` hardcodes `_GDR_CHUNK_SIZE = 64` with a `# TODO: deal with
this`. Chunk size sets the arithmetic intensity of a chunked linear attention:
intra-chunk work is O(C) per token while state work is O(d_k*d_v/C). With
d_k = d_v = 128, 64 may be well below optimal.

*Experiment:* sweep chunk size in a standalone benchmark of the fla kernel at the
9B head shapes, before touching the model.

---

### 6.5 Revision 3: the measured decomposition

Two instruments, neither of them a profiler.

**`models/tests/bench_layers.py`** builds a real `DecoderLayer` at the real config
and times it with CUDA events, sweeping `tp` by dividing head counts and the MLP
intermediate -- which is exactly what a rank sees, since this TP plan replicates
activations. Jobs `1779321`, `1779604`, one GH200.

```
tp layer                n   fwd ms  fwd+bwd   x n (ms)
 1 linear_attention    24     6.66    19.83      475.9
 1 full_attention       8     6.69    19.88      159.1
 4 linear_attention    24     3.51     9.56      229.4
 4 full_attention       8     2.30     6.47       51.8

tp=4: layers 0.281 s + lm_head 0.016 s + loss 0.025 s = 0.322 s
      measured fwd_bwd_time = 1.154 s
```

**Section timers** (`QWEN_SECTION_TIMING=1`, section 9.7) attribute the real
forward. Job `1779630`, steady-state step:

```
fwd sections (418.1ms): layers 243.6ms 58%  visual 140.0ms 33%
                        lm_head 16.0ms 4%   loss 13.1ms 3%
                        rope 3.8ms 1%       prologue 1.7ms 0%
```

#### The layers are fine

Hand-counting matmuls per layer at tp=1: a linear-attention layer is 2.68 TFLOP
forward (in_proj_qkv 412 GF, in_proj_z 206, out_proj 206, MLP 1856), so 8.05
TFLOP fwd+bwd against 19.83 ms -- **411 TFLOP/s, 42% of the 989 peak**. A
full-attention layer is 8.66 TFLOP against 19.88 ms -- **445 TFLOP/s, 45%**.

That is a normally-performing transformer. Every revision before this one was
looking for a kernel problem that does not exist.

#### But they cost 2.4x that in the real model

Standalone at tp=4 the 32 layers are 102.6 ms of forward. In the real model they
are 243.6 ms. Same shapes, same kernels. The 141 ms difference is eager DTensor
dispatch, the 64 per-step TP all-reduces (2 per layer x 48 MiB), and FSDP.

#### And a third of the step is the vision tower

140 ms of a 418 ms forward. Scaling by the benchmark's own bwd/fwd ratio (~2.7),
roughly 0.38 s of the 1.154 s fwd+bwd.

`configs/jupiter/scaling/qwen3_5_9b.toml` sets `train_vit = true` with
`lr_vit = 0.000001`. The tower is being trained at a learning rate near zero. If
that is not deliberate, `train_vit = false` deletes its backward pass -- an
estimated **0.24 s off the step for one config line**, larger than any other lever
currently identified.

#### The whole step

| | fwd | est. fwd+bwd | share |
|---|---|---|---|
| decoder layers | 243.6 ms | ~0.66 s | 57% |
| -- of which kernels | 102.6 ms | ~0.28 s | 24% |
| -- of which dispatch + collectives | 141 ms | ~0.38 s | 33% |
| vision tower | 140.0 ms | ~0.38 s | 33% |
| lm_head + loss | 29.1 ms | ~0.07 s | 6% |

Sums to ~1.11 s against 1.154 s measured; the decomposition closes to within 4%.

#### One confirmed theory that does not matter

TP shards GDN heads, so at tp=4 each rank runs the delta rule with 8 value heads
instead of 32. Measured: **1.11 ms at tp=4 against 1.07 ms at tp=1, for a quarter
of the work.** The kernel is occupancy-starved on the head dimension exactly as
predicted -- and it costs about 22 ms a step, because the whole kernel is 29 ms.
Recorded because it will matter at larger `tp`, not because it matters now.

---

## 7. Memory: why 8192 OOMs and how to get 10240

### 7.1 The OOM, exactly

`nn0xkqv9`, `seq_len = 8192`, tp=4, compile=true, `ac_memory_budget = 1.0`:

```
File ".../train/train_qwen.py", line 593, in train_step
  scaled_loss.backward()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.58 GiB.
GPU 0 has a total capacity of 95.00 GiB of which 3.09 GiB is free.
Of the allocated memory 74.85 GiB is allocated by PyTorch,
and 15.02 GiB is reserved by PyTorch but unallocated.
```

**7.58 GiB is exactly `8192 * 248320 * 4` bytes** — the fp32 gradient of the
logits. The allocation that failed is the loss, not the model.

Note also the 15.02 GiB of reserved-but-unallocated memory: 17% of the card lost
to fragmentation, despite `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
being set in both `debug_1node.sh:49` and `jup_scaling.sbatch:59`.

### 7.2 8192 already works in a different configuration

`tvuxaa5b`: `seq_len = 8192`, tp=4, `compile = false`, `ac_memory_budget = 0.0`,
bf16 master, `foreach`. **Finished 40 steps at 81.77 GiB peak, 40.9 TFLOP/s** —
the best 8192 throughput measured.

The difference is not seq_len. It is that with `compile = false` the
`ac_memory_budget` is inert (the partitioner budget only applies inside compiled
regions), so that run used plain eager autograd, which saves *less* than
inductor's `budget = 1.0` "save everything" partition. Turning compile on while
leaving the budget at 1.0 is the most memory-hungry combination the config can
express, and it is the current default.

### 7.3 A memory model

Two same-configuration points, tp=4, dp=1, no compile, fp32 master:

```
es0dler6   4096 tokens -> 44.45 GiB
6fkapml5   8192 tokens -> 84.53 GiB
```

Fitting:

```
peak_allocated  ~=  4.3 GiB  +  9.8 MiB per token
```

At 10240 that projects to **~104.7 GiB** on a 95 GiB card. The OOM is not
marginal; it is 10 GiB over.

### 7.4 Scaling out does not help

Sweep A peaks: 60.76 GiB at 4 nodes, 62.27 GiB at 16 nodes. Peak went **up**
while dp went from 4 to 16.

The reason: at tp=4, sharded static state is 8 bytes/param / 16 = 4.4 GiB at
dp=4 and 1.1 GiB at dp=16. A 3 GiB difference, invisible against ~55 GiB of
activations that are identical on every rank regardless of world size.

**Adding nodes will never buy seq_len.** Only recompute, the loss, and seq_len
itself move that number.

### 7.5 The ~17 GiB per-rank spread

Within a single job, peak memory varies 43.85 to 60.76 GiB across ranks. With
`batch_size = 0` the packer emits exactly `seq_len` tokens per rank, so this is
not sequence length — it is image count. `train_vit = true` and vision loading is
on, so ranks that draw image-heavy batches run a larger vision tower.

The rank that OOMs is the heaviest one, and it sits ~17 GiB above the rank you
see if you only watch rank 0. This is why memory is now printed per step (§9.3)
*and* why `perf_topk/mem_gib_*` remains necessary.

### 7.6 Getting to 10240, in order of value per line of code

**(a) Port the chunked cross-entropy. ~19-28 GiB.**

`models/qwen3_vl/model.py:139` already contains a complete `_ChunkedCrossEntropy`
autograd Function, driven by `loss_chunk_mb`, that never materialises a
vocab-wide fp32 tensor:

```python
shift_logits = logits[..., :-1, :]        # a view, no copy
...
for lo in range(0, logits.shape[0], chunk):
    lg = logits[lo:hi].float()
    row = torch.logsumexp(lg, dim=-1) - tgt
    total = total + torch.where(keep, row, zeros).sum()
```

with a matching chunked backward that writes the gradient in `logits.dtype`.
Qwen3.5 has no equivalent, and `train_qwen.py:197` only calls
`set_loss_chunk_mb` for `ModelType.Qwen3_vl`. At 10240 x 248320 this removes the
fp32 logits copy, the saved log-softmax, and the fp32 gradient. **This alone
should be enough for 10240.**

**(b) `compile = true` with `ac_memory_budget` below 1.0. ~15-20 GiB.**
The current setting is the save-everything end of the range. 0.5 is a sensible
first probe. Note the interaction: the budget does nothing while compile is off,
so these two flags must move together.

**(c) `loss_parallel = True`. ~7 GiB plus a 4x shrink downstream.**
Keeps the logits `Shard(-1)` so each rank holds 62080 columns. Requires running
the loss inside `torch.distributed.tensor.parallel.loss_parallel()` and threading
the flag through `apply_tp`, which currently hardcodes `False`. TorchTitan does
this by default; at vocab 248320 it matters more here than in most models.

**(d) `master_dtype = "bfloat16"`. 8 bytes/param instead of 16.**
Already the default (§9.1). Worth 17.5 GiB of static state at tp=4/dp=1, though
only ~2.8 GiB of *peak*, because peak occurs at the activation high-water mark
where the transient dominates.

---

## 8. Can the gap to Megatron-LM / TorchTitan be closed?

### 8.1 Reframe the target

Megatron and TorchTitan publish 35-50% MFU for **dense transformers** — models
where FlashAttention and large fused GEMMs dominate the profile. Qwen3.5-9B is
3/4 gated-delta-net. A chunked recurrence has fundamentally lower arithmetic
intensity than softmax attention: with d_k = d_v = 128 and chunk 64, the kernel
is bound by materialising and updating the recurrent state, not by matmul
throughput. Published Mamba- and GDN-class training runs typically report 15-25%
MFU.

**Comparing to Megatron-on-Llama is a category error.** The right target for this
architecture is 20-25%.

That said: 3.8% is still 5-6x below that target, and every factor identified in
this document is a configuration choice or a kernel-boundary decision. None of it
is architectural.

### 8.2 What TorchTitan has that this codebase does not

| capability | TorchTitan | here |
|---|---|---|
| whole-block compilation with no kernel boundaries | yes | 3 opaque Triton kernels |
| selective activation checkpointing (save matmuls, recompute the rest) | yes | **deleted** — only the all-or-nothing partitioner budget remains |
| `loss_parallel` | default on | hardcoded off |
| async tensor parallel | supported | `async_tp = false` |
| foreach/fused optimizer | default | per-parameter compiled loop |
| metric reduction every N steps | yes | six collectives every step |

The selective-AC deletion deserves emphasis. At `5c521e0`, `train/infra.py` had
`ACConfig`, `_apply_ac_to_transformer_block`, and:

```python
_op_sac_save_list = {torch.ops.aten.mm.default}
```

— save the matmul outputs, recompute everything else. That is the standard way
to buy sequence length cheaply, because matmul outputs are expensive to recompute
and cheap to store while elementwise results are the reverse. All of it was
removed in favour of `torch._functorch.config.activation_memory_budget`, which is
a single scalar with no notion of *which* ops to keep, and which only applies
inside compiled regions.

### 8.3 A staged plan, with predicted numbers

Predictions are extrapolations from the measured decomposition, not guarantees.

**Stage 1 — configuration only, no new code.**

| change | effect at seq 6144 | status |
|---|---|---|
| `adamw_impl = "foreach_sr"` | tail 0.596 -> 0.155 s, step 1.790 -> 1.267 s | **done, measured** |
| ~~`adamw_impl = "foreach"`~~ | ~~tail -> 0.094 s~~ | **rejected**: does not train at lr 2e-5 (§4.1) |
| remove the per-step host syncs | included in the above | **done** (§5, §9.5, §9.6) |
| `train_vit = false` | est. -0.24 s | **not tried** -- the largest remaining config lever (§6.5) |
| `compile = true` | fwd/bwd 1.53 -> ~0.95 s | **measured earlier**, attacks the dispatch overhead in §6.5 |
| ~~`native_kernels = true`~~ | ~~-0.20 s (T1)~~ | **dead**: the whole GDN kernel is 29 ms |
| **step time so far** | **1.790 -> 1.267 s** | **measured** (`1779631` vs `1781056`) |
| **with `train_vit = false`** | **~1.10 s** | estimate |

The two tail rows have collapsed into one since revision 1: the logging work
shipped and recovered nothing at one node, so the whole tail now rides on
`adamw_impl`. The stage-1 total is unchanged because the tail was always going to
end up near 0.05 s; only the attribution moved.

Caveat: `foreach` gives up stochastic rounding, so it needs the §4 loss check
before it becomes the default for real training rather than for benchmarking.

**Stage 2 — small, local code changes.**

- Port `_ChunkedCrossEntropy` to `models/qwen3_5/utils.py` and wire
  `set_loss_chunk_mb` for `ModelType.Qwen3_5`. Unlocks 10240.
- Hoist `seq_idx` out of the GDN layer loop.
- Replace `repeat_interleave` with a view where the kernel accepts it.

Plausibly another 1.3-1.5x, plus the seq_len.

**Stage 3 — real but bounded work.**

- `loss_parallel = True` end to end.
- Restore selective activation checkpointing with an op save-list.
- `async_tp = true`, which requires compile and needs validating against the
  DTensor bugs documented in `compile_model`'s docstring.
- Tune the GDN chunk size against the 9B head shapes.

**Stage 4 — the actual project.**

Making the three Triton kernels fusable with their neighbours, or replacing them
with implementations inductor can see through. This is where the remaining 2-3x
lives. Weeks of work, and the correct thing to attempt only after Stages 1-3 have
been measured, because it is the only item on this list that cannot be reverted
with a config flag.

### 8.4 The honest answer

Yes, the gap can be closed — to the extent that "closed" means reaching 20-25%
MFU for a hybrid-attention model rather than matching a dense-transformer number.
The 10x looks alarming because four independent 1.4-2x factors multiply. Removing
them is mostly unwinding decisions that were made for correctness reasons and
never revisited for cost.

**Revision 3 answers the question this paragraph posed.** It said the assessment
would change if the non-GEMM time turned out to be inside the fla kernels rather
than around them, and called getting a trace the highest-priority action.

It is not inside the kernels, and it did not need a trace. A whole
linear-attention layer runs at 411 TFLOP/s and the delta-rule kernel is 29 ms of a
1154 ms step (§6.5). The ceiling is not set by fla. It is set by three things that
are all ordinary engineering:

1. the vision tower, a third of the step, never examined until now
2. a 2.4x dispatch-and-collective multiplier around layers whose kernels are fine
3. `compile = false`, which is most of what feeds (2)

That is a better position than revision 1 described, and none of it requires
writing a Triton kernel. The remaining reason to capture a trace is to split the
2.4x multiplier into its DTensor and collective parts -- useful, no longer
blocking.

---

## 9. Changes already in the tree

All uncommitted, in `/e/project1/open-sci-mm/ockier1/vlm-training`.

### 9.1 bf16 master weights with stochastic rounding

`adamw_stochastic_round` was already plumbed to `build_adamw`
(`train_qwen.py:355`); the missing piece was a non-quantised torchao optimizer.
torchao 0.18.0 exports `_AdamW`, whose `block_size=inf` makes `_new_buffer` fall
through to `torch.zeros_like`, so moments are plain tensors in the parameter
dtype — no `OptimState8bit` subclass, hence none of the
`AttributeError: 'Tensor' object has no attribute 'codes'` that kills `8bit` on a
dp x tp mesh.

```python
TORCHAO_ADAMW = {"torchao": "_AdamW", "fp8": "AdamWFp8",
                 "8bit": "AdamW8bit", "4bit": "AdamW4bit"}
```

Defaults changed: `adamw_impl = "torchao"`, `adamw_stochastic_round = True`,
`master_dtype = "bfloat16"`. The `stochastic_round` guard now warns rather than
raising, since the flag defaults on and every `foreach` config would otherwise
fail to build.

Verified: `models/tests/test_adamw_stochastic_round.py`, 4 passed.

FSDP interaction checked in torch 2.14 — `_fsdp_collectives.py:687` does
`reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)`, so
`reduce_dtype = float32` upcasts only the transient reduce-scatter buffer and the
stored sharded gradient comes back at the parameter dtype. fp32 reduction
accuracy at no standing memory cost.

### 9.2 `native_kernels`, and the stale `*_native` ops

`compile_ops.py` shipped `gated_delta_rule_native`, `causal_conv1d_native` and
`rms_norm_gated_native` — unused, and broken against the installed fla:

```
TypeError: chunk_gated_delta_rule_fwd() got an unexpected keyword argument 'transpose_state_layout'
```

Against fla-core 0.5.2 there are three mismatches: no `transpose_state_layout`
parameter; `chunk_gated_delta_rule_fwd` returns six values
(`g, o, A, final_state, initial_state, g_input`) where `_gdr_native_fwd` unpacks
five; `chunk_gated_delta_rule_bwd` returns eight where it unpacks six.

Rather than rewire them — which means threading `initial_state` and `g_input`
through a strictly-typed custom op — `native_kernels = true` now calls the
library entry points directly inside `@torch.compiler.disable(recursive=True)`.
That gets the actual goal (fla's `ChunkGatedDeltaRuleFunction` saves `A`,
`g_input` and the l2norm rstds in forward and reads them in backward, no
recompute) for one line. The price is one graph break per call, which is what the
code cost before the custom ops existed. The stale ops are left in place, unused,
with a comment explaining why.

### 9.3 Memory in the step log

`log()` now reads both counters and prints them:

```
... time 2.175s fwd 1.510s mem 60.8/76.1G data_pct 0.03% nsamples 22 batch_util 99.6%
```

Allocated and reserved, because allocated alone hid the 15 GiB of fragmentation
in the OOM. `perf/mem_gib` and `perf/mem_reserved_gib` added to `log_metrics` —
memory was previously reachable only through `perf_topk/*`, which needs
`log_topk = true` and an all-gather.

### 9.4 Config levers

Everything previously hardcoded or hidden behind an environment variable is now
in the TOML, so combinations can be swept without editing code:

| option | was |
|---|---|
| `native_kernels` | hardcoded `_ops.gated_delta_rule` |
| `compile_head_mode` | hardcoded `max-autotune-no-cudagraphs` |
| `compile_block_mode` | hardcoded `'default'` |
| `compile_gdn` | env `QWEN_COMPILE_GDN` |
| `compile_dynamic` | env `QWEN_COMPILE_DYNAMIC` |
| `grad_reduce_dtype` | hardcoded `torch.float32` |
| `log_graph_code` | unconditional at module scope |

That last one was `torch._logging.set_logs(graph_code=True)` at
`train_qwen.py:74`, dumping every traced FX graph's source on every rank — 5 MB
per rank at 16 nodes, 42 MB at 256, written during the dynamo tracing that
already dominates startup. Now defaults off.

Also fixed: the inductor FX graph cache bakes process-group *names* into compiled
collectives, and those names are assigned in creation order, so they depend on
the mesh shape. A graph cached by a 4-GPU run replays "group 35" inside a 512-GPU
job that never registered it:

```
RuntimeError: Could not resolve the process group registered under the name 35
```

That killed the 16n and 32n points of sweep `20260912_223302` at ~18 minutes.
`jup_scaling.sbatch` now keys `TORCHINDUCTOR_CACHE_DIR` on the world size.

### 9.5 Every per-step host sync removed

Landed after revision 1, verified by run `1779125`. See §5 for why it did not pay
off at one node and why it is kept anyway.

- `train/utils.py`: `_dist_reduce`, `dist_mean`, `dist_max`, `dist_sum` deleted —
  all four were typed `-> float` and ended in `.item()`. Replaced by
  `dist_sum_max`, one SUM all-reduce and one MAX, returning device tensors. A mean
  is a sum over the group size, computed on the host; it does not need its own
  collective.
- `train/train_qwen.py`: `_stage_log` / `_flush_log`. The reduced counters are
  copied into a pinned buffer with a recorded event and read on the *next* step,
  so the record for step N is emitted during step N+1 and labelled N. `end_run`
  flushes the last one.
- Token counters (`tokens_seen`, `ntokens_since_last_log`, ...) are device
  tensors; `int()` is called only at checkpoint save.
- `_all_ranks_have_batch` runs on a dedicated stream. Its answer is needed before
  the step so it cannot be deferred, but the flag has no data dependency on the
  model, so keeping it off the training stream means its `.item()` waits on a
  one-int all-reduce rather than on the previous step.
- `topk_interval` (default 50) gates `_gather_perf`'s WORLD all-gather;
  `topk_metrics` does one `gathered.cpu()` instead of 96 `.item()` calls.
- `clip_grad_norm_mixed`'s multi-mesh path keeps the squared norm as a tensor and
  delegates clipping to `torch.nn.utils.clip_grads_with_norm_`. Revision 1
  proposed logging `len(groups)` to find out whether that path even runs; making
  both paths sync-free was cheaper than finding out.
- New: `perf/log_wait_ms` and `perf/log_emit_ms` in wandb. `log_wait` is the
  canary — above ~0.05 ms means something has started draining the stream again.

Tests: `models/tests/test_deferred_log.py` pins the eight pinned-buffer slot
indices through `log()`; `models/tests/test_topk_metrics.py` pins the topk
ordering and that no tensor leaks into the metrics dict. Both CPU-only.

### 9.6 The forward's own host syncs

Section 9.5 removed eight syncs from the training loop. The model forward still
had five, and they sat at the *top* of the step where they drain the previous
backward:

- the `cu_seqlens` contract check, two `.item()` calls -> `torch._assert_async`,
  guard kept, no sync (a failure becomes a device-side abort, not a message)
- `max_seqlen` -> computed in `batch_generator` while cu_seqlens is still on the
  host and passed as `batch['max_seqlen']`; `varlen_attn` needs a Python int, so
  this one can only be moved, not deferred. The in-model computation stays as a
  fallback for tests and parity scripts.
- the vision tower's own `max_seqlen` -> same treatment, `batch['vision_max_seqlen']`
- the image-token-count assert -> `torch._assert_async`; a mismatch also fails in
  `masked_scatter` two lines later, so the sync only ever bought a nicer message
- `causal_lm_loss`'s `if (flat_labels != ignore_index).sum() == 0:` -> replaced by
  `reduction="sum"` over `count.clamp(min=1)`, the same mean with no branch on a
  device value. It sat immediately after the fp32 logits materialization, the
  worst place in the step to drain the stream.

Left alone: the video branch's `combined.sum().item()`, which genuinely needs a
host int for `new_zeros` and only fires with video data.

No single-node gain (§5, revision 3). Verified equivalent: step 1 of `1779631` is
bit-identical to step 1 of `1779125`, and `models/tests/test_forward_syncs.py`
pins `packed_positions` against the loop it replaced and the new loss against the
branching version, including the all-ignored case and a single-supervised-token
case that would catch a denominator averaging over ignored positions.

### 9.7 Instrumentation added for this investigation

- `models/tests/bench_layers.py` -- per-module CUDA-event benchmark at the real
  config. Sweeps `tp` by dividing head counts and the MLP intermediate, which is
  what a rank actually sees under this TP plan. Also benchmarks `lm_head`, the
  loss, and the raw delta-rule kernel across chunk sizes. One GPU, ~4 minutes.
- `QWEN_SECTION_TIMING=1` -- CUDA-event section timers in
  `Qwen3_5ForCausalLM.forward` (`prologue`, `visual`, `rope`, `layers`, `lm_head`,
  `loss`). Implemented as `_SectionTimer.mark()` calls rather than nested context
  managers so the forward keeps its indentation. A no-op when the variable is
  unset. Prints one line per step and logs `perf_fwd/*_ms`. Forward only:
  backward is one fused `.backward()` and cannot be split this way.
- `perf/log_wait_ms` and `perf/log_emit_ms` -- the canaries from §9.5. `log_wait`
  above ~0.05 ms means something has started draining the stream again.

---

## 10. Two coupling traps

Worth stating plainly, because either one silently wastes a run:

1. **`ac_memory_budget` only applies inside compiled regions.** It is a no-op
   while `compile = false`. The current scaling config sets a budget and disables
   compile, so the budget does nothing — and this is why 8192 fits in that
   configuration but not with compile on.
2. **`adamw_stochastic_round` needs both a torchao `adamw_impl` and
   `master_dtype = "bfloat16"`.** With any other optimizer it is ignored with a
   warning; with fp32 master weights it is meaningless.

---

## 11. What is not yet known

Listed explicitly so the next person does not mistake a theory for a result.

- ~~**No profile exists.**~~ Superseded. The microbenchmark and section timers in
  §6.5 decomposed the step without one, and the answer was that the kernels are
  fine. A trace is still the way to split the 2.4x dispatch/collective multiplier
  into its parts, but it is no longer blocking anything.
- **The vision tower is measured but not understood.** 140 ms of forward, and no
  breakdown of it by block, image count, or resolution. Its cost varies step to
  step with the packed image count, which is also why step time wobbles.
- **`train_vit = false` has not been tried.** The 0.24 s is an estimate from the
  measured forward share and the benchmark's bwd/fwd ratio, not a measurement.
- **The 2.4x multiplier is a subtraction, not an attribution.** 243.6 ms real
  against 102.6 ms standalone. How that splits between eager DTensor dispatch, the
  64 TP all-reduces and FSDP is unknown.
- **`native_kernels = true` has never completed a step.** The rewire imports
  cleanly and configs parse, but the login node has no GPU. Its first real run is
  also its first correctness check: the two paths are mathematically identical,
  so any loss difference is a bug.
- **The 0.95 s sequence-independent term inside tp=2 fwd/bwd is attributed to
  FSDP collectives on circumstantial evidence.** The tp=2 and tp=4 runs differ in
  more than dp.
- **The guard-chain mechanism behind the torchao tail is inferred** from its
  invariance to problem size, not from a Python profile. Timing
  `optimizer.step()` between synchronize calls, with and without
  `torch._dynamo.disable` on `single_param_adam`, separates guard cost from launch
  cost and takes ten minutes.
- ~~**bf16 master with round-to-nearest matched over 1500 steps**~~ Resolved, and
  not in the direction that reading suggested: at `lr_llm = 2e-5` round-to-nearest
  does not move a bf16 weight of typical magnitude *at all* (§4.1). The matching
  curve was an artifact of a dataset on which neither arm needed to learn.
- **`AdamWSR` has not been validated on data that requires learning.** Its
  correctness rests on the ULP arithmetic and the unit tests, not on a loss curve;
  the synthetic dataset structurally cannot discriminate between an optimizer that
  trains and one that does not. Running it against real data is the outstanding
  check.
- ~~**Whether the sync removal is worth anything at 16+ nodes.**~~ Answered in
  13.3: the tail is flat at ~0.2 s across 4n/8n/16n, against sweep A's
  0.665/0.680/0.706 before the work.
- **Whether the FSDP collectives are exposed or merely slow.** The accelerating
  cost with node count (13.3) is consistent with failed overlap but does not
  demonstrate it. A profiler trace at 8 or 16 nodes would show whether the
  all-gathers sit in gaps.
- **Whether the vision tower is efficient.** It is ~28% of the step and no
  measurement exists of its achieved TFLOP/s, only its wall time.
- **Whether DDP still beats FSDP once compile is on** (14.7). Under DDP the GDN
  layers do not compile, so three quarters of the model stays eager.
- **What the remaining `[0/32]` frame in the `compile_vision = "off"` arm was.**
  The `dynamic` arm's remaining driver was the `DecoderLayer` branch, now fixed;
  the `off` arm reported the same reason on a different frame id and was not
  chased further.
- **The 0.15 s GEMM floor assumes 400-500 TFLOP/s achieved** on the relevant
  shapes. It has not been benchmarked on this hardware at these shapes.

---

## 13. Revision 5: 10240 works, and what it cost

### 13.1 The sequence-length sweep, 4 nodes

`foreach_sr`, bf16 master, tp=4, compile off, 60 steps.

```
                          step    fwd_bwd   tail    MFU   TF/s   mem max        outcome
6144   no chunking       1.682 s  1.529 s  0.153 s  4.8%  47.9   63.9/72.6 G    clean
8192   no chunking       1.850 s  1.683 s  0.167 s  5.5%  54.1   82.0/91.8 G    OOM at step 49
8192   chunked CE        1.882 s  1.706 s  0.176 s  5.8%  57.9   66.9/73.9 G    clean
10240  chunked CE        2.121 s  1.924 s  0.197 s  6.6%  65.4   82.3/90.2 G    clean
```

Two things fall out of this table.

**MFU rises with sequence length: 4.8% -> 5.8% -> 6.6%.** The per-step cost of the
FSDP collectives is fixed, so more tokens per step amortizes it. This inverts the
single-node reading in section 6.5, where 8192 bought nothing over 6144 -- at one
node there is no cross-node collective to amortize. **Sequence length is worth
more at scale than the single-node numbers suggested.**

**The unchunked 8192 failure is instructive.** It survived 48 steps with reserved
pinned at 91.8 GiB while allocated swung 62.7 -> 77.8 GiB with the packed image
count. It was not sitting at a peak, it was sampling from a distribution, and step
49 drew a batch that did not fit. The error came from NCCL (`Cuda failure 'out of
memory'`) rather than a torch tensor: with reserved at 91.8 GiB there was nothing
left outside the caching allocator for the communicator. A configuration this
close to the ceiling fails eventually rather than immediately, which is the worst
failure mode for a long run.

### 13.2 Chunked cross-entropy

Ported from `models/qwen3_vl/model.py:139` into `models/qwen3_5/utils.py`, with
`set_loss_chunk_mb` and `loss_chunk_mb` in the config. **1.7% of step time for
18 GiB** (8192: step 1.850 -> 1.882 s, reserved 91.8 -> 73.9 GiB).

Two savings, not one:

- the chunk loop keeps the fp32 working set proportional to the chunk instead of
  the packed row, and its backward recomputes softmax from the saved bf16 logits
  rather than storing an fp32 log-softmax
- `causal_lm_loss` was also doing `logits[..., :-1, :].contiguous().float()`.
  `logits` is `(1, T, V)`, so dropping the last row leaves a contiguous view --
  the `.contiguous()` bought nothing and the `.float()` made a vocab-sized fp32
  copy before the loss started. 8.14 GB at 8192, on its own.

One change from the qwen3-VL original: it guards an all-ignored batch with
`if (flat_labels != ignore_index).sum() == 0`, a host sync. The chunked forward
already divides by `n_valid.clamp_min(1)`, so the branch is redundant.

**A bug found on the way.** `set_loss_chunk_mb` never existed in
`models/qwen3_vl/model.py` -- not in the working tree and not in `de38c80`, the
commit that added the chunking. `train/train_qwen.py:202` imports it
unconditionally inside the `ModelType.Qwen3_vl` branch, so **every Qwen3-VL run
has been failing at startup with ImportError**, and the feature has never been
switched on by any config. The machinery was complete; only the switch was
missing. Added.

### 13.3 Weak scaling at 10240

Same config, per-GPU tokens fixed, global batch growing.

```
nodes  GPUs  dp   step_med  fwd_bwd   tail    MFU   TF/s   mem max      efficiency
  4     16    4    2.121 s  1.924 s  0.197 s  6.6%  65.4   82.3/90.2 G    100%
  8     32    8    2.222 s  2.033 s  0.189 s  6.3%  62.2   80.7/87.8 G     95%
 16     64   16    2.462 s  2.239 s  0.223 s  5.6%  55.1   80.0/87.5 G     86%
```

**The tail is flat: 0.197 / 0.189 / 0.223 s.** Section 11 listed "whether the sync
removal is worth anything above one node" as unknown. It is: sweep A, before any
of that work, grew 0.665 -> 0.706 s across the same 4n -> 16n span. The tail is now
a third of that in absolute terms and grows by 0.026 s rather than 0.041 s.
Answered, and in favour.

**All the degradation is in `fwd_bwd`: 1.924 -> 2.033 -> 2.239 s**, and it
accelerates -- **+4.8%** from 4 to 8 nodes, **+10.8%** from 8 to 16. A
constant-cost collective would be flat and a log-scaling one would decelerate. An
accelerating curve says the all-gathers are increasingly failing to overlap with
compute as the ring lengthens. Extrapolating that shape to 256 nodes is a reason
to measure, not a number to quote.

**Memory barely moves with node count: 82.3 -> 80.7 -> 80.0 GiB.** Parameters shard
over dp and fall 4.7 -> 2.35 -> 1.18 GB/GPU, but activations dominate and do not
shard. This re-confirms 7.4 from a completely different direction: scaling out
never buys sequence length.

**Jitter grows sharply at 16 nodes.** Steps range 2.339-3.021 s against 8 nodes'
2.135-2.436 s, and the two slowest steps of the run are the last two. The
`perf_topk` rows sampled at step 50 would name the ranks.

### 13.4 What this changes

`train_vit` is settled and not by this document: the vision tower's cost is the
price of a capability that will be judged on downstream benchmark results, and it
will very probably stay on. Its ~28% is therefore a *make it faster* problem, not
a lever. Nothing has yet measured whether the tower runs at the 42-45% of peak
that the decoder layers manage.

Selective activation checkpointing was called "the item 10240 depends on" in
revisions 1-4. It was not: 10240 runs without it. It remains absent and still
worth having -- 80 of 95 GiB is usable but not generous, and it is what opens
12288 and beyond -- but it is no longer blocking anything.

**FSDP communication is now the priority.** It is 0.42 s/step at 4 nodes (from the
6144 1n -> 4n comparison: `fwd_bwd` 1.112 -> 1.55 s at identical sequence length),
and it is the only cost measured this session that grows with the machine.

---

## 14. Revision 6: torch.compile and its recompilations

### 14.1 The workflow

```bash
pip install tlparse                       # 0.4.3, aarch64 wheel

# jup_scaling.sbatch, opt-in and keyed on job id so runs do not interleave
if [ -n "${TORCH_TRACE_ROOT:-}" ]; then
    export TORCH_TRACE="${TORCH_TRACE_ROOT}/${SLURM_JOB_ID}"
    mkdir -p "$TORCH_TRACE"
fi

tlparse $TORCH_TRACE/dedicated_log_torch_trace_rank_0_*.log -o $OUT --overwrite
cat $OUT/*/recompile_reasons_*.json       # the machine-readable form
```

`tlparse` renders an HTML report for a browser, but it also drops
`recompile_reasons_*.json` into each compile directory. Job `1781599` produced
15,329 trace events across 120 compile entries; aggregating those JSON files is
how you get the reasons without clicking through.

### 14.2 Compile is a large memory saving, not only a speed one

At 6144 / tp=4 / 1 node, `foreach_sr`, chunked CE:

```
compile = false    step 1.267 s    reserved 82.7 GiB
compile = true     step ~1.21 s    reserved 69.2 GiB
```

**13.5 GiB.** Nothing in revisions 1-5 predicted this; compile was ranked purely
as an attack on the 2.4x dispatch multiplier from 6.5. Inductor fuses away
intermediates that eager materialises, and at this activation-dominated working
set that is worth more than the time it saves.

It also matters for the DDP-versus-FSDP question in 13.4: DDP's memory penalty
is ~11 GiB, which is about the size of this saving.

### 14.3 Three recompilation causes, from the trace

```
size mismatch at index 0. expected 8, actual 9          <- cu_seqlens length
size mismatch at index 0. expected 9, actual 4
size mismatch at index 0. expected 19132, actual 14492  <- vision patch count
size mismatch at index 0. expected 17988, actual 3328
size mismatch at index 0. expected 14492, actual 544
KeyError on self._modules['self_attn']                  <- the DecoderLayer branch
```

All three are genuinely dynamic properties of packed VLM training: the document
count per row, the image count per row, and the alternation between the two layer
types. `VisionBlock.forward` (`models/qwen3_5/model.py:498`) hit
`recompile_limit (8)` on the patch count by itself.

### 14.4 The lesson: input marking only reaches the compiled unit's own arguments

The first fix marked the varying tensors with `maybe_mark_dynamic`. It worked for
`cu_seqlens` and **did nothing** for the vision tower. Job `1781654`, with the
marks and `recompile_limit = 32`:

```
1:71.79 2:6.25 ... 32:4.47  33:1.34 34:1.33 ... 40:1.20
last reason: 0/31: tensor 'x' size mismatch at index 0. expected 16388, actual 19856
```

Thirty steps at 3-6.8 s, then a sudden drop at step 33 -- which is not the fix
working, it is dynamo exhausting the 32-entry cache and falling back to eager.

The reason the marks failed: `compile_model` compiles *blocks*, not the top-level
forward. `pixel_values` was annotated in `Qwen3_5ForCausalLM.forward`, but
`patch_embed` runs eagerly and produces a **new** tensor, and the annotation does
not survive that. Each compiled `VisionBlock` receives a fresh `x` with no
marking. `cu_seqlens` worked precisely because it is passed through unchanged and
is itself an argument to the compiled block.

**When the compiled unit is a submodule, `dynamic=True` on the module is the
lever; marking a tensor several eager ops upstream is not.**

### 14.5 The vision tower gets its own setting

`compile_model` was compiling the vision blocks with the decoder blocks'
`dynamic` and `mode`, unconditionally and without a log line. The two towers do
not behave alike: decoder blocks see a fixed hidden shape because `seq_len` is
padded to a constant, while vision blocks have the packed patch count as their
leading dimension.

New `compile_vision` config field -- `"dynamic"` (default) / `"static"` /
`"off"` -- applied to `visual.blocks` and `visual.merger`. Measured at 6144 / 1
node, 40 steps:

```
                 steps 3-16    steady (17-40)   reserved   frames hitting the limit
static           2.2-2.8 s     ~1.26 s          64.1 GiB   2
off              1.5-1.9 s     ~1.24 s          69.2 GiB   1
dynamic          1.5-2.0 s     ~1.20 s          66.8 GiB   1
```

Compiling the tower is worth it, but only symbolically. `static` spent 30 steps
at 2.2-2.8 s and became fast only at step 33, by exhausting its cache and falling
back to eager -- the same non-fix as 14.4.

### 14.6 `DecoderLayer` split into two code objects

With the tower fixed, `tlparse` named one remaining driver in both arms:
`models/qwen3_5/model.py:352`, `KeyError on self._modules['self_attn']`, 32
recompiles then eager fallback.

Dynamo keys its cache on the **code object**. A single `forward` shared by both
layer types gives them one cache, and

```python
attn = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
```

guards on a submodule that exists for 8 layers and not for the other 24, so every
alternation is a miss.

Now `_DecoderLayerBase` holds construction and `FullAttentionDecoderLayer` /
`LinearAttentionDecoderLayer` each carry their own `forward`; `DecoderLayer`
becomes a factory so the call sites and tests are unchanged. Verified: distinct
code objects, identical state-dict keys (`self_attn.k_proj.weight`,
`linear_attn.A_log`), and `hasattr(block, "self_attn")` still discriminates for
`apply_tp` and `compile_model`.

### 14.6a All four drivers closed: the measured result

Job `1781985`, 6144 / 1 node, with the vision fix, the layer split and the
`max_seqlen` quantisation:

```
1:162.5  2:6.27  3:0.919  4:11.52  5:0.924 ... 12:5.62 ... 21:3.21 ... 40:1.001
steady ~1.00 s   MFU 7.4-8.6%   72.9-84.6 TF/s   mem 50.2/62.6 GiB
zero recompile-limit hits
```

The three remaining spikes are new power-of-two `max_seqlen` buckets compiling,
which is the intended behaviour.

```
compile off, all fixes    step 1.267 s   MFU 6.4%   82.7 GiB
compile on,  all fixes    step ~1.00 s   MFU ~8.2%  62.6 GiB
```

**-21% step, +28% MFU, -20.1 GiB.**

The progression is worth keeping because the intermediate numbers were all
misleading:

```
probe1  compile, no fixes           ~1.21 s   69.2 G   limit hit
probe2  + input marks, limit 32     thrash    69.3 G   limit hit
        + compile_vision=dynamic    ~1.20 s   66.8 G   limit hit (1 frame)
        + DecoderLayer split        ~1.20 s   66.8 G   limit hit (2 frames)
        + max_seqlen quantised      ~1.00 s   62.6 G   none
```

The last fix bought 0.20 s and 4.2 GiB by itself, far more than a recompile fix
should be worth. The reason: **while `recompile_limit` is being hit, dynamo gives
up on that frame and runs it eagerly.** Every earlier "compiled" run had eager
decoder blocks. A run that hits the limit is not a slow compiled run, it is a
fast-looking eager one, and compile's benefit cannot be read off it at all.

#### The `max_seqlen` driver was self-inflicted

`max_seqlen` became a Python int in 9.6, to remove a device sync. Dynamo
specialises on int values, and it is the longest document in a packed row, so it
takes a new value most steps: `last reason: max_seqlen == 786` against both
decoder frames. `train/utils.py:round_max_seqlen` now rounds it to a power of two
-- the varlen kernels use it to size the block grid and need only an upper bound,
so the value space collapses to about seven cached possibilities for under 2x of
launched-but-idle blocks.

The sync removal still earns its place at scale (13.3, the flat tail), but this
part of it was a net negative until the quantisation landed.

### 14.7 An interaction that is not yet measured

`compile_gdn = "auto"` resolves to *on* only under FSDP: `train/config.py:228`
records that with TP alone the linear-attention layers die in the DTensor
backward with `AttributeError: 'Tensor' object has no attribute '_local_tensor'`.

**So under DDP the 24 GatedDeltaNet layers are not compiled at all** -- three
quarters of the model. Every compile measurement in this section was taken under
FSDP. DDP won the 8192 comparison in 13.4 on communication grounds; whether it
still wins once compile is on is an open question, because it may be forfeiting
most of compile's benefit.

---

## 15. Revision 7: the 16-node sweep

Full stack throughout: `foreach_sr`, bf16 master, chunked CE, `compile = true`,
`compile_vision = "dynamic"`, `compile_head_mode = "off"`, quantised
`max_seqlen`. 60 steps.

```
16 nodes, 10240                     step      fwd_bwd   MFU    TF/s   resv
a  FSDP  reshard="default"         2.433 s   2.017 s   5.8%   57.2   75.2 G
b  FSDP  reshard="never"           1.751 s   1.417 s   8.0%   79.4   79.2 G
f  FSDP  8192, reshard="default"   1.961 s   1.665 s   5.6%   55.9   61.6 G
c  DDP   8192                      failed at step 0
d  DDP   10240                     failed at step 1

reference   16n FSDP 10240, no compile   2.462 s   5.6%
            4n  FSDP 10240 + compile     1.780 s   7.9%   78.8 G
            4n  DDP  10240 + compile     1.615 s   8.7%   91.6 G  (see 15.4)
```

### 15.1 `reshard_after_forward = "never"` is worth 28%

2.433 -> 1.751 s, MFU 5.8% -> 8.0%, for 4 GiB of extra residency (75.2 -> 79.2).

`"default"` resolves to `True` in `apply_fsdp_qwen3_vl`: parameters are resharded
after forward and **all-gathered a second time during backward**. `"never"`
deletes that second gather. The gathered parameters are bf16 only -- not grads,
not moments -- so `9.41e9 / tp_size x 2 B = 4.7 GB`, which matches the measured
4 GiB almost exactly.

This is the single largest configuration win in the document, and it is one line.

### 15.2 Correction: compile and communication are not independent

Revisions 5 and 6 listed "compile" and "FSDP communication" as separate items
with additive value. They are not.

**Compile alone bought nothing at 16 nodes**: 2.462 s uncompiled -> 2.433 s
compiled, which is noise. At 1 node the same change was worth 21%. When 0.4 s of
every step is an exposed all-gather, making the compute faster does not move the
step -- the collective is what the step is waiting on.

Only both together give 1.751 s. Any plan that prices these separately will
mis-rank them, and this document did.

### 15.3 Weak scaling is NOT flat — a correction

An earlier draft of this section claimed `reshard = "never"` had flattened weak
scaling, on the strength of 16 nodes at 1.751 s against 4 nodes at 1.780 s. That
compared **16n with `never` against 4n with `default`** -- two different configs.

With the configs matched (`1782108`, `1782109`):

```
reshard="never" + compile, 10240     step      MFU    efficiency
 4 nodes                            1.497 s   9.4%    100%
16 nodes                            1.751 s   8.0%     85.5%
32 nodes                            2.027 s   6.9%     73.8%

uncompiled, reshard="default"        4n 2.121 s  16n 2.462 s   86.1%
```

**The slope is unchanged, and it accelerates**: ~93% per doubling from 4 to 16,
~86% from 16 to 32. `reshard = "never"` shifts the whole curve down ~28% without
changing its shape. The scaling concern from 13.3 stands exactly as written --
what improved is the absolute number at every point.

At this slope, 256 nodes projects to roughly 4-5% MFU. That is an extrapolation
over three more doublings from the last measured point and should be treated as a
reason to measure, not a number to plan against.

Also confirmed a third time: **8192 is worse than 10240 at scale** (5.6% vs 8.0%
at 16 nodes). The fixed per-step communication amortizes better over more tokens,
so sequence length is a throughput lever at scale and was not at one node (6.5).

Also confirmed a third time: **8192 is worse than 10240 at scale** (5.6% vs 8.0%
at 16 nodes). The fixed per-step communication amortizes better over more tokens,
so sequence length is a throughput lever at scale and was not at one node (6.5).

### 15.4 DDP at 10240 trains, and cannot checkpoint

The fastest single configuration measured: **1.615 s, 8.7% MFU, 86.1 TFLOP/s** at
4 nodes, 10% faster than FSDP with the same stack. It completed all 60 steps.

Then:

```
step 60 completes 22:42:11 at 87.4/91.6 GiB
22:42:13  ncclCuMemAlloc ... NCCL WARN Cuda failure 2 'out of memory'
          NCCL version 2.30.7 x3      <- new communicators
[scaling] done: exit 143
```

Not a training-loop OOM. Checkpointing creates new NCCL communicators, and those
allocate **outside** the caching allocator; with 91.6 of 95 GiB reserved there was
nothing left for them. Same failure class as the unchunked 8192 run in 13.1 and as
the flash_qla cubin note in `jup_scaling.sbatch`.

Two fixes were tried. `torch.cuda.synchronize(); torch.cuda.empty_cache()` before
the save in `save_checkpoint` is applied and retesting as `1782171`.

**`ac_memory_budget < 1.0` does not work on this model** (`1782110`, died at step
0):

```
RuntimeError: Cannot compute the size of
<class 'torch._library.fake_class_registry.FakeScriptObject'> on node primals_2.
A ScriptObject may hold tensors internally and the partitioner has no general way
to measure its memory.
```

Any value below 1.0 activates inductor's memory-budget partitioner, which must
estimate every node's memory -- and the `qwen3_5::gated_delta_rule`,
`causal_conv1d` and `rms_norm_gated` custom ops in `compile_ops.py` present as
`ScriptObject`s it cannot measure. So the "selective AC arrives free once compile
is on" route assumed by revisions 6 and 7 **is closed**. Real
`checkpoint_wrapper`-based AC is the only remaining path to activation-memory
reduction; it does not go through the partitioner's estimator.

DDP at 16 nodes failed at step 0-1 with a SIGTERM cascade; the originating error
did not surface in the logs and is not diagnosed. The 4-node result stands alone.

### 15.5 Where this leaves the parallelism question

13.4 argued FSDP was mostly tax. That was right about `reshard = "default"` and
wrong about FSDP as such:

```
10240, 4 nodes, compile        step     MFU    checkpoints
DDP                            1.615 s  8.7%   no (15.4, fix pending)
FSDP reshard="never" (16n)     1.751 s  8.0%   yes
FSDP reshard="default"         1.780 s  7.9%   yes
```

DDP is still ~8% faster, but FSDP with one flag is close, has 12 GiB more
headroom, and checkpoints today. The decision now turns on whether `1782110`
clears DDP's save path, and on whether DDP's 16-node failure is fixable.

---

## 16. Revision 8: the sweep to 256 nodes

Winning config throughout (`qwen3_5_9b_16n_b_fsdp_10240_noreshard.toml`):
`foreach_sr`, bf16 master, chunked CE, compile on, `compile_vision = "dynamic"`,
`compile_head_mode = "off"`, `reshard_after_forward = "never"`, 10240, 60 steps.

### 16.1 The finding that outranks the rest: `nan`

```
  4n /  8n /  16n   60 steps, no nan
 64n  (1782269)     descends to 0.0302, nan at step 25, nan for the last 35
128n  (1782270)     clean through step 43
256n  (1782543)     descends to 0.0571, nan at step 13
```

The loss descends normally first in every case -- this is not a diverging run, it
is a healthy run that hits one bad step and dies. `gnorm` goes `nan` in the same
step as the loss, so the gradients are genuinely non-finite; it is not the
logging artefact that `a87d0a2` fixed.

**Correction.** Revision 5 recorded 10240 as running "60 steps clean" and
revision 7's scaling table treated 64n as a healthy point. The 64-node run was
`nan` for its last 35 steps. The MFU and step-time numbers taken from it are
still valid -- throughput does not care whether the numbers are finite -- but the
run was not healthy and the table should not have implied it was.

Rank-steps to the first `nan`:

```
 64n   256 ranks x 25 steps =  6.4k
256n  1024 ranks x 13 steps = 13.3k
128n  512 ranks x 43 steps  = 22.0k   (survived)
```

Same order of magnitude, not a constant. That is the signature of a rare
per-rank-per-step event: the 4/8/16-node runs never hit it because 60 steps x 64
ranks is 4k rank-steps, below the threshold where it becomes likely. **Every
small-scale result in this document was collected under the detection limit of
this bug.**

#### Why one bad step is permanent

There is no non-finite guard anywhere in the step. `grep -nE "isfinite|isnan"
train/train_qwen.py` returns one hit and it is a comment. So:

1. one rank produces a `nan` gradient
2. the gradient all-reduce spreads it to every rank
3. `clip_grad_norm_mixed` computes `nan`, and `clip_grads_with_norm_` scales
   every gradient by `nan`
4. the optimizer writes `nan` into the weights, and the moments
5. every subsequent step is `nan`

Steps 2-5 are the repairable part and they are what turns a single bad sample
into a dead 10,000-step run. The fix must not introduce a host sync or a
rank-divergent branch -- `train/train_qwen.py:623` already records that "a
collective one rank skips is a hang". `gnorm` is an all-reduced global scalar, so
every rank already agrees on it; scaling the gradients by
`torch.isfinite(gnorm)` device-side skips the update on every rank at once, with
no sync and no divergence. Zeroed gradients also leave the moments decaying
rather than poisoned, which `lr = 0` alone would not.

**What produces the first `nan` is not diagnosed.** Candidates, untested: a
packed row with no loss tokens surviving the `n.clamp(min=1)` guard by a
different path; a GDN kernel edge case; bf16 master-weight overflow. The 256n
run logged `batch_util 25.5%` at step 8, so the data pipeline does produce odd
batches once 1024 ranks are pulling from it. `perf_topk` at the failing step
would name the rank and is the cheapest next measurement.

### 16.2 DDP's ceiling above 4 nodes is memory

Every DDP point at 16 nodes and above failed. Two different failures wearing the
same `exit 143`, which is why it looked like one mystery.

The real one, from `1782427` (16n, `NCCL_DEBUG=INFO`, fresh cache):

```
torch.OutOfMemoryError: Tried to allocate 4.74 GiB.
GPU 1 has 95.00 GiB of which 4.69 GiB is free. This process has 90.30 GiB in
use. Of the allocated memory 83.01 GiB is allocated by PyTorch, and 876.64 MiB
is reserved but unallocated.
```
in `_engine_run_backward`. Three numbers identify it:

- **4.74 GiB is not an activation.** `9.41e9 / tp_size x 2 B = 4.7 GiB` -- the
  flat bf16 gradient bucket `replicate()` allocates.
- **6.4 GiB is non-PyTorch** (90.30 in use minus 83.9 reserved): NCCL
  communicator buffers. At 4 nodes that figure was ~3.4 GiB. More peers, more
  channels -- **DDP's memory grows with node count**, the opposite of what
  scaling needs.
- **876 MiB reserved-unallocated**, so this is not fragmentation. There is
  nothing left.

The arithmetic explains the 4-node/16-node split. At tp=4 DDP holds params 4.7 +
grads 4.7 + bf16 moments 9.4 = **18.8 GiB regardless of scale**; FSDP at dp=16
shards grads and moments: 4.7 + 14.1/16 = **5.6 GiB**. That ~13 GiB is the whole
headroom difference, and DDP spends it on the flat bucket at exactly the moment
NCCL is also asking for more.

### 16.3 DDP is not broken at scale -- it cannot afford 10240

`1782544`, DDP at 16 nodes and **8192**, completed 60 steps, exit 0:

```
16 nodes, matched at 8192        step      MFU    TF/s   resv
DDP   (1782544)                  1.679 s   6.6%   ~66    75.8 G
FSDP  (1781996)                  1.961 s   5.6%   55.9   61.6 G

for comparison
FSDP reshard="never" @ 10240     1.751 s   8.0%   79.4   79.2 G
```

**At equal sequence length DDP wins by 17%.** The 14 GiB it spends on unsharded
grads and moments buys real speed, and this is the first clean measurement of the
FSDP tax at 16 nodes -- 13.4 estimated it from a 4-node comparison.

**And FSDP at 10240 still wins on MFU**, 8.0% against 6.6%. The slower
configuration wins by running a longer sequence, and it can run that sequence
only because sharding freed the memory.

So FSDP is not chosen for being faster -- it is not. It is chosen because the
14 GiB it returns converts into sequence length, and at these scales sequence
length is worth more than the 17% it costs. That is the "superscaling" question
from the parallelism discussion, and 16.2 plus this table is the first pair of
runs that answers it rather than estimating it.

One caveat on the DDP number: max 4.716 s against a 1.679 s median, so one late
spike. 8192 leaves ~14 GiB of headroom at 16 nodes and the image-count swing is
what killed the unchunked run at step 49 in 13.1.

### 16.4 The inductor cache bakes in process-group names, again

`1782272` (DDP 16n) died before step 1 with a different error:

```
File ".../compilecache/d16/torch/inductor/ws64/5b/c5bsph....py", line 113
  torch.ops._c10d_functional.all_reduce_.default(..., 'sum', '16')
RuntimeError: Could not resolve the process group registered under the name 16
```

Rank 15 is in TP group 3. It was running TP group 15's kernel.

Inductor writes the process-group *name* into the generated collective, and a
group is registered only on its member ranks. Each TP group therefore emits its
own kernel source -- that part is correct and expected, and a healthy cache holds
one per TP group:

```
f128/ws512    128 group names (5..132)    one per TP group, 128 nodes
f64/ws256      64 group names (5..68)     one per TP group,  64 nodes
ddpdbg/ws64    16 group names (5..20)     one per TP group,  16 nodes
```

The bug is that the FX graph cache **key** does not separate them: two ranks with
structurally identical graphs collide on one entry, and one is handed the other's
artifact. It is a race -- `d16` lost it, `ddpdbg` on a fresh directory did not --
which is why every FSDP run above was exposed to this and none happened to fire.

The sbatch already keys the cache on world size, from the previous instance of
this bug (a 1-node cache poisoning a 512-GPU job, `jup_scaling.sbatch:70`). World
size does not separate TP groups. Fixed in `numa_wrapper.sh`, the only hook that
runs per rank and knows the node index:

```bash
if [ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/n${GROUP_RANK:-${SLURM_NODEID:-0}}"
fi
```

One cache per node is one per TP group only while `tp_size == NGPUS`, which is
true here because TP is intra-node. If `tp_size` ever drops below 4 this needs
the TP group index instead; the comment in the file says so.

### 16.5 256 nodes runs, and has no usable number yet

Three attempts. The first two never reached the model:

```
rank593: torch.cuda.set_device(self.local_rank)
torch.AcceleratorError: CUDA error: context is destroyed
```

`Trainer.__init__:99`, before anything is built. `cudaErrorContextIsDestroyed` at
`set_device` is a sick GPU -- both were bad-node draws, not a limit at 256.

The third (`1782543`) ran, and hit the 25-minute walltime at step 42 with step
times **still falling**: 12.5 s at step 2, 9.2 by step 11, 7.0-7.5 at step 42.
Nothing here is a steady state, and the run was `nan` from step 13 in any case.
**There is no 256-node data point.** A rerun needs a longer walltime and 16.1
fixed first.

### 16.6 The scaling curve, with the correction applied

```
FSDP, 10240, reshard="never", compile      step      MFU    weak-scaling eff.
  4 nodes                                  1.497 s   9.4%   100%
 16 nodes                                  1.751 s   8.0%    85.5%
 32 nodes                                  2.027 s   6.9%    73.8%
 64 nodes                                  2.424 s   5.8%    61.8%   nan from step 25
128 nodes                                  3.728 s   3.8%    40.2%
256 nodes                                  no steady state reached
```

Memory is flat at 77-82 GiB across all of them, which is the useful part: the
configuration does not get closer to OOM as it scales. The knee is at 128 and it
is steep -- 61.8% to 40.2% for one doubling. Diagnosing it needs a profiler trace
at 64 and 128 nodes, and that is now second in line behind 16.1.

---

## 17. Revision 9: the `nan` closed, and qwen3-vl

### 17.1 The guard, and what it revealed

`train/utils.py:zero_grads_if_nonfinite_`, called after clipping (which is what
turns one non-finite gradient into all of them):

```python
n = total_norm
if isinstance(n, DTensor):
    n = n.full_tensor()
bad = ~torch.isfinite(torch.as_tensor(n).reshape(()))

for p in parameters:
    if p.grad is None:
        continue
    g = p.grad
    local = g.to_local() if isinstance(g, DTensor) else g
    local.masked_fill_(bad, 0.0)
```

Three decisions worth keeping:

- **`masked_fill_`, not a multiply.** Scaling by `isfinite(gnorm)` is the obvious
  implementation and it is wrong: `nan * 0.0` is `nan`, so it would look like a
  fix and leave the poison in place. There is a test pinning exactly this.
- **No host sync, no divergent branch.** `total_norm` is already a global
  reduction, so every rank fills identically. An `if` on a host float would be
  both a sync and the hang `train_qwen.py:623` warns about.
- **The optimizer still steps.** Zeroed gradients mean a momentum-only step: the
  moments decay and the parameters move along the existing `exp_avg`. A true skip
  needs the host to know, i.e. a sync. A test asserts `exp_avg` comes back as
  `good * 0.9` and everything stays finite.

Config: `skip_nonfinite_grads` (default true). It needs `max_grad_norm > 0`,
which is where the global norm comes from, and warns at init if that does not
hold. Counter is device-side, read a step late with everything else
(`_LOG_SLOTS` 8 -> 9), shown as `skips N` only when non-zero.

Validation on the two configurations that had died:

```
                          before guard      with guard
synth  16n x300           nan @ step 16     300 steps, 0 nan, skips 1, loss -> 0.0205
plotqa 16n x300           nan @ step 283    300 steps, 0 nan, skips 1, loss -> 0.1015
```

**Exactly one skipped step per 300, on each of two independent datasets.** Step
1.747 s against the 1.751 s baseline -- no measurable cost.

So the `nan` is a **rare transient, not a corruption**: training resumes normally
the next step, weights and moments intact. That is what 16.1 could not
distinguish, and it means root-causing is now optional rather than blocking.

### 17.2 Correction to 16.1: scale was not the isolated variable

16.1 presented the `nan` as scale-correlated. Every run in that section also used
`foreach_sr` with bf16 master weights -- verified from the logged config of all
five -- so scale and the optimizer co-varied and the section should have said so.

Controls, plotqa 16n x300, one variable each:

```
SR on,  bf16 master     nan @ 283     (baseline)
SR off, bf16 master     nan @ 176
SR off, float32 master  nan @  57
```

All three die, and **fp32 master dies earliest**. Stochastic rounding and bf16
master precision are both exonerated. So is the dataset: plotqa and synth both
fail. And the rank-steps model from 16.1 is dead too -- 16 nodes reproduced at
step 16 (1k rank-steps) where 128 nodes survived 22k. The 57/176/283 spread under
identical data and learning rate is a random per-step event, not a threshold.

What remains unexplained: something emits one non-finite gradient per ~300 steps
at 64 ranks. `train/nonfinite_skips` is the watch -- a steady tick is the known
behaviour, a steep climb means the cause changed.

### 17.3 A real dataset: `plotqa_cot`

The published subset is two halves energon cannot join: `media/` is a
CrudeWebdataset of the *images* (8,212, keyed by bare filename) and
`plotqa_cot.jsonl` holds 16,256 conversations referencing them. The `.nv-meta`
describes the image store, not training samples.

`utils/prepare_nemotron_energon.py` joins them into shards of `{key}.png` +
`{key}.json` (16,256 written, 0 skipped, 761 MB); `cooker_nemotron` in
`data/energon_dataloader.py` strips the filename and metadata off the image part
and keeps `<think>` spans verbatim. Generalises to the other Nemotron subsets.

```
plotqa, 4 nodes, 10240    step 1.18-1.26 s   MFU 11.2-11.9%   mem 55.5/76.2 GiB
synth,  4 nodes, 10240    step 1.497 s       MFU  9.4%        mem 82.4 GiB
```

**Do not read that MFU as a 25% gain.** `flops_per_token` is computed once from
config and `seq_len` (`train_qwen.py:150`), and `vision_flops` sizes the tower
from `seq_len`, not from the actual image count -- so a run with fewer images per
row is billed for vision work it never did. MFU is not comparable across datasets
of different image density. The **26 GiB of headroom is measured** and real, but
17.6 shows it is partly a tokenizer artifact and does not transfer across models.

### 17.4 qwen3-vl migration

Six qwen3.5 forward-path optimisations ported to `models/qwen3_vl/model.py`:
section timing, the sync-free loss, `_assert_async` guards, `max_seqlen` from the
trainer with a quantised fallback, `maybe_mark_dynamic`, and `packed_positions`.
32 existing plus 16 new tests pass.

The trainer was **already computing** `max_seqlen` and `vision_max_seqlen`
host-side for every model (`train_qwen.py:538,544`); qwen3-vl's `**kwargs` swallowed
both and recomputed them with `.item()`. Half that fix had been in the tree unused.

`compile_model` turned out to be structural rather than model-typed -- it walks
`language_model.layers`, `visual.blocks`, `visual.merger`, `lm_head` -- so compile,
`reshard_after_forward` and the FSDP branch already applied to qwen3-vl. The
audit's "`apply_fsdp` has no `Qwen3_5` branch" note came from a stale local
checkout and is wrong; `infra.py:248` has handled both all along.

**Compile on qwen3-vl is a memory lever, not a speed one:**

```
Qwen3-VL-8B, 4 nodes, 10240, plotqa       step      peak alloc/resv
compile=true,  compile_vision="dynamic"   5.752 s   47.7 / 61.1 GiB
compile=true,  compile_vision="off"       5.746 s   47.9 / 62.5 GiB
compile=false                             5.906 s   65.0 / 79.8 GiB
```

2.6% of step time, **18.7 GiB** of peak reserved -- the same shape as the 13.5 GiB
compile was worth on qwen3.5 (14.2). And `compile_vision` does nothing for VL
either way, where on qwen3.5 it was what stopped the recompile thrash.

First forward decomposition qwen3-vl has ever had:

```
8B, 10240   layers 355.1ms 70%  visual 93.6ms 19%  lm_head 26.4ms 5%  loss 19.1ms 4%
2B,  4096   visual  85.1ms 47%  layers 79.0ms 43%  lm_head  6.9ms 4%  loss  7.8ms 4%
```

**At 2B the vision tower is the single largest forward cost** -- larger than the
whole language model. The 2B pairs a 24-layer tower with a 28-layer text model, so
the tower barely shrinks while the text side does. The "is the vision tower
efficient" question that has been open since revision 5 is a 2B question, not an
8B one.

### 17.5 Architecture ablation: the hybrid is the *slower* configuration

`from_pretrained` took `load_weights`, wired to `random_init` in `train/utils.py`.
It had loaded safetensors unconditionally while `random_init` re-initialised a
moment later, and for an architecture ablation it could never have worked at all:
an all-full-attention model and the hybrid checkpoint do not share state-dict keys
(`self_attn.*` vs `linear_attn.*`), and a missing key is a hard error, correctly.
`full_attention_interval = 1` then gives all-full-attention with no model change
(`model.py:423`). The model dir is `config.json` plus symlinked tokenizer files.

```
hybrid          9,409,813,744 params   24 LinearAttention + 8 FullAttention
all-full-attn   9,201,416,944 params   32 FullAttention           -2.2%
```

GDN layers are *larger* than full-attention ones at 4 KV heads, so removing them
shrinks the model slightly. Both arms random-init, identical config, identical
packing (nsamples 23, batch_util 99.1%):

```
4 nodes, 10240, plotqa        step      MFU     TF/s    decoder layers
all-full-attention (32)       1.182 s   12.9%   127.4   217.2 ms
hybrid (24 GDN + 8 full)      1.319 s   10.7%   105.4   281.0 ms
```

**All full attention is 10% faster overall and 23% faster in the decoder.** The
reason is document length: 23 documents in a 10240 row averages ~445 tokens, and
varlen attention is per-document, so full attention pays O(445^2) -- trivial.
GatedDeltaNet's O(T) advantage only appears on long documents; on short packed
ones it is simply a more expensive kernel.

The hybrid's value is **contingent on document length**, not free. Which
architecture wins at SFT lengths is now a measurable question, and the ablation
harness exists.

### 17.6 Why Qwen3-VL-8B costs 5x Qwen3.5-9B: the tokenizer

Five hypotheses died by measurement before the right one: the section timer
(5.745 vs 5.752 s with it off), the 27-layer dynamic-compiled vision tower
(17.4), FSDP resharding (`train_qwen.py:281` does pass the policy through), the
architecture (17.5 -- full attention is *faster*), and image-token density
(identical, 687 tokens/image at patch 16 / merge 2 for all three models).

The answer is the vocabulary. Same samples, same images:

```
Qwen3.5-9B    vocab 248,077   median 1818 tok/sample   -> ~6.2 samples per 10240 row
Qwen3-VL-8B   vocab 151,669   median 5157 tok/sample   -> ~1.8 samples per 10240 row
```

3.4x, which matches the observed `nsamples` 23 vs 7 exactly. And because varlen
attention is **per-document and O(L^2)**, longer documents compound with the
all-full-attention stack:

```
              docs/row   avg doc   attn work per layer   full-attn layers
Qwen3.5-9B       6.2      ~1650    6.2 x 1650^2 = 17M           8
Qwen3-VL-8B      1.8      ~5700    1.8 x 5700^2 = 59M          36
```

3.5x per layer times 4.5x the layers: **~15x more attention work for the same
10240 tokens**. Not a defect, and nothing to do with the migration.

This is also why 17.5's ablation did not reproduce it: that model kept qwen3.5's
tokenizer, so its documents stayed short, where full attention is cheap. The two
factors are multiplicative and only one was varied.

**Consequence for the scaling numbers in this document: they are tokenizer-bound
as much as model-bound.** A step time at "seq_len = 10240" means different amounts
of text for different models, and comparing MFU across them is comparing different
workloads.

### 17.7 MareNostrum 5

Production qwen3-vl on MN5 is at `5c355a9`, which predates `de38c80` -- so it does
not have chunked cross-entropy, written *for* qwen3-vl and worth 18 GiB at 8192.
It is 0 commits behind its own `origin/main`; it has simply not fetched.

The test there runs from an isolated copy at
`/gpfs/scratch/ehpc543/tockier/vlmtest/repo`, not the production checkout, which
is untouched. MN5 has no outbound network, so the dataset went across as a
1000-sample single-shard subset (58 MB).

**The environments are not comparable for compile work:** JUPITER is torch
2.14.0+cu130, MN5 is torch 2.11.0+cu126. Three minor versions apart.

---

## 12. Appendix: reproduction

Runs are in `bsc_runs/scaling_9b`; wandb operates offline on JUPITER
(`WANDB_MODE=offline`, `jup_scaling.sbatch:39`) with run directories under
`/e/project1/open-sci-mm/ockier1/cache/wandb/runs/wandb`. Sync with:

```bash
cd /e/project1/open-sci-mm/ockier1/cache/wandb/runs/wandb
for r in $(ls -dtr offline-run-*); do
    ls "$r"/run-*.wandb.synced >/dev/null 2>&1 && continue
    wandb sync "$r"
done
```

A live run reports `transactionlog: error reading: unexpected EOF` because its
transaction log has no terminating record. The data uploads regardless, and the
`.synced` marker is only written once the job exits — so re-running the loop
afterwards picks up the tail.

Single-node probe:

```bash
salloc --account=open-sci-mm --partition=booster --nodes=1 \
       --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 --time=02:00:00
./scripts/scaling/debug_1node.sh --training.adamw-impl foreach --training.compile true
```

Multi-node sweep:

```bash
SCALING_TIME=25 ./scripts/scaling/submit_sweep.sh 16 32 64 128 256
```

Model facts used throughout: 9,409,813,744 parameters; vocab 248320; hidden 4096;
intermediate 12288; 32 layers with `full_attention_interval = 4`, so 8 full
attention and 24 linear; head_dim 256 with 16 query and 4 KV heads; GDN with 16
key heads x 128 and 32 value heads x 128; peak bf16 taken as 989.4 TFLOP/s per
GPU (`train_qwen.py:156`).
