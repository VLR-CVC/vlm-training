# Closing the performance gap — execution plan

Companion to `PERFORMANCE.md`. That document says what was measured and why. This
one says what to do next.

**Revision 9 (2026-09-14).** **Item 0 is done.** The `nan` guard is in and
validated on both datasets: one skipped step per 300, training continues normally,
no measurable cost. Production is unblocked. The cause is still unknown but is now
a counter rather than a fatality.

Three things changed the shape of this list. **The hybrid architecture is the
slower configuration** at short document lengths — all-full-attention is 10%
faster (`PERFORMANCE.md` §17.5). **Step times are tokenizer-bound**, not just
model-bound: qwen3.5's 248k vocab packs 3.4× more text per row than qwen3-vl's
152k, and with O(L²) varlen attention that is most of why VL-8B costs 5× (§17.6).
And **at 2B the vision tower is the largest single forward cost** (47%), which
finally makes item 3 concrete.

**Revision 8 (2026-09-14).** The sweep reached 256 nodes and found that **runs
go `nan` and never recover**, at 64 nodes (step 25) and 256 nodes (step 13).
Nothing else on this list matters until that is fixed — a configuration that
cannot survive 60 steps at 64 nodes cannot run 10,000 at any scale. Item 0.

The parallelism question is also settled, in FSDP's favour but not for the reason
this document assumed: DDP is **17% faster than FSDP at the same sequence
length** (1.679 s vs 1.961 s at 16 nodes / 8192) and simply cannot afford 10240
above 4 nodes — a memory ceiling, not a communication one. FSDP wins because the
14 GiB it returns buys the sequence length that wins on MFU. `PERFORMANCE.md`
§16.

**Revision 7 (2026-09-13).** `reshard_after_forward = "never"` is worth **~28% at
every scale** — and at 4 nodes it gives the session's best result: **10240,
1.497 s, 9.4% MFU, 92.9 TFLOP/s**, which beats DDP *and* checkpoints. It does
**not** flatten weak scaling; an earlier draft of this line said it did, from
comparing two different configs (`PERFORMANCE.md` §15.3). Two corrections worth
carrying: compile and communication are not independent items — compile alone
bought *nothing* at 16 nodes because an exposed all-gather was hiding it (§15.2).

**Revision 6 (2026-09-13).** `torch.compile` is on and its recompilations are
fixed — two structural causes, both found with `TORCH_TRACE` + `tlparse` and both
now closed (`PERFORMANCE.md` §14). Compile is also a **13.5 GiB memory saving**,
which no earlier revision predicted and which changes the DDP-vs-FSDP arithmetic.

**Revision 5 (2026-09-13). `seq_len = 10240` runs**, at 4, 8 and 16 nodes, 60
steps each, clean. That was the goal the investigation started from. Best
configuration measured: **10240 at 4 nodes, 6.6% MFU, 65.4 TFLOP/s/GPU** — against
3.8% at the start, at 1.7× the sequence length.

The bottleneck has moved twice today. It was never the kernels (revision 3); it
was briefly the optimizer (revision 4, fixed); it is now **FSDP communication**,
which is the only cost measured this session that grows as hardware is added.

Predicted numbers are **[P]**, measured **[M]** with the job that produced them.

---

## The winning configuration

```toml
adamw_impl = "foreach_sr"        # train/adamw_sr.py
adamw_stochastic_round = true    # not optional at lr 2e-5, see PERFORMANCE.md §4.1
master_dtype = "bfloat16"        # 8 B/param
loss_chunk_mb = 512              # chunked cross-entropy
data_parallel = 'fsdp'
tp_size = 4
seq_len = 10240
compile = true                   # §14: -20 GiB and -21% step once recompiles are fixed
compile_vision = "dynamic"       # the tower's patch count changes every step
compile_head_mode = "off"        # skips autotuning the 248320-wide lm_head GEMM
reshard_after_forward = "never"  # §15.1: -28% at 16 nodes for 4 GiB
```

`configs/jupiter/scaling/qwen3_5_9b_16n_b_fsdp_10240_noreshard.toml`.

```
10240, full stack          step      MFU    resv      checkpoints
 4n FSDP reshard=never    1.497 s   9.4%   82.4 G    yes   <- best
 4n DDP                   1.615 s   8.7%   91.6 G    no — NCCL OOM at save (§15.4)
16n FSDP reshard=never    1.751 s   8.0%   79.2 G    yes
 4n FSDP reshard=default  1.780 s   7.9%   78.8 G    yes
```

**FSDP with one flag now beats DDP** on speed, headroom and checkpointing at
once, which reverses the §13.4 conclusion. The parallelism question is settled
for 4 nodes unless DDP's save path is fixed and its 16-node failure diagnosed.

`configs/jupiter/scaling/qwen3_5_9b_sr_10240_ce.toml`.

## Where the step goes at 10240 / 4 nodes (2.121 s)

| | est. | share |
|---|---|---|
| decoder layer kernels | ~0.45 s | 21% |
| **FSDP + TP collectives + dispatch** | **~0.80 s** | **38%** |
| vision tower | ~0.55 s | 26% |
| lm_head + loss | ~0.12 s | 6% |
| optimizer + logging tail | 0.197 s | 9% |

Scaled from the 6144 decomposition in `PERFORMANCE.md` §6.5 plus the 1n→4n delta
in §13.4. The collective share is the one that grows with node count.

---

## 0. ~~The `nan`~~ — done, and it was a transient

**[M]** Guard in `train/utils.py:zero_grads_if_nonfinite_`, on by default
(`skip_nonfinite_grads`). Validated on the two configurations that had died:

```
                      before guard      with guard
synth  16n x300       nan @ step 16     300 steps, 0 nan, skips 1
plotqa 16n x300       nan @ step 283    300 steps, 0 nan, skips 1
```

One skipped step per 300 on each, loss keeps descending, 1.747 s against the
1.751 s baseline. **The `nan` is a rare transient, not a corruption** — the next
step trains normally, so the weights and moments were never damaged.

Controls ruled out everything that was suspected: stochastic rounding (nan @ 176
without it), bf16 master weights (fp32 died *earliest*, @ 57), and the dataset
(both fail). Cause still unknown; `train/nonfinite_skips` is the watch. A steady
tick is expected, a steep climb means something changed.

**What is left of this item:** nothing blocking. Finding the cause is worth doing
when convenient — `perf_topk` at a skipped step would name the rank — but a
10,000-step run now costs ~30 skipped steps instead of dying.

## 0b. Old item 0, kept for the reasoning

**[M]** 64 nodes `nan` at step 25 and stays `nan` for 35 steps; 256 nodes `nan`
at step 13; 128 nodes clean through 43; 4/8/16 nodes clean for 60. The loss
descends normally first in every case, so this is one bad step in a healthy run,
not divergence. `gnorm` goes `nan` in the same step, so the gradients really are
non-finite.

Rank-steps to first `nan`: 6.4k (64n), 13.3k (256n), 22k survived (128n). A rare
per-rank-per-step event. Every measurement in `PERFORMANCE.md` below 32 nodes was
taken under this bug's detection limit.

**0.1 Make one bad step survivable — small, do it first.** There is no
non-finite guard in the step, so a single `nan` gradient is all-reduced to every
rank, multiplied through `clip_grads_with_norm_`, and written into the weights
and the moments. Scaling the gradients by `torch.isfinite(gnorm)` device-side
skips the update on every rank at once: `gnorm` is already an all-reduced global
scalar so every rank agrees, and there is no host sync and no rank-divergent
branch (`train/train_qwen.py:623` — "a collective one rank skips is a hang").
Zeroing the gradients rather than setting `lr = 0` also leaves the moments
decaying instead of poisoned. ~10 lines.

This does not fix the cause. It converts a dead run into a logged skipped step,
which is the difference between a 10,000-step job finishing and not.

**0.2 Find the cause.** Not diagnosed. `perf_topk` at the failing step names the
rank; that is the cheapest next measurement. Candidates, untested: a packed row
with no loss tokens reaching the loss by a path the `n.clamp(min=1)` guard does
not cover; a GDN kernel edge case; bf16 master-weight overflow. The 256n run
logged `batch_util 25.5%` at step 8, so the pipeline does produce odd batches
once 1024 ranks pull from it.

**0.3 Then re-run the curve.** Every scaling number above 32 nodes was collected
from a run that was or would become `nan`. The step times stand; the claim that
the configuration is production-ready does not.

## 1. ~~FSDP communication~~ — mostly solved by one flag

`reshard_after_forward = "never"`: **2.433 → 1.751 s at 16 nodes, MFU 5.8 → 8.0%,
for 4 GiB.** `"default"` re-gathers every parameter a second time during
backward; `"never"` deletes that gather, and the gathered params are bf16 only
(`9.41e9 / tp × 2 B = 4.7 GB`, matching the measured 4 GiB).

Weak scaling is **unchanged and accelerating**: 100% / 85.5% / 73.8% at 4 / 16 /
32 nodes with the flag, against 86.1% at 16n without it. The flag moves the curve
down, not flat. §15.3's concern about extrapolating to 256 nodes stands and is now
measured one doubling further out.

Remaining, in order:

- **1.2 Prefetch depth** — still unexplored, and now the main lever if the 32-node
  point bends.
- **1.3 A profiler trace at 16 or 32 nodes** — to see whether what remains is
  exposed or bandwidth-bound.
- **1.4 HSDP** — only if the curve still degrades after 1.2.

## 1b. The old framing of this item

**[M]** 0.42 s/step at 4 nodes, from the 6144 comparison at fixed sequence length
(`fwd_bwd` 1.112 s at 1 node → 1.55 s at 4). Weak scaling at 10240 shows it
accelerating: efficiency 100% → 95% → 86% across 4 → 8 → 16 nodes, with **all** of
the loss in `fwd_bwd` and none in the tail.

With `tp_size = 4` on 4-GPU nodes, TP consumes the intra-node NVLink domain
entirely, so **every FSDP rank is on a different node and every parameter
all-gather crosses InfiniBand.** That is the structural reason this is expensive.

In order of effort:

**1.1 `reshard_after_forward = "never"` — one config line, try first.**
Currently `"default"`, which resolves to `True` in `apply_fsdp_qwen3_vl`, meaning
parameters are re-sharded after forward and **all-gathered a second time during
backward**. Setting it to `"never"` removes that second gather entirely.

The cost is holding gathered parameters resident. Gathered params are bf16 only
(not grads, not moments): `9.41e9 / tp_size × 2 B = 4.7 GB`, against 0.29 GB
sharded at dp=16 — so **[P]** ~+4.4 GB against ~15 GB of headroom at 10240. It
should fit, and it is already exposed in the toml.

**[P]** up to half the parameter all-gather traffic. Measure at 4 and 16 nodes;
the benefit should grow with node count while the memory cost shrinks.

**1.2 Prefetch depth.** FSDP2 prefetches one module ahead by default. On a long
IB ring one module of compute may not cover one all-gather. `set_modules_to_forward_prefetch`
and `set_modules_to_backward_prefetch` deepen it explicitly. ~20 lines in
`apply_fsdp_qwen3_vl`, no memory cost beyond the extra in-flight buckets.

**1.3 A profiler trace at 8 or 16 nodes.** The accelerating cost is *consistent
with* failed overlap but does not demonstrate it. This is the measurement that
tells us whether 1.1 and 1.2 are the right fixes or whether the collectives are
simply bandwidth-bound. Do it before 1.4.

**1.4 HSDP.** Shard parameters within a group of nodes and replicate across
groups, so all-gathers stay inside the group and only the gradient reduce goes
wide. `fully_shard` takes a 2D mesh for this. Real work, and only justified if
1.1–1.3 leave the curve still accelerating.

## 2. ~~`compile = true`~~ — done, with two structural fixes

**Landed.** At 6144/1n: step 1.267 → ~1.20 s and reserved memory **82.7 → 69.2
GiB**. The memory was the surprise; compile had been ranked purely as an attack
on the dispatch multiplier.

Two recompilation causes, both structural rather than tuning:

- **The vision tower was compiled with the decoder blocks' settings.** Its
  leading dimension is the packed patch count (544 → 19132 within one 40-step
  run), so a static trace recompiles until it evicts. New `compile_vision`
  field, default `"dynamic"`. Measured: dynamic ~1.20 s steady vs off ~1.24 vs
  static ~1.26 (and static only got fast by exhausting its cache and falling
  back to eager).
- **`DecoderLayer.forward` was one code object for two layer types.** Dynamo
  caches on the code object, so the `self.self_attn if ... else self.linear_attn`
  branch guarded on a submodule missing from 24 of 32 layers — a miss on every
  alternation. Split into `FullAttentionDecoderLayer` /
  `LinearAttentionDecoderLayer` with a `DecoderLayer` factory; state-dict keys
  unchanged.

The generalisable lesson: **`maybe_mark_dynamic` only reaches tensors that are
arguments of the compiled unit.** It fixed `cu_seqlens` (passed straight into the
blocks) and did nothing for `pixel_values`, because `patch_embed` runs eagerly and
hands the block a fresh, unannotated tensor. For a compiled submodule the lever is
`dynamic=True` on the module.

**Closed, negatively:** `ac_memory_budget` became live once compile was on, but
any value below 1.0 fails to compile on this model — inductor's memory-budget
partitioner cannot size the `qwen3_5::*` custom ops. It must stay at `1.0`.

## 2b. The interaction that needs measuring next

`compile_gdn = "auto"` resolves to *on* only under FSDP; `train/config.py:228`
records that with TP alone the linear-attention layers die in the DTensor backward
(`AttributeError: 'Tensor' object has no attribute '_local_tensor'`).

**So under DDP the 24 GatedDeltaNet layers do not compile at all.** Every compile
number above was taken under FSDP. DDP won the 8192 comparison on communication;
whether it still wins with compile on is unmeasured, and it is the next thing to
run — it decides the production parallelism choice.

Worth revisiting now for three reasons: it attacks the 2.4× dispatch multiplier
measured in §6.5 (141 ms of forward at 6144); it makes `ac_memory_budget` live, so
selective activation checkpointing arrives without new code; and
`torch._inductor.config.reorder_for_compute_comm_overlap` is a compile-only pass
that targets exactly item 1.

**Known recompilation triggers in this codebase**, in the order I'd expect them to
bite:

- **The vision tower.** `pixel_values` shape varies with the packed image count,
  which swings from 2 to 184 samples per batch. This is the worst offender and the
  reason to compile *blocks*, not the top-level forward. `compile_model` already
  compiles per decoder block; keep it that way and leave `visual` in eager.
- **`cu_seqlens` length** varies with the document count per packed row. It is an
  input to every block. Mark it dynamic (`torch._dynamo.mark_dynamic`) rather than
  letting dynamo specialise then recompile.
- **`max_seqlen`, a Python int.** I made this a host int this afternoon to remove
  a device sync, which under compile turns it into a specialisation point.
  `torch._dynamo.config.specialize_int` defaults to False so it should become a
  symint after the first recompile rather than one graph per value — but this is
  exactly the interaction to verify, not assume.
- **The `lm_head` / loss region**, whose shapes follow `seq_len` (stable) but whose
  chunk count follows the vocabulary (stable). Low risk. `compile_head_mode = "off"`
  exists if it misbehaves.

**The experiment, in this order:**

1. `compile = true`, `compile_head_mode = "off"`, `TORCH_LOGS="recompiles"`, 60
   steps at 10240 / 4 nodes. Read the recompile reasons — do not guess at them.
2. Whatever the log names, fix it: `mark_dynamic` on `cu_seqlens`, raise
   `torch._dynamo.config.cache_size_limit` (default 8) and
   `accumulated_cache_size_limit` (default 256), or exclude a region.
3. Only then set `ac_memory_budget` deliberately — it has been sitting at `1.0`
   (recompute nothing) and inert, and turning compile on makes it live. Leaving it
   at 1.0 while enabling compile changes memory behaviour by accident.
4. Then try `reorder_for_compute_comm_overlap` against item 1.

Note the inductor cache hazard already documented in `jup_scaling.sbatch`:
compiled collectives bake process-group *names* in, so the cache must stay keyed
on world size. It already is.

## 3. The vision tower — measure it before touching it

~26–28% of the step. `train_vit` stays on: it will be judged on downstream
benchmark results, and the expectation is that it earns its place. So the question
is not whether to run it but whether it runs *well* — and nothing has measured
that. The decoder layers hit 42–45% of peak; if the tower is at 15% there is a
real problem, and if it is at 45% then 26% is simply what it costs.

Extend `bench_layers.py` to the vision blocks, or add finer section marks inside
`VisionModel.forward`. Cheap, and it decides whether this is an item at all.

## 4. Selective activation checkpointing — no longer blocking

Still absent (`grep` finds one hit for `checkpoint_wrapper` and it is a docstring
at `train/config.py:188`). Revisions 1–4 called it the item 10240 depended on. It
was not — 10240 runs without it.

Still worth having: 80 of 95 GiB is usable but not generous, the image-count swing
is what killed the unchunked 8192 run at step 49, and this is what opens 12288 and
beyond. **[P]** selective matmul-only policy cuts activation memory ~2× for ~8%
compute. ~80 lines. Also arrives free via `ac_memory_budget` once item 2 lands, so
sequence them together.

## 5. TP + sequence parallelism

No SP today: norms are `NoParallel()` and both `out_proj` and `mlp.down_proj` use
`output_layouts=Replicate()`, which emits an **all-reduce**. Two consequences:
every norm and residual is computed and stored 4× at tp=4, and **`async_tp = true`
is a no-op** because `_micro_pipeline_tp` matches `all_gather → mm` and
`mm → reduce_scatter`, never an all-reduce.

Moderate work in `_apply_tp_to_decoder_qwen3_5` and `_shard_gated_delta_net`. Do it
after 1 and 2.

---

## Done, with measurements

| | result |
|---|---|
| `seq_len = 10240` | **[M]** runs at 4/8/16 nodes, 60 steps clean (`1781312`/`1781429`/`1781430`) |
| `AdamWSR` (`train/adamw_sr.py`) | **[M]** step 1.790 → 1.267 s single-node, +41% throughput, keeps stochastic rounding |
| chunked cross-entropy | **[M]** 18 GiB for 1.7% of step time; turned an OOM at step 49 into a clean run |
| thirteen host syncs removed | **[M]** tail flat at ~0.2 s across 4n/8n/16n, against sweep A's 0.665 → 0.706 |
| bf16 master weights | **[M]** 8 B/param instead of 16 |
| `compile = true` + recompile fixes | **[M]** step 1.267 → ~1.20 s and **13.5 GiB** reserved, at 6144/1n |
| `compile_vision` lever | **[M]** dynamic 1.20 s vs off 1.24 vs static 1.26 |
| `DecoderLayer` split into two code objects | **[M]** removes a recompile driver |
| `round_max_seqlen` quantisation | **[M]** the last driver; worth 0.20 s and 4.2 GiB because the limit had been forcing eager fallback |
| `reshard_after_forward = "never"` | **[M]** −28% at every scale; 4n/10240 → 1.497 s, 9.4% MFU |
| `empty_cache()` before checkpoint | fixes the NCCL-OOM-at-save that killed DDP 10240 (§15.4) |
| `set_loss_chunk_mb` in qwen3-VL | **[M]** every Qwen3-VL run had been dying at startup with ImportError |

## Dead — do not spend time here

| | why |
|---|---|
| GDN chunk size | 64/128/256 → 1.07/1.06/1.07 ms **[M]** |
| Layout churn (`repeat_interleave`, transposes) | ~13 GB/forward ≈ 4.4 ms ≈ 1% |
| `native_kernels` / custom-op backward | the whole GDN kernel is 29 ms of the step |
| Rewriting the fla kernels | a linear-attention layer runs at 411 TFLOP/s, 42% of peak |
| Trimming the console log | 0.70 ms of a 1737 ms step **[M]** |
| `adamw_impl = "foreach"` (no SR) | 0.06 s faster than `foreach_sr` and does not train at lr 2e-5 |
| `train_vit = false` | not a lever — the capability is wanted, see item 3 |

---

## Order of work

1. **Re-run the scaling curve with the guard in.** Every point above 32 nodes was
   collected from a run that was or would become `nan`, so the step times stand
   but nothing above 32 was a healthy run. 256 also needs a longer walltime — the
   only attempt that ran hit 25 minutes at step 42 with step times still falling.
2. **The knee at 128 nodes.** 61.8% weak-scaling efficiency at 64 to 40.2% at
   128, one doubling. Profiler trace at 64 and 128 is the measurement; prefetch
   depth (1.2) and HSDP (1.4) are the candidate fixes.
3. **The vision tower, at 2B.** No longer speculative: it is 47% of the forward
   there, more than the whole language model (`PERFORMANCE.md` §17.4). At 8B it
   is 19%. Measure its achieved TFLOP/s before optimising anything else in the
   VL stack.
4. **Decide the attention architecture per workload.** §17.5 measured the hybrid
   as 10% *slower* than all-full-attention at ~445-token documents, and §17.6
   shows the crossover is driven by document length. Pretraining and SFT may want
   different answers. The ablation harness exists now (`random_init` +
   `load_weights=False`, `full_attention_interval = 1`), so this is a one-job
   question at any document length.
5. **Sync MN5 and re-test.** Production qwen3-vl there is at `5c355a9`, missing
   chunked cross-entropy — written for qwen3-vl, worth 18 GiB at 8192 — plus the
   guard and the migration. It is 0 commits behind its own `origin/main` and has
   simply not fetched.
6. **Find the `nan` cause** (0). Not blocking any more.
6. **Selective AC is now required, not optional.** `ac_memory_budget < 1.0` is
   incompatible with this model: it activates inductor's memory-budget
   partitioner, which cannot size the `qwen3_5::*` custom ops
   (`FakeScriptObject`, job `1782110`). The free route is closed;
   `checkpoint_wrapper` is the only path left to activation-memory reduction.
7. TP+SP, then `async_tp`.

## Still not known

- Whether the FSDP collectives are exposed or bandwidth-bound. The accelerating
  curve is consistent with both.
- Whether the vision tower is efficient, now sharpened: at 2B it is 47% of the
  forward, more than the whole language model, and no measurement of its achieved
  TFLOP/s exists at either size.
- Where the hybrid/full-attention crossover sits as document length grows. §17.5
  measured one point (~445-token documents, full attention wins by 10%).
- Whether `foreach_sr` trains correctly on real data. The mechanism is right and
  covered by 9 unit tests; the synthetic dataset cannot produce a discriminating
  loss curve.
- What causes the jitter at 16 nodes (2.339–3.021 s against 8 nodes' 2.135–2.436).
  `perf_topk` at step 50 would name the ranks.
- Whether 10240 survives 10,000 steps. 60 does not prove it: allocated peaks at
  82.3 GiB and swings with the image count — and item 0 says 60 steps is not even
  survivable at 64 nodes.
- **What produces the non-finite gradient.** Not the optimizer, not the dataset,
  not scale (§17.2 rules out all three). Now survivable, still unexplained — the
  guard turned it from the most important open question into a counter to watch.
- Whether the 128-node knee is exposed collectives or bandwidth. Same ambiguity
  as the first entry, now with a much steeper curve to explain.
- Whether DDP at 8192 holds for a long run. It completed 60 steps at 16 nodes and
  is 17% faster than FSDP at that shape, but max 4.716 s against a 1.679 s median
  says the image-count swing still bites.
