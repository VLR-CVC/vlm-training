# Attention variants at scale — benchmark project

Companion to `PERFORMANCE.md` (what the training stack costs) and
`OPTIMIZATION_PLAN.md` (what to do about it). This document owns one question:

> Does sparse attention beat dense attention for Qwen3.5 training at scale, and
> under what conditions?

Started 2026-09-14 from `ATTN_PLAN.md`. **Status: closed. QSA does not pay off
for Qwen3.5 — not in training (§5, §7) and not in inference (§8). The recommended
configuration is the existing `hybrid`: 8 dense full-attention layers plus 24
GatedDeltaNet.**

---

## 1. Where this stands in one paragraph

Four attention variants were built and measured across 16–64 nodes on two
datasets. QSA lost every arm by 18–59%. It then turned out the benchmark was
measuring an **untrained** indexer, which selects key blocks independently per
query — the worst possible input to a block-sparse kernel, because
`flex_attention` skips work per *(query-tile, key-tile)* pair and a 128-query
tile must visit the union of what all 128 of its queries want.

That confound is real, and fixing it does not save QSA. Measuring an *oracle*
indexer — top-k over the true dense attention of Qwen3.5-9B, which is the ceiling
on what any distilled indexer could learn — gives a union of **53.8%** of causal
tiles, against 7.6% for a pure locality prior. The measured cost-vs-union curve
puts break-even near 20% union for a single 16k document and below 7% in clevr's
actual 2-documents-per-row regime. At 53.8% the kernel runs at **~2.7× dense**.
QSA cannot win here even with a perfect indexer; real attention is simply not
local enough for 128 neighbouring queries to agree on what to keep. See §7.

---

## 2. The four variants

Built as config-only model dirs under `/e/project1/reformo/ockier1/qwen_models/`
— `config.json` plus symlinked tokenizer, no weights, driven by
`random_init = true`. `Qwen3_5ForCausalLM.from_pretrained(load_weights=False)`
zeroes params instead of reading safetensors, so a variant needs no checkpoint.

| dir | params | full-attn | GDN | attention |
|---|---|---|---|---|
| `qwen3_5_9b` (+ `_hybrid_ri`) | 9,409,813,744 | 8 | 24 | dense varlen (baseline) |
| `qwen3_5_9b_fullattn` | 9,201,416,944 | 32 | 0 | dense varlen |
| `qwen3_5_9b_qsa512` | 9,430,787,312 | 8 | 24 | QSA, 128 blocks kept |
| `qwen3_5_9b_allqsa512` | 9,285,311,216 | 32 | 0 | QSA, 128 blocks kept |

Parameter counts agree to within 2.4%. `qwen3_5_9b_qsa2048` exists (qwen4's
shipped `budget=2048`) but was dropped: at these document lengths it selects
everything and measures pure indexer overhead.

### QSA port

Qwen Sparse Attention was imported from `models/qwen4/model.py` into
`models/qwen3_5/model.py` — the attention modules only, not the Qwen4 merge.
Extracted by AST so the block is byte-identical to its source: `QSAIndexer`,
`run_flex_attention`, `_create_block_mask`, `_flex_attention`,
`_resolve_flex_kernel_options`, `_segment_ids`, `_FLEX_BLOCK_SIZE`,
`_FLEX_BWD_BLOCKS`, `_qsa_query_chunk`.

Integration points:

- `SelfAttention.__init__` gains `self.indexer = QSAIndexer(cfg) if cfg.use_qsa else None`
- `SelfAttention.forward` builds a BlockMask when the indexer is present and
  routes to `_run_flex_attn`; otherwise the existing `varlen_attn` path is
  untouched
- `models/qwen3_5/config.py` gains `indexer_*` fields, `use_qsa`, `rotary_dim`
  and `validate_qsa()` (all-or-nothing, `kv_heads == 1`,
  `budget % ratio == 0`, `indexer_head_dim >= rotary_dim`)
- `models/qwen3_5/utils.py:apply_rope` accepts `k=None` (the indexer ropes `q`
  alone). Qwen3.5's DTensor guard is preserved — qwen4's copy lacks it
- `train/utils.py:set_model_qwen3_5` freezes every `.indexer.` parameter

**Why the indexer is frozen.** Its outputs are `selected` (bool),
`block_of_token` and `tail_start` (int64) — all `requires_grad=False`, feeding
only `mask_mod`. No float path returns, so it can never receive gradient.
Freezing buys DDP correctness (`replicate()` raises on params that never get
grads) and truthful accounting (LR grouping, param counts and checkpoint
filtering all read `requires_grad`). It costs nothing, because unfrozen ≠
trained: training the indexer needs Qwen4's auxiliary distillation loss, which
is not implemented here.

> **This is where the confound enters.** Freezing is correct. Combining it with
> `random_init = true` is what produced a random indexer, and §5 shows that is
> not a neutral choice.

---

## 3. Datasets

### The `<think>` fix

The stock Qwen chat template emits `<think>` spans only for assistant turns
after the final user query (`chat_template.jinja:100`,
`loop.index0 > ns.last_query_index`). Correct for serving — you do not re-feed
history CoT to a model about to generate — and wrong for SFT on multi-turn CoT
data. It also contradicts `cooker_nemotron`'s docstring, which claims the spans
are kept verbatim.

Across the Nemotron corpus `<think>` is 7.15 B of 10.14 B text tokens.

An SFT variant lives at `qwen_models/chat_template_sft.jinja`:

```jinja
- {%- if loop.index0 > ns.last_query_index %}
+ {%- if reasoning_content %}
```

Gating on whether reasoning *exists* rather than on turn position, so turns
without a span do not get an empty `<think></think>` wrapper. Symlinked into the
four ablation model dirs **only**; production `qwen3_5_9b` keeps its original
file. `qwen3_5_9b_hybrid_ri` exists solely so the baseline stops pointing at the
production dir.

Measured with Qwen3.5's own processor, 400 samples:

| dataset | think blocks | mean doc | p50 | vision tok | QSA density |
|---|---|---|---|---|---|
| plotqa_cot, stock template | 1.0 / 12.4 | 1,627 | 1,662 | 593 | 31% |
| plotqa_cot, SFT template | 12.4 / 12.4 | 5,434 | 4,824 | 593 | 9% |
| clevr_1, SFT template | 9.2 / 9.2 | 14,085 | 12,534 | 150 | 4% |

Density is `indexer_budget 512 / doc_len` — QSA attends a fixed 512 tokens
regardless of document length, so density falls as documents grow.

### clevr_1

Chosen from 76 subsets of `nvidia/Nemotron-Image-Training-v3`, of which only 11
ship their own media (the rest reference an upstream corpus). Selected because
its 480×320 images cost 150 visual tokens against plotqa's 593, so the vision
tower drops out and the step becomes attention-dominated — necessary for an
attention benchmark to resolve anything.

Prepared to `/e/scratch/open-sci-mm/ockier1/vlm_datasets/clevr_1`:
70,005 samples, 36 shards, 0 skipped.

```bash
hf download nvidia/Nemotron-Image-Training-v3 --repo-type dataset \
    --include "clevr_1/*" --local-dir $D/_raw
python utils/prepare_nemotron_energon.py $D/_raw/clevr_1 $D/clevr_1 --per-shard 2000
energon prepare $D/clevr_1 --non-interactive --sample-type CrudeWebdataset --split-ratio 1,0,0
printf '__module__: megatron.energon\n__class__: CrudeWebdataset\nsubflavors:\n    type_dataset: nemotron\n' \
    > $D/clevr_1/.nv-meta/dataset.yaml
```

Two fixes were needed and both are in the tree:

- `prepare_nemotron_energon.py` held the entire media store in RAM. Fine at
  plotqa's 271 MB, not at clevr's 13.4 GB. Now indexes tar members and seeks.
- v3 interleaves bare strings into `content` where v2 used dicts throughout.
  One `isinstance` guard in the repacker and in `cooker_nemotron` covers both.
  `cooker_nemotron` is now also registered for a `nemotron` subflavor, so
  further Nemotron subsets need no code.

---

## 4. Results

`random_init = true` throughout, so loss values are meaningless — these measure
step time only. Median of the last 20 of 40 steps; early steps are compilation,
and the QSA arms recompile `create_block_mask` whenever the block grid widens
(deterministic spikes at steps 3, 6, 7, 9, 11, 13, 17, 18, 29, 30).

**`MFU` and `TF/s` are not comparable between QSA and dense arms.**
`flops_per_token` is derived from config and is blind to sparsity, so it charges
every arm the dense FLOP count. Step time is the honest column.

### clevr, seq 16384, tp=4, FSDP

| nodes | hybrid | fullattn | qsa512 | allqsa512 |
|---|---|---|---|---|
| 16 | **1.531** | 1.685 | 1.809 | 2.675 |
| 32 | **1.615** | 1.761 | 2.171 | 2.817 |
| 64 | 2.171 | **2.023** | 2.505 | 3.929 |
| eff @64n | 70.5% | 83.3% | 72.2% | 68.1% |

Jobs `1789869`–`1789890`.

### plotqa, seq 10240, 16 nodes

| variant | step | vs dense twin |
|---|---|---|
| hybrid | 1.468 | — |
| qsa512 | 1.895 | +29% |
| fullattn | **1.358** | — |
| allqsa512 | 1.801 | +33% |

Jobs `1790465`–`1790468`. plotqa at 16384 OOMs (§6), hence 10240 — so the two
datasets sit at different sequence lengths. Within-dataset comparisons are
clean; cross-dataset absolute step times were never comparable anyway.

### Earlier row, stock template, 4 nodes, seq 10240

Kept because it is the only row connecting to the pre-existing `PERFORMANCE.md`
record, and it is the short-document extreme (`L≈1,627`, 31% density).

| variant | step | layers ms |
|---|---|---|
| hybrid | 1.280 | 271.2 |
| fullattn | 1.173 | 218.2 |
| qsa512 | 1.347 | 338.0 |
| allqsa512 | 1.520 | 418.4 |

---

## 5. Why QSA lost — the union

A prediction was made before the runs, from the config geometry (`H=16`,
`head_dim=256`, `hidden=4096`, `intermediate=12288`):

```
per full-attention layer, document length L
  MLP (SwiGLU)         2·3·L·4096·12288 = 3.02e8 · L
  qkv + o projections  2·L·4096·10240   = 0.84e8 · L
  attention (QK + AV)  4·L²·16·256      = 1.64e4 · L²
  attention share = L / (23,537 + L)
```

predicting `allqsa512` ~35% *faster* than `fullattn` at clevr's lengths. It came
back ~59% *slower*. The FLOP analysis was correct and irrelevant: at `L≈10,000`
QSA does ~6.7% of dense attention's arithmetic and still loses.

### The microbenchmark

`models/tests/bench_qsa_overhead.py`, single GPU, real 9B attention geometry,
fwd+bwd, against `flash_attn_varlen_func`. Jobs `1790478`, `1790517`.

```
T=16384, L=16384        dense    build mask   flex kernel   vs dense   tiles/query
ratio=  4  random      29.63 ms    1.21 ms     147.73 ms      5.03x       78/128
ratio=128  random      28.65 ms    1.16 ms     137.31 ms      4.83x        4/128
ratio=  4  local       29.44 ms    1.20 ms       8.06 ms      0.31x        5/128
ratio=128  local       29.20 ms    1.18 ms       5.72 ms      0.24x        4/128
```

Two candidate explanations die here:

- **`create_block_mask` is not the bottleneck** — ~1.2 ms flat, under 1%.
- **Granularity is not the variable.** `ratio=128 random` touches 4 tiles per
  query and costs 4.83×; `ratio=4 local` touches 5 and costs 0.31×. Same
  per-query sparsity, 15× apart.

### The mechanism

A BlockMask is sparse over *(query-tile, key-tile)* pairs and a query tile spans
`_FLEX_BLOCK_SIZE = 128` queries. Those 128 queries share one row of the mask, so
the kernel visits the **union** of everything they select. Per-query sparsity is
not what gets skipped.

Measured on the mask alone, no kernel involved (`/tmp/union.py` idiom, trivially
reproducible):

| scatter | ratio | tiles per query | union per 128-query tile | of causal-visible |
|---|---|---|---|---|
| random | 4 | 3.9 | 63.4 | **98.4%** |
| local | 4 | 3.9 | 4.9 | 7.6% |
| random | 128 | 4.0 | 63.3 | 98.1% |
| local | 128 | 4.0 | 4.0 | 6.1% |

A random indexer visits 98.4% of causal tiles: the kernel runs fully dense *and*
pays per-element mask evaluation. That is the measured ~4.8×, with no appeal to
kernel quality.

**The corollary is the useful part.** With clustered selection `flex_attention`
runs at 0.24–0.31× dense — a 3–4× speedup, roughly what QSA promises.

---

## 6. Findings independent of QSA

### Sequence length is gated by vocabulary, not attention

Two configurations died on the same allocation:

```
clevr  @ 24576   OOM: tried to allocate 11.37 GiB   = 24576 × 248320 × 2 B
plotqa @ 16384   OOM: tried to allocate  7.58 GiB   = 16384 × 248320 × 2 B
```

`models/qwen3_5/model.py:1348` materializes the full logits tensor before the
loss sees it. Chunked CE chunks the fp32 upcast *inside* the loss — worth 18 GiB
historically — but the bf16 logits already exist by then. With a 248,320-token
vocabulary that is the ceiling on `seq_len`, which is the axis sparse attention
lives on. **Fused linear cross-entropy is the unblock.**

### GatedDeltaNet scales worse than plain full attention

On clevr the dense pair inverts with node count: `hybrid` wins at 16 nodes
(1.531 vs 1.685) and loses by 64 (2.171 vs 2.023), holding 70.5% weak-scaling
efficiency against `fullattn`'s 83.3%. Nothing to do with QSA, and not something
the earlier scaling work surfaced.

---

## 7. The verdict: an oracle indexer does not save it

### 7.1 What an optimal selection looks like

QSA's indexer is trained by distillation against the dense attention
distribution, so top-k over the *real* dense attention is the ceiling on what any
indexer for this model could learn. `models/tests/oracle_mask.py` loads
Qwen3.5-9B with real weights, runs real packed clevr batches, hooks
`SelfAttention._run_varlen_attn`, and compares three selections at identical
per-query density.

```bash
sbatch scripts/scaling/oracle_job.sbatch      # job 1791719
```

T = 16384, 16 full-attention layers over 2 batches. Fraction of causal tiles the
kernel must visit:

| selection | union | |
|---|---|---|
| `oracle` — top-k over true attention mass | **53.8%** | per-layer 51.0 – 58.7 |
| `local` — pure locality prior | 7.6% | |
| `random` — what the ladder ran | 100.0% | |

Real attention is not local enough for 128 neighbouring queries to agree on what
to keep. An oracle is only ~2× better than random, and 7× worse than locality.

### 7.2 Where break-even actually sits

`models/tests/bench_union_curve.py` (job `1791784`) walks the union from 7.6% to
100% with a locality window and times the kernel at each point.

```
T=16384 L=16384 (1 doc/row)      union    mask+flex    vs dense
  window   512                    7.6%     12.93 ms      0.45x
  window  1024                   13.5%     21.27 ms      0.74x
  window  2048                   24.7%     37.31 ms      1.30x   <- break-even ~20%
  window  4096                   44.8%     65.47 ms      2.28x
  window  8192                   75.6%    110.63 ms      3.86x
  window 16384                  100.0%    149.79 ms      5.22x

T=16384 L=6144 (2 docs/row, clevr's actual regime)
  window   512                    7.6%     12.48 ms      1.18x   <- already losing
  window  4096                   44.8%     48.42 ms      4.51x
```

**Break-even needs union below ~20% for a single 16k document, and below ~7% at
clevr's real document lengths. The oracle produces 53.8%**, interpolating to
~2.7× dense. QSA loses on this kernel with a perfect indexer.

Fitting `flex = a + b·L` at fixed union across the two document lengths gives
`a ≈ 37.7 ms` of fixed overhead and `b ≈ 0.00167 ms/token` against dense's
`0.00175` — so at 44.8% union the ratio tends to ~0.95× as `L → ∞`. Even at
unbounded document length an oracle indexer reaches only break-even. That is two
points and should be read as indicative; the gap between break-even and the
oracle's union is the solid part.

### 7.3 What is still worth doing

**Inference was the remaining hope and it did not pay off either — see §8.**
Decode has one query per step, so there is no 128-query tile and no union; the
training argument genuinely does not transfer. QSA still loses, for an unrelated
reason (kernel launch latency against a dense path already running at peak HBM
bandwidth).

**A kernel with finer query tiles** would change the training answer, since the
whole effect is the 128-query union. Qwen4 ships FlashQLA; whether its sparse
path uses a narrower query tile than `flex_attention` is worth checking before
concluding anything about QSA as an architecture rather than about
QSA-on-flex_attention.

### Other open items

| | |
|---|---|
| **Fused linear CE** | Blocks the whole seq_len axis (§6). Project `lm_head` and reduce per chunk; never materialize `(T, 248320)`. |
| **128- and 256-node rungs** | `ATTN_PLAN.md` asks for them; not run. With §7 settled there is little reason to spend allocation on the QSA arms; the dense pair is still worth extending for the GatedDeltaNet scaling question. |
| **Trained indexer weights** | `Qwen/Qwen3.8-Flash-Next` is downloaded at `/data/151-2/users/tockier/models/qwen4` (336 GB, 131 shards) and carries 39 real indexer tensors over 13 QSA layers. **Not transferable**: Qwen4 is `hidden_size 2560`, so `index_qk_proj` is `(640, 2560)` where Qwen3.5 needs `(640, 4096)` — and it was distilled against a different representation space. Running Qwen4-Exp natively needs its full MoE + hyper-connection + PLE stack. |

---

## 8. Reproducing

```bash
# one job per (variant, node count); all four arms on identical parallelism
./scripts/scaling/submit_ladder.sh clevr 16384 16 32 64
./scripts/scaling/submit_ladder.sh plotqa 10240 16

# assemble the table; medians of the last 20 steps, flags nan/OOM/flex-fallback
python scripts/scaling/collect_attn.py

# kernel-level: where the overhead lives
sbatch --nodes=1 --wrap "python models/tests/bench_qsa_overhead.py"
```

Config matrix is generated, not hand-maintained — the four variants differ only
in `model_dir`, so dataset and seq_len are a textual substitution producing
`qwen3_5_9b_{variant}_{dataset}_{seq_len}.toml`.

`submit_ladder.sh` exports `QWEN_SECTION_TIMING=1`. The 16/32/64-node rows above
were submitted before that was added, so they have no `layers_ms` column; the
microbenchmark supersedes it for attribution.

---

## 8. Inference

`models/tests/bench_infer_attn.py`, job `1791922`. The repo has no KV cache, no
`generate` and no decode path — these are training-only varlen implementations —
so this measures the one thing that differs between the four variants, the
per-layer attention op, at inference shapes. MLP, projections and norms are
identical across variants.

Decode QSA is deliberately **not** `flex_attention`: `run_flex_attention`
requires `total % 128 == 0` and would fall back to the eager path at `Q_LEN=1`.
The real decode op is score pooled keys, top-k, gather the selected KV, attend
over them. That is what is timed.

### 8.1 Decode — 1 query against a KV cache of S

| S | dense | QSA | speedup | traffic saved |
|---|---|---|---|---|
| 4,096 | 0.105 ms (16 MiB) | 0.259 ms (2.3 MiB) | 0.41× | 7.1× |
| 16,384 | 0.072 ms (64 MiB) | 0.258 ms (3.0 MiB) | 0.28× | 21.3× |
| 65,536 | 0.100 ms (256 MiB) | 0.250 ms (6.0 MiB) | 0.40× | 42.7× |
| 131,072 | 0.179 ms (512 MiB) | 0.296 ms (10.0 MiB) | 0.60× | 51.2× |

**QSA moves 51× fewer bytes and is still 1.7× slower.** Dense reads 512 MiB in
0.179 ms — 2.86 TB/s, essentially peak HBM3 on GH200 — as a single fused kernel.
QSA's four launches (score, top-k, gather, attend) sit at a flat ~0.25 ms
regardless of context: **launch-latency bound, not bandwidth bound**, so the
traffic advantage never converts. Extrapolating dense's doubling above 64k puts
crossover near **200-250k context**.

### 8.2 Prefill — S queries at once

| S | dense | QSA | slowdown |
|---|---|---|---|
| 4,096 | 0.559 ms | 1.741 ms | 3.12× |
| 8,192 | 1.983 ms | 6.280 ms | 3.17× |
| 16,384 | 7.812 ms | 18.588 ms | 2.38× |
| 32,768 | 30.826 ms | 42.667 ms | 1.38× |

Prefill is the training regime and behaves like §7. The narrowing trend is
**flattered by the test**: a fixed 8192-token locality window means the union
fraction falls as S grows, so at 32k the mask is sparser than the oracle's real
53.8%. Treat 1.38× as optimistic.

### 8.3 Model-level, attention only, per decode step

| S | `hybrid` (8 dense) | `qsa512` (8 QSA) | `fullattn` (32) | `allqsa512` (32) |
|---|---|---|---|---|
| 16,384 | **0.57 ms** | 2.06 ms | 2.29 ms | 8.24 ms |
| 131,072 | **1.43 ms** | 2.37 ms | 5.72 ms | 9.48 ms |

`hybrid` wins at every context length. The 24 GatedDeltaNet layers already solve
long-context decode with O(1) recurrent state, while **QSA still stores the
entire KV cache** and only reduces reads from it. QSA buys bandwidth; GDN buys
memory *and* bandwidth.

### 8.4 The caveat that would change this

The decode QSA path measured here is **unfused** — four kernel launches. A fused
decode kernel would collapse the ~0.25 ms overhead, and then the 51× traffic
advantage would start to matter. Qwen4 ships FlashQLA; if its sparse decode path
is fused, the decode numbers above are a floor rather than a ceiling. This
measures the implementation in this tree, not QSA's limit.

---

## 9. Conclusion

QSA does not pay off for Qwen3.5 at any point measured:

- **Training**: 18-59% slower across 16-64 nodes (§4), and an oracle indexer
  still unions to 53.8% of tiles against a ~20% break-even (§7).
- **Prefill**: 1.4-3.2× slower, and the best number is optimistic (§8.2).
- **Decode**: 1.7-3.6× slower despite 51× less memory traffic (§8.1).

**Recommendation: keep `hybrid`** — 8 dense full-attention layers plus 24
GatedDeltaNet. It is the fastest configuration in training below 32 nodes, the
fastest in decode at every context length, and needs no indexer, no distillation
loss and no sparse kernel.

Two things would reopen this, and neither is speculative:

1. **A kernel with a narrower query tile.** The entire training result is the
   128-query union. Check whether FlashQLA's sparse path tiles more finely
   before concluding anything about QSA as an architecture rather than about
   QSA-on-`flex_attention`.
2. **A fused decode kernel** (§8.4), which would let the 51× traffic saving
   reach the clock.

Independently of QSA, two results here are worth acting on: fused linear
cross-entropy unblocks the whole sequence-length axis (§6), and GatedDeltaNet
scaling worse than plain full attention above 32 nodes deserves its own look.

---

### Gotchas worth keeping

- `PackedBatchEncoder.encode_sample` **drops** oversized samples
  (`data/energon_dataloader.py:398`, `raise SkipSample()`) — it does not
  truncate. clevr retention is 31% at 10240, 73% at 16384, 93% at 24576. Below
  16384 the dataloader discards exactly the long tail clevr was chosen for, which
  biases the benchmark *against* QSA rather than merely wasting data.
- `flex_attention` requires `total % 128 == 0`. 10240 and 16384 both qualify.
- Per-node `TORCHINDUCTOR_CACHE_DIR` is required (`numa_wrapper.sh`); compiled
  collectives bake process-group names into the cache.
