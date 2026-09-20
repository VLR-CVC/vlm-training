# How throughput is reported

Three numbers get quoted from a run: tokens/s, TFLOP/s and MFU. This is what each one
counts and where it comes from. Code: `train/flops_estimation.py` (the FLOPs) and
`Trainer.log` / `Trainer._gather_perf` in `train/train_qwen.py` (the reporting).

## Scope

Every metric is emitted in exactly one scope, and the aggregation follows the metric's type:

| type | aggregation | metrics |
|---|---|---|
| data/additive | SUM over the **DP group** | tokens, FLOPs, samples, loss |
| global | MAX over **WORLD** | peak memory |
| intensive | derived from two additives, never averaged | MFU, tok/s/GPU |

### What "the DP group" means

Counts are added up across DP ranks only. **not** across all ranks. Ranks in the same TP
group are working on the *same* micro-batch, so adding them in would count those tokens twice.

In general a DP group holds `world_size / (pp * cp * tp)` ranks.

### perf_topk agrees with the headline

The per-rank rows under `perf_topk/*` average to the per-GPU headline, exactly. They are a
breakdown of that number, not a second measurement of it.
`models/tests/test_deferred_log.py::test_perf_topk_decomposes_the_headline` checks this.

## tokens/s

```
perf/tokens_per_second          = dp_sum(ntokens) / time_delta
perf/tokens_per_second_per_gpu  = perf/tokens_per_second / world_size
```

`ntokens` counts **real tokens only**, padding is excluded.

The per-GPU number divides by *every* GPU, TP ranks included. Two GPUs that split the work on
one token get half a token each. That is deliberate: it keeps the figure comparable whether a
run uses TP or not.

## TFLOP/s and MFU

FLOPs are counted **from the batch**, not from a per-token constant:

```
flops = dense_per_token * n_tokens              # projections, MLP, logits
      + text_attn_pair  * sum(positions + 1)    # exact causal pair count
      + vision_per_patch * sum(P)               # P = patches per image
      + vision_attn_pair * sum(P^2)             # ViT attention is bidirectional
```

Why the batch and not a constant: training is varlen-packed, so one row holds many documents
and **attention never spans the row**. A 32768-token row of 11 documents does roughly 1/11 of
the attention work the row length suggests. The ViT is the same story — it attends inside one
image, a few hundred patches, not across the row.

```
perf/tflops_per_second = dp_sum(flops) * tp / time_delta / world_size / 1e12
perf/mfu               = perf/tflops_per_second / peak_tflops_per_gpu * 100
```

The `* tp` puts back what was divided out earlier. Each rank counted `batch_flops / tp` as its
own share, and both ranks of a TP group hold the same share, so multiplying the DP sum by `tp`
gives the job total — without a second collective.

Both numbers describe the **whole job**, spread over every GPU. Neither is rank 0's.

`peak_tflops_per_gpu = 989.4` is a GH200's bf16 **dense** peak. The 1979 TFLOP/s NVIDIA
advertises is the sparsity number and does not apply here.

## Traps

- **`perf/step_time` is per micro-batch**, not per optimizer step: it is
  `time_delta / len(batches)`. Multiply by `tpi_multiplier` for the optimizer step.
- **tokens/s skips padding, MFU does not.** The GPU really does run the projections and MLP
  over padding, so those FLOPs are real work. Watch `train/batch_efficiency`: when it drops,
  tokens/s falls but MFU holds, and the difference between them is the padding.
- **MFU is not the average of the per-rank MFUs.** It is `sum(flops) / (world_size * peak)`.

## Sanity checks

`python -m train.flops_estimation` verifies the formula against the Megatron per-token
constant it replaced, in the one case that constant assumed (a single document filling the
row, ViT at row length), for all five model configs.

`sacct` and the log agree on wall clock; if `perf/log_wait_ms` is not ~0 the deferred-logging
path is stalling on the device and the step timings are suspect.