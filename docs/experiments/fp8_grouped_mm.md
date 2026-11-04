# FP8 grouped GEMM for the MoE experts (shelved)

Status: **shelved 2026-11-04**. Correct, but much slower than bf16 on Qwen3.5-35B-A3B.
The code lives on branch **`exp/fp8-grouped-mm`**, one commit on top of the MoE commit,
so `git diff feat/qwen35-moe exp/fp8-grouped-mm` is exactly the fp8 work and
`git cherry-pick` brings it back.

## What it did

`model.moe_grouped_mm = "fp8_rowwise"` switched the three expert GEMMs (w1, w3, w2) of
every MoE layer from `torch._grouped_mm` (bf16) to torchao's rowwise-quantized
`torch._scaled_grouped_mm`. Two things had to change together, so one function,
`models.common.moe.set_moe_grouped_mm(config, "fp8_rowwise")`, applied both at build time:

1. **The kernel.** `GroupedExperts.grouped_mm = "fp8_rowwise"` routes `_grouped_mm` to
   ```python
   from torchao.prototype.moe_training.fp8_grouped_mm import (
       _to_fp8_rowwise_then_scaled_grouped_mm,
   )
   return _to_fp8_rowwise_then_scaled_grouped_mm(
       A, B_t, offs, out_dtype=torch.bfloat16,  # offs = cumsum of PADDED counts
       pad_token_groups_for_grouped_mm=False,  # torchao 0.18.0 asserts on True
   )
   ```
2. **The dispatcher.** `TorchAOTokenDispatcher` (subclass of `AllToAllTokenDispatcher`)
   pads each expert's token group to `pad_multiple = 16`, using torchao's triton
   `permute_and_pad` (`torchao.prototype.moe_training.ep.permute`). An unaligned
   group gives a wrong result, not an error, and torchao 0.18.0 cannot pad
   internally.

Guards: compute capability 9.0 or 10.0 only (GH200 yes, L40S / workstation
Blackwell no); MoE models only (`build_meta` raises otherwise).

### Files on the branch

| file | what |
|---|---|
| `models/common/moe.py` | `GroupedExperts.Config.grouped_mm`, fp8 branch of `_grouped_mm`, `set_moe_grouped_mm`, `_FP8_GROUPED_MM_CAPS` |
| `models/common/token_dispatcher.py` | `TorchAOTokenDispatcher` |
| `models/qwen3_5/checkpoint.py` | `build_meta(..., moe_grouped_mm=)` |
| `train/config.py`, `train/train_qwen.py` | `Model.moe_grouped_mm`, passed to `build_meta` |
| `models/tests/test_fp8_grouped_mm.py` | wiring tests (CPU, fake sm_90) |
| `models/tests/fp8_grouped_mm_check.py`, `scripts/scaling/jup_fp8_check.sbatch` | GH200 numerical check |
| `models/tests/fp8_autotune_probe.py` | per-shape GEMM timing probe |
| `models/tests/test_ep_dispatcher.py`, `test_dispatcher_permute_output_size.py` | AO dispatcher cases |
| `configs/jupiter/qwen3_5_35b_a3b_exp.toml` | the 8-node fp8 arm |

## Results (JUPITER, GH200, torch 2.14, torchao 0.18.0)

**Correctness: passed.** Job 2218641 (`jup_fp8_check.sbatch`, 1 GH200): padding
invariance exact, error unbiased, cosine vs bf16 0.9993 forward / 0.9988 backward.

**Throughput: about 115x slower.** Same debug model (`qwen3_5_moe_debug_1n`, 1 node / 4 GPUs),
same config:

| arm | job | step time | MFU | tok/s/GPU |
|---|---|---|---|---|
| bf16 | 2219865 | 0.31 s | 16.8-18.0% | 25,550 |
| fp8 | 2219866 | 37.2 s | 0.1% | 216 |

The 8-node 35B-A3B fp8 arm (2218725, then 2219216 with `precompile = false`) ran at about
200 s/step and was cancelled. The bf16 baseline at 64 GPUs is 6.16% MFU, 2810 tok/s/GPU
(wandb `gk9dlwsv`).

**GEMM alone** (`fp8_autotune_probe.py`): 0.5 ms fp8 vs 0.1 ms bf16 at the model's
expert width `moe_intermediate_size = 512`. These experts are too narrow for fp8 to beat
cuBLAS bf16, before any overhead.

### What the 37 s is not

Ruled out by measurement: precompile off, per-step dynamo recompiles (bounded, warmup
only), per-shape triton autotune (a new M costs 0.5 ms, the same as a repeat), graph breaks
(`fullgraph=True` would raise), and recompiles driven by the padded total. Not timed: the
`permute_and_pad` call, once per layer per forward. It is the leading suspect and the
first thing to measure if this is picked up again.

### Why it was dropped rather than fixed

The 35B-A3B is limited by communication. The profile has all_gather + reduce_scatter at
about 71% of GPU time, all compute at about 8-16%, and the expert GEMMs at about 3%. A perfect fp8
GEMM would chase that 3%, and the weight all-gather stays bf16 either way. Levers
ranked ahead of it at the time: NCCL protocol (`NCCL_PROTO=LL128`/`Simple`; the big
collectives were on `LL`), a bigger micro-batch (unblocked by AC),
`reshard_after_forward = "never"`, and a lower-precision weight all-gather.

## When to revisit

- A MoE with wide experts (`moe_intermediate_size` in the thousands), where the fp8 GEMM
  can beat bf16 at all. Rerun `fp8_autotune_probe.py` at that width first.
- A compute-bound setup (single node, or after the communication fixes above).
- A torchao release that pads inside the grouped GEMM, which would let
  `TorchAOTokenDispatcher` go.

To restore: `git cherry-pick exp/fp8-grouped-mm` onto the MoE branch, then
`moe_grouped_mm = "fp8_rowwise"` in `[model]`, then run `jup_fp8_check.sbatch` before any
throughput run.
