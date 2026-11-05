## Qwen3-VL-2B @ JUPITER — single node (2026-09-18)

One node, 4x GH200. `configs/jupiter/qwen3_vl_2b.toml` as committed:
- `seq_len = 32768`
- `tp_size = 2`
- `data_parallel = fsdp`

Averaged over steps 10-40:

| metric | value |
|---|---|
| tokens/s (job) | 61,161 |
| tokens/s per GPU | 15,290 |
| TFLOP/s per GPU | 215.7 |
| MFU | 21.8% |
| step time | 1.249 s |
| documents per row | 11.5 |
| mean document length | 2,853 tokens |
| batch utilisation | 99.4% |
| peak memory | 72.1 GiB allocated / 75.9 reserved |

MFU is against a GH200's 989.4 TFLOP/s bf16 dense peak. Tokens/s counts non-padding tokens
and is summed over the DP axis only; the per-GPU figure divides by all 4 GPUs, including the
TP partner that works on the same micro-batch.

## Qwen3.5-2B @ JUPITER — 64 nodes (2026-10-11)

64 nodes, 256x GH200. Qwen3.5-2B, instruct SFT on the OneVision-Instruct + Nemotron-VLM-v2 blend:
- `seq_len = 16384`
- `tokens_per_microbatch = 65536` (MBS 4)
- `tp_size = 1`
- `data_parallel = fsdp`, `reshard_after_forward = "never"`, `selective_ac_freq = 1`

At the end of the run:

| metric | value |
|---|---|
| tokens/s (job) | 3,152,000 |
| tokens/s per GPU | 12,310 |
| TFLOP/s per GPU | 282.0 |
| MFU | 28.5% |
| step time | 4.66 s |
| documents per row | not recorded |
| mean document length | not recorded |
| batch utilisation | 87.5% |
| peak memory | 32.4 GiB allocated / 35.4 reserved |

MFU is the logged value. TFLOP/s is MFU x 989.4; tokens/s divides that by the 22.9 GFLOP/token
of `train/flops_estimation.py`; step time and batch utilisation come from 16.72B tokens over 1139
steps against a 16.8M-token global batch (256 x 4 x 16384).

## Qwen3.5-35B-A3B @ JUPITER — 64 nodes (2026-10-11)

64 nodes, 256x GH200. Qwen3.5-35B-A3B (256 experts, top-8), instruct SFT on the
OneVision-Instruct + Nemotron-VLM-v2 blend. `configs/jupiter/qwen3_5_35b_a3b.toml`:
- `seq_len = 16384`
- `tokens_per_microbatch = 65536` (MBS 4)
- `tp_size = 2`, `ep_size = 4`
- `data_parallel = fsdp`, `reshard_after_forward = "never"`, `selective_ac_freq = 1`

At the end of the run:

| metric | value |
|---|---|
| tokens/s (job) | 913,600 |
| tokens/s per GPU | 3,569 |
| TFLOP/s per GPU | 99.9 |
| MFU | 10.1% |
| step time | 8.04 s |
| documents per row | not recorded |
| mean document length | not recorded |
| batch utilisation | 87.5% |
| peak memory | 48.6 GiB allocated / 51.6 reserved |

MFU is the logged value. TFLOP/s is MFU x 989.4; tokens/s divides that by the 28.0 GFLOP/token
of `train/flops_estimation.py` (routed experts at top-8, shared expert and router; dispatch and
combine are not counted); step time and batch utilisation come from 16.70B tokens over 2274 steps
against an 8.4M-token global batch (128 data ranks x 4 x 16384).

## Qwen3.5-2B @ JUPITER — single node, 128K (2026-10-21)

One node, 4x GH200. Qwen3.5-2B on the long-document blend (multi-page PDFs, up to 120 images
per sample):
- `seq_len = 131072`
- `tokens_per_microbatch = 131072` (MBS 1)
- `tp_size = 1`
- `data_parallel = fsdp`, `reshard_after_forward = "never"`, `selective_ac_freq = 1`
- `packing_buffer_size = 16`, `num_workers = 2`

Over 20 steps:

| metric | value |
|---|---|
| tokens/s (job) | ~29,000 |
| tokens/s per GPU | ~7,250 |
| TFLOP/s per GPU | 188-208 |
| MFU | 19-21% |
| step time | 13.8-17.5 s |
| documents per row | not recorded |
| mean document length | ~20,000 tokens |
| batch utilisation | 33-98% |
| peak memory | 67.7 GiB allocated / 72.1 reserved |

Batch utilisation swings because the longest samples (p95 117-119K tokens) fill a row on their
own. Peak memory is 72.1 of 95 GiB; going from 64K to 128K cost 1.7x memory, so 256K at MBS 1
would need roughly 120 GiB and does not fit on one node.
