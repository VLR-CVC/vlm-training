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
