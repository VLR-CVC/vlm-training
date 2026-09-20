# Visual Language Models Large-Scale Training
**Computer Vision Center | VLR Group**

> Contact: [tockier@cvc.uab.cat](mailto:tockier@cvc.uab.cat)

Massive-scale VLM pre-training and finetuning on HPC. Designed and tested for
**MareNostrum 5** and **JUPITER**.

Works like torchtitan: pure-torch model definitions, native torch distributed,
no `transformers` in the training path. Architectures are read from an HF-format
`config.json` and weights load straight from an HF snapshot directory into the
already-sharded state dict.

## Documentation

| Document | What is in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how models are defined, how parallelism is applied |
| [docs/USAGE.md](docs/USAGE.md) | configuring and launching a run |
| [docs/METRICS.md](docs/METRICS.md) | what tokens/s, TFLOP/s and MFU actually count |
| [INSTALL.md](INSTALL.md) | the environment, and why each pin exists |
| [BENCHMARKS.md](BENCHMARKS.md) | measured throughput, scaling tables |
| [SCALABILITY.md](SCALABILITY.md) | single-node reference numbers |
| `train/config.py` | **the reference for every config field** |

## Key features

* **Architectures:** Qwen3.5 (hybrid GatedDeltaNet + gated full attention) and
  Qwen3-VL, both with a ViT tower. Defined in `models/`, config-driven.
* **2D parallelism:** FSDP / HSDP / DDP × Tensor Parallelism, with optional
  sequence parallelism. Measured 4 → 32 nodes (16 → 128 GH200) at **90%
  strong-scaling efficiency**.
* **Packed varlen batches:** no padded rectangles — documents are first-fit-decreasing
  packed into rows and attention runs per document.
* **Compile:** per-block `torch.compile(fullgraph=True)`, with a synthetic
  pre-compile pass that removes the multi-rank recompile stall (warmup 416 steps
  → 1 at 32 nodes).
* **Memory:** chunked cross-entropy, so full `[tokens, vocab]` logits never exist.
* **Dataloading:** NVIDIA Energon with online packing.
* **Checkpointing:** fully distributed model, optimizer, scheduler and dataloader state.

## Environment

The same environment runs on MN5, JUPITER and local clusters. See
[INSTALL.md](INSTALL.md); `requirements.txt` is the lock, `requirements.in`
records why each pin exists.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

- `torch==2.14.0`, `transformers==5.16.1`, python 3.13
- `flash-linear-attention` + `causal-conv1d` for the Qwen3.5 linear-attention layers
- `flash_qla` (optional): FLA dispatches the gated delta rule to it automatically,
  worth -20% forward / -30% forward+backward on a GH200
- `flash_attn` (optional): worth 7-8% forward on a GH200. It *does* build on
  JUPITER's ARM nodes — INSTALL.md has the two workarounds. Without it the models
  fall back to `torch.nn.attention.varlen.varlen_attn`.

Both optional packages are CUDA extensions and must be rebuilt on every torch
upgrade.

## Quick start

1. Download weights to a shared filesystem with `utils/down.py` (login node —
   compute nodes have no internet).
2. Pick or write a TOML under `configs/`. `configs/models/*.json` holds the
   architectures; `configs/{local,mn5,jupiter}/` hold the run configs.
3. Launch:

```bash
./scripts/finetune.sh      --config configs/local/qwen3_5_2b.toml    # local, single node
./scripts/mn5_finetune.sh  --config configs/mn5/qwen3_5_9b.toml      # MareNostrum 5
./scripts/jup_finetune.sh  --config configs/jupiter/qwen3_vl_2b.toml # JUPITER
```

Any field can be overridden on the command line: `--data.seq_len 8192`.

`scripts/` holds both direct CLI launchers and SLURM batch scripts;
`scripts/scaling/` holds the sweep and benchmarking harness.

## Tests

```bash
pytest models/tests/test_configs_parse.py   # every config under configs/ still parses
pytest models/tests/test_precompile.py      # synthetic pre-compile batches
pytest models/tests/test_pack_rows.py       # packing
```

The parity tests are the ones that matter for correctness — `test_tp_parity.py`,
`test_dp_parity.py` and the two `*_parity.py` against `transformers` (those need
a local snapshot). Parallelism bugs are silent: the loss curve looks fine and the
gradients are wrong by a constant factor. Run them after touching anything under
`train/parallel/` or `models/*/sharding.py`.

## Parity status

**Last run 2026-09-20 on JUPITER**, one node / 4× GH200.

### Layer 1 — against `transformers`

Single GPU, our model vs the HF reference on the same snapshot. Gates: top-1
agreement on decisive text positions **must be 1.0000**, mean `|Δlogit|` < 0.1.
Both sides run bf16 compute, so exact equality is not the target; a reference
top-2 gap under 1.0 is a bf16 coin flip and is counted as a tie, reported
separately.

| Check | Qwen3.5-9B | Qwen3-VL-2B |
|---|---|---|
| state dict `from_hf(to_hf(sd))` round trip | PASS | PASS |
| every HF tensor consumed | PASS | PASS |
| vision tower, fp32 | rel mean 6.99e-06 | rel mean 3.66e-06 |
| text logits | mean 2.06e-02, **top1 1.0000** | mean 3.26e-02, **top1 1.0000** |
| multimodal logits, text positions | mean 4.99e-02, **top1 1.0000** | mean 5.78e-02, **top1 1.0000** |
| packed doc vs the same doc alone | mean 2.78e-02, top1 1.0000 | mean 0.00e+00, top1 1.0000 |
| masked loss | rel 1.65e-03 | rel 2.99e-03 |
| eager vs compiled, fp32 GEMMs | mean 7.65e-03, **top1 1.0000** | mean 1.00e-02, **top1 1.0000** |
| compiled backward | 932 grads, all finite | 693 grads, all finite |
| `mrope_positions` == HF `get_rope_index` | — | PASS |
| DeepStack mergers receive gradient | — | PASS |

### Layer 2 — TP / SP / FSDP against TP=1

Same model, same batch, five layouts, compared to the single-GPU result. Run in
**fp32 GEMMs**. Gates: top-1 **1.0000**, loss rel < 2e-3, decoder gradient norm
rel < 3e-2, and worst per-parameter-**type** median gradient ratio within 5%.

| Mode | Qwen3.5-9B | Qwen3-VL-2B |
|---|---|---|
| `tp2` | PASS — top1 1.0000, loss 3.4e-04, gnorm 1.9e-04, worst type 0.996 | PASS — top1 1.0000, loss 1.3e-03, gnorm 2.6e-03, worst type 1.003 |
| `tp2sp` | PASS — worst type 0.996 | PASS — worst type 1.003 |
| `dp2tp2sp` (FSDP × TP) | PASS — worst type 0.997 | PASS — worst type 1.003 |
| `tp4sp` | PASS — worst type 0.993 | PASS — worst type 1.004 |
| `dp2tp2sp` compiled | PASS — worst type 0.995 | PASS — worst type 0.996 |
| `tp4sp` compiled | PASS — worst type 0.989 | PASS — worst type 1.001 |
| vision-only `tp2` | PASS — every grad ratio 2.0 within **2.6e-04** | PASS — within **1.1e-03** |

### What these numbers do *not* cover

Being explicit, because several lines in the logs read like results and are not:

- **`tp2` and `tp2sp` are skipped in the compiled block.** By design — the
  compiled pass runs `tp1`, `dp2tp2sp` and `tp4sp` only. The compiled matrix is
  three layouts, not five.
- **Image-position logits are reported, not gated.** `multimodal image positions:
  top1=0.8163` (9B) and `0.9184` (VL-2B) are `[INFO]`. Only text positions are
  gated; vision-position agreement at bf16 is not a pass/fail signal here.
- **Eager vs compiled in bf16 is reported, not gated** — `top1=0.9362` (9B),
  `0.9778` (VL-2B). The gated comparison is the fp32-GEMM one in the table.
- **`async_tp` is not exercised.** It rewrites collectives and has never been
  through this gate.
- Nothing above is a *throughput* result. See [BENCHMARKS.md](BENCHMARKS.md).

### Local suite

`pytest models/tests` on local compute (CVC): **71 passed, 1 skipped** (the skip needs
`QWEN3_TEXT_SNAPSHOT` / `SIGLIP2_SNAPSHOT`).

## Known issues and TODOs

* **Context, pipeline and expert parallelism are not supported.** The mesh axes
  exist but nothing drives them; CP raises, because GatedDeltaNet needs the full
  sequence.
* The NCCL timeout override `QWEN_NCCL_TIMEOUT_S` only reaches
  `init_process_group`; sub-meshes built by `ParallelDims.build_mesh()` still use
  torch's 600 s default. A slow first step on a large allocation dies there.