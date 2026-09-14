# Visual Language Models Large-Scale Training
**Computer Vision Center | VLR Group**

> Contact: [tockier@cvc.uab.cat](mailto:tockier@cvc.uab.cat)

Massive-scale VLM pre-training and finetuning on HPC environments. It is specifically designed and tested for **Marenostrum 5** and **JUPITER**.
Works similary to torchtitan, only relying on native torch code for the distributed implementation. Compatibilty with HF state-dict, loads weights from HF snapshot directory.

See SCALABILITY.md and USAGE.md for more details.

## Key Features
* **Supported Architectures:** **Qwen4-Exp** (`Qwen/Qwen3.8-Flash-Next`), **Qwen3.5**, Qwen3-VL and Qwen3 (text).
* **2D Parallelism:** FSDP/DDP (Single & Multi-node) and Tensor Parallelism (TP) support. Tested scaling up to 256 GPUs.
* **Optimized Dataloading:** Nvidia Energon integration with offline data packing for high-throughput data ingestion.
* **State Management:** Fully distributed model, optimizer, and scheduler checkpointing.

## Environment
We are using the same environment in both MN5 and JUPITER, as well as our local clusters.

Relies on the `torch.nn.attention.varlen.varlen_attn` implementation of `torch=2.11.0` ([see here](https://docs.pytorch.org/docs/2.11/nn.attention.varlen.html)) for the attention in Qwen3.5, we do not require `flash_attn` since its difficult to install in JUPITER (ARM system).

To use `torch=2.10.0` you MUST install `flash_attention`, [see here for the CUDA kernels](https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels).

Support for ROCm systems (LUMI) is work in progress.

#### Qwen3-VL/Qwen3
- `torch=2.11.0` ideally, also works with `torch=2.10.0 + flash_attn`
- `transformers=5.3.0`

#### Qwen3.5
- `torch=2.11.0`
- `flash-linear-attention`
- `causal-conv1d`
- `transformers=5.6.0`

#### Qwen4-Exp
Same requirements as Qwen3.5, plus `transformers>=5.16.0` for the parity tests
(that is the first release shipping `transformers.models.qwen4_exp`).

Qwen4-Exp adds four things on top of Qwen3.5, all implemented in `models/qwen4`:
* **MoE** — 512 routed experts, top-10, plus a gated shared expert. The training
  path uses `torch._grouped_mm`; a reference loop is kept for CPU and for tests.
* **Hyper-connections** — the residual stream is carried as `hc_count=4` parallel
  streams, mixed per block by `GatedResidual` and collapsed at the end by
  `hyper_connection_mixer` (which replaces the final RMSNorm).
* **QSA** — sparse attention. An indexer scores mean-pooled key blocks and keeps
  the top `indexer_budget // indexer_compress_ratio` per query. Because the
  selection is a mask, these layers run through `flex_attention` with a
  `BlockMask` rather than `varlen_attn`. The indexer receives no gradient (its
  top-k is non-differentiable and Qwen trains it with a separate distillation
  loss), so `set_model_qwen4` freezes it.
* **PLE** — hashed n-gram per-layer embeddings on the layers named by
  `ple_layer_ids`. The n-gram table is ~90 GiB and the checkpoint stores it as
  `split_ngram_parts` shards, which `load_safetensors_into` concatenates.

The MTP head present in the checkpoint (`mtp.*`) is not implemented, matching
`transformers`.

### FlashQLA (optional GDN kernels)

[FlashQLA](https://github.com/QwenLM/FlashQLA) is the Qwen team's TileLang
implementation of the chunked gated delta rule — the same operator FLA provides,
reported at 2-3x forward and 2x backward on Hopper and Blackwell. Its entry point
takes the same arguments as FLA's, so `models/qwen4` wraps both behind matching
`torch.library` custom ops.

```bash
pip install flash-qla   # pulls tilelang 0.1.9, additive: nothing else is upgraded
```

Selection happens once, after the model reaches its GPU:

```python
from models.qwen4.model import set_gdn_backend
set_gdn_backend("auto")      # or "fla" / "flashqla"
```

`train/train_qwen.py` does this for `ModelType.Qwen4` and logs the choice;
`QWEN4_GDN_BACKEND` overrides it. `auto` takes FlashQLA only when it is usable
**and** has a backward kernel, so it never silently picks something that cannot
train:

| arch | | |
|---|---|---|
| SM90 (H100, GH200) | forward + backward | `auto` -> flashqla |
| SM100 / SM103 | forward + backward | `auto` -> flashqla |
| SM120 / SM121 | forward only | `auto` -> fla |
| SM89 and below | unsupported (import raises) | `auto` -> fla |

Two constraints worth knowing before you debug a wall of TileLang output:

* **`linear_key_head_dim` must be 128.** FlashQLA's `kkt_solve` asserts it. The
  released Qwen4-Exp config uses 128/128, so this only bites on toy configs.
* **TileLang JITs through `nvcc`**, and the whole toolchain has to agree —
  `nvcc`, `nvvm`, `ptxas` and the CCCL headers all at the same CUDA version, new
  enough for the target arch (12.8+ for SM120). A mismatch shows up as
  `Unsupported .version` from ptxas, `cccl/cuda/std/utility: No such file`, or
  `CUDA compiler and CUDA toolkit headers are incompatible`. On a box without a
  suitable system CUDA:

  ```bash
  pip install nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 \
              nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85
  export CUDA_HOME=$(python -c "import site;print(site.getsitepackages()[0])")/nvidia/cu13
  export PATH=$CUDA_HOME/bin:$PATH
  ```

  Pin those to whatever `nvidia-cuda-runtime` is already installed. On MN5 and
  JUPITER, load the system CUDA module instead.

Parity against FLA lives in `models/tests/test_qwen4_flashqla.py` (forward,
packed forward, per-document isolation, and backward where the arch has it).
It skips with the reason when FlashQLA cannot build.

### float8 training (torchao)

`training.float8 = true` swaps every eligible `nn.Linear` for torchao's
`Float8Linear`, so its GEMMs run in fp8 with dynamic scaling. Weights,
gradients and optimizer state stay high precision -- only the matmul operands
are quantized. `training.float8_recipe` picks `tensorwise` (default),
`rowwise` or `rowwise_with_gw_hp`.

The swap happens before TP, compile and FSDP, the way torchtitan orders it.

Left in high precision on purpose: `lm_head` (fp8's dynamic range lands
straight in the loss), the QSA indexer (frozen, and its scores feed a `topk`
where rounding changes *which* blocks are selected), the vision tower, and
anything whose weight dims are not divisible by 16.

A second bite comes out of it from `compile_model`: a QSA layer is compiled
piecewise and `self_attn` is not one of the pieces, so those projections do
their fp8 scaling in eager.

The filter also has to know the TP width. It runs before `apply_tp`, so a
4096x48 projection passes a plain divisible-by-16 test and then fails at trace
time under TP=2, where the kernel sees 4096x24 (*"Expected both dimensions of
mat2 to be divisible by 16"*). `apply_float8` takes `tp_size` and requires
`16 * tp_size`.

#### The MoE experts: `training.float8_moe`

The experts are 3D parameters behind `torch._grouped_mm`, not linears, so
`float8` cannot reach them -- and on a MoE model that is where most of the
FLOPs are. `training.float8_moe = true` swaps their data for torchao's
`ScaledGroupedMMTensor`, which overrides `torch._grouped_mm` with a
differentiable scaled grouped GEMM. `training.float8_moe_recipe` picks
`fp8_rowwise` (default), `mxfp8` or `mxfp8_wgrad_with_hp`; there is no
tensorwise option on this path.

It is enabled independently of `float8` -- the two cover disjoint parts of the
model.

**The architecture gate is a set, not a floor.** `torch._scaled_grouped_mm`:

    torch._scaled_grouped_mm is only supported on CUDA devices with
    compute capability = [9.0, 10.0], or ROCm MI300+

SM90 (H100, GH200) and SM100 (B100/B200, GB200) are in. SM120 (RTX PRO 6000)
is *not*, despite being numerically higher than both -- which is why this box
cannot run it. `mxfp8` narrows that further to SM100. `apply_float8_moe`
checks at startup and raises with the reason rather than failing mid-step.

**This path has not run its kernel.** What is tested on SM120 is the guard, the
parameter selection and the swap (`models/tests/test_qwen4_float8.py` fakes the
capability to reach the swap). Numerics and throughput are unverified until it
runs on GH200.

#### Measured, 2x RTX PRO 6000 (SM120), tensorwise

Loss tracks bf16 closely; throughput does not pay off at these model widths.

| config | bf16 | fp8 | step 10-12 loss (bf16 -> fp8) |
|---|---|---|---|
| 615M, tp=2, seq 4096 | 0.351 s/step | 0.418 s/step | 12.1212 -> 12.1226 |
| 615M, fsdp=2, seq 4096 | 0.302 s/step | 0.308 s/step | 12.1085 -> 12.1096 |
| 9B, tp=2, seq 2048 | 1.052 s/step | 1.328 s/step | 12.5978 -> 12.6109 |

9B is the largest local datapoint: 27B does not fit on two cards at all, with
or without fp8. `train_qwen.py` builds the whole model per rank and upcasts it
to fp32 before `apply_tp` shards anything, and that peak OOMs at 94 GiB. Use
`pp_size` for models that big.

Isolated two-linear stacks, compiled, forward+backward, show where the
crossover actually is:

| T x K x N | bf16 | fp8 | speedup |
|---|---|---|---|
| 2048 x 2048 x 2048 | 0.62 ms | 1.22 ms | 0.51x |
| 4096 x 4096 x 4096 | 3.07 ms | 2.11 ms | 1.45x |
| 8192 x 4096 x 4096 | 6.12 ms | 3.96 ms | 1.54x |
| 8192 x 8192 x 8192 | 28.3 ms | 17.1 ms | 1.65x |

Qwen4-Exp is a *narrow* MoE: even the released checkpoint is
`hidden_size = 2560`, `moe_intermediate_size = 640`, with 512 experts. Its
dense projections sit on the losing side of that crossover, and the width that
would pay is in the experts fp8 cannot reach here. Hence `float8 = false` by
default.

#### Where the crossover is, on a real training step

`configs/cvc/qwen4/wide{2,3,4,6,8}k.toml` exist for this measurement: 4 layers,
8 experts, seq_len 4096, TP=2, everything fixed except `hidden_size`. Generate
each with `python -m utils.make_qwen4 --size wide4k --out ...`. Median step time
over 20 steps, warmup dropped:

| hidden_size | params | bf16 | fp8 | speedup |
|---|---|---|---|---|
| 2048 | 1.56B | 0.329 s | 0.356 s | 0.92x |
| 3072 | 2.49B | 0.437 s | 0.453 s | 0.96x |
| 4096 | 3.54B | 0.550 s | 0.565 s | 0.97x |
| 6144 | 5.95B | 0.823 s | 0.822 s | 1.00x |
| 8192 | 8.79B | 1.248 s | 1.228 s | 1.02x |

Break-even is `hidden_size` ~6144 and the curve is still nearly flat at 8192 --
a whole-model step is not the isolated-GEMM microbenchmark, and the parts fp8
never touches (experts, embeddings, `lm_head`, norms, attention itself) set the
ceiling on what it can win. On this hardware, the linear swap alone is not the
lever; `float8_moe` on SM90/SM100 is.

Eager fp8 is always a loss: on a 4x4096 linear stack, bf16 12.2 ms, eager fp8
22.8 ms, compiled fp8 7.7 ms. `float8` without `compile` logs a warning.

Tests: `models/tests/test_qwen4_float8.py`.

### Parity against `transformers`, and how far it goes

`models/tests/test_qwen4_checkpoint.py` runs against the real
`Qwen/Qwen3.8-Flash-Next` weights. Point `QWEN4_SNAPSHOT` at the snapshot; the
suite skips cleanly without one. Two tests are opt-in because of their
footprint:

| env | what it does | cost |
|---|---|---|
| `QWEN4_TEST_PLE_TABLE=1` | loads the full n-gram table through `load_safetensors_into` and checks shard placement | ~95 GiB RAM, ~17 min |
| `QWEN4_TEST_FULL_MODEL=1` | all 48 layers vs `transformers`, both models in host RAM, layers streamed to the GPU | ~470 GiB RAM, ~9 min |

The whole-model test streams layers to the GPU rather than running on CPU
because 36 of the 48 layers are `linear_attention`, whose gated delta rule is
Triton-only.

**Expect the whole-model logits to differ.** Every layer is individually exact
-- `test_layer_wiring_exact_in_fp32` holds each to ~5e-7 relative in fp32 --
but our `linear_attention` calls FLA's Triton chunked kernel while
`transformers` calls its own `torch_chunk_gated_delta_rule`. The two accumulate
differently: ~5e-3 to 4e-2 relative in fp32, ~3e-2 in bf16, *per linear layer*.
Qwen4 has no final RMSNorm to rescale the residual stream, so that compounds to
~3e-1 by layer 47. On this checkpoint at `seq=2560` the result is top-1 token
agreement 0.95 and logit correlation 0.989.

This is a property of the kernel choice, not a defect, and it is why the
bf16 layer tests use `atol=0.15`. The tight check that would actually catch a
wiring bug is `test_layer_wiring_exact_in_fp32`, which swaps in HF's own
reference kernel so the only remaining difference is our wiring.

## Datasets and Dataloading
Datasets are expected to be as a CrudeWebdataset. With https://github.com/NVIDIA/Megatron-Energon we handle the raw data and tokenize it on the fly. It is an asynchrnos process that does not have an impact on model performance. **Online datapacking is used by default.** Support for Metadatasets (multiple sources).

## Model Weights & Offline Loading
Use `utils/down.py` on a login node to pre-download model weights and tokenizers to a shared filesystem. The models' archicture configuration relies on what is downloaded. 

**Loading Mechanism:** During training, models are instantiated directly from these local paths. The architecture is initialized purely in PyTorch, and the offline weights are mapped and loaded directly into the native state dictionary.

## Usage
1. Ensure your datasets are formatted as Nvidia Energon webdatasets.
2. Configure your hyperparameters and environment variables in the `configs/` directory.
3. Launch the distributed training job using the environment-specific script:

```bash
# For Marenostrum 5
./scripts/mn5_finetune.sh --config [toml file]

# For JUPITER
./scripts/jup_finetune.sh --config [toml file]
```
In `configs/` you can find several examples. Look into the `jup` and `mn5` directories to see the configs for the respective HPC systems.

*Note: The `scripts/` directory contains both direct CLI launch scripts and SLURM batch scripts.*

## Scalability Results
The codebase demonstrates linear scaling up to 256 GPUs using FSDP and Tensor Parallelism.
For a detailed breakdown of throughput, GPU efficiency, and scaling characteristics, please refer to [SCALABILITY.md](SCALABILITY.md).

## Known Issues & TODOs
* The entire workflow `training -> checkpoints -> eval/usage` needs a lot of work.
* Static shape compilation (`torch.compile` with `fullgraph=True`) is pending.
* A better data packing implemented is needed.
