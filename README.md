# Visual Language Models Large-Scale Training
**Computer Vision Center | VLR Group**

> Contact: [tockier@cvc.uab.cat](mailto:tockier@cvc.uab.cat)

Massive-scale VLM pre-training and finetuning on HPC environments. It is specifically designed and tested for **Marenostrum 5** and **JUPITER**.
Works similary to torchtitan, only relying on native torch code for the distributed implementation. Compatibilty with HF state-dict, loads weights from HF snapshot directory.

See SCALABILITY.md and USAGE.md for more details.

## Key Features
* **Supported Architectures:** **Qwen3.5**, Qwen3-VL and Qwen3 (text).
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
