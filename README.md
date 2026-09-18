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
The same environment runs on MN5, JUPITER and our local clusters. **See
[INSTALL.md](INSTALL.md)** — `requirements.txt` is the lock, `requirements.in`
records why each pin exists.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
pytest models/tests/test_dependencies.py -v -rs    # verifies the install
```

- `torch=2.14.0`, `transformers=5.16.1`, python 3.13
- `flash-linear-attention` + `causal-conv1d` for the Qwen3.5/Qwen4 linear attention
- `flash_qla` (optional): FLA dispatches the gated delta rule to it automatically,
  worth -20% forward / -30% forward+backward on a GH200
- `flash_attn` (optional): worth 7-8% forward on a GH200. It *does* build on
  JUPITER's ARM nodes — INSTALL.md has the two workarounds. Without it the models
  fall back to `torch.nn.attention.varlen.varlen_attn`.

Both optional packages are CUDA extensions and must be rebuilt on every torch
upgrade; INSTALL.md explains why and the dependency test catches it when you forget.

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
