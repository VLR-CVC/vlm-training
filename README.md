# Visual Language Models Large-Scale Training
**Computer Vision Center | VLR Group**

> Contact: [tockier@cvc.uab.cat](mailto:tockier@cvc.uab.cat)

This codebase is optimized for massive-scale VLM pre-training and finetuning on HPC environments. It is specifically designed and tested for distributed training on **Marenostrum 5** and **JUPITER**.
Works similary to torchtitan, only relying on native torch code for the distributed implementation.

## Key Features
* **Supported Architectures:** **Qwen3.5**, Qwen3-VL and Qwen3 (text).
* **Distributed Training:** FSDP (Single & Multi-node) and Tensor Parallelism (TP) support. Tested scaling up to 256 GPUs.
* **Optimized Dataloading:** Nvidia Energon integration with offline data packing for high-throughput data ingestion.
* **State Management:** Fully distributed model, optimizer, and scheduler checkpointing.

## Environment
We are using the same environment in both MN5 and JUPITER, as well as our local clusters.
The only big requirement would be `torch 2.11`, as we use their `varlen` implementation for data-packing. The FlashAttention package is not used because requires compilation for JUPITER (ARM system). However, FlashLinearAttention is needed to run the Qwen3.5 models.

## Datasets and Dataloading
Datasets are expected to be as a CrudeWebdataset. With https://github.com/NVIDIA/Megatron-Energon we handle the raw data and tokenize it on the fly. It is an asynchrnos process that does not have an impact on model performance.

**Online datapacking is not yet supported** (no particular issue related to the HPC system, its skill issue on my part). We believe that data packing is a must-have for a visual-language training codebase with native resolution, as the varying image sizes on the datasets can be handled easily.

## Model Weights & Offline Loading
Use `utils/down.py` on a login node to pre-download model weights and tokenizers to a shared filesystem:

```bash
python utils/down.py --model_id "Qwen/Qwen3-VL-8B" --save_path "/path/to/shared/storage"
```

**Loading Mechanism:** During training, models are instantiated directly from these local paths. For Native Torch models, the architecture is initialized purely in PyTorch, and the offline weights are mapped and loaded directly into the native state dictionary.

## Usage
1. Ensure your datasets are formatted as Nvidia Energon webdatasets.
2. Configure your hyperparameters and environment variables in the `configs/` directory.
3. Launch the distributed training job using the environment-specific script:

```bash
# For Marenostrum 5
./scripts/mn5_finetune.sh

# For JUPITER
./scripts/jup_finetune.sh
```

*Note: The `scripts/` directory contains both direct CLI launch scripts and SLURM batch scripts.*

## Scalability Results
The codebase demonstrates near-linear scaling up to 256 GPUs using FSDP and Tensor Parallelism.
For a detailed breakdown of throughput, GPU efficiency, and scaling characteristics, please refer to [SCALABILITY.md](SCALABILITY.md).

## Known Issues & TODOs
* Tensor Parallelism (TP) is currently missing for native Torch models.
* Online data packing for Energon dataloading is not yet supported.
* Static shape compilation (`torch.compile` with `fullgraph=True`) is pending.
