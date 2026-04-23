# Usage Guide

This document provides detailed instructions on how to use the VLM training codebase for pre-training and fine-tuning Visual Language Models on HPC environments.

## MUST READ
For details on the configuration see `train/config.py`. That file serves as the DOCS for model and run configuration.

## Installation & Setup

### Environment Requirements

The codebase requires:
- **PyTorch 2.11+** (for varlen implementation used in data-packing)
- Nvidia Energon for data loading
- FlashLinearAttention and Causal-Conv1d for Qwen3.5 models

### Dependencies
Install all required packages:

```bash
torch==2.11.0
megatron-core
transformers
nvidia-energon
wandb

# for qwen3.5 only
flash-linear-attention
causal-conv1d
```

For ARM systems (like JUPITER), use precompiled wheels from GitHub releases:

```bash
pip install --no-build-isolation flashlinearattention causal-conv1d
```

## Preparing Datasets

### Dataset Format

Datasets must be formatted as **Nvidia Energon webdatasets**. This format enables efficient distributed data loading with on-the-fly tokenization.
We expect CrudeDadasets that are formatted using Energon's cookers ([see here](https://nvidia.github.io/Megatron-Energon/advanced/crude_datasets.html)). This allows for easy implementation of medatadasets with serveral data sources.

### Data Packing

Online data packing is implemented with no impact in throughput yet detected. However the implementation is very naive and results in inefficient packing when the sample buffer is nearly empty. Increating `data.packing_buffer_size` should aid.

## Model Weights & Offline Loading

Currently, models are configuration via their HF config (`config.json`). To use a model you MUST downloaded locally first (see `utils/down.py`).

### Downloading Model Weights

Use the provided `utils/down.py` script to pre-download model weights, configuration and tokenizers.
```bash
python utils/down.py
```

Edit the script to change which models and destination paths. Examples:
```python
snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    local_dir="/path/to/shared/cache/qwen35_9b",
)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    local_dir="/path/to/shared/cache/qwen3_vl",
)
```

### Supported Models

The codebase supports the following architectures:

- **Qwen3.5-2B / 9B**: Native Torch implementation with optimized attention
- **Qwen3-VL**: Vision-Language variant
- **Qwen3**: Text-only variant

Originally the models were implemented using HF and Qwen3-VL supports them, however this feature will be deprecated.

### Loading Mechanism

During training:
1. Models are instantiated directly from the local paths specified in config
2. For native Torch models, the architecture is initialized purely in PyTorch
3. Offline weights are mapped and loaded directly into the native state dictionary
4. No network access required during training

This approach is essential for HPC environments where compute nodes have limited internet access.

## Configuration

See `train/config.py` for more details.

### Configuration Structure

All hyperparameters and environment variables are specified in TOML files located in the `configs/` directory:

```
configs/
├── cvc/              # CVC cluster configs
│   ├── qwen3_5_2b.toml
│   ├── qwen3_5_9b.toml
│   ├── qwen3_2b.toml
│   └── ...
├── mn5/              # Marenostrum 5 configs
│   ├── mn5_config.toml
│   ├── mn5_8b_64_nodes.toml
│   └── ...
└── jupiter/          # JUPITER cluster configs
    ├── qwen3_5_2b.toml
    └── qwen3_5_9b.toml
```

### Configuration Parameters

#### Model Configuration

```toml
[model]
model_name = "Qwen/Qwen3.5-2B"     # HuggingFace model identifier, its used to select the model class!!
model_impl = "native"              # Implementation type: "native" or "hf"
```

#### Training Configuration

```toml
[training]
model_dir = "/path/to/model/cache"           # Local path where model weights are cached
output_dir = "/path/to/checkpoints"          # Path for saving training checkpoints

save_steps = 10000                           # Frequency of checkpoint saves
total_steps = 100000                         # Total training steps
random_init = false                          # Initialize MLP layers randomly
compile = false                              # Enable torch.compile
tp_size = 2                                  # Tensor Parallelism size
data_parallel = "ddp"                        # Data parallelism: "ddp" or "fsdp"
```

#### Data Configuration

```toml
[data]
data_path = "/path/to/datasets"    # Path to Energon webdataset
seq_len = 2048                     # Maximum sequence length
```
Online datapacking is configured by default. To change the amount of samples per sequence use `data.batch_size`.

#### Weights & Biases (W&B) Configuration

```toml
[wandb]
run_name = "exp_name"              # Experiment name for tracking
project_name = "vlm_training"      # W&B project name
entity_name = "your_entity"        # W&B workspace/team name
```

### Tensor Parallelism Configuration

For large models, configure tensor parallelism to split the model across multiple GPUs:

```toml
[training]
tp_size = 4                        # Split model across 4 GPUs
data_parallel = "ddp"              # Use DDP for data parallelism
```

Total GPUs = `tp_size × dp_group`

### Example Configurations

**Single GPU, Small Model (Debugging)**
```toml
[model]
model_name = "Qwen/Qwen3.5-2B"
model_impl = "native"

[training]
tp_size = 1
data_parallel = "ddp"
batch_size = 4
total_steps = 1000

[data]
seq_len = 2048
```

**Multi-GPU with Tensor Parallelism (Production)**
```toml
[training]
tp_size = 4
data_parallel = "ddp"
total_steps = "insert very high number here"
compile = true
```

**Large-Scale Multi-Node (64 Nodes)**
```toml
[training]
tp_size = 2
data_parallel = "fsdp"
total_steps = "insert very high number here"
compile = true
```

The torch compiler is one core feature that right now needs a lot of work. The compilation may take up more than 2 min due to many recompilations.

## Running Training

The environtment variables need a lot of polish and new features ([see JUPITER docs](https://apps.fz-juelich.de/jsc/hps/jupiter/buildup.html#tuning-for-large-scale-execution)).

### Local Single-Node Training

For debugging on local machines with multiple GPUs:

```bash
./scripts/finetune.sh --config configs/cvc/qwen3_5_2b.toml
```

This script:
1. Auto-detects available GPUs via `CUDA_VISIBLE_DEVICES` or nvidia-smi
2. Launches torchrun with correct process group settings
3. Sets `OMP_NUM_THREADS` for optimal CPU performance

### Marenostrum 5

#### Direct Launch (for development)

```bash
./scripts/mn5_finetune.sh --config configs/mn5/mn5_config.toml
```

#### SLURM Batch Job

Submit a batch job with the SLURM script:

```bash
sbatch scripts/mn5_finetune.sh --config configs/mn5/mn5_8b_64_nodes.toml
```

The script handles:
- Multi-node setup across SLURM allocation
- Environment variable configuration
- Distributed process group initialization

### JUPITER Cluster

#### Direct Launch

```bash
./scripts/jup_finetune.sh --config configs/jupiter/qwen3_5_9b.toml
```

#### Key Differences from Marenostrum 5

- Uses FlashLinearAttention (ARM-compatible)
- No FlashAttention support (requires ARM compilation)

### Multi-Node Training

For 64+ GPU setups across multiple nodes on Marenostrum 5:

```bash
./scripts/multinode_mn5.sh --config configs/mn5/mn5_8b_64_nodes.toml
```

### Command-Line Overrides

Override config values from the command line:

```bash
./scripts/finetune.sh \
    --config configs/cvc/qwen3_5_2b.toml \
    --data.seq_len 4096
```

See `train/config.py` for more info about the variables. Every aspect of the run can be overrided via the CLI.

## Monitoring Training

### Weights & Biases Integration

All training runs are automatically logged to Weights & Biases. Configure logging in your TOML config:

```toml
[wandb]
run_name = "qwen3.5_2b_exp1"
project_name = "vlm_training"
entity_name = "your_team"
```

Then view real-time metrics:
- Training loss
- Learning rate schedule
- GPU memory usage
- Throughput (tokens/second)
- Gradient norms
- Distributed training statistics (TP/DP group sizes)

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Decrease `seq_len` only if it doesn't reduce samples per batch
2. Use `data_parallel = 'fsdp'` if not already
3. See if the torch compiler decreases memory
4. Increase `tp_size` to split model across more GPUs
5. Enable gradient checkpointing in model config using `ac_mode = full`
