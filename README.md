# VLR Vision Language Model | Large Scale Training

## FINETUNING
- Go to `finetune.sh` and change the model type
- Using finevision (data path should be a directory with parquet files.
- Just run `./finetune.sh`

The code is optimized for the Marenostrum5 HPC system, with H100s.

### Features
- [x] Qwen2.5-VL & Qwen3-VL Support
- [x] distributed checkpoints
- [x] optimizer & scheduler checkpoints
- [x] compile
- [x] deterministic
- [x] better args + config
- [x] data parallel
- [x] FSDP
- [x] compile
- [ ] static shape compile (fullgraph)
- [x] FSDP multinode
- [x] data packing

### Quality of life ISSUES/To be improved
- Webdatasets/Nvidia Energon dataloader not supported, we just iterate over a raw list of parquet files
- We do not keep track of the parquet files/data consumed (see above)
- UPDATE PENDING (added TP support)

### Models Supported
- Qwen3-VL series
- Qwen2.5-VL series
- Qwen2-VL series

### DISCLAIMER
This code was originally the Qwen3-VL codebase developed by Qwen team, Alibaba Cloud. We didnt change the license.
