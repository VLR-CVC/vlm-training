# VLR Vision Language Model | Large Scale Training Repository

## FINETUNING
- Go to `finetune.sh` and change the model type
- Using finevision (local path already in)
- Just run `./finetune.sh`

The code is optimize for the Marenostrum 5 HPC system, with H100s. 


### Features
- [x] distributed checkpoints
- [x] compile
- [x] deterministic
- [x] better args + config
- [x] data parallel
- [x] FSDP
- [x] compile + checkpoints
- [ ] static shape compile
- [x] FSDP multinode

### Models Supported
- Qwen2.5-VL series
- Qwen3-VL series

### DISCLAIMER
This code was originally the Qwen3-VL codebase developed by Qwen team, Alibaba Cloud. We didnt change the license.
