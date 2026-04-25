## Qwen3.5-2B @ JUPITER
- 16 node (64 H200 96GB) tested
- +15,000 tks/sec/gpu

## Qwen3.5-9B @ JUPITER
- scaling test from 16 to 256 nodes
- +500 TFLOPS/s/gpu
<img width="700" height="599" alt="image" src="https://github.com/user-attachments/assets/9ce70706-a081-4ba4-b018-734503fac241" />


## Qwen3-VL-8B @ JUPITER
- ~380 TFLOPS with 4 nodes (16 GH200 96GB)
- More testing needs to be performed
- seq_len: 12288 (50% increase in batch size over MN5)
- Fully Sharded Data Parallel (FSDP) + Tensor Parallelism (TP = 4)

## Qwen3-VL-8B @ Marenostrum 5
### Setup Configuration
- Fully Sharded Data Parallel (FSDP) + Tensor Parallelism (TP = 4)
- Scaled across 64, 128, and 256 GPUs.
- seq_len: ~7k to 8192 tokens.

### Results
Scalability throughput with 8B model on Marenostrum 5:
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/186567ce-5a76-4625-9e1c-587d0f44c24c" />
