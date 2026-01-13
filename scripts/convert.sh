source /gpfs/projects/ehpc391/env_variables.sh
source /gpfs/projects/ehpc391/envs/qwen3/bin/activate

python3 \
    convertion_script.py \
    --base_model "/gpfs/scratch/ehpc391/qwen_finetune/cache/" \
    --checkpoint "/gpfs/scratch/ehpc391/qwen_finetune/checkpoints_32_packed/checkpoint-step-2000" \
    --output "/gpfs/scratch/ehpc391/qwen_finetune/checkpoints_32_packed/final_model" \
