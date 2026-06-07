from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B-Base",
    local_dir="/data/151-1/users/tockier/qwen_finetune/cache/qwen3_1_7b",
)
