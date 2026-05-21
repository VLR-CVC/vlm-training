from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    local_dir="/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)
