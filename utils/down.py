from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    local_dir="/data/151-1/users/tockier/qwen_finetune/cache/qwen35_35b_a3",
)
