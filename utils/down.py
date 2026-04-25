from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-27B",
    local_dir="/e/project1/reformo/ockier1/qwen_models/qwen3_5_27b",
)
