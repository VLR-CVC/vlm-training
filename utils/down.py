from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-397B-A17B",
    local_dir="/e/project1/reformo/ockier1/qwen_models/qwen35_397b_a17b",
)
