from dataclasses import dataclass, field

@dataclass
class Model:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    run_name: str = "default"
    project_name: str = "test_151_qwen_vl"
    entity_name: str = "bsc_runs"

    train_llm: bool = True
    train_mlp: bool = True
    train_vit: bool = False

@dataclass
class Training:
    resume: bool = True

    output_dir: str = "checkpoints"
    cache_dir: str = "/data/users/tockier/qwen_finetune/cache"

    bfloat16: bool = True

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    save_steps: int = 1000
    garbage_steps: int = 100

    tpi_multiplier: float = 1.0
    eps: float =  1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    scheduler_steps: int = 10_000

    compile: bool = True
    shard: bool = True

@dataclass
class Data:
    data_path: str = "/gpfs/scratch/ehpc391/fv_parquet/"

    seq_len: float = 6144
    queue_len: int = 32

    start_idx: int = 0
    end_idx: int = 0

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)

    config: str = '/home-local/vlm-training/cvc_config.toml'
