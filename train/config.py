from dataclasses import dataclass, field

@dataclass
class Model:
    model_name: str = "NULL"

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
    cache_dir: str = "NULL"
    text_cache_dir: str = "NULL"
    load_text_model: bool = False

    bfloat16: bool = True
    ac: bool = True

    split_vit_attn: bool = False

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    save_steps: int = 1000
    garbage_steps: int = 100

    tpi_multiplier: float = 1.0
    eps: float =  1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # SCHEDULER --
    # "wsd" or "cosine"
    scheduler_type: str = "wsd"
    scheduler_steps: int = 1_000
    warmup_steps: int = 50

    # percentage of final decay steps, only for WSD
    wsd_decay_ratio: float = 0.1

    # percentage of minumum lr to decay, only for COSINE
    min_lr_ratio: float = 0.1
    # --

    data_parallel: str = "ddp" # fsdp, ddp, hybrid
    tp_size: int = 1 # 1 means disabled
    # (world_size // tp_size, tp_size)
    # first dim is handled by data parallel

    compile: bool = True
    random_init_mlp: bool = False

@dataclass
class Data:
    data_path: str = "NULL"

    seq_len: float = 4096
    queue_len: int = 32

    start_idx: int = 0
    end_idx: int = 0

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)

    config: str = '/home-local/vlm-training/cvc_config.toml'
