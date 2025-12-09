from dataclasses import dataclass, field

@dataclass
class Model:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    train_llm: bool = True
    train_mlp: bool = True
    train_vit: bool = False

@dataclass
class Training:
    output_dir: str = "checkpoints"
    cache_dir: str = "/gpfs/scratch/ehpc391/qwen_finetune/cache"

    bfloat16: bool = True

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    #gradient_accumulation_steps: int = 1 NOT supported

    save_steps: int = 1000
    garbage_steps: int = 100

    eps: float =  1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    compile: bool = True
    shard: bool = True

@dataclass
class Data:
    data_path: str = "/gpfs/scratch/ehpc391/fv_parquet/"

    seq_len: float = 6144
    data_flatten: bool = False
    data_packing: bool = False

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
