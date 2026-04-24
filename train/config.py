from dataclasses import dataclass, field
from enum import Enum, auto

class ModelType(Enum):
    Qwen3_5 = auto()
    Qwen3_vl = auto()
    Qwen3_text = auto()

@dataclass
class Model:
    # this defines the CLASS to initilize the model
    model_name: str = "NULL"
    """
    Supported:
    - Qwen3-VL
    - Qwen3.5
    """

    # which implementation of the model to use.
    # "hf"     -> HuggingFace transformers classes (default)
    # "native" -> our torch-native models under models/
    model_impl: str = "hf"

    # freeze model parts, its used by `utils.set_model`
    train_llm: bool = True
    train_mlp: bool = True
    train_vit: bool = False

@dataclass
class Wandb:
    run_name: str = "default"
    project_name: str = "test_151_qwen_vl"
    entity_name: str = "bsc_runs"

@dataclass
class Training:

    # ALWAYS CHANGE
    model_dir: str = "NULL"
    """
    This defines the model to be used. We perform `.from_pretrained`
    from this directory. The `AutoProcessor` is also defined with this.
    """

    # where to checkpoint
    output_dir: str = "checkpoints"

    # whether or not to load the text model
    load_text_model: bool = False
    text_model_dir: str = "NULL"

    # whether to resume from previous checkpoints or not
    resume_checkpoint: bool = False

    # "will checkpoint each `save_steps`"
    save_steps: int = 1000

    # execute with mixed precision
    bfloat16: bool = True

    lr_llm: float = 2e-6
    lr_mlp: float = 1e-5
    lr_vit: float = 1e-6

    # init of the projecter and deepstack layers
    random_init: bool = False

    # gradient accumulation
    tpi_multiplier: float = 1.0

    # more training args
    eps: float =  1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # SCHEDULER -----
    # "wsd" or "cosine"
    scheduler_type: str = "wsd"

    # the run will end
    # it defines the lenght of the scheduler
    total_steps: int = 1_000
    warmup_steps: int = 50

    # percentage of final decay steps, only for WSD
    wsd_decay_ratio: float = 0.1

    # percentage of minumum lr to decay, only for COSINE
    min_lr_ratio: float = 0.1
    # ---------------

    data_parallel: str = "ddp" # fsdp, ddp
    tp_size: int = 1 # 1 means disabled
    pp_size: int = 1 # 1 means disabled; supported values: 1, 2, 4
    """
    Use `fsdp` when you want to decrease usage to increase seq_len/batch_size.
    """


    # compiler flag for TP (goes faster)
    async_tp: bool = True

    # torch dynamo compiler
    compile: bool = True
    """
    Always on by default, unless you have an error.
    """

    # activation checkpointing
    ac_mode: str = "off"
    """
    ``ac_mode`` selects the policy:
      - "off"  : no AC
      - "full" : checkpoint the whole decoder block (max memory savings)
      - "sac"  : selective-op AC, saves ops in ``_op_sac_save_list``
    """

@dataclass
class Data:
    # must be an energon dataset. currently only CrudeWebdatasets are expected
    data_path: str = "NULL"

    shuffle_buffer_size: int = 100
    max_samples_per_sequence: int = 100
    packing_buffer_size: int = 0

    batch_size: int = 4
    """
    this currently determines if we use online datapacking or not. Default = sequence packing (4 batch size).
    given a non-zero integer, the energon task encoder builds the sequences with that number of samples.
    flash attention varlen with cu_seqlens is used either way, with a single sequence batch.

    Dispatch:
        data.text_dataset == True                 -> QwenTextEncoder
        data.text_dataset == False, batch_size>0  -> SingleBatchEncoder
        data.text_dataset == False, batch_size==0 -> PackedBatchEncoder (online datapacking)

    DO NOT forget to define `packing_buffer_size` if using online datapacking.
    """

    text_dataset: bool = False
    """
    when true, uses the text-ony task encoder (QwenTextEncoder)
    when false, dispatch according to everything above
    """

    seq_len: float = 4096
    """
    maximum sequence lenght used when building the batches. with a large batch size, the sequence may
    exceed this number. tune both parameters when using batch_size.

    you always want to have a fixed sized input into the decoder, as it helps with compilation.
    """

@dataclass
class Config:
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    wandb: Wandb = field(default_factory=Wandb)

    config: str = '/home/tockier/vlm-training/configs/cvc_config.toml'
