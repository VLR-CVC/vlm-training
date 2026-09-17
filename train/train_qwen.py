import datetime
import faulthandler
import os
import signal
import sys
import torch
import wandb
import transformers
from itertools import cycle

import time

from transformers import AutoProcessor

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.replicate import replicate
from torch.distributed.checkpoint.state_dict import _init_optim_state

from torch.profiler import record_function

# data imports
from megatron.energon import get_train_dataset, get_loader, get_savable_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

# training imports
from train.config_manager import ConfigManager
from train.config import Config, ModelType

TITAN_MODEL_TYPES = (ModelType.Qwen3_5_TT, ModelType.Qwen3_VL_TT)
from train.logger import init_logger, redirect_rank_io, logger, Color

# Pinned-buffer slot order for the deferred log record, written by `_stage_log`
# and read by `log`: loss_sum, tokens_seen, tokens_seen_assistant, samples
# (those four SUM-reduced over dp), loss_max (MAX-reduced), then three local
# values -- ntokens_since_last_log, ntokens_last_batch, grad_norm.
_LOG_SLOTS = 9
from train.infra import (
    get_mesh,
    get_tp_group,
    get_dp_group,
    apply_fsdp,
    apply_tp,
    compile_model,
)
from train.utils import (
    build_adamw,
    MASTER_DTYPES,
    clip_grad_norm_mixed,
    zero_grads_if_nonfinite_,
    round_max_seqlen,
    cast_master_weights,
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,

    init_qwen35,
    init_qwen3vl,

    dist_sum_max,
    dist_all_gather,

    select_text_model,
    select_vision_model,
    select_model_class,
    set_model,
    set_model_titan,
    load_text_model,
    load_vision_model,

    build_optimizer_param_groups,
    topk_metrics,
)
from train.checkpoint import (
    save_distributed_checkpoint,
    load_distributed_checkpoint,
    find_latest_checkpoint_step,
    save_dataloader_state,
    load_dataloader_state,
)
from train.training_debug import (
    write_batch_stats,
    dump_cprofile,
    build_debug_profiler,
)
from train.flops_estimation import get_dense_model_nparams_and_flops

torch._inductor.config.fx_graph_cache = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def _apply_chat_template(processor, path: str) -> None:
    """Override the processor's chat template, and warn about the Qwen default.

    Qwen's shipped template renders `<think>` spans only for turns after the last
    user query -- correct for generation, silently destructive for SFT on
    reasoning data. It cost every qwen3.5 plotqa measurement in `PERFORMANCE.md`
    roughly three quarters of its text before anyone noticed, so a run that keeps
    the default gets told once, loudly, rather than being left to find out from a
    token count months later.
    """
    if path and path != "NULL":
        with open(path) as f:
            processor.chat_template = f.read()
        logger.info(f"chat template overridden from {path}")
        return

    # The comparison, not the bare name: `ns.last_query_index` is still computed
    # in the fixed template (it is used for tool-call handling), so matching the
    # variable alone warns on a template that is already correct.
    template = getattr(processor, "chat_template", None) or ""
    if "> ns.last_query_index" in template:
        c = Color()
        logger.warning(
            f"{c.red}chat template in {processor.__class__.__name__} drops "
            f"<think> spans from all but the last turn. Fine for inference, "
            f"wrong for SFT on reasoning data. Set "
            f"training.chat_template=assets/chat_template_sft.jinja if this "
            f"dataset has reasoning.{c.reset}"
        )

class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
        self.wandb_args = cfg.wandb
        self.debug_mode = bool(os.environ.get("DEBUG", False))

        # QWEN_NCCL_TIMEOUT_S: the first step compiles every block on every rank, and a
        # rank whose data forces a recompile can lag the rest past NCCL's 10 min default
        # (TITAN_MIGRATION_v2.md S4, jobs 1842180/1842181). Mesh groups split from this
        # one inherit it.
        timeout = datetime.timedelta(seconds=int(os.environ.get("QWEN_NCCL_TIMEOUT_S", 600)))
        torch.distributed.init_process_group(backend='nccl', timeout=timeout)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        self.mesh = get_mesh(self.training_args, self.world_size)
        self.tp_group = get_tp_group(self.mesh)
        self.dp_group = get_dp_group(self.mesh)

        self.device = torch.device(f"cuda:{self.local_rank}")
        if self.if_log_rank():
            wandb.init(
                name=self.wandb_args.run_name,
                project=self.wandb_args.project_name,
                entity=self.wandb_args.entity_name,
                config={
                    **vars(self.model_args),
                    **vars(self.training_args),
                    **vars(self.data_args),
                    "mesh": self.mesh,
                    "world_size": self.world_size,
                    "dp_group": self.dp_group,
                    "tp_group": self.tp_group,
                },
            )

            logger.info('using directory:')
            logger.info(os.getcwd())
            logger.info(f"world_size: {self.world_size}")
            logger.info("starting finetune job")
            logger.info(f"mesh: {self.mesh}")

            logger.info(self.model_args)
            logger.info(self.training_args)
            logger.info(self.data_args)

        set_determinism(seed=42 + self.local_rank, deterministic=self.training_args.deterministic, world_mesh=self.mesh, debug_mode=self.debug_mode)

        if self.rank() == 0:
            if not os.path.exists(self.training_args.output_dir):
                os.makedirs(self.training_args.output_dir)

        if "Qwen3.5" in self.model_args.model_name and self.model_args.impl == "titan":
            self.model_type = ModelType.Qwen3_5_TT
        elif "Qwen3-VL" in self.model_args.model_name and self.model_args.impl == "titan":
            self.model_type = ModelType.Qwen3_VL_TT
        elif "Qwen3.5" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_5
        elif "Qwen3-VL" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_vl
        elif "Qwen3" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_text
        else:
            raise NotImplementedError(f"model not supported: {self.model_args.model_name}")

        if (
            self.training_args.skip_nonfinite_grads
            and self.training_args.max_grad_norm <= 0
        ):
            logger.warning(
                "skip_nonfinite_grads is on but max_grad_norm <= 0, so no global "
                "grad norm is computed and the guard cannot run. One non-finite "
                "gradient will end this run."
            )

        if self.model_type in TITAN_MODEL_TYPES:
            self._setup_titan_model()
        else:
            self._setup_native_model()

        # get rank of local GPU that belongs to the DP group
        data_rank = self.dp_group.get_local_rank()
        data_world_size = self.dp_group.size()

        # ranks sharing a data_rank (a TP group) read identical data, so only the
        # TP-group leader persists the (shared) dataloader state on checkpoint.
        self.data_rank = data_rank
        self.is_data_leader = self.tp_group is None or self.tp_group.get_local_rank() == 0

        logger.info('sharding/parallelism applied')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.training_args.model_dir,
            model_max_length=int(self.data_args.seq_len),
            padding_side="right",
            use_fast=False,
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.processor = AutoProcessor.from_pretrained(
            self.training_args.model_dir,
            max_pixels=1048576,
        )
        _apply_chat_template(self.processor, self.training_args.chat_template)

        if self.model_type not in TITAN_MODEL_TYPES:
            self.model = set_model(self.model_type, self.model_args, self.model)

        worker_config = WorkerConfig(
            rank=data_rank,
            world_size=data_world_size,
            data_parallel_group=self.dp_group,
            num_workers=2,
        )

        task_encoder, extra_ds_kwargs = build_task_encoder(
            self.data_args,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        ds = get_train_dataset(
            self.data_args.data_path,
            batch_size=1,
            repeat=self.data_args.repeat,
            shuffle_buffer_size=self.data_args.shuffle_buffer_size,
            max_samples_per_sequence=self.data_args.max_samples_per_sequence,
            task_encoder=task_encoder,
            worker_config=worker_config,
            **extra_ds_kwargs,
        )

        if self.training_args.debug_batch_stats:
            self._batch_stats_dir = os.path.join(
                self.training_args.output_dir, "batch_debug", f"rank_{self.rank()}"
            )
            os.makedirs(self._batch_stats_dir, exist_ok=True)

        # creation of dataloader
        if self.data_args.save_dataloader_state:
            self.data_loader = get_savable_loader(ds)
        else:
            self.data_loader = get_loader(ds)

        self.setup_accumulation(self.training_args.tpi_multiplier)

        self.global_step = 0
        self.micro_step = 0

        # Token counters live on the device. They come out of reductions over
        # the batch (`(input_ids != pad).sum()`), so reading them on the host is
        # a sync at the top of every step -- it drains the stream and the CPU
        # stops running ahead of the GPU. Nothing in the step needs their value;
        # only the logger does, and it can have it a step late.
        self.tokens_seen = torch.zeros((), dtype=torch.int64, device=self.device)
        self.tokens_seen_assistant = torch.zeros((), dtype=torch.int64, device=self.device)
        self.ntokens_since_last_log = torch.zeros((), dtype=torch.int64, device=self.device)
        self.ntokens_last_batch = torch.zeros((), dtype=torch.int64, device=self.device)
        self.grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        # cumulative, device-side: read a step late with everything else
        self.nonfinite_skips = torch.zeros((), dtype=torch.float32, device=self.device)

        # host-side already: one is `+= seq_len`, the other comes from a tensor
        # *shape*. Neither reads device memory.
        self.total_ntokens_since_last_log = 0
        self.samples_since_last_log = 0

        # Deferred logging -- see `_stage_log` / `_flush_log`.
        self._log_host = torch.zeros(_LOG_SLOTS, dtype=torch.float64).pin_memory()
        self._log_event = torch.cuda.Event()
        self._log_pending = None
        self._log_wait = 0.0   # host wait for the staged copy; should stay ~0
        self._log_emit = 0.0   # formatting + wandb.log
        self._flag_stream = None

        self.time_last_log = time.perf_counter()
        self.color = Color()

    def _setup_native_model(self):
        self.model, self.cfg_model = select_model_class(self.model_type, self.model_args, self.training_args)

        # we calculate the flops per token used to get the MFU number
        num_params, self.flops_per_token = get_dense_model_nparams_and_flops(
            self.model_type,
            self.cfg_model,
            self.model,
            seq_len=int(self.data_args.seq_len),
        )

        self.flops_per_token = self.flops_per_token / self.training_args.tp_size

        # peak bf16 TFLOPs per GPU, used for the MFU number
        # SXM H100/GH200 (MN5): 989.4 ; L40S: 362
        self.peak_tflops_per_gpu = 989.4

        logger.info(f"Number params: {num_params}")

        if self.training_args.load_text_model:
            self.text_model = select_text_model(self.training_args)
            self.model = load_text_model(self.model, self.text_model)
            del self.text_model

        if self.training_args.load_vision_model:
            self.vision_model = select_vision_model(self.training_args)
            self.model = load_vision_model(self.model, self.vision_model)
            del self.vision_model

        # MOVE TO cuda:{self.local_rank}
        self.model.to(self.device)
        
        if self.training_args.random_init:
            if self.model_type == ModelType.Qwen3_5:
                logger.info('initilizing decoder and projecter of Qwen3.5')
                init_qwen35(self.model)
            elif self.model_type == ModelType.Qwen3_vl:
                logger.info('initilizing projector of Qwen3-VL')
                init_qwen3vl(self.model)
            else:
                logger.info('model not initlized, incompatible')

        # replace flash_attn
        self.model.train()
        self.optimizer = None # its defined later on

        # we only cast the weights
        master_dtype = cast_master_weights(self.model, self.training_args.master_dtype)
        if master_dtype is torch.bfloat16 and not self.training_args.bf16_compute:
            logger.warning(
                "master_dtype='bfloat16' with bf16_compute=false: nothing is in "
                "fp32 any more. Autocast is what keeps softmax, the norms and "
                "the loss in fp32, and it also gates the FSDP "
                "MixedPrecisionPolicy below, so gradients get reduced in bf16 too."
            )

        if self.model_type == ModelType.Qwen3_vl:
            from models.qwen3_vl.model import set_loss_chunk_mb

            set_loss_chunk_mb(self.training_args.loss_chunk_mb)

        if self.model_type == ModelType.Qwen3_5:
            from models.qwen3_5.utils import set_loss_chunk_mb

            set_loss_chunk_mb(self.training_args.loss_chunk_mb)

            from models.qwen3_5.compile_ops import set_native_kernels

            set_native_kernels(self.training_args.native_kernels)
            logger.info(
                "qwen3.5 fused-kernel backward: "
                + ("native (saved intermediates)" if self.training_args.native_kernels
                   else "recompute-in-backward")
            )

        logger.info("model loaded")

        if self.training_args.tp_size > 1:
            apply_tp(self.model, self.model_type, self.tp_group, self.training_args.async_tp)

        ac_memory_budget = getattr(self.training_args, "ac_memory_budget", None)
        if ac_memory_budget is not None:
            import torch._functorch.config as functorch_config
            functorch_config.activation_memory_budget = ac_memory_budget
            logger.info(f"activation memory budget set to {ac_memory_budget}")

        if self.training_args.log_graph_code:
            # One FX-graph source dump per rank. 42 MB at 256 nodes, and it is
            # written while dynamo is tracing, i.e. during the part of startup
            # that is already the bottleneck.
            torch._logging.set_logs(graph_code=True)

        # Compile before sharding, the way torchtitan orders it.
        if self.training_args.dynamo_recompile_limit > 0:
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = self.training_args.dynamo_recompile_limit
            logger.info(
                f"dynamo recompile_limit = {self.training_args.dynamo_recompile_limit}"
            )

        if self.training_args.compile:
            compile_model(
                self.model,
                fsdp=self.training_args.data_parallel == 'fsdp',
                dynamic=self.training_args.compile_dynamic,
                compile_gdn=self.training_args.compile_gdn,
                block_mode=self.training_args.compile_block_mode,
                head_mode=self.training_args.compile_head_mode,
                vision_mode=self.training_args.compile_vision,
            )
            logger.info("model (will be) compiled")

        if self.training_args.data_parallel == 'fsdp':
            # bf16 compute + comms, fp32 master shards + fp32 gradient reduce
            mp_policy = None
            if self.training_args.bf16_compute:
                from torch.distributed.fsdp import MixedPrecisionPolicy
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=MASTER_DTYPES[self.training_args.grad_reduce_dtype],
                )
            apply_fsdp(
                self.model_type, self.model, mesh=self.dp_group, mp_policy=mp_policy,
                reshard_after_forward_policy=self.training_args.reshard_after_forward,
            )
        elif self.training_args.data_parallel == 'ddp':
            # params stay at master_dtype; torch.autocast in train_step handles bf16 compute
            self.model = replicate(self.model, device_mesh=self.dp_group)
        else:
            raise Exception('invalid sharding strategy for Data Parallel')

    def _setup_titan_model(self):
        """Qwen3.5 via `models/qwen3_5_tt`, Qwen3-VL via `models/qwen3_vl_tt`
        (TITAN_MIGRATION_v2.md).

        torchtitan's order: build on meta -> freeze -> cast master dtype ->
        per-block compile -> FSDP/replicate -> `to_empty` -> load HF (DCP straight
        into the shards) or init. No rank ever holds a materialised full model.
        """
        from models.qwen3_5.config import Qwen3_5Config
        from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
        from models.qwen3_5_tt.configs import resolve_model_config
        from train.parallel.parallel_dims import ParallelDims
        from train.parallel.parallelize import parallelize_qwen3_5

        tp = self.training_args.tp_size
        # spmd_types looks the TP group up from a thread-local mesh; backward must
        # run on the thread that set it (torchtitan init_distributed does the same)
        torch.autograd.set_multithreading_enabled(False)
        # ponytail: a second set of process groups next to get_mesh()'s (dp, tp) mesh,
        # which the dataloader and logging still use. Both place rank r in DP group
        # r // tp. Fold them into one mesh when the native path is retired.
        self.parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=-1, cp=1, tp=tp, pp=1, ep=1, world_size=self.world_size
        )
        self.parallel_dims.build_mesh()

        model_dir = self.training_args.model_dir
        config_path = resolve_model_config(
            self.model_args.model_config, model_dir, self.model_args.use_model_dir_config
        )
        logger.info(f"titan: architecture from {config_path}")
        if self.model_type == ModelType.Qwen3_VL_TT:
            from models.qwen3_vl.model import Qwen3VLConfig

            self.cfg_model = Qwen3VLConfig.from_json(str(config_path))
            flops_model_type = ModelType.Qwen3_vl
        else:
            self.cfg_model = Qwen3_5Config.from_json(str(config_path))
            flops_model_type = ModelType.Qwen3_5
        seq_len = int(self.data_args.seq_len)
        enable_sp = tp > 1 and self.training_args.sequence_parallel
        if enable_sp and seq_len % tp:
            raise ValueError(f"sequence parallel needs seq_len ({seq_len}) divisible by tp ({tp})")
        self.model = build_meta(
            config_path, seq_len=seq_len, tp=tp, enable_sp=enable_sp,
            attn_backend=self.model_args.attn_backend, decoder_mask=self.model_args.decoder_mask,
        )

        # flops are a function of the config; the native estimator reads the same fields
        num_params = sum(p.numel() for p in self.model.parameters())
        _, self.flops_per_token = get_dense_model_nparams_and_flops(
            flops_model_type, self.cfg_model, self.model, seq_len=seq_len
        )
        # per GPU, as the native path does: tokens are counted per TP group
        self.flops_per_token = self.flops_per_token / tp
        self.peak_tflops_per_gpu = 989.4
        logger.info(f"Number params: {num_params}")

        set_model_titan(self.model_args, self.model)

        master_dtype = cast_master_weights(self.model, self.training_args.master_dtype)
        logger.info(f"titan: master weights in {master_dtype}")

        if self.training_args.dynamo_recompile_limit > 0:
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = self.training_args.dynamo_recompile_limit

        # TP (Module.parallelize) -> per-block fullgraph compile -> FSDP on the
        # storage mesh. attn_gym's fused GDN kernel takes fp16/bf16 only, so compute
        # is bf16 regardless of `bf16_compute`.
        parallelize_qwen3_5(
            self.model,
            self.parallel_dims,
            mode=self.training_args.data_parallel,
            compile=self.training_args.compile,
            param_dtype=torch.bfloat16,
            reduce_dtype=MASTER_DTYPES[self.training_args.grad_reduce_dtype],
            reshard_after_forward=self.training_args.reshard_after_forward == "always",
        )
        logger.info(f"titan: tp={tp} sp={enable_sp} compile={self.training_args.compile} "
                    f"data_parallel={self.training_args.data_parallel}")

        materialize(self.model, self.device)
        if self.training_args.random_init:
            with torch.no_grad():
                self.model.init_states(buffer_device=self.device)
            logger.info("titan: random init (init_states)")
        else:
            load_hf(self.model, model_dir)
            logger.info(f"titan: loaded HF weights from {model_dir}")
        self.model.train()
        self.optimizer = None

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        return self.rank() == 0

    def _all_ranks_have_batch(self, local_has_batch: bool) -> bool:
        """Collective agreement on whether EVERY rank still has data.

        The answer is needed before the step runs, so unlike the logging
        counters this one cannot be deferred. It can be kept off the training
        stream, though: the flag is built from a host bool and has no data
        dependency on anything the model computed. Running it on its own stream
        means the `.item()` waits for a one-int all-reduce instead of for every
        kernel the previous step queued.
        """
        if self._flag_stream is None:
            self._flag_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self._flag_stream):
            flag = torch.tensor(
                [1 if local_has_batch else 0], dtype=torch.int32, device=self.device
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            return bool(flag.item())

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        weight_decay = self.training_args.weight_decay
        lr_by_group = {
            "mlp": self.training_args.lr_mlp,
            "vit": self.training_args.lr_vit,
            "llm": self.training_args.lr_llm,
        }

        optimizer_grouped_parameters = build_optimizer_param_groups(
            self.model.named_parameters(),
            lr_by_group,
            weight_decay,
            log=self.if_log_rank(),
        )

        # different types of AdamW supported
        self.optimizer = build_adamw(
            optimizer_grouped_parameters,
            lr=self.training_args.lr_llm,
            weight_decay=weight_decay,
            impl=self.training_args.adamw_impl,
            stochastic_round=self.training_args.adamw_stochastic_round,
            betas=self.training_args.adam_betas,
            eps=self.training_args.eps,
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.training_args
        )
        return self.optimizer, self.scheduler

    def save_checkpoint(self):
        # Hand reserved-but-unallocated blocks back to the driver first.
        # Checkpointing creates new NCCL communicators, and those allocate
        # *outside* the caching allocator. Job 1782009 finished all 60 steps at
        # 10240 with 91.6 GiB reserved and then died in `ncclCuMemAlloc` with
        # "Cuda failure 2 'out of memory'" -- the training loop fit, the
        # communicator did not. This costs a sync and some re-allocation on the
        # next step, once per save.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        state_dict = {
            "model": self.model,
            "step": self.global_step,
            "tokens_seen": int(self.tokens_seen),
            "tokens_seen_assistant": int(self.tokens_seen_assistant),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }
        save_distributed_checkpoint(
            self.training_args.output_dir,
            self.global_step,
            state_dict,
            self.rank(),
            self.if_log_rank(),
        )

        if self.data_args.save_dataloader_state:
            save_dataloader_state(
                self.training_args.output_dir,
                self.global_step,
                self.data_loader,
                self.data_rank,
                self.is_data_leader,
                self.if_log_rank(),
            )

    def load_checkpoint(self, step_num, ckpt_dir=None):
        # where to read the resume checkpoint from; defaults to output_dir
        ckpt_dir = ckpt_dir or self.training_args.output_dir

        # init AdamW state by calling step() with zero grads
        _init_optim_state(self.optimizer)

        state_dict = {
            "model": self.model,
            "step": step_num,
            "tokens_seen": None,
            "tokens_seen_assistant": None,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        loaded = load_distributed_checkpoint(
            ckpt_dir, step_num, state_dict, self.rank()
        )
        if loaded is None:
            return

        self.tokens_seen.fill_(loaded['tokens_seen'])
        self.tokens_seen_assistant.fill_(loaded['tokens_seen_assistant'])
        self.global_step = loaded['step']
        self.optimizer = loaded['optimizer']
        self.scheduler = loaded['scheduler']

        if self.data_args.save_dataloader_state:
            if self.data_args.restore_dataloader_state:
                load_dataloader_state(
                    ckpt_dir,
                    step_num,
                    self.data_loader,
                    self.data_rank,
                )
            elif self.if_log_rank():
                logger.info("restore_dataloader_state=false; data stream starts from scratch")

        if self.if_log_rank():
            logger.info(f"{self.color.red}load checkpoint at step {self.global_step}{self.color.reset}")
        return self.optimizer, self.scheduler

    def may_save(self):
        if self.global_step % self.training_args.save_steps == 0:
            return True
        return False

    def batch_generator(self):
        data_iter = iter(self.data_loader)

        while True:
            data_start_time = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                return

            if batch['cu_seqlens'].ndim > 1:
                batch['cu_seqlens'].squeeze_()

            if 'image_grid_thw' in batch and batch['image_grid_thw'].ndim > 1:
                # do not use squeeze because we need to have two dims
                batch['image_grid_thw'] = batch['image_grid_thw'][0]

            if self.model_type in TITAN_MODEL_TYPES:
                yield self._titan_batch(batch, data_start_time)
                continue

            batch['attention_mask'], batch['original_mask'] = batch['cu_seqlens'], batch['attention_mask']

            # While cu_seqlens is still on the host. The model needs this as a
            # Python int for `varlen_attn`; computing it after the H2D copy
            # below costs a device sync at the top of every step.
            cu = batch['attention_mask']
            # rounded to a power of two: an exact value recompiles the decoder
            # blocks on nearly every step, see `round_max_seqlen`
            batch['max_seqlen'] = round_max_seqlen(int((cu[1:] - cu[:-1]).max()))

            grid = batch.get('image_grid_thw')
            if grid is not None and grid.numel():
                # the vision tower's own varlen max, same reasoning
                vis_seg = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0])
                batch['vision_max_seqlen'] = round_max_seqlen(int(vis_seg.max()))

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=torch.cuda.current_device(), non_blocking=True)

            # the first and last numbers in cu_seqlens do not count towards the sample count
            # (pun intented)
            batch_samples = batch['attention_mask'].shape[0] - 2
            
            # no `.item()`: these two used to be the first host sync of the
            # step, draining everything the previous step had queued.
            # `batch_efficiency` is derived from `ntokens_last_batch` at log time.
            ntokens_batch = (batch['input_ids'] != self.pad_token_id).sum()
            ntokens_batch_assistant = (batch['labels'] != -100).sum()

            self.ntokens_last_batch.copy_(ntokens_batch)
            self.tokens_seen_assistant.add_(ntokens_batch_assistant)
            self.tokens_seen.add_(ntokens_batch)
            self.ntokens_since_last_log.add_(ntokens_batch)
            self.total_ntokens_since_last_log += self.data_args.seq_len
            self.samples_since_last_log += batch_samples

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch

    def _titan_batch(self, batch, data_start_time):
        """Energon row -> `models/qwen3_5_tt` input, on the host, then H2D.

        Same counters as the native path, from the same tensors: `ntokens` counts
        non-pad input ids, the assistant count uses the *unshifted* labels.
        """
        from data.titan_batch import to_titan_batch

        input_ids = batch['input_ids'].reshape(-1)
        cu = batch['cu_seqlens'].reshape(-1)
        ntokens_batch = (input_ids != self.pad_token_id).sum()
        ntokens_batch_assistant = (batch['labels'] != -100).sum()

        out = to_titan_batch(
            batch,
            image_token_id=self.cfg_model.image_token_id,
            video_token_id=self.cfg_model.video_token_id,
            spatial_merge_size=self.cfg_model.vision.spatial_merge_size,
        )
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device=self.device, non_blocking=True)
        if 'pixel_values' in out:
            out['pixel_values'] = out['pixel_values'].to(torch.bfloat16)

        self.ntokens_last_batch.fill_(int(ntokens_batch))
        self.tokens_seen_assistant.add_(int(ntokens_batch_assistant))
        self.tokens_seen.add_(int(ntokens_batch))
        self.ntokens_since_last_log.add_(int(ntokens_batch))
        self.total_ntokens_since_last_log += self.data_args.seq_len
        self.samples_since_last_log += cu.shape[0] - 2
        self.data_time_delta = time.perf_counter() - data_start_time
        return out

    def train_step_titan(self, batches, optimizer):
        """One optimizer step over `batches` (the accumulation window), torchtitan's
        way (`trainer.py:872`): count valid tokens across every micro-batch and every
        DP rank first, then give each token the weight 1/global_tokens."""
        from train.titan_step import forward_backward

        s_model = time.perf_counter()
        accumulated, local_tokens, denom = forward_backward(
            self.model,
            batches,
            dp_group=self.dp_group,
            special_tokens={"image_id": self.cfg_model.image_token_id},
            ddp=self.training_args.data_parallel == "ddp",
            loss_chunks=self.training_args.titan_loss_chunks,
            compile_loss=self.training_args.compile,
            parallel_dims=self.parallel_dims,
        )
        self.fwd_bwd_time = time.perf_counter() - s_model

        with record_function("optimizer_step"):
            params = [p for p in self.model.parameters() if p.grad is not None]
            if self.training_args.max_grad_norm > 0:
                self.grad_norm = clip_grad_norm_mixed(params, self.training_args.max_grad_norm)
            else:
                self.grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
            gn = self.grad_norm.full_tensor() if hasattr(self.grad_norm, "full_tensor") else self.grad_norm
            # torchtitan's check: stop before the optimizer can write a non-finite
            # update, without a host sync (TITAN_MIGRATION_v2.md D5)
            torch._assert_async(
                torch.isfinite(accumulated) & torch.isfinite(gn),
                "loss or gradient norm is not finite; stopping before the optimizer step",
            )
            optimizer.step()
            optimizer.zero_grad()

        n = self.training_args.clear_cache_vram
        if n > 0 and self.global_step % n == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        lr = optimizer.param_groups[0]['lr']
        self.global_step += 1
        time_delta = time.perf_counter() - self.time_last_log
        self.train_step_delta = time_delta / len(batches)

        gathered = None
        topk_interval = max(1, self.wandb_args.topk_interval)
        if self.wandb_args.log_topk and self.global_step % topk_interval == 0:
            gathered = self._gather_perf(time_delta)

        # `accumulated` is local_sum / global_tokens, so the SUM over DP is the exact
        # global mean. `_stage_log` divides its summed slot by dp_size, hence * dp.
        # The max slot gets this rank's own per-token mean.
        dp_size = self.dp_group.size()
        local_mean = accumulated * denom / max(local_tokens, 1)
        self._flush_log()
        self._stage_log(accumulated * dp_size, lr, time_delta, gathered, loss_max=local_mean)

        self.total_ntokens_since_last_log = 0
        self.ntokens_since_last_log.zero_()
        self.samples_since_last_log = 0
        torch.cuda.reset_peak_memory_stats(self.device)
        self.time_last_log = time.perf_counter()
        self.current_accum_count = 0
        self.current_accum_target = next(self.accum_schedule)

    def _gather_perf(self, time_delta):
        """Per-rank perf row, gathered across the world.

        Assembled on device. `ntokens_since_last_log` is a device tensor now, so
        building this row on the host would reintroduce the sync that
        `_stage_log` exists to avoid. The returned tensor is read on the host by
        `topk_metrics`, one step later, on the log rank only.
        """
        flops_per_sec = (self.flops_per_token * self.total_ntokens_since_last_log) / time_delta
        tflops_per_sec = flops_per_sec / 1e12
        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100
        peak_mem_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        local = torch.empty(6, dtype=torch.float32, device=self.device)
        local[0] = self.ntokens_since_last_log / time_delta  # tps
        local[1] = self.train_step_delta
        local[2] = self.fwd_bwd_time
        local[3] = tflops_per_sec
        local[4] = mfu
        local[5] = peak_mem_gib
        return dist_all_gather(local, torch.distributed.group.WORLD)

    def _stage_log(self, loss, lr, time_delta, gathered, loss_max=None):
        """Reduce this step's counters and start a non-blocking copy of the
        result into pinned memory. Touches the host for nothing that came off
        the device; `_flush_log` reads the values on the next step, by which
        time the copy has long landed.

        Both collectives here are unavoidable -- the loss and the token counts
        are genuinely per-rank. What was avoidable was reading them.
        """
        loss64 = loss.detach().to(torch.float64).reshape(1)
        sums = torch.cat([
            loss64,
            self.tokens_seen.to(torch.float64).reshape(1),
            self.tokens_seen_assistant.to(torch.float64).reshape(1),
            torch.full(
                (1,), float(self.samples_since_last_log),
                dtype=torch.float64, device=self.device,
            ),
        ])
        mx_in = loss64 if loss_max is None else loss_max.detach().to(torch.float64).reshape(1)
        sums, mx = dist_sum_max(sums, mx_in, self.dp_group)

        # Popped on every rank so the dict does not grow on the ones that
        # never read it. Empty unless QWEN_SECTION_TIMING=1.
        sections = {}
        if self.model_type == ModelType.Qwen3_5:
            from models.qwen3_5.model import pop_section_ms

            sections = pop_section_ms()
        elif self.model_type == ModelType.Qwen3_vl:
            from models.qwen3_vl.model import pop_section_ms

            sections = pop_section_ms()

        # on every rank, not just the log rank: `full_tensor()` on a sharded
        # DTensor is a collective, and a collective one rank skips is a hang.
        # It is a no-op when the norm comes back replicated, which is the
        # normal case, but the uniform call is what makes that safe to assume.
        grad_norm = self.grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        # every rank ran the collectives above; only the log rank reads values
        if not self.if_log_rank():
            return

        grad_norm = torch.as_tensor(
            grad_norm, dtype=torch.float64, device=self.device
        ).reshape(1)

        vec = torch.cat([
            sums, mx,
            self.ntokens_since_last_log.to(torch.float64).reshape(1),
            self.ntokens_last_batch.to(torch.float64).reshape(1),
            grad_norm,
            self.nonfinite_skips.to(torch.float64).reshape(1),
        ])
        self._log_host.copy_(vec, non_blocking=True)
        self._log_event.record()

        gib = 1024 ** 3
        self._log_pending = {
            "step": self.global_step,
            "sections": sections,
            "lr": lr,
            "time_delta": time_delta,
            "train_step_delta": self.train_step_delta,
            "fwd_bwd_time": self.fwd_bwd_time,
            "data_time_delta": self.data_time_delta,
            "total_ntokens": self.total_ntokens_since_last_log,
            # peak since the previous log; `reset_peak_memory_stats` runs just
            # after this call. Host-side bookkeeping, not a device read.
            "peak_alloc": torch.cuda.max_memory_allocated(self.device) / gib,
            "peak_resv": torch.cuda.max_memory_reserved(self.device) / gib,
            # cost of the flush that ran immediately before this stage
            "log_wait": self._log_wait,
            "log_emit": self._log_emit,
            "gathered": gathered,
        }

    def _flush_log(self):
        """Emit the previous step's record. The copy it reads was queued a full
        step ago, so the event wait below is already satisfied and returns
        without stalling."""
        rec = self._log_pending
        if rec is None:
            return
        self._log_pending = None

        # Two numbers, both reported on the *next* record. `log_wait` is the
        # host waiting on a copy that was queued a full step ago -- if it is
        # not ~0, the deferral is not buying anything and something upstream
        # is still draining the stream. `log_emit` is the part people assume
        # is expensive: string formatting and `wandb.log`. Measure before
        # trimming what gets printed.
        t0 = time.perf_counter()
        self._log_event.synchronize()
        t1 = time.perf_counter()
        self.log(rec, self._log_host.tolist())
        self._log_wait = t1 - t0
        self._log_emit = time.perf_counter() - t1

    def log(self, rec, h):
        """`rec` is the host-side snapshot taken when the step was staged; `h`
        is the reduced counter vector read back out of pinned memory. Slot order
        is fixed by `_stage_log`."""
        time_delta = rec["time_delta"]

        # SUM/n rather than a second collective on ReduceOp.AVG. `dp_group` is
        # None on a tp-only mesh, where the reductions above ran over WORLD.
        dp_size = (
            self.dp_group.size() if self.dp_group is not None
            else torch.distributed.get_world_size()
        )
        avg_loss = h[0] / dp_size
        global_tokens = int(h[1])
        global_assistant_tokens = int(h[2])
        global_samples = int(h[3])
        max_loss = h[4]
        tps = h[5] / time_delta
        batch_efficiency = (h[6] / self.data_args.seq_len) * 100
        grad_norm = h[7]
        nonfinite_skips = int(h[8])

        step_flops = self.flops_per_token * rec["total_ntokens"]
        flops_per_sec = step_flops / time_delta
        tflops_per_sec = flops_per_sec / 1e12

        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100

        color = self.color

        data_time_pct = (rec["data_time_delta"] / time_delta) * 100

        # Peak since the log before this one. Rank 0 only, which is the point:
        # it is the rank the console shows, and an OOM elsewhere is invisible
        # here. `perf_topk/mem_gib_*` carries the cross-rank spread, which on
        # the 9B sweep is ~17 GiB wide because vision-token counts differ per
        # packed batch. Reserved as well as allocated: the 8192 OOM had 15 GiB
        # sitting in reserved-but-unallocated blocks, and allocated alone does
        # not show it.
        peak_alloc = rec["peak_alloc"]
        peak_resv = rec["peak_resv"]

        logger.info(
            f"{color.red}{rec['step']}{color.reset} - "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.blue}tps {tps:.2f} "
                f"{color.magenta}mfu {mfu:.1f}% "
                f"{color.cyan}tflops {tflops_per_sec:.1f} "
                f"{color.reset}"
                f"gnorm {grad_norm:.3f} "
                # only when it fires: a permanent "skips 0" is noise, and
                # a non-zero count is the one thing worth noticing here
                + (f"{color.red}skips {nonfinite_skips}{color.reset} " if nonfinite_skips else "")
                + f"time {rec['train_step_delta']:.3f}s "
                f"fwd {rec['fwd_bwd_time']:.3f}s "
                f"mem {peak_alloc:.1f}/{peak_resv:.1f}G "
                f"data_pct {data_time_pct:.2f}% "
                f"nsamples {global_samples} "
                f"batch_util {batch_efficiency:.1f}% "
        )

        log_metrics = {
            "train/loss": avg_loss,
            "train/max_loss": max_loss,
            "train/tokens_seen": global_tokens,
            "train/assistant_tokens_seen": global_assistant_tokens,
            "train/num_samples": global_samples,
            "train/lr": rec["lr"],
            "train/grad_norm": grad_norm,
            "train/nonfinite_skips": nonfinite_skips,
            "train/batch_efficiency": batch_efficiency,

            # performance related
            "perf/tokens_per_second": tps,
            "perf/data_time_pct": data_time_pct,
            "perf/step_time": rec["train_step_delta"],
            "perf/fwd_bwd_time": rec["fwd_bwd_time"],
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
            "perf/log_wait_ms": rec["log_wait"] * 1e3,
            "perf/log_emit_ms": rec["log_emit"] * 1e3,
            "perf/mem_gib": peak_alloc,
            "perf/mem_reserved_gib": peak_resv,
        }

        if rec["sections"]:
            # forward only, and only under QWEN_SECTION_TIMING=1
            total = sum(rec["sections"].values())
            parts = "  ".join(
                f"{k} {v:.1f}ms {100*v/total:.0f}%"
                for k, v in sorted(rec["sections"].items(), key=lambda kv: -kv[1])
            )
            logger.info(f"  fwd sections ({total:.1f}ms): {parts}")
            log_metrics.update(
                {f"perf_fwd/{k}_ms": v for k, v in rec["sections"].items()}
            )

        if rec["gathered"] is not None:
            log_metrics.update(topk_metrics(rec["gathered"], self.wandb_args.top_k))

        wandb.log(log_metrics, step=rec["step"])

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train_step(self, batch, optimizer):
        if self.training_args.debug_batch_stats:
            write_batch_stats(
                batch, self._batch_stats_dir, self.global_step, self.current_accum_count
            )

        s_model = time.perf_counter()
        with record_function("forward_pass"):
            with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bf16_compute):
                outputs = self.model(
                    **batch
                )
                loss = outputs.loss

        with record_function("backward_pass"):
            scaled_loss = loss / self.current_accum_target
            with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bf16_compute):
                scaled_loss.backward()

        self.fwd_bwd_time = time.perf_counter() - s_model

        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
            with record_function("optimizer_step"):
                if self.training_args.max_grad_norm > 0:
                    self.grad_norm = clip_grad_norm_mixed(
                        self.model.parameters(), self.training_args.max_grad_norm
                    )
                    if self.training_args.skip_nonfinite_grads:
                        # after clipping, which is what turns one non-finite
                        # gradient into all of them
                        self.nonfinite_skips += zero_grads_if_nonfinite_(
                            self.model.parameters(), self.grad_norm
                        )
                optimizer.step()
                optimizer.zero_grad()

            n = self.training_args.clear_cache_vram
            if n > 0 and self.global_step % n == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]['lr']

            self.global_step += 1

            time_delta = time.perf_counter() - self.time_last_log
            self.train_step_delta = time_delta / self.current_accum_target

            gathered = None
            topk_interval = max(1, self.wandb_args.topk_interval)
            if self.wandb_args.log_topk and self.global_step % topk_interval == 0:
                # collective: every rank takes this branch or none does.
                # `global_step` is incremented above and is rank-invariant.
                gathered = self._gather_perf(time_delta)

            # flush first: `_stage_log` overwrites the pinned buffer that
            # `_flush_log` reads.
            self._flush_log()
            self._stage_log(loss, lr, time_delta, gathered)

            self.total_ntokens_since_last_log = 0
            self.ntokens_since_last_log.zero_()
            self.samples_since_last_log = 0
            torch.cuda.reset_peak_memory_stats(self.device)
            self.time_last_log = time.perf_counter()

            self.current_accum_count = 0
            self.current_accum_target = next(self.accum_schedule)

            return True

        return False

    def train(self):
        data_iterator = self.batch_generator()

        optimizer, scheduler = self.create_optimizer()
        if self.training_args.resume_checkpoint:
            load_dir = self.training_args.load_dir
            if load_dir in ("NULL", "", None):
                load_dir = self.training_args.output_dir
            resume_step = self.training_args.start_step
            if resume_step <= 0:
                resume_step = find_latest_checkpoint_step(load_dir)
            if resume_step is None:
                logger.info('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")
            optimizer, scheduler = self.load_checkpoint(resume_step, load_dir)

        prof_ctx, _cprof, _CPROF_START, _CPROF_STOP = build_debug_profiler(
            self.debug_mode, self.training_args.output_dir, self.rank(), self.if_log_rank()
        )
        profile_dir = os.environ.get("PROFILE_DIR")
        if profile_dir and not self.debug_mode:
            # Kernel-level trace without DEBUG=1, which also changes determinism
            # settings and so the run being measured. Steps are optimizer steps
            # (`prof.step()` below runs once per loop iteration; the titan path does
            # a whole accumulation window per iteration).
            from torch.profiler import ProfilerActivity, profile, schedule
            from train.training_debug import make_trace_handler

            os.makedirs(profile_dir, exist_ok=True)
            wait = int(os.environ.get("PROFILE_WAIT", "40"))
            prof_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=3, active=2, repeat=1),
                on_trace_ready=make_trace_handler(profile_dir, self.rank(), self.if_log_rank()),
            )
        _cprof_active = False

        with prof_ctx as prof:
            
            # training loop
            while self.global_step < self.training_args.total_steps:
                self.micro_step += 1

                if self.debug_mode and not _cprof_active and self.global_step >= _CPROF_START:
                    _cprof.enable()
                    _cprof_active = True

                try:
                    # retrieve the batch
                    batch = next(data_iterator)
                    local_has_batch = True
                except StopIteration:
                    batch = None
                    local_has_batch = False

                # check for data exhaustion
                if not self.data_args.repeat:
                    if not self._all_ranks_have_batch(local_has_batch):
                        if self.if_log_rank():
                            logger.info(f"data exhausted on at least one rank at step {self.global_step}; stopping")
                        break
                elif not local_has_batch:
                    break

                # TRAINING STEP
                if self.model_type in TITAN_MODEL_TYPES:
                    # the whole accumulation window up front: the loss normaliser
                    # is the step's global valid-token count
                    batches = [batch]
                    have_window = True
                    for _ in range(self.current_accum_target - 1):
                        try:
                            batches.append(next(data_iterator))
                        except StopIteration:
                            have_window = False
                            break
                    # Same collective agreement as for the first micro-batch above:
                    # a rank that stops alone leaves the others waiting in the
                    # token-count all_reduce of `forward_backward`.
                    if not self.data_args.repeat:
                        if not self._all_ranks_have_batch(have_window):
                            if self.if_log_rank():
                                logger.info(f"data exhausted mid-window on at least one rank at step {self.global_step}; stopping")
                            break
                    elif not have_window:
                        break
                    self.train_step_titan(batches, optimizer)
                    optimizer_updated = True
                else:
                    optimizer_updated = self.train_step(batch, optimizer)

                if prof is not None:
                    prof.step()

                # this is a flag because of grad_accm
                if optimizer_updated:
                    scheduler.step()

                    if _cprof_active and self.global_step >= _CPROF_STOP:
                        _cprof.disable()
                        _cprof_active = False
                        dump_cprofile(_cprof, self.training_args.output_dir, self.rank())

                    if self.may_save() and self.global_step < self.training_args.total_steps:
                        self.save_checkpoint()

        self.end_run()

    def end_run(self):
        self._flush_log()  # the last step staged a record nobody has emitted yet
        if self.if_log_rank():
            logger.info(f"finalizing run at step {self.global_step}")
            logger.info(f"tokens seen: {int(self.tokens_seen)}")
            logger.info(f"assistant tokens seen: {int(self.tokens_seen_assistant)}")
            logger.info("saving final checkpoint...")

        self.save_checkpoint()

        # make sure every rank has finished writing before tearing down NCCL
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # `kill -USR1 <pid>` prints every thread's Python stack to stderr: how to see where
    # a hung rank is without py-spy
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # patch how error are reported
    #real_stdout = redirect_rank_io()

    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    #init_logger(stream=real_stdout)
    init_logger()

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
