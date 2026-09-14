import os
import sys
import torch
import wandb
import transformers
from itertools import cycle

import time

from transformers import AutoProcessor

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.replicate import replicate

from torch.profiler import record_function

# data imports
from megatron.energon import get_train_dataset, get_loader, get_savable_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

# training imports
from train.config_manager import ConfigManager
from train.config import Config, ModelType
from torch.distributed.pipelining.microbatch import _Replicate
from train.logger import init_logger, redirect_rank_io, logger, Color
from train.infra import (
    get_mesh,
    get_tp_group,
    get_dp_group,
    get_pp_group,
    apply_float8,
    apply_float8_moe,
    apply_fsdp,
    apply_tp,
    apply_pp_qwen4,
    compile_model,
)
from train.utils import (
    clip_grad_norm_mixed,
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,

    init_qwen35,
    init_qwen3vl,
    init_qwen4,

    dist_mean,
    dist_max,
    dist_sum,
    dist_all_gather,

    select_text_model,
    select_vision_model,
    select_model_class,
    set_model,
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

torch._logging.set_logs(graph_code=True)
torch._inductor.config.fx_graph_cache = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
        self.wandb_args = cfg.wandb
        self.debug_mode = bool(os.environ.get("DEBUG", False))

        torch.distributed.init_process_group(backend='nccl')
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        self.mesh = get_mesh(self.training_args, self.world_size)
        self.tp_group = get_tp_group(self.mesh)
        self.dp_group = get_dp_group(self.mesh)
        self.pp_group = get_pp_group(self.mesh)
        self.pp_size = getattr(self.training_args, "pp_size", 1)

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

        set_determinism(seed=42 + self.local_rank, deterministic=True, world_mesh=self.mesh, debug_mode=self.debug_mode)

        if self.rank() == 0:
            if not os.path.exists(self.training_args.output_dir):
                os.makedirs(self.training_args.output_dir)

        if "Qwen4" in self.model_args.model_name or "Qwen3.8" in self.model_args.model_name:
            self.model_type = ModelType.Qwen4
        elif "Qwen3.5" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_5
        elif "Qwen3-VL" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_vl
        elif "Qwen3" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_text
        else:
            raise NotImplementedError(f"model not supported: {self.model_args.model_name}")

        self.model, self.cfg_model = select_model_class(self.model_type, self.model_args, self.training_args)

        # we calculate the flops per token used to get the MFU number
        num_params, self.flops_per_token = get_dense_model_nparams_and_flops(
            self.model_type,
            self.cfg_model,
            self.model,
            seq_len=int(self.data_args.seq_len),
        )

        # each rank does 1/tp of every layer and 1/pp of the layers, so its
        # share of the model's FLOPs is divided by both
        self.flops_per_token = self.flops_per_token / (
            self.training_args.tp_size * self.pp_size
        )

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

        # -- PIPELINE PARALLEL
        # Must run before the model reaches the GPU: `apply_pp_qwen4` drops the
        # modules this rank does not own while everything is still on CPU, and
        # moves only the stage across. That is what keeps the build-then-upcast
        # peak proportional to the stage instead of the whole model.
        self.pp_schedule = None
        self.pp_has_first_stage = True
        self.pp_has_last_stage = True
        if self.pp_size > 1:
            if self.model_type != ModelType.Qwen4:
                raise NotImplementedError(
                    "pipeline parallelism is only wired up for Qwen4 "
                    f"(got {self.model_type})"
                )

            from models.qwen4.utils import causal_lm_loss

            def pp_loss_fn(logits, labels):
                # The schedule calls backward on what this returns, so the
                # gradient-accumulation scaling has to happen here rather than
                # in `train_step`.
                return causal_lm_loss(logits, labels) / self.current_accum_target

            (
                self.pp_microbatches,
                self.pp_schedule,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = apply_pp_qwen4(
                self.model, self.mesh, self.training_args, self.device, pp_loss_fn
            )
        else:
            # MOVE TO cuda:{self.local_rank}
            self.model.to(self.device)

        if self.model_type == ModelType.Qwen4:
            # FlashQLA's architecture check runs against the current CUDA device
            # at import time, so the backend can only be resolved once the model
            # is on its GPU. `auto` takes FlashQLA when it is usable *and* has a
            # backward kernel (SM90 Hopper, SM100/103), otherwise FLA.
            from models.qwen4.model import set_gdn_backend

            backend = set_gdn_backend(os.environ.get("QWEN4_GDN_BACKEND", "auto"))
            logger.info(f"Qwen4 GatedDeltaNet kernel backend: {backend}")
        
        if self.training_args.random_init:
            if self.model_type == ModelType.Qwen4:
                logger.info('initilizing decoder and projecter of Qwen4')
                # the broadcast inside is a world collective; under PP the
                # ranks hold different trees, so it must be off everywhere
                init_qwen4(self.model, broadcast=self.pp_size == 1)
            elif self.model_type == ModelType.Qwen3_5:
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

        # The optimizer's master copy. Compute is bf16 regardless (autocast, and
        # FSDP's MixedPrecisionPolicy); this is the precision the update lands
        # in. See `Training.master_dtype`.
        master_dtype = getattr(self.training_args, "master_dtype", "float32")
        if master_dtype == "float32":
            self.model = self.model.float()
        elif master_dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)
        else:
            raise ValueError(
                f"master_dtype must be 'float32' or 'bfloat16', got {master_dtype!r}"
            )

        logger.info("model loaded")

        # Before TP / compile / FSDP: the swap replaces `nn.Linear` modules, so
        # the parallelism plans and the compiled graphs have to see the ones
        # that will actually run.
        if self.training_args.float8:
            converted, total = apply_float8(
                self.model,
                self.training_args.float8_recipe,
                tp_size=self.training_args.tp_size,
            )
            logger.info(
                f"float8 ({self.training_args.float8_recipe}): converted {converted} "
                f"of {total} linear layers"
            )
            if not self.training_args.compile:
                logger.warning(
                    "float8 without compile is slower than bf16: the scaling ops "
                    "stay unfused. Set compile = true."
                )

        if self.training_args.float8_moe:
            swapped, blocks = apply_float8_moe(
                self.model, self.training_args.float8_moe_recipe
            )
            logger.info(
                f"float8 MoE ({self.training_args.float8_moe_recipe}): swapped "
                f"{swapped} expert parameters across {blocks} expert blocks"
            )

        if self.training_args.tp_size > 1:
            apply_tp(self.model, self.model_type, self.tp_group, self.training_args.async_tp)

        ac_memory_budget = getattr(self.training_args, "ac_memory_budget", None)
        if ac_memory_budget is not None:
            import torch._functorch.config as functorch_config
            functorch_config.activation_memory_budget = ac_memory_budget
            logger.info(f"activation memory budget set to {ac_memory_budget}")

        # `Module.compile` on a stage would have to be re-applied per stage and
        # the schedule's own graph breaks make it mostly moot; leave it off.
        if self.pp_size > 1 and self.training_args.compile:
            logger.info("compile disabled: not supported together with pp_size > 1")

        # Compile before sharding, the way torchtitan orders it. `Module.compile`
        # wraps `_call_impl`, so hooks installed later stay outside the compiled
        # region; wrapping FSDP first puts its pre-forward hook inside, where
        # Dynamo hits the `torch._dynamo.disable` it uses to skip FSDP hooks.
        if self.training_args.compile and self.pp_size == 1:
            compile_model(self.model)
            logger.info("model (will be) compiled")

        skip_fsdp = (
            self.training_args.data_parallel == 'fsdp'
            and self.dp_group.size() == 1
            and self.training_args.tp_size == 1
        )
        if skip_fsdp:
            # A 1-rank data-parallel mesh shards nothing, but `fully_shard` still
            # allocates a reduce-scatter buffer the size of each parameter
            # group's gradients and runs the collective into it. Under PP=4 on
            # 4 GPUs (dp=1, tp=1) that buffer is what runs a stage out of memory
            # in `post_backward`, to move data from a rank to itself.
            #
            # The `tp_size == 1` half is not optional. On a 1-rank mesh
            # `fully_shard` is *not* a no-op: it turns every parameter into a
            # DTensor, and under TP that is what makes the parameter set
            # uniform. Skip it there and the TP-sharded parameters are DTensors
            # while everything outside the TP plan stays a plain tensor, so the
            # first optimizer step dies with "aten._foreach_mul_.Scalar: got
            # mixed torch.Tensor and DTensor".
            logger.info(
                "data_parallel='fsdp' skipped: dp mesh has 1 rank and tp_size "
                "is 1, so sharding is a no-op and its collective buffers are "
                "pure cost"
            )
        elif self.training_args.data_parallel == 'fsdp':
            # bf16 compute + comms, fp32 master shards + fp32 gradient reduce
            mp_policy = None
            if self.training_args.bfloat16:
                from torch.distributed.fsdp import MixedPrecisionPolicy
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                )
            apply_fsdp(self.model_type, self.model, mesh=self.dp_group, mp_policy=mp_policy)
        elif self.training_args.data_parallel == 'ddp':
            if self.training_args.tp_size > 1:
                # `replicate` is FSDP2-based in torch 2.11 and does not manage
                # parameters that TP already turned into DTensors: they vanish
                # from `model.parameters()` after the first forward, never
                # receive a gradient, and the optimizer silently updates
                # nothing. Only the modules left out of the TP plan would train.
                raise ValueError(
                    "data_parallel='ddp' is not supported together with tp_size > 1; "
                    "use data_parallel='fsdp'."
                )
            # params stay fp32; torch.autocast in train_step handles bf16 compute
            self.model = replicate(self.model, device_mesh=self.dp_group)
        else:
            raise Exception('invalid sharding strategy for Data Parallel')

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

        self.tokens_seen = 0
        self.tokens_seen_assistant = 0

        self.ntokens_since_last_log = 0
        self.total_ntokens_since_last_log = 0
        self.samples_since_last_log = 0
        self.grad_norm = 0.0

        self.time_last_log = time.perf_counter()
        self.color = Color()

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        return self.rank() == 0

    def _all_ranks_have_batch(self, local_has_batch: bool) -> bool:
        """Collective agreement on whether EVERY rank still has data."""
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

        # the "global learning rate" is the LLM learning rate
        # See `Training.adamw_impl`: "foreach" allocates a temporary as large as
        # the parameter set inside the step, which is what caps the model size.
        impl = getattr(self.training_args, "adamw_impl", "foreach")
        _TORCHAO_ADAMW = {"fp8": "AdamWFp8", "8bit": "AdamW8bit", "4bit": "AdamW4bit"}
        if impl not in ("fused", "foreach", "forloop", *_TORCHAO_ADAMW):
            raise ValueError(
                "adamw_impl must be one of 'fused', 'foreach', 'forloop', "
                f"{sorted(_TORCHAO_ADAMW)}, got {impl!r}"
            )
        if impl in _TORCHAO_ADAMW:
            # Quantizes the two moments (scale per 256-element block) and leaves
            # the master weights alone: 10 bytes per parameter at fp8/8bit
            # against 16, which is what lifts the 4-GPU size ceiling.
            import torchao.optim as ao_optim

            self.optimizer = getattr(ao_optim, _TORCHAO_ADAMW[impl])(
                optimizer_grouped_parameters,
                lr=self.training_args.lr_llm,
                weight_decay=weight_decay,
                bf16_stochastic_round=self.training_args.adamw_stochastic_round,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.training_args.lr_llm,
                foreach=impl == "foreach",
                fused=impl == "fused",
                weight_decay=weight_decay,
            )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.training_args
        )
        return self.optimizer, self.scheduler

    def save_checkpoint(self):
        if self.pp_size > 1:
            # `dcp.save` is handed the bare optimizer, so its state is keyed by
            # the parameter's *position* in `param_groups` ("optimizer.state.2"),
            # not by name. Every rank holds the same parameters under TP and
            # FSDP so the positions agree; under PP they do not, and two stages
            # write different tensors to the same key -- torch catches it as
            # "key has overlapping chunks" only because the shapes happen to
            # differ. Writing a checkpoint that silently mixes two stages'
            # optimizer state is worse than not writing one.
            #
            # The fix is `torch.distributed.checkpoint.state_dict.get_state_dict`
            # / `set_state_dict`, which key optimizer state by parameter FQN.
            # That changes the on-disk layout, so existing checkpoints would no
            # longer resume -- not a decision to make silently.
            if self.if_log_rank():
                logger.info(
                    "checkpointing is skipped under pp_size > 1: the optimizer "
                    "state is keyed by parameter index, which collides across "
                    "pipeline stages"
                )
            return

        state_dict = {
            "model": self.model,
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "tokens_seen_assistant": self.tokens_seen_assistant,
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

    def load_checkpoint(self, step_num):
        state_dict = {
            "model": self.model,
            "step": step_num,
            "tokens_seen": None,
            "tokens_seen_assistant": None,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        loaded = load_distributed_checkpoint(
            self.training_args.output_dir, step_num, state_dict, self.rank()
        )
        if loaded is None:
            return

        self.tokens_seen = loaded['tokens_seen']
        self.tokens_seen_assistant = loaded['tokens_seen_assistant']
        self.global_step = loaded['step']
        self.optimizer = loaded['optimizer']
        self.scheduler = loaded['scheduler']

        if self.data_args.save_dataloader_state:
            load_dataloader_state(
                self.training_args.output_dir,
                step_num,
                self.data_loader,
                self.data_rank,
            )

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

            batch['attention_mask'], batch['original_mask'] = batch['cu_seqlens'], batch['attention_mask']

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=torch.cuda.current_device(), non_blocking=True)

            # the first and last numbers in cu_seqlens do not count towards the sample count
            # (pun intented)
            batch_samples = batch['attention_mask'].shape[0] - 2
            
            ntokens_batch = (batch['input_ids'] != self.pad_token_id).sum().item()
            ntokens_batch_assistant = (batch['labels'] != -100).sum().item()

            self.batch_efficiency = (ntokens_batch / self.data_args.seq_len ) * 100
            self.tokens_seen_assistant += ntokens_batch_assistant
            self.tokens_seen += ntokens_batch
            self.ntokens_since_last_log += ntokens_batch
            self.total_ntokens_since_last_log += self.data_args.seq_len
            self.samples_since_last_log += batch_samples

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch

    def _gather_perf(self, time_delta):
        tps = self.ntokens_since_last_log / time_delta
        flops_per_sec = (self.flops_per_token * self.total_ntokens_since_last_log) / time_delta
        tflops_per_sec = flops_per_sec / 1e12
        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100
        peak_mem_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        local = torch.tensor(
            [tps, self.train_step_delta, self.fwd_bwd_time, tflops_per_sec, mfu, peak_mem_gib],
            dtype=torch.float32,
            device=self.device,
        )
        return dist_all_gather(local, torch.distributed.group.WORLD)

    def log(self, avg_loss, max_loss, global_tokens, global_assistant_tokens, global_samples, lr, time_delta, gathered=None):
        tps = self.ntokens_since_last_log / time_delta

        step_flops = self.flops_per_token * self.total_ntokens_since_last_log
        flops_per_sec = step_flops / time_delta
        tflops_per_sec = flops_per_sec / 1e12

        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}{self.global_step}{color.reset} - "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.blue}tps {tps:.2f} "
                f"{color.magenta}mfu {mfu:.1f}% "
                f"{color.cyan}tflops {tflops_per_sec:.1f} "
                f"{color.reset}"
                f"gnorm {self.grad_norm:.3f} "
                f"time {self.train_step_delta:.3f}s "
                f"fwd {self.fwd_bwd_time:.3f}s "
                f"data_pct {data_time_pct:.2f}% "
                f"nsamples {global_samples} "
                f"batch_util {self.batch_efficiency:.1f}% "
        )

        log_metrics = {
            "train/loss": avg_loss,
            "train/max_loss": max_loss,
            "train/tokens_seen": global_tokens,
            "train/assistant_tokens_seen": global_assistant_tokens,
            "train/num_samples": global_samples,
            "train/lr": lr,
            "train/grad_norm": self.grad_norm,
            "train/batch_efficiency": self.batch_efficiency,

            # performance related
            "perf/tokens_per_second": tps,
            "perf/data_time_pct": data_time_pct,
            "perf/step_time": self.train_step_delta,
            "perf/fwd_bwd_time": self.fwd_bwd_time,
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
        }

        if gathered is not None:
            log_metrics.update(topk_metrics(gathered, self.wandb_args.top_k))

        wandb.log(log_metrics, step=self.global_step)

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def _pp_forward_backward(self, batch):
        """One pipeline step. Returns the unscaled loss, broadcast to all stages.

        The schedule runs forward and backward itself and calls `pp_loss_fn`,
        which already divides by the accumulation target -- so unlike the
        single-stage path there is no `.backward()` here.
        """
        batch = dict(batch)
        labels = batch.pop("labels", None)
        input_ids = batch.pop("input_ids")
        # every stage needs the token ids (MRoPE, image scatter, PLE), so they
        # go back in as a replicated kwarg on top of the positional input
        batch["input_ids"] = input_ids

        # The schedule chunks the positional input and the target along dim 0.
        # The dataloader emits one packed (1, total) row, so n > 1 tiles the
        # same row -- enough to exercise the schedule, not to train on.
        n = self.pp_microbatches
        tiled_input_ids = input_ids.repeat(n, 1) if n > 1 else input_ids
        tiled_labels = (
            labels.repeat(n, 1) if (n > 1 and labels is not None) else labels
        )

        losses = [] if self.pp_has_last_stage else None
        target = tiled_labels if self.pp_has_last_stage else None

        # Every kwarg is per-batch metadata that each microbatch needs whole:
        # `attention_mask` is cu_seqlens for the packed row and the pixel
        # tensors are indexed by a mask over it. The spec must name exactly the
        # keys this step passes, and the dataloader's key set varies with the
        # sample, so it is rebuilt here rather than fixed at construction.
        self.pp_schedule._kwargs_chunk_spec = {k: _Replicate() for k in batch}

        with record_function("pp_forward_backward"):
            with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bfloat16):
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        tiled_input_ids, **batch, target=target, losses=losses
                    )
                else:
                    self.pp_schedule.step(**batch, target=target, losses=losses)

        # `losses` holds the per-microbatch scaled losses and only the last
        # stage has them; undo the accumulation scaling and share the number so
        # every rank logs and checkpoints the same value.
        if losses:
            loss = torch.stack(losses).sum() * self.current_accum_target
        else:
            loss = torch.zeros((), device=self.device, dtype=torch.float32)
        torch.distributed.all_reduce(loss, group=self.pp_group.get_group())
        return loss

    def train_step(self, batch, optimizer):
        if self.training_args.debug_batch_stats:
            write_batch_stats(
                batch, self._batch_stats_dir, self.global_step, self.current_accum_count
            )

        s_model = time.perf_counter()
        if self.pp_size > 1:
            loss = self._pp_forward_backward(batch)
        else:
            with record_function("forward_pass"):
                with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bfloat16):
                    outputs = self.model(
                        **batch
                    )
                    loss = outputs.loss

            with record_function("backward_pass"):
                scaled_loss = loss / self.current_accum_target
                with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bfloat16):
                    scaled_loss.backward()

        self.fwd_bwd_time = time.perf_counter() - s_model

        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
            with record_function("optimizer_step"):
                if self.training_args.max_grad_norm > 0:
                    # returns the pre-clip global norm as a float, and copes with
                    # grads spread over several meshes (TP + modules left out of
                    # the TP plan).
                    # Under PP this norm is stage-local: each stage clips
                    # against its own gradients, not the pipeline's. Matching
                    # single-stage behaviour needs an all-reduce of the squared
                    # norm across `pp_group` before scaling.
                    # TODO: make the clip pipeline-global.
                    self.grad_norm = clip_grad_norm_mixed(
                        self.model.parameters(), self.training_args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            n = self.training_args.clear_cache_vram
            if n > 0 and self.global_step % n == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]['lr']

            self.global_step += 1

            avg_loss, max_loss, global_tokens, global_assistant, global_samples = (
                dist_mean(loss, self.dp_group),
                dist_max(loss, self.dp_group),
                dist_sum(
                    torch.tensor(
                        self.tokens_seen, dtype=torch.int64, device=self.device
                    ),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(
                        self.tokens_seen_assistant, dtype=torch.int64, device=self.device
                    ),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(self.samples_since_last_log, dtype=torch.int32, device=self.device),
                    self.dp_group,
                )
            )

            time_delta = time.perf_counter() - self.time_last_log
            self.train_step_delta = time_delta / self.current_accum_target

            gathered = None
            if self.wandb_args.log_topk:
                gathered = self._gather_perf(time_delta)

            if self.if_log_rank():
                self.log(avg_loss, max_loss, global_tokens, global_assistant, global_samples, lr, time_delta, gathered)

            self.total_ntokens_since_last_log = 0
            self.ntokens_since_last_log = 0
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
            largest_step = find_latest_checkpoint_step(self.training_args.output_dir)
            if largest_step is None:
                logger.info('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")
            optimizer, scheduler = self.load_checkpoint(largest_step)

        prof_ctx, _cprof, _CPROF_START, _CPROF_STOP = build_debug_profiler(
            self.debug_mode, self.training_args.output_dir, self.rank(), self.if_log_rank()
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
        if self.if_log_rank():
            logger.info(f"finalizing run at step {self.global_step}")
            logger.info(f"tokens seen: {self.tokens_seen}")
            logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
            logger.info("saving final checkpoint...")

        self.save_checkpoint()

        # make sure every rank has finished writing before tearing down NCCL
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # patch how error are reported
    real_stdout = redirect_rank_io()

    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    init_logger(stream=real_stdout)

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
