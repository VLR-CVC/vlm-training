from dataclasses import dataclass
from functools import partial

from train.config import ModelType

import torch
import torch._inductor.config

from torch.distributed.device_mesh import init_device_mesh

from train.logger import logger
from models.qwen4.utils import iter_layers
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
}

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.microbatch import _Replicate
from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layout: Placement | None,
        desired_input_layout: Placement | None,
        mod: nn.Module,
        inputs,
        device_mesh: DeviceMesh,
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert input_layout is not None
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            assert input_layout is not None
            assert desired_input_layout is not None
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(
        output_layout: Placement,
        use_local_output: bool,
        mod: nn.Module,
        outputs,
        device_mesh: DeviceMesh,
    ):
        # `GatedResidual` returns `(mixed_input, hyper_input, injection_weights)`,
        # so a module under this style can hand back a tuple rather than a
        # single tensor.
        def _one(out):
            if not isinstance(out, DTensor):
                return out
            if out.placements != (output_layout,):
                out = out.redistribute(placements=(output_layout,), async_op=True)
            # back to local tensor
            return out.to_local() if use_local_output else out

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(_one(o) for o in outputs)
        return _one(outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn,
                self.input_layout,
                self.desired_input_layout,
            ),
            partial(
                self._prepare_output_fn,
                self.output_layout,
                self.use_local_output,
            ),
        )

def get_mesh(training_args, world_size):
    """
    Creates a DeviceMesh from tp_size, pp_size and world_size.

    Returns ('dp', 'tp') when PP is off and ('pp', 'dp', 'tp') when it is on.
    The 2D shape is kept verbatim for pp_size == 1 so the TP/FSDP paths see
    exactly the mesh they were verified against.
    """
    tp_size = training_args.tp_size
    pp_size = getattr(training_args, "pp_size", 1)

    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(
            f"World size {world_size} is not divisible by tp_size * pp_size "
            f"({tp_size} * {pp_size})"
        )

    dp_size = world_size // (tp_size * pp_size)

    if pp_size == 1:
        return init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    # `pp` is outermost so a pipeline stage owns a contiguous rank block: rank
    # r holds stage r // (dp*tp), which keeps each stage's dp/tp collectives
    # inside one node before they cross a stage boundary.
    return init_device_mesh(
        "cuda", (pp_size, dp_size, tp_size), mesh_dim_names=("pp", "dp", "tp")
    )

def get_tp_group(mesh):
    if "tp" in mesh.mesh_dim_names:
        return mesh['tp']
    return None

def get_dp_group(mesh):
    if "dp" in mesh.mesh_dim_names:
        return mesh['dp']
    return None

def get_pp_group(mesh):
    if "pp" in mesh.mesh_dim_names:
        return mesh['pp']
    return None

# Linear layers that stay in high precision. Matched against the fully
# qualified name, so a substring covers every layer.
_FLOAT8_SKIP = (
    # the vision tower is a small fraction of the FLOPs and runs its own
    # attention path
    "visual",
    # the output projection: `vocab_size` wide, and quantizing the logits is
    # the one place where fp8's dynamic range shows up directly in the loss
    "lm_head",
    # Qwen4's QSA indexer. It is frozen (`set_model_qwen4` clears requires_grad)
    # so there is no backward GEMM to speed up, and its scores feed a `topk` --
    # fp8 rounding there changes which blocks are selected, not just by how much.
    "indexer",
)


def module_filter_float8_fn(mod: torch.nn.Module, fqn: str, divisor: int = 16):
    if any(skip in fqn for skip in _FLOAT8_SKIP):
        return False

    # fp8 GEMMs need both weight dimensions divisible by 16. `divisor` carries
    # the TP width on top of that, because this runs *before* `apply_tp`: a
    # colwise-sharded 4096x48 projection is 4096x24 by the time the kernel sees
    # it, and `torch._scaled_mm` then fails at trace time with
    # "Expected both dimensions of mat2 to be divisible by 16".
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % divisor != 0 or mod.out_features % divisor != 0:
            return False
    return True


def apply_float8(model, recipe: str = "tensorwise", tp_size: int = 1):
    """Swap `nn.Linear` for torchao's `Float8Linear`, returning how many moved.

    Call this before `apply_tp` / `compile_model` / `apply_fsdp`, the way
    torchtitan orders it: the swap replaces modules, and the parallelism plans
    and the compiled graphs have to see the modules they will actually run.

    What is *not* covered: the MoE experts. They are 3D `nn.Parameter`s behind
    `torch._grouped_mm`, not linear layers, and they hold most of the model's
    FLOPs. torchao has `MoETrainingConfig` for exactly that shape, but it lowers
    to `torch._scaled_grouped_mm`, which is SM90/SM100 only -- on SM120 it
    raises "only supported on CUDA devices with compute capability = [9.0,
    10.0]". So on these cards fp8 buys the attention, GDN, shared-expert,
    hyper-connection and PLE projections, and nothing else.

    One more thing that eats the win on Qwen4 specifically: `compile_model`
    compiles a QSA layer piecewise (`create_block_mask` is not traceable), and
    `self_attn` is not one of the pieces. The q/k/v/o projections in those
    layers therefore run their fp8 scaling unfused, in eager. With the default
    `full_attention_interval = 4` that is a quarter of the layers.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("float8 training needs a CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"float8 training needs SM89 or newer, this device is SM{major}{minor}"
        )

    config = Float8LinearConfig.from_recipe_name(recipe)
    before = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    convert_to_float8_training(
        model,
        module_filter_fn=partial(
            module_filter_float8_fn, divisor=16 * max(1, tp_size)
        ),
        config=config,
    )
    from torchao.float8.float8_linear import Float8Linear

    converted = sum(isinstance(m, Float8Linear) for m in model.modules())
    return converted, before


# The 3D parameters a `torch._grouped_mm` expert block owns. Ours mirror HF's
# names, which is also what torchao's MoE handler expects to find.
_MOE_EXPERT_PARAMS = ("gate_up_proj", "down_proj")

# Which architectures each MoE scaling type has a kernel for. fp8_rowwise goes
# through `torch._scaled_grouped_mm`, whose gate is a *set*, not a floor: SM90
# (H100, GH200) and SM100 (B100/B200, GB200) are in, and SM120 -- numerically
# higher than both -- is not. mxfp8 is Blackwell-datacenter only.
_MOE_SCALING_ARCHS = {
    "fp8_rowwise": {(9, 0), (10, 0)},
    "mxfp8": {(10, 0)},
    "mxfp8_wgrad_with_hp": {(10, 0)},
}


def _is_grouped_mm_experts(mod: torch.nn.Module, fqn: str) -> bool:
    """A stacked expert block: owns `gate_up_proj`/`down_proj` as 3D params."""
    own = {name for name, _ in mod.named_parameters(recurse=False)}
    if not set(_MOE_EXPERT_PARAMS) <= own:
        return False
    return all(mod.get_parameter(name).ndim == 3 for name in _MOE_EXPERT_PARAMS)


def _require_grouped_mm_fp8(recipe: str) -> None:
    archs = _MOE_SCALING_ARCHS.get(recipe)
    if archs is None:
        raise ValueError(
            f"unknown MoE scaling type {recipe!r}; expected one of "
            f"{sorted(_MOE_SCALING_ARCHS)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("float8 MoE training needs a CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability not in archs:
        wanted = " or ".join(f"SM{a}{b}" for a, b in sorted(archs))
        raise RuntimeError(
            f"float8 MoE training with {recipe!r} needs {wanted}; this device is "
            f"SM{capability[0]}{capability[1]}. `torch._scaled_grouped_mm` accepts "
            f"an exact set of architectures, so a newer card is not automatically "
            f"in it -- SM120 (RTX PRO 6000) is the case that catches people out. "
            f"Leave `float8_moe` off here; `float8` alone still covers the linears."
        )


def apply_float8_moe(model, recipe: str = "fp8_rowwise"):
    """Put the stacked MoE experts in fp8, returning how many params moved.

    The experts are 3D `nn.Parameter`s consumed by `torch._grouped_mm`, not
    linear layers, so `apply_float8` cannot see them -- and on a MoE model they
    hold most of the FLOPs. torchao's `MoETrainingConfig` swaps their data for a
    `ScaledGroupedMMTensor`, a subclass that overrides `torch._grouped_mm` with
    a differentiable scaled grouped GEMM and behaves like a plain tensor
    everywhere else.

    Call it next to `apply_float8`, before the parallelism plans: `apply_tp`
    turns these same parameters into DTensors, and the subclass has to be
    underneath that wrapper, not on top of it.

    **This path has never run its kernel in this repo.** `torch._scaled_grouped_mm`
    is SM90/SM100 only, and the dev box is SM120, so what is verified here is
    the guard, the parameter selection and the swap -- not the numerics and not
    the throughput. First real run will be on GH200 (SM90).
    """
    _require_grouped_mm_fp8(recipe)

    from torchao.prototype.moe_training.conversion_utils import (
        MoEScalingType,
        MoETrainingConfig,
    )
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
    from torchao.quantization import quantize_

    targets = [
        fqn for fqn, mod in model.named_modules() if _is_grouped_mm_experts(mod, fqn)
    ]
    if not targets:
        raise ValueError(
            "float8_moe is on but the model has no stacked expert block "
            f"(a module owning 3D {' and '.join(_MOE_EXPERT_PARAMS)} parameters)"
        )

    quantize_(
        model,
        MoETrainingConfig(scaling_type=MoEScalingType(recipe)),
        filter_fn=_is_grouped_mm_experts,
    )

    swapped = sum(
        isinstance(p.data, ScaledGroupedMMTensor) for p in model.parameters()
    )
    return swapped, len(targets)


@dataclass
class ACConfig:
    enabled: bool = True
    full: bool = False


def _make_sac_context_fn(save_list):
    def policy_fn(ctx, op, *args, **kwargs):
        if op in save_list:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def context_fn():
        return create_selective_checkpoint_contexts(policy_fn)

    return context_fn


def _apply_ac_to_transformer_block(
    block: torch.nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str = "",
    model_compile_enabled: bool = False,
    op_sac_save_list: set | None = None,
) -> torch.nn.Module:
    """Wrap one decoder block with activation checkpointing.

    ``ac_config.full=True``  → recompute the whole block in backward.
    ``ac_config.full=False`` → selective AC that saves the ops in
    ``op_sac_save_list`` and recomputes everything else.
    """
    if ac_config.full or not op_sac_save_list:
        return ptd_checkpoint_wrapper(
            block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
    return ptd_checkpoint_wrapper(
        block,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=_make_sac_context_fn(op_sac_save_list),
    )


def apply_ac(
    model: torch.nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.
    Returns:
        None
    """
    # see: https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    if ac_config.enabled:

        if not ac_config.full: op_sac_save_list = _op_sac_save_list
        else: op_sac_save_list = set()

        layers = model.get_submodule("layers")
        for layer_id, transformer_block in layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block,
                ac_config,
                base_fqn=f"layers.{layer_id}",
                model_compile_enabled=model_compile_enabled,
                op_sac_save_list=op_sac_save_list,
            )
            layers.register_module(layer_id, transformer_block)

def compile_model(model: torch.nn.Module):
    inner = model.model

    qwen4 = getattr(inner.language_model, "hyper_connection_mixer", None) is not None

    # `dynamic=True` marks every dimension symbolic, including the hidden size.
    # DTensor's sharding propagation then fails on the hyper-connection reshapes
    # ("s50*Max(1, (3072//s51)) is not tracked with proxy"). Automatic dynamic
    # shapes keep the constants static and only generalize the dims that vary.
    dynamic = None if qwen4 else True

    for transformer_block in iter_layers(inner.language_model.layers):
        if qwen4 and hasattr(transformer_block, "self_attn"):
            # A QSA layer builds its flex-attention `BlockMask` inside the
            # layer, and `create_block_mask` calls `warnings.warn`, which Dynamo
            # refuses to trace. Allowing the graph break instead is worse: the
            # resumed graph re-enters the FSDP-wrapped module and trips over
            # `_dynamo.disable` in FSDP's pre-forward hook. So compile the parts
            # of the layer that sit either side of the mask instead.
            for part in ("attn_hyper_connection", "mlp_hyper_connection", "mlp", "ple"):
                submodule = getattr(transformer_block, part, None)
                if submodule is not None:
                    submodule.compile(dynamic=dynamic, fullgraph=True, mode='default')
        else:
            transformer_block.compile(dynamic=dynamic, fullgraph=True, mode='default')

    # Qwen4 replaces the final RMSNorm with the hyper-connection mixer
    if getattr(inner.language_model, "norm", None) is not None:
        inner.language_model.norm = torch.compile(inner.language_model.norm, dynamic=True, fullgraph=True, mode='max-autotune-no-cudagraphs')
    elif getattr(inner.language_model, "hyper_connection_mixer", None) is not None:
        inner.language_model.hyper_connection_mixer = torch.compile(inner.language_model.hyper_connection_mixer, dynamic=dynamic, fullgraph=True, mode='max-autotune-no-cudagraphs')
    model.lm_head = torch.compile(model.lm_head, dynamic=dynamic, fullgraph=True, mode='max-autotune-no-cudagraphs')

    for transformer_block in inner.visual.blocks:
        transformer_block.compile(dynamic=True, fullgraph=False, mode='default')

    inner.visual.merger = torch.compile(inner.visual.merger, dynamic=True, fullgraph=True, mode='max-autotune-no-cudagraphs')

def apply_fsdp(model_type, model, **kwargs):
    if model_type == ModelType.Qwen4:
        apply_fsdp_qwen4(model, **kwargs)
    elif model_type == ModelType.Qwen3_text:
        apply_fsdp_qwen3(model, **kwargs)
    elif model_type == ModelType.Qwen3_vl:
        apply_fsdp_qwen3_vl(model, **kwargs)

def apply_fsdp_qwen3(model, mesh, reshard_after_forward_policy='never', mp_policy=None):
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()  # no-op: keeps params in their loaded dtype
    model = model.model

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = True
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # text decoder
    for transformer_block in model.layers:
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )

    fully_shard(
        [model.norm, model.embed_tokens],
        mesh=mesh,
        reshard_after_forward=reshard_after_forward_policy == "always",
        mp_policy=mp_policy,
    )

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

def apply_fsdp_qwen3_vl(model, mesh, reshard_after_forward_policy='never', mp_policy=None):
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()  # no-op: keeps params in their loaded dtype

    fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=False, mp_policy=mp_policy)

    model = model.model

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped

            # to be implemented (likely not)
            reshard_after_forward = True
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # text decoder
    for transformer_block in model.language_model.layers:
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )

    # vision encoder blocks
    for transformer_block in model.visual.blocks:
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )

    for mod in [model.visual.patch_embed, model.visual.pos_embed, model.visual.merger]:
        fully_shard(mod, mesh=mesh, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy)
    for deepstack_merger in model.visual.deepstack_merger_list:
        fully_shard(
            deepstack_merger,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )

    fully_shard(
        model.language_model.norm,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward_policy == "always",
        mp_policy=mp_policy,
    )

    fully_shard(
            model.language_model.embed_tokens,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward_policy == "always",
            mp_policy=mp_policy,
    )

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

def apply_tp(
        model,
        model_type: ModelType,
        tp_mesh,
        enable_tp_async,
):
    outer = model

    if getattr(outer, "cfg", None) is not None and outer.cfg.tie_word_embeddings:
        raise ValueError(
            "Tensor Parallelism is not supported for models with tie_word_embeddings=True. "
            "Use tp_size=1 for small models (e.g. 2B) that tie lm_head and embed_tokens."
        )

    if model_type == ModelType.Qwen4:
        _tp_decoder = _apply_tp_to_decoder_qwen4
    elif model_type == ModelType.Qwen3_5:
        _tp_decoder = _apply_tp_to_decoder_qwen3_5
    elif model_type == ModelType.Qwen3_vl:
        _tp_decoder = _apply_tp_to_decoder_qwen3_vl
    else:
        raise NotImplementedError()
    _tp_decoder(outer.model, tp_mesh, False, enable_tp_async)

    # `lm_head` is None on every pipeline stage but the last.
    if getattr(outer, "lm_head", None) is not None:
        parallelize_module(
            outer,
            tp_mesh,
            {
                "lm_head": ColwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Replicate(),
                    use_local_output=True,
                ),
            },
        )

    # they share the same ViT -- not implemented yet
    #_to_visual_encoder(model.visual, tp_mesh)

def _apply_tp_to_decoder_qwen3_vl(
    model,
    tp_mesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to the decoder without SequenceParallel.

    Unlike Qwen3's apply_non_moe_tp which uses SequenceParallel (hidden states
    are Shard(1) between blocks), this keeps hidden states as Replicate. This is
    necessary for VLM because vision scatter and DeepStack operate on the full
    sequence with boolean masks that aren't DTensor-aware.

    The trade-off is slightly higher activation memory (full sequence on each
    rank instead of 1/TP), but it avoids costly all-gather/re-shard at every
    vision scatter and DeepStack layer.
    """
    # Parallelize embedding, norm, and output — no SequenceParallel
    top_level_plan = {
        "language_model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "language_model.norm": NoParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    }
    parallelize_module(model, tp_mesh, top_level_plan)


    rowwise_parallel, colwise_parallel = (
        RowwiseParallel,
        ColwiseParallel,
    )

    # Apply TP to every transformer block's linear layers.
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP) and inserts I/O hooks that
    # convert local tensor ↔ DTensor at the norm boundary, keeping the block's
    # data path in local-tensor space as RowwiseParallel(use_local_output=True)
    # expects.

    model = model.language_model
    for transformer_block in model.layers:
        layer_plan = {
            "input_layernorm": NoParallel(),
            "post_attention_layernorm": NoParallel(),
            # Wrap attention inputs so rope_cache becomes a Replicate DTensor,
            # needed because wq/wk/wv outputs are DTensors and apply_rotary_emb
            # multiplies them with cos/sin from rope_cache.
            "self_attn": PrepareModuleInput(
                input_kwarg_layouts={
                    "hidden_states": Replicate(),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                },
            ), 
            "self_attn.q_proj": colwise_parallel(use_local_output=False),
            "self_attn.k_proj": colwise_parallel(use_local_output=False),
            "self_attn.v_proj": colwise_parallel(use_local_output=False),
            "self_attn.q_norm": SequenceParallel(sequence_dim=2),
            "self_attn.k_norm": SequenceParallel(sequence_dim=2),
            "self_attn.o_proj": rowwise_parallel(output_layouts=Replicate()),
        }

        layer_plan.update(
            {
                "mlp.gate_proj": colwise_parallel(),
                "mlp.down_proj": rowwise_parallel(output_layouts=Replicate()),
                "mlp.up_proj": colwise_parallel(),
            }
        )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

def _register_tp_sum_hook(param, tp_mesh):
    """All-reduce SUM a parameter's grad on the TP process group.

    Needed for replicated weights that are used inside custom kernels (or
    otherwise unwrapped to local), where each rank produces a *partial*
    gradient (sum over its own head/sequence subset) and autograd doesn't
    propagate a Partial placement back up through the `to_local()` boundary.
    """
    import torch.distributed as _dist
    _tp_group = tp_mesh.get_group()

    def _reduce_tp(p):
        if p.grad is None:
            return
        g = p.grad
        if isinstance(g, DTensor):
            g = g.to_local()
        _dist.all_reduce(g, op=_dist.ReduceOp.SUM, group=_tp_group)

    param.register_post_accumulate_grad_hook(_reduce_tp)


def _shard_gated_delta_net(layer, tp_mesh, colwise_parallel, rowwise_parallel):
    """Apply tensor parallelism to a ``DecoderLayer`` whose attention is a
    :class:`GatedDeltaNet` (linear attention).

    Heads are partitioned across TP ranks: each rank owns
    ``n_key_heads // tp`` and ``n_value_heads // tp`` heads. Because
    ``in_proj_qkv`` and ``conv1d`` are fused along
    ``[q_heads | k_heads | v_heads]``, a plain row-shard would split the
    concatenation boundary, not the head dimension. We permute both weights
    into a rank-grouped layout first, after which ``ColwiseParallel(Shard(0))``
    naturally gives each rank its ``[q_local | k_local | v_local]`` slab.

    ``A_log``, ``dt_bias`` and the permuted ``conv1d.weight`` are not inside
    ``nn.Linear`` modules, so we shard them manually via ``distribute_tensor``.
    ``n_key_heads`` / ``n_value_heads`` on the module are overwritten with the
    local counts so the forward's ``.view`` / ``.split`` compute local shapes.
    """
    gdn = layer.linear_attn
    tp_size = tp_mesh.size()
    if tp_size == 1:
        return

    n_key = gdn.n_key_heads
    n_val = gdn.n_value_heads
    key_hd = gdn.key_head_dim
    val_hd = gdn.value_head_dim
    key_dim = n_key * key_hd
    val_dim = n_val * val_hd

    assert n_key % tp_size == 0, f"n_key_heads={n_key} not divisible by tp={tp_size}"
    assert n_val % tp_size == 0, f"n_value_heads={n_val} not divisible by tp={tp_size}"

    n_key_per = n_key // tp_size
    n_val_per = n_val // tp_size

    with torch.no_grad():
        Wqkv = gdn.in_proj_qkv.weight.data
        hidden = Wqkv.shape[1]
        Wq = Wqkv[:key_dim].view(n_key, key_hd, hidden)
        Wk = Wqkv[key_dim : 2 * key_dim].view(n_key, key_hd, hidden)
        # we do not use the val_dim because we just take all to the end of the tensor
        Wv = Wqkv[2 * key_dim :].view(n_val, val_hd, hidden)

        chunks = []
        for r in range(tp_size):
            rank_heads_qk = slice(r * n_key_per, (r + 1) * n_key_per)
            rank_heads_v  = slice(r * n_val_per, (r + 1) * n_val_per)

            chunks.append(Wq[rank_heads_qk].reshape(-1, hidden))
            chunks.append(Wk[rank_heads_qk].reshape(-1, hidden))
            chunks.append(Wv[rank_heads_v].reshape(-1, hidden))

        # re-concatenate into the weight
        gdn.in_proj_qkv.weight.data.copy_(torch.cat(chunks, dim=0))

        # the same is performated to the Conv1D weight
        # since it acts on a per-head basis
        Cw = gdn.conv1d.weight.data
        K = Cw.shape[-1]
        Cq = Cw[:key_dim].view(n_key, key_hd, 1, K)
        Ck = Cw[key_dim : 2 * key_dim].view(n_key, key_hd, 1, K)
        Cv = Cw[2 * key_dim :].view(n_val, val_hd, 1, K)

        chunks = []
        for r in range(tp_size):
            rank_heads_qk = slice(r * n_key_per, (r + 1) * n_key_per)
            rank_heads_v  = slice(r * n_val_per, (r + 1) * n_val_per)

            chunks.append(Cq[rank_heads_qk].reshape(-1, 1, K))
            chunks.append(Ck[rank_heads_qk].reshape(-1, 1, K))
            chunks.append(Cv[rank_heads_v].reshape(-1, 1, K))

        # re-concatenate into the weight
        gdn.conv1d.weight.data.copy_(torch.cat(chunks, dim=0))

    # like standard attention, we only rowwise the output projection
    plan = {
        "in_proj_qkv": colwise_parallel(use_local_output=False),
        "in_proj_z": colwise_parallel(use_local_output=False),
        "in_proj_a": colwise_parallel(use_local_output=False),
        "in_proj_b": colwise_parallel(use_local_output=False),
        "out_proj": rowwise_parallel(output_layouts=Replicate()),
    }
    parallelize_module(gdn, tp_mesh, plan)

    # sharded on the head dimension
    gdn.A_log = nn.Parameter(
        distribute_tensor(gdn.A_log.data, tp_mesh, [Shard(0)])
    )
    gdn.dt_bias = nn.Parameter(
        distribute_tensor(gdn.dt_bias.data, tp_mesh, [Shard(0)])
    )

    # the permuted weights are sharded according to the head dim
    # each rank uses the conv1d that acts on its heads
    gdn.conv1d.weight = nn.Parameter(
        distribute_tensor(gdn.conv1d.weight.data, tp_mesh, [Shard(0)])
    )

    # norm.weight is replicated across ranks, but the gradient is NOT.
    # RMSNormGated runs as a custom Triton autograd.Function on the LOCAL weight
    # (we unwrap via _local() to feed the kernel), so each rank ends up with a
    # partial gradient for its own head subset. Sum across TP explicitly.
    _register_tp_sum_hook(gdn.norm.weight, tp_mesh)

    # Rewrite head counts so forward computes local (B, L, n_local, head_dim).
    gdn.n_key_heads = n_key_per
    gdn.n_value_heads = n_val_per

def _shard_moe_experts(layer, tp_mesh):
    """Shard a `SparseMoeBlock`'s stacked expert weights on the intermediate dim.

    `gate_up_proj` is `(E, 2*I, H)` with the gate and up halves concatenated, so
    a plain shard of dim 1 would hand rank 0 the gate rows and rank 1 the up
    rows. As with the fused GDN qkv projection, the weight is first permuted
    into per-rank `[gate_local | up_local]` slabs; after that a straight even
    split along dim 1 gives each rank a coherent slice.

    `down_proj` is `(E, H, I)`, sharded on the intermediate dim, so each rank
    produces a *partial* sum over hidden. `Experts.forward` runs on local
    tensors (grouped_mm / index_add), which DTensor cannot see through, so the
    reduction is done explicitly via `tp_group` on the module.
    """
    tp_size = tp_mesh.size()
    if tp_size == 1:
        return
    experts = layer.mlp.experts
    inter = experts.intermediate_dim
    assert inter % tp_size == 0, f"moe_intermediate_size={inter} not divisible by tp={tp_size}"
    per = inter // tp_size

    with torch.no_grad():
        gu = experts.gate_up_proj.data                      # (E, 2I, H)
        gate, up = gu[:, :inter], gu[:, inter:]
        experts.gate_up_proj.data.copy_(
            torch.cat(
                [
                    torch.cat([gate[:, r * per : (r + 1) * per],
                               up[:, r * per : (r + 1) * per]], dim=1)
                    for r in range(tp_size)
                ],
                dim=1,
            )
        )

    experts.gate_up_proj = nn.Parameter(
        distribute_tensor(experts.gate_up_proj.data, tp_mesh, [Shard(1)])
    )
    experts.down_proj = nn.Parameter(
        distribute_tensor(experts.down_proj.data, tp_mesh, [Shard(2)])
    )
    experts.intermediate_dim = per
    experts.tp_group = tp_mesh.get_group()


def _shard_ple(layer, tp_mesh):
    """Head-shard a `PLELayer`: n-gram table by head, its two consumers rowwise.

    The table is block-diagonal in the n-gram head -- head `h` owns its own
    vocabulary rows and lands in exactly its own `head_dim` columns of the
    flattened output -- so one shard on the flat head index is row- and
    column-parallel at once. `NGramEmbedding` already built only this rank's
    heads (see its docstring: at the released `ngram_vocab_size_base` the table
    is ~95 GiB, far too large to build whole and split afterwards), so all that
    is left here is to declare the layout and to shard the consumers.

    `key_proj` and `value_proj` are rowwise on their *input* dim, which is the
    sharded `ple_embed_dim`. That is the whole point of the pairing: each rank
    multiplies its own column block and the partial sums are reduced once, so
    the collective is one all-reduce of `(T, hc_hidden)` + `(T, hidden)`
    instead of an all-gather of the embedding itself.

    Everything downstream of those projections (`norm_*`, `conv1d`) is
    full-width again and stays unparallelized, alongside the n-gram hashing
    buffers -- those are int64 constants multiplied against plain token ids and
    raise "mixed torch.Tensor and DTensor" if replicated as DTensors. Plain
    weights are correct here for the same reason they are on the router: every
    rank sees the same replicated activations and computes the same gradient.
    """
    tp_size = tp_mesh.size()
    if tp_size == 1:
        return
    ple = layer.ple
    ngram = ple.ple_embedding
    if ngram.tp_size != tp_size:
        raise ValueError(
            f"the PLE table was built for ple_tp_size={ngram.tp_size} but TP is "
            f"{tp_size}. It is sharded at construction, so the size has to be "
            f"known before the model is built."
        )

    # Local rows are equal on every rank by construction (`rows_per_head` pads
    # each head to a common size), so this is an even `Shard(0)`.
    weight = ngram.ngram_embedding.weight
    ngram.ngram_embedding.weight = nn.Parameter(
        DTensor.from_local(weight.data, tp_mesh, [Shard(0)], run_check=False),
        requires_grad=weight.requires_grad,
    )

    parallelize_module(
        module=ple,
        device_mesh=tp_mesh,
        parallelize_plan={
            # the gather returns this rank's column block of `ple_embed_dim`
            "ple_embedding": PrepareModuleOutput(
                output_layouts=Shard(-1),
                desired_output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "key_proj": RowwiseParallel(
                input_layouts=Shard(-1), output_layouts=Replicate()
            ),
            "value_proj": RowwiseParallel(
                input_layouts=Shard(-1), output_layouts=Replicate()
            ),
        },
    )


def _apply_tp_to_decoder_qwen4(
    model,
    tp_mesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    top_level_plan = {}
    # Under PP a stage may own neither end of the model, so each entry is
    # conditional on the module actually being here.
    if model.language_model.embed_tokens is not None:
        top_level_plan["language_model.embed_tokens"] = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        )
    # hyper-connection mixers stay replicated: they are small (hc_lowrank)
    # and sharding them would need a `rowwise_split_input` style plan.
    if model.language_model.hyper_connection_mixer is not None:
        top_level_plan["language_model.hyper_connection_mixer"] = NoParallel()
    # `lm_head` is not here: this runs on `Qwen4Inner`, which does not own
    # it. `apply_tp` shards it separately on the outer module.
    parallelize_module(model, tp_mesh, top_level_plan)

    rowwise_parallel, colwise_parallel = RowwiseParallel, ColwiseParallel
    model_lm = model.language_model

    for transformer_block in iter_layers(model_lm.layers):
        full_attention = hasattr(transformer_block, "self_attn")

        layer_plan = {
            "attn_hyper_connection": NoParallel(),
            "mlp_hyper_connection": NoParallel(),
            # Routed experts are sharded below; the shared expert follows the
            # usual colwise/rowwise pattern. The router is left out of the plan
            # on purpose: under `NoParallel` its `topk` runs on DTensors while
            # the expert forward consumes the local top-k tensors, and the
            # topk backward then scatters a DTensor grad into a plain zeros
            # buffer. Plain weights are correct here -- every TP rank routes
            # the same replicated tokens and so computes the same gradient.
            "mlp.shared_expert.gate_proj": colwise_parallel(),
            "mlp.shared_expert.up_proj": colwise_parallel(),
            "mlp.shared_expert.down_proj": rowwise_parallel(output_layouts=Replicate()),
            "mlp.shared_expert_gate": NoParallel(),
        }

        if full_attention:
            layer_plan.update({
                "self_attn": PrepareModuleInput(
                    input_kwarg_layouts={"hidden_states": Replicate()},
                    desired_input_kwarg_layouts={"hidden_states": Replicate()},
                ),
                "self_attn.q_proj": colwise_parallel(use_local_output=False),
                "self_attn.k_proj": colwise_parallel(use_local_output=False),
                "self_attn.v_proj": colwise_parallel(use_local_output=False),
                "self_attn.q_norm": SequenceParallel(sequence_dim=2),
                "self_attn.k_norm": SequenceParallel(sequence_dim=2),
                "self_attn.o_proj": rowwise_parallel(output_layouts=Replicate()),
                # The indexer is frozen, gradient-free and cheap, so it is
                # left out of the plan: its weights stay plain and it unwraps
                # its DTensor input itself. Replicating it via `NoParallel`
                # would make its norm weights DTensors while the activations
                # it computes are local.
            })

        # PLE is absent from `layer_plan` on purpose -- it is head-sharded
        # separately by `_shard_ple` below, which needs the layer's other
        # parallelism already in place.

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        _shard_moe_experts(transformer_block, tp_mesh)

        if getattr(transformer_block, "ple", None) is not None:
            _shard_ple(transformer_block, tp_mesh)

        if not full_attention:
            _shard_gated_delta_net(
                transformer_block, tp_mesh, colwise_parallel, rowwise_parallel
            )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True


def apply_fsdp_qwen4(model, mesh, reshard_after_forward_policy='never', mp_policy=None):
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()  # no-op: keeps params in their loaded dtype

    # Under PP a stage owns only part of the tree; everything it does not own
    # is None (see `apply_pp_qwen4`).
    if model.lm_head is not None:
        fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=False,
                    mp_policy=mp_policy)

    model = model.model

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = True
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    for transformer_block in iter_layers(model.language_model.layers):
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
    if model.visual is not None:
        for block in model.visual.blocks:
            fully_shard(
                block,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        # `Qwen4ForCausalLM.forward` calls `self.model.visual(...)` and
        # `self.model.language_model(...)` directly, so the root group wrapping
        # `model` never runs its pre-forward hook and anything left in it stays
        # sharded. Every module holding parameters therefore needs its own group.
        for mod in (model.visual.patch_embed, model.visual.pos_embed, model.visual.merger):
            fully_shard(mod, mesh=mesh, reshard_after_forward=reshard_after_forward,
                        mp_policy=mp_policy)

    if model.language_model.embed_tokens is not None:
        fully_shard(
            model.language_model.embed_tokens,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward_policy == "always",
            mp_policy=mp_policy,
        )
    # the mixer is tiny and used at the very end of every step
    if model.language_model.hyper_connection_mixer is not None:
        fully_shard(model.language_model.hyper_connection_mixer, mesh=mesh,
                    reshard_after_forward=False, mp_policy=mp_policy)

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)


def _apply_tp_to_decoder_qwen3_5(
    model,
    tp_mesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    top_level_plan = {
        "language_model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "language_model.norm": NoParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    }
    parallelize_module(model, tp_mesh, top_level_plan)

    rowwise_parallel, colwise_parallel = RowwiseParallel, ColwiseParallel
    model_lm = model.language_model

    for transformer_block in model_lm.layers:
        full_attention = hasattr(transformer_block, "self_attn")

        if full_attention:
            layer_plan = {
                "input_layernorm": NoParallel(),
                "post_attention_layernorm": NoParallel(),
                "self_attn": PrepareModuleInput(
                    input_kwarg_layouts={"hidden_states": Replicate()},
                    desired_input_kwarg_layouts={"hidden_states": Replicate()},
                ),
                "self_attn.q_proj": colwise_parallel(use_local_output=False),
                "self_attn.k_proj": colwise_parallel(use_local_output=False),
                "self_attn.v_proj": colwise_parallel(use_local_output=False),
                "self_attn.q_norm": SequenceParallel(sequence_dim=2),
                "self_attn.k_norm": SequenceParallel(sequence_dim=2),
                "self_attn.o_proj": rowwise_parallel(output_layouts=Replicate()),
            }
        else:
            layer_plan = {
                "input_layernorm": NoParallel(),
                "post_attention_layernorm": NoParallel(),
            }

        layer_plan.update({
            "mlp.gate_proj": colwise_parallel(),
            "mlp.down_proj": rowwise_parallel(output_layouts=Replicate()),
            "mlp.up_proj": colwise_parallel(),
        })
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
        if full_attention:
            # SequenceParallel wraps q_norm.weight / k_norm.weight as Replicate,
            # but their input gets resharded from head-split (q_proj output) to
            # Shard(num_heads). Each rank's backward only sees its own head
            # subset, producing a partial grad that the DTensor→local→DTensor
            # transitions around varlen_attn don't all-reduce. Force it.
            _register_tp_sum_hook(
                transformer_block.self_attn.q_norm.weight, tp_mesh
            )
            _register_tp_sum_hook(
                transformer_block.self_attn.k_norm.weight, tp_mesh
            )
        else:
            _shard_gated_delta_net(
                transformer_block, tp_mesh, colwise_parallel, rowwise_parallel
            )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True


def pp_layer_split(
    num_layers: int,
    pp_size: int,
    num_first: int = 0,
    num_last: int = 0,
) -> list[tuple[int, int]]:
    """Decoder-layer range owned by each pipeline stage, as `(start, end)`.

    `num_first`/`num_last` pin the first and last stages; 0 means "even split".
    They exist because rank 0 also carries the vision tower and `embed_tokens`
    and the last rank the mixer and `lm_head`, so an even split by layer count
    is not an even split by memory.

    Unlike the version this was ported from, stage 0 gets layers. Leaving it
    with only the embedding silently drops the first half of the network for
    `pp_size == 2`.
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if num_first < 0 or num_last < 0:
        raise ValueError("pp_num_layers_first/last must be >= 0")
    if pp_size == 1:
        return [(0, num_layers)]

    first = num_first or None
    last = num_last or None
    if pp_size == 2:
        if first is not None and last is not None and first + last != num_layers:
            raise ValueError(
                f"pp_num_layers_first ({first}) + pp_num_layers_last ({last}) "
                f"!= num_hidden_layers ({num_layers})"
            )
        if first is None:
            first = num_layers - last if last is not None else (num_layers + 1) // 2
        return [(0, first), (first, num_layers)]

    pinned = (first or 0) + (last or 0)
    middle_ranks = pp_size - 2
    middle = num_layers - pinned if pinned else None
    if first is None or last is None:
        # even split first, then override whichever end was pinned
        base, rem = divmod(num_layers, pp_size)
        counts = [base + (1 if i < rem else 0) for i in range(pp_size)]
        if first is not None:
            counts[0] = first
        if last is not None:
            counts[-1] = last
        if first is not None or last is not None:
            spare = num_layers - sum(counts)
            # push the difference into the middle stages, one layer at a time
            i = 1
            while spare != 0:
                step = 1 if spare > 0 else -1
                if counts[i] + step >= 0:
                    counts[i] += step
                    spare -= step
                i = 1 + (i % middle_ranks)
    else:
        if middle < 0:
            raise ValueError(
                f"pp_num_layers_first + pp_num_layers_last ({pinned}) exceeds "
                f"num_hidden_layers ({num_layers})"
            )
        base, rem = divmod(middle, middle_ranks)
        counts = [first] + [base + (1 if i < rem else 0) for i in range(middle_ranks)] + [last]

    if any(c <= 0 for c in counts):
        raise ValueError(
            f"pp split leaves an empty stage: {counts} for {num_layers} layers "
            f"over {pp_size} stages"
        )

    bounds, start = [], 0
    for c in counts:
        bounds.append((start, start + c))
        start += c
    return bounds


def apply_pp_qwen4(
    model,
    mesh,
    training_args,
    device,
    pp_loss_fn,
):
    """Split `Qwen4ForCausalLM` into this rank's pipeline stage.

    Runs while the model is still on CPU. Everything the stage does not own is
    replaced with `None` and dropped, so only the stage reaches the GPU -- that
    is the whole point of PP here, and the reason a model that cannot be
    materialized whole on one card becomes reachable.

    `Qwen4ForCausalLM.forward` reads those `None`s to decide what it is:
    `embed_tokens` present means first stage (its positional input is token
    ids), `lm_head` present means last (it returns logits). Everything between
    passes the `(1, total, hc_count * hidden_size)` hyper-connection stream.
    """
    pp_mesh = mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    total_layers = model.cfg.text.num_hidden_layers
    bounds = pp_layer_split(
        total_layers,
        pp_size,
        num_first=getattr(training_args, "pp_num_layers_first", 0),
        num_last=getattr(training_args, "pp_num_layers_last", 0),
    )
    start_idx, end_idx = bounds[pp_rank]
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1

    logger.info(
        f"PP rank {pp_rank}/{pp_size}: layers [{start_idx}, {end_idx}) "
        f"of {total_layers}; split={bounds}"
    )

    language_model = model.model.language_model

    # Keep the layers this stage owns. The kept `DecoderLayer`s hold their
    # original `layer_idx`/`ple_index`, so the PLE layer stays wired to the
    # right n-gram table wherever it lands.
    # Keyed by the *original* layer index, not renumbered from zero: a stage's
    # parameters have to keep the names they have in the unsplit model, or two
    # stages write the same `layers.0.*` keys into one checkpoint and DCP
    # either rejects the plan or silently keeps one of them. `iter_layers`
    # reads either container.
    language_model.layers = torch.nn.ModuleDict(
        {str(i): layer for i, layer in enumerate(language_model.layers)
         if start_idx <= i < end_idx}
    )

    if not is_first:
        model.model.visual = None
        language_model.embed_tokens = None
    if not is_last:
        # Qwen4 has no final norm; the mixer is the last op before `lm_head`.
        language_model.hyper_connection_mixer = None
        model.lm_head = None

    model.to(device=device)

    pp_stage = PipelineStage(
        model,
        stage_index=pp_rank,
        num_stages=pp_size,
        device=device,
        group=pp_mesh.get_group(),
    )

    schedule_name = getattr(training_args, "pp_schedule", "gpipe").lower()
    n_microbatches = getattr(training_args, "pp_microbatches", 1)
    schedule_cls = {"gpipe": ScheduleGPipe, "1f1b": Schedule1F1B}.get(schedule_name)
    if schedule_cls is None:
        raise ValueError(
            f"unknown pp_schedule={schedule_name!r}; expected one of: gpipe, 1f1b"
        )
    if schedule_cls is Schedule1F1B:
        # The plumbing is here (schedule construction, kwargs_chunk_spec, the
        # tiled input in `_train_step_pp`), but 1F1B needs
        # n_microbatches >= pp_size and the dataloader emits one packed
        # (1, total) row per step -- so the microbatches are tiled copies of
        # the same content and the loss means nothing. Re-enable once the data
        # path produces n_microbatches independent packed rows per step
        # (per-row cu_seqlens, labels, image scatter).
        raise NotImplementedError(
            "pp_schedule='1f1b' is disabled until the data path supports "
            "n_microbatches independent packed rows per step. Use 'gpipe'."
        )

    # The chunk spec has to name exactly the kwargs a step passes -- the
    # dataloader's key set varies with the sample (`mm_token_type_ids` and the
    # video tensors come and go), and `_shard_dict_of_args` asserts the two key
    # sets are equal. `_train_step_pp` therefore rebuilds it per step; every
    # kwarg is replicated anyway, since only the positional stage input is
    # chunked.
    pp_schedule = schedule_cls(
        pp_stage,
        n_microbatches=n_microbatches,
        loss_fn=pp_loss_fn,
    )
    logger.info(f"PP schedule: {schedule_name} (n_microbatches={n_microbatches})")

    return n_microbatches, pp_schedule, is_first, is_last
