from dataclasses import dataclass
from functools import partial

from train.config import ModelType

import torch
import torch._inductor.config

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, distribute_module, DTensor, Replicate
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
}

_softplus_registered = False

from torchao.float8 import convert_to_float8_training
from torch.distributed.fsdp import fully_shard

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
        outputs: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor | DTensor:
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                # TODO: this is pytorch distribute_module typing issue.
                # pyrefly: ignore [bad-argument-type]
                self._prepare_input_fn,
                self.input_layout,
                self.desired_input_layout,
            ),
            partial(
                # TODO: this is pytorch distribute_module typing issue.
                # pyrefly: ignore [bad-argument-type]
                self._prepare_output_fn,
                self.output_layout,
                self.use_local_output,
            ),
        )

class _DTensorSafeConv1d(nn.Module):
    """Conv1d wrapper that bypasses DTensor dispatch for depthwise conv.

    DTensor's _tp_conv handler doesn't support depthwise conv (groups > 1).
    This wrapper stores weight as a Replicate DTensor (for mesh consistency
    needed by gradient norm clipping) but runs F.conv1d on local tensors.
    """

    def __init__(self, original: nn.Conv1d, tp_mesh: DeviceMesh):
        super().__init__()
        self.weight = nn.Parameter(
            DTensor.from_local(
                original.weight.data, tp_mesh, [Replicate()], run_check=False
            ),
            requires_grad=original.weight.requires_grad,
        )
        self.bias: nn.Parameter | None = None
        if original.bias is not None:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    original.bias.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=original.bias.requires_grad,
            )
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_dtensor = isinstance(x, DTensor)
        x_local = x.to_local() if is_dtensor else x
        w_local = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        b_local = None
        if self.bias is not None:
            b_local = (
                self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            )
        out = torch.nn.functional.conv1d(
            x_local,
            w_local,
            b_local,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if is_dtensor:
            out = DTensor.from_local(out, x.device_mesh, x.placements, run_check=False)
        return out


_fla_dispatch_patched = False

def _install_dtensor_safe_fla_dispatch() -> None:
    """Monkey-patch FLA's chunk_gated_delta_rule to handle DTensor inputs."""
    global _fla_dispatch_patched
    if _fla_dispatch_patched:
        return
    _fla_dispatch_patched = True

    import fla.ops.gated_delta_rule.chunk as fla_chunk_module
    original_fn = fla_chunk_module.chunk_gated_delta_rule

    def _dtensor_safe_chunk_gated_delta_rule(
        q, k, v, g, beta,
        **kwargs,
    ):
        is_dtensor = isinstance(q, DTensor)
        if is_dtensor:
            mesh, placements = q.device_mesh, q.placements

            def _unwrap(x):
                if isinstance(x, DTensor):
                    return x.to_local()
                return x

            safe_kwargs = {key: _unwrap(val) for key, val in kwargs.items()}

            result = original_fn(
                _unwrap(q), _unwrap(k), _unwrap(v), _unwrap(g), _unwrap(beta),
                **safe_kwargs,
            )

            if isinstance(result, tuple):
                o = DTensor.from_local(result[0], mesh, placements, run_check=False)
                rest = tuple(
                    DTensor.from_local(r, mesh, placements, run_check=False)
                    if isinstance(r, torch.Tensor) else r
                    for r in result[1:]
                )
                return (o, *rest)
            return DTensor.from_local(result, mesh, placements, run_check=False)

        return original_fn(q, k, v, g, beta, **kwargs)

    fla_chunk_module.chunk_gated_delta_rule = _dtensor_safe_chunk_gated_delta_rule

_softplus_registered = False

def _register_dtensor_softplus() -> None:
    global _softplus_registered
    if _softplus_registered:
        return
    _softplus_registered = True

    from torch.distributed.tensor.experimental import register_sharding
    from torch.distributed.tensor import Replicate, Shard

    @register_sharding(torch.ops.aten.softplus.default)
    def softplus_sharding(x, beta=1, threshold=20):
        # Pointwise op: any placement in → same placement out
        acceptable = [([Replicate()], [Replicate(), None, None])]
        for dim in range(x.ndim):
            acceptable.append(
                ([Shard(dim)], [Shard(dim), None, None])
            )
        return acceptable

    @register_sharding(torch.ops.aten.softplus_backward.default)
    def softplus_backward_sharding(grad_output, x, beta, threshold):
        acceptable = [([Replicate()], [Replicate(), Replicate(), None, None])]
        for dim in range(x.ndim):
            acceptable.append(
                ([Shard(dim)], [Shard(dim), Shard(dim), None, None])
            )
        return acceptable



def get_mesh(training_args, world_size):
    """
    Creates a 2D DeviceMesh based on tp_size and world_size.
    Always returns ('dp', 'tp').
    """
    tp_size = training_args.tp_size
    
    if world_size % tp_size != 0:
        raise ValueError(f"World size {world_size} is not divisible by TP size {tp_size}")

    dp_size = world_size // tp_size

    # Always 2D: Handles Pure DP (X, 1), Pure TP (1, X), and Hybrid (X, Y)
    return init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

def get_tp_group(mesh):
    if "tp" in mesh.mesh_dim_names:
        return mesh['tp']
    return None

def get_dp_group(mesh):
    if "dp" in mesh.mesh_dim_names:
        return mesh['dp']
    return None

def module_filter_float8_fn(mod: torch.nn.Module, fqn: str):
    if "visual" in fqn:
        return False

    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

def apply_float8(model):
    convert_to_float8_training(
        model,
        module_filter_fn=module_filter_float8_fn,
    )

@dataclass
class ACConfig:
    enabled: bool = True
    full: bool = False


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
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP, and Flex:
    #
    # When forward runs with a second PP microbatch, it triggers recompilation with dynamic
    # shapes enabled. Now there are two valid compiled graphs. By default, dynamo selects
    # the latest one (the dynamic shapes version), so the runtime wrapper expects an extra
    # symint output. When SAC caches the inductor HOP output from the static graph for
    # batch_idx=0, it would miss that symint and cause an assertion failure. The workaround
    # here is to disable the LRU cache, and select graphs in insertion order instead.
    #
    # Also see: https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    if ac_config.enabled:

        if not ac_config.full: op_sac_save_list = _op_sac_save_list
        else: op_sac_save_list = {}

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
    model = model.model
    model.language_model = torch.compile(
        model.language_model, fullgraph=False, mode='default',
    )
    model.visual = torch.compile(
        model.visual, fullgraph=False, mode='default',
    )
    model.visual.merger = torch.compile(
        model.visual.merger, fullgraph=False, mode='default',
    )
    model = torch.compile(
        model, mode='default',
    ) 

def apply_fsdp(model_type, model, **kwargs):
    if model_type == ModelType.Qwen3_text:
        apply_fsdp_qwen3(model, **kwargs)
    elif model_type == ModelType.Qwen3_vl:
        apply_fsdp_qwen3_vl(model, **kwargs)

def apply_fsdp_qwen3(model, mesh, reshard_after_forward_policy='never'):
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
    for transformer_block in model.layers:
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    fully_shard(
        [model.norm, model.embed_tokens],
        mesh=mesh,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )

    fully_shard(model, mesh=mesh)

def apply_fsdp_qwen3_vl(model, mesh, reshard_after_forward_policy='never'):
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
        )

    # vision encoder
    for transformer_block in model.visual.blocks:
        fully_shard(
            transformer_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    fully_shard(
        [model.language_model.norm, model.language_model.embed_tokens],
        mesh=mesh,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )

    fully_shard(model, mesh=mesh)

def apply_tp(
        model,
        model_type: ModelType,
        tp_mesh,
        enable_tp_async,
):
    model = model.model

    if model_type == ModelType.Qwen3_5:
        _tp_decoder = _apply_tp_to_decoder_qwen3_5
    elif model_type == ModelType.Qwen3_vl:
        _tp_decoder = _apply_tp_to_decoder_qwen3_vl
    else:
        raise NotImplementedError()
    _tp_decoder(model, tp_mesh, False, enable_tp_async)

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


def _apply_tp_to_decoder_qwen3_5(
    model,
    tp_mesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    """
    See this torchtitan PR (Qwen3.5 MoE implementation): https://github.com/pytorch/torchtitan/pull/2545
    """
    _register_dtensor_softplus()
    _install_dtensor_safe_fla_dispatch()

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
                "mlp.gate_proj": colwise_parallel(),
                "mlp.down_proj": rowwise_parallel(output_layouts=Replicate()),
                "mlp.up_proj": colwise_parallel(),
            }
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        else:
            layer_plan = {
                "input_layernorm": NoParallel(),
                "post_attention_layernorm": NoParallel(),
                "linear_attn.in_proj_qkv": NoParallel(),
                "linear_attn.in_proj_z": NoParallel(),
                "linear_attn.in_proj_a": NoParallel(),
                "linear_attn.in_proj_b": NoParallel(),
                "linear_attn.out_proj": NoParallel(),
                "mlp.gate_proj": colwise_parallel(),
                "mlp.down_proj": rowwise_parallel(output_layouts=Replicate()),
                "mlp.up_proj": colwise_parallel(),
            }
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

            # Standalone params need to be on the mesh too
            attn = transformer_block.linear_attn
            attn.conv1d = _DTensorSafeConv1d(attn.conv1d, tp_mesh)

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True