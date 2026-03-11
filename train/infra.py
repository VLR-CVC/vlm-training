from dataclasses import dataclass
from functools import partial

import torch
import torch._inductor.config

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
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

from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torch.distributed.fsdp import fully_shard

# for activation checkpoiting
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper

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

def apply_fsdp(model, mesh, reshard_after_forward_policy='never'):

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

def _replicate_module_params(module: torch.nn.Module, mesh):
    """Convert a module's direct (non-recursive) parameters to Replicate DTensors."""
    for name, param in module.named_parameters(recurse=False):
        replicated = torch.nn.Parameter(
            distribute_tensor(param.data, mesh, [Replicate()]),
            requires_grad=param.requires_grad,
        )
        setattr(module, name, replicated)

def compile_model(model: torch.nn.Module):
    model.visual = torch.compile(
        model.visual, fullgraph=False, mode='default',
    )
    model.language_model = torch.compile(
        model.language_model, fullgraph=True, mode='default',
    )
    model.visual.merger = torch.compile(
        model.visual.merger, fullgraph=True, mode='default',
    )
    model = torch.compile(
        model, mode='default',
    )

def apply_tp_complex(
        model,
        tp_mesh,
):
    _apply_tp_to_decoder(model, tp_mesh, False, False, False)
    #_apply_tp_to_visual(model.visual, tp_mesh)

def _apply_tp_to_decoder(
    model,
    tp_mesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
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

    if False:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )
    else:
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

def _apply_tp_to_visual(
    visual,
    tp_mesh,
):

    raise NotImplementedError # WIP
    """Apply tensor parallelism to the vision encoder.

    Uses TP without SequenceParallel: data between blocks stays Replicate
    (all ranks hold full hidden_states). This is simpler because norms and
    position embeddings don't need DTensor conversion, and vision encoder
    sequence lengths are short enough that redundant norm computation is cheap.
    Memory savings come from sharding the linear layer weights.
    """
    # Use NoParallel on patch_embed so its params become Replicate DTensors
    # on tp_mesh (ensuring a consistent (fsdp, tp) mesh after FSDP), while
    # its I/O hooks convert plain pixel_values to DTensor on entry and back
    # to local tensor on exit — avoiding mixed tensor/DTensor errors with
    # pos_embeds (which are computed as plain tensors).
    parallelize_module(visual, tp_mesh, {"patch_embed": NoParallel()})

    # pos_embed.weight is accessed directly (not through forward), so we
    # just need its weight to be a DTensor on tp_mesh for mesh consistency.
    # The vision encoder's compute_position_embeddings() already calls
    # .to_local() on it, so the downstream pos_embeds stays a plain tensor.
    _replicate_module_params(visual.pos_embed, tp_mesh)

    # TP plan for each vision transformer block.
    # NoParallel on norms sets their params as Replicate DTensors on tp_mesh
    # (for consistent (fsdp, tp) mesh after FSDP) and inserts I/O hooks that
    # convert local tensor → DTensor on entry and DTensor → local tensor on
    # exit. This keeps the block's data path in local-tensor space (as
    # ColwiseParallel/RowwiseParallel with use_local_output=True expect).

    for transformer_block in visual.blocks:
        layer_plan = {
            "norm1": NoParallel(),
            "norm2": NoParallel(),
            "attn.qkv": ColwiseParallel(),
            "attn.proj": RowwiseParallel(),
            "mlp.linear_fc1": ColwiseParallel(),
            "mlp.linear_fc2": RowwiseParallel(),
        }
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack).
    merger_plan = {
        "norm": NoParallel(),
        "linear_fc1": ColwiseParallel(),
        "linear_fc2": RowwiseParallel(),
    }

    parallelize_module(visual.merger, tp_mesh, merger_plan)
    for merger in visual.deepstack_merger_list:
        parallelize_module(merger, tp_mesh, merger_plan)