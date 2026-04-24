from dataclasses import dataclass
from functools import partial
from typing import Optional

from train.config import ModelType

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn.functional as F

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

from torchao.float8 import convert_to_float8_training
from torch.distributed.fsdp import fully_shard
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
    tp_size = training_args.tp_size
    pp_size = getattr(training_args, "pp_size", 1)

    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(
            f"world_size {world_size} not divisible by tp_size*pp_size={tp_size * pp_size}"
        )

    dp_size = world_size // (tp_size * pp_size)

    if pp_size > 1:
        return init_device_mesh(
            "cuda", (dp_size, pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp")
        )
    return init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

def get_tp_group(mesh):
    if "tp" in mesh.mesh_dim_names:
        return mesh["tp"]
    return None

def get_dp_group(mesh):
    if "dp" in mesh.mesh_dim_names:
        return mesh["dp"]
    return None

def get_pp_group(mesh):
    if "pp" in mesh.mesh_dim_names:
        return mesh["pp"]
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
    model = model.model
    model.language_model = torch.compile(model.language_model, fullgraph=False, mode='default')
    model.visual = torch.compile(model.visual, fullgraph=False, mode='default')
    model.visual.merger = torch.compile(model.visual.merger, fullgraph=False, mode='default',)
    #model = torch.compile(model, mode='default')

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
    outer = model

    if getattr(outer, "cfg", None) is not None and outer.cfg.tie_word_embeddings:
        raise ValueError(
            "Tensor Parallelism is not supported for models with tie_word_embeddings=True. "
            "Use tp_size=1 for small models (e.g. 2B) that tie lm_head and embed_tokens."
        )

    if model_type == ModelType.Qwen3_5:
        _tp_decoder = _apply_tp_to_decoder_qwen3_5
    elif model_type == ModelType.Qwen3_vl:
        _tp_decoder = _apply_tp_to_decoder_qwen3_vl
    else:
        raise NotImplementedError()
    _tp_decoder(outer.model, tp_mesh, False, enable_tp_async)

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


# ---------------------------------------------------------------------------
# Pipeline Parallel for Qwen3.5 (native impl, dense model)
# ---------------------------------------------------------------------------

# cu_seqlens is padded to this fixed size so PipelineStage sees constant shapes.
_PP_MAX_SEQS: int = 256


def _pp_layer_ranges(
    n_layers: int,
    pp_size: int,
    first_virtual: float = 1.0,
    last_virtual: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Distribute n_layers across pp_size stages for balanced memory.

    first_virtual / last_virtual: overhead of the non-layer modules on the
    first/last stage expressed in units of a single transformer layer.
    Computed from actual parameter counts so the split automatically adapts
    to any model size.

    Optimal layers per stage:
        target = (n_layers + first_virtual + last_virtual) / pp_size
        first_n = round(target - first_virtual)
        last_n  = round(target - last_virtual)   [pp_size == 4 only]
        middle  = evenly distributed remainder
    """
    target  = (n_layers + first_virtual + last_virtual) / pp_size
    first_n = max(1, round(target - first_virtual))

    if pp_size == 2:
        return [(0, first_n), (first_n, n_layers)]

    last_n    = max(1, round(target - last_virtual))
    remaining = n_layers - first_n - last_n
    assert remaining >= 2, (
        f"Not enough layers for middle stages with "
        f"first_n={first_n}, last_n={last_n}, n_layers={n_layers}"
    )
    # all minus 2 (last and first)
    mid, extra = divmod(remaining, pp_size - 2)
    ranges, pos = [(0, first_n)], first_n
    for i in range(pp_size - 2):
        n = mid + (1 if i < extra else 0)
        ranges.append((pos, pos + n))
        pos += n
    ranges.append((pos, n_layers))
    return ranges


class PPStageModule(nn.Module):
    """
    A single PP stage for Qwen3.5ForCausalLM (native impl).

    Owns decoder layers[layer_start:layer_end].
    Stage 0 (is_first=True)  also owns the visual encoder and embed_tokens.
    Last stage (is_last=True) also owns norm and lm_head.

    Inter-stage tensor protocol (all fixed shapes):
        hidden_states  : (1, seq_len, hidden_size)  dtype
        cos, sin       : (1, seq_len, rope_dim)      dtype
        cu_seqlens_pad : (_PP_MAX_SEQS+1,)           int32
        n_seqs         : ()                           int64
    """

    def __init__(
        self,
        full_model: nn.Module,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        lm = full_model.model.language_model

        self.is_first = is_first
        self.is_last = is_last
        self.layers = nn.ModuleList(list(lm.layers[layer_start:layer_end]))

        if is_first:
            n_ds = len(full_model.model.visual.deepstack_visual_indexes)
            assert n_ds <= (layer_end - layer_start), (
                f"Stage 0 has {layer_end - layer_start} layers but {n_ds} deepstack "
                "injections — all must fit in stage 0. Reduce pp_size or use a "
                "larger first-stage split."
            )
            self.visual = full_model.model.visual
            self.embed_tokens = lm.embed_tokens
            self.register_buffer("text_inv_freq", full_model.text_inv_freq.clone())
            self.mrope_section = list(full_model.mrope_section)
            self.image_token_id = full_model.cfg.image_token_id
            self.video_token_id = full_model.cfg.video_token_id
            self.spatial_merge_size = full_model.cfg.vision.spatial_merge_size
            # Populated by preprocess() before each forward
            self._vis_masks: Optional[torch.Tensor] = None
            self._ds_embeds: Optional[list] = None

        if is_last:
            self.norm = lm.norm
            self.lm_head = full_model.lm_head

    # ------------------------------------------------------------------
    def preprocess(self, batch: dict) -> tuple:
        """
        Stage-0 only: visual encoding + token embedding + RoPE.

        Call this on pp_rank=0 before schedule.step() to produce the
        fixed-shape tensors that enter the pipeline.
        """
        from models.qwen3_5.utils import mrope_cos_sin

        input_ids      = batch["input_ids"]       # (1, seq_len)
        pixel_values   = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        video_grid_thw = batch.get("video_grid_thw")
        cu_seqlens     = batch["attention_mask"]   # (n_seqs+1,) int32
        device = input_ids.device
        total  = input_ids.shape[1]

        x = self.embed_tokens(input_ids)          # (1, total, H)

        self._vis_masks = None
        self._ds_embeds = None
        has_img = pixel_values is not None and pixel_values.numel() > 0
        has_vid = video_grid_thw is not None and video_grid_thw.numel() > 0

        if has_img:
            merged, ds = self.visual(pixel_values, image_grid_thw)
            merged = merged.to(x.dtype)
            mask = input_ids == self.image_token_id
            x = x.masked_scatter(mask.unsqueeze(-1).expand_as(x), merged)
            self._vis_masks, self._ds_embeds = mask, ds

        if has_vid:
            merged_v, ds_v = self.visual(batch["pixel_values_videos"], video_grid_thw)
            merged_v = merged_v.to(x.dtype)
            vmask = input_ids == self.video_token_id
            x = x.masked_scatter(vmask.unsqueeze(-1).expand_as(x), merged_v)
            if self._vis_masks is None:
                self._vis_masks, self._ds_embeds = vmask, ds_v
            else:
                combined = self._vis_masks | vmask
                merged_ds = []
                for a, b in zip(self._ds_embeds, ds_v):
                    e = a.new_zeros(combined.sum().item(), a.shape[-1])
                    e[self._vis_masks[combined]] = a
                    e[vmask[combined]] = b
                    merged_ds.append(e)
                self._vis_masks, self._ds_embeds = combined, merged_ds

        # position ids → cos/sin
        if has_img or has_vid:
            pos = _mrope_position_ids(
                input_ids, cu_seqlens,
                image_grid_thw if has_img else None,
                video_grid_thw if has_vid else None,
                self.image_token_id, self.video_token_id, self.spatial_merge_size,
            )
        else:
            pos = torch.zeros(total, dtype=torch.int64, device=device)
            for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                pos[s:e] = torch.arange(e - s, device=device)
            pos = pos.view(1, 1, -1).expand(3, 1, -1)

        cos, sin = mrope_cos_sin(self.text_inv_freq, pos, self.mrope_section)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        # pad cu_seqlens to fixed size
        n = cu_seqlens.shape[0]
        assert n <= _PP_MAX_SEQS + 1, f"cu_seqlens len {n} > _PP_MAX_SEQS={_PP_MAX_SEQS}"
        cu_pad = F.pad(cu_seqlens, (0, _PP_MAX_SEQS + 1 - n))
        n_t    = torch.tensor(n, dtype=torch.int64, device=device)

        return x, cos, sin, cu_pad, n_t

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_pad: torch.Tensor,
        n_t: torch.Tensor,
    ) -> tuple | torch.Tensor:
        n = int(n_t.item())
        cu = cu_pad[:n].to(torch.int32)
        max_s = int((cu[1:] - cu[:-1]).max().item())

        x = hidden
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cu, max_s)
            if self.is_first and self._ds_embeds is not None and i < len(self._ds_embeds):
                x = x.clone()
                x[self._vis_masks] = x[self._vis_masks] + self._ds_embeds[i].to(x.dtype)

        if self.is_last:
            return self.lm_head(self.norm(x))   # (1, seq_len, vocab_size)

        return x, cos, sin, cu_pad, n_t


class _ScaledLoss:
    """Stateful loss callable — update .accum_target each step."""

    def __init__(self) -> None:
        self.accum_target: int = 1

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        from models.qwen3_5.utils import causal_lm_loss
        return causal_lm_loss(logits, labels) / self.accum_target


def _mrope_position_ids(
    input_ids, cu_seqlens, image_grid_thw, video_grid_thw,
    image_token_id, video_token_id, spatial_merge_size,
) -> torch.Tensor:
    """3-D MRoPE positions — mirrors Qwen3_5ForCausalLM.get_rope_index."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1

    _, S = input_ids.shape
    device = input_ids.device
    mm = torch.zeros(S, dtype=torch.int64, device=device)
    mm[input_ids[0] == image_token_id] = 1
    mm[input_ids[0] == video_token_id] = 2
    types = mm.tolist()
    img_it = iter(image_grid_thw) if image_grid_thw is not None else None
    vid_it = iter(video_grid_thw) if video_grid_thw is not None else None
    out = torch.zeros(3, 1, S, dtype=torch.int64, device=device)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        if start == end:
            continue
        seg = types[start:end]
        parts, cur, j = [], 0, 0
        while j < len(seg):
            k = j
            while k < len(seg) and seg[k] == seg[j]:
                k += 1
            key, length = seg[j], k - j
            if key == 0:
                parts.append(torch.arange(length, device=device).view(1, -1).expand(3, -1) + cur)
                cur += length
            else:
                g = next(img_it if key == 1 else vid_it)
                t, h, w = int(g[0]), int(g[1]), int(g[2])
                lh, lw = h // spatial_merge_size, w // spatial_merge_size
                n = t * lh * lw
                pw = torch.arange(cur, cur + lw, device=device).repeat(lh * t)
                ph = torch.arange(cur, cur + lh, device=device).repeat_interleave(lw * t)
                pt = torch.full((n,), cur, device=device, dtype=torch.int64)
                parts.append(torch.stack([pt, ph, pw]))
                cur += max(lh, lw)
            j = k
        out[:, 0, start:end] = torch.cat(parts, dim=1)
    return out


class _PPSchedule:
    """
    Single-microbatch pipeline schedule using blocking P2P comms.

    Implements GPipe-style all-forward then all-backward for a linear
    pp_size-stage pipeline with packed (flash-attn varlen) sequences.

    Inter-stage tensor layout:
        metadata : (1,) int64   — n_tokens (variable, sent before hidden/cos/sin)
        hidden   : (1, n_tokens, H)   bfloat16
        cos      : (1, n_tokens, R)   bfloat16
        sin      : (1, n_tokens, R)   bfloat16
        cu_pad   : (_PP_MAX_SEQS+1,)  int32     (fixed size)
        n_t      : ()                 int64     (scalar)
    Backward sends only grad_hidden : (1, n_tokens, H) bfloat16.
    """

    def __init__(
        self,
        stage_module: nn.Module,
        pp_rank: int,
        pp_size: int,
        pp_group,           # DeviceMesh sub-mesh (pp dimension)
        hidden_size: int,
        rope_dim: int,
        dtype: torch.dtype,
        loss_fn: Optional["_ScaledLoss"] = None,
    ):
        self.stage    = stage_module
        self.pp_rank  = pp_rank
        self.pp_size  = pp_size
        self.is_first = pp_rank == 0
        self.is_last  = pp_rank == pp_size - 1
        self._group   = pp_group.get_group()
        self._H       = hidden_size
        self._R       = rope_dim
        self._dt      = dtype
        self._device  = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.loss_fn  = loss_fn

        # Fixed-size recv buffers (cu_pad, n_t, metadata always fixed)
        self._cu_buf   = torch.empty(_PP_MAX_SEQS + 1, device=self._device, dtype=torch.int32)
        self._nt_buf   = torch.empty((),               device=self._device, dtype=torch.int64)
        self._meta_buf = torch.empty(1,                device=self._device, dtype=torch.int64)

    # ------------------------------------------------------------------
    def _send_fwd(self, x, cos, sin, cu_pad, n_t, dst: int):
        n_tok = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int64)
        dist.send(n_tok,              dst=dst, group=self._group)
        dist.send(x.contiguous(),     dst=dst, group=self._group)
        dist.send(cos.contiguous(),   dst=dst, group=self._group)
        dist.send(sin.contiguous(),   dst=dst, group=self._group)
        dist.send(cu_pad.contiguous(), dst=dst, group=self._group)
        dist.send(n_t.contiguous(),   dst=dst, group=self._group)

    def _recv_fwd(self, src: int):
        dist.recv(self._meta_buf, src=src, group=self._group)
        n = int(self._meta_buf.item())
        x   = torch.empty(1, n, self._H, device=self._device, dtype=self._dt)
        cos = torch.empty(1, n, self._R, device=self._device, dtype=self._dt)
        sin = torch.empty(1, n, self._R, device=self._device, dtype=self._dt)
        dist.recv(x,            src=src, group=self._group)
        dist.recv(cos,          src=src, group=self._group)
        dist.recv(sin,          src=src, group=self._group)
        dist.recv(self._cu_buf, src=src, group=self._group)
        dist.recv(self._nt_buf, src=src, group=self._group)
        return x, cos, sin, self._cu_buf, self._nt_buf

    # ------------------------------------------------------------------
    def step(self, *args, target=None, losses=None):
        if self.is_first and not self.is_last:
            x, cos, sin, cu_pad, n_t = self.stage(*args)
            self._send_fwd(x, cos, sin, cu_pad, n_t, dst=self.pp_rank + 1)
            grad_x = torch.empty_like(x)
            dist.recv(grad_x, src=self.pp_rank + 1, group=self._group)
            x.backward(grad_x)

        elif not self.is_first and self.is_last:
            x_in, cos, sin, cu_pad, n_t = self._recv_fwd(src=self.pp_rank - 1)
            x_leaf = x_in.detach().requires_grad_(True)
            logits = self.stage(x_leaf, cos, sin, cu_pad, n_t)
            loss = self.loss_fn(logits, target)
            if losses is not None:
                losses.append(loss.detach())
            loss.backward()
            dist.send(x_leaf.grad.contiguous(), dst=self.pp_rank - 1, group=self._group)

        elif not self.is_first and not self.is_last:
            # Middle stage (pp_size=4)
            x_in, cos, sin, cu_pad, n_t = self._recv_fwd(src=self.pp_rank - 1)
            x_leaf = x_in.detach().requires_grad_(True)
            x_out, cos_out, sin_out, cu_out, nt_out = self.stage(x_leaf, cos, sin, cu_pad, n_t)
            self._send_fwd(x_out, cos_out, sin_out, cu_out, nt_out, dst=self.pp_rank + 1)
            grad_x_out = torch.empty_like(x_out)
            dist.recv(grad_x_out, src=self.pp_rank + 1, group=self._group)
            x_out.backward(grad_x_out)
            dist.send(x_leaf.grad.contiguous(), dst=self.pp_rank - 1, group=self._group)

        else:
            raise RuntimeError("_PPSchedule used with pp_size=1; use regular training path instead")


def apply_pp_qwen35(
    model: nn.Module,
    pp_group,
    seq_len: int,
    *,
    snapshot_dir=None,  # Path | None — when set, model is on meta; weights loaded per-rank
    device=None,        # torch.device | None — required when snapshot_dir is set
    dtype=None,         # torch.dtype  | None — required when snapshot_dir is set
) -> tuple:
    """
    Split Qwen3_5ForCausalLM across PP ranks (pp_size must be 2 or 4).
    Returns (stage_module, None, schedule, loss_fn, pp_rank, pp_size, is_last).

    Meta-loading path (large models):
        Pass ``snapshot_dir``, ``device``, ``dtype``.  The model must be on
        ``torch.device("meta")`` with no weights.  Each rank materialises only
        its stage slice and loads the corresponding weights directly onto
        ``device`` via ``load_stage_weights`` — no full-model CPU or GPU copy.

    Legacy path:
        Omit those kwargs.  The full model must already reside on the target
        device with weights loaded (original behaviour, kept for compatibility).
    """
    pp_rank: int = pp_group.get_local_rank()
    pp_size: int = pp_group.size()
    meta_load = snapshot_dir is not None

    if not meta_load:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    lm = model.model.language_model
    n_layers     = len(lm.layers)
    layer_params = sum(p.numel() for p in lm.layers[0].parameters())
    embed_params = lm.embed_tokens.weight.numel()
    visual       = getattr(model.model, "visual", None)
    visual_params = sum(p.numel() for p in visual.parameters()) if visual is not None else 0
    lm_head_params = model.lm_head.weight.numel()

    first_virtual = (embed_params + visual_params) / layer_params
    last_virtual  = lm_head_params / layer_params

    ranges = _pp_layer_ranges(n_layers, pp_size, first_virtual, last_virtual)
    if pp_rank == 0:
        counts = [e - s for s, e in ranges]
        print(
            f"[PP] layer split {counts}  "
            f"(first_virtual={first_virtual:.2f}  last_virtual={last_virtual:.2f})",
            flush=True,
        )
    ls, le   = ranges[pp_rank]
    is_first = pp_rank == 0
    is_last  = pp_rank == pp_size - 1

    # Read config values before to_empty() modifies parameters.
    H = model.model.language_model.cfg.hidden_size
    R = model.text_inv_freq.shape[0] * 2   # rope_dim = 2 * len(inv_freq)

    stage_module = PPStageModule(model, ls, le, is_first, is_last)

    if meta_load:
        from models.qwen3_5.utils import load_stage_weights
        stage_module.to_empty(device=device)
        stage_module.to(dtype)
        load_stage_weights(stage_module, snapshot_dir, ls, le, is_first, is_last, device, dtype)
    else:
        dtype = next(model.parameters()).dtype
        stage_module = stage_module.to(device)

    loss_fn  = _ScaledLoss()
    schedule = _PPSchedule(
        stage_module, pp_rank, pp_size, pp_group,
        hidden_size=H, rope_dim=R, dtype=dtype,
        loss_fn=loss_fn if is_last else None,
    )

    return stage_module, None, schedule, loss_fn, pp_rank, pp_size, is_last
