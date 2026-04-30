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
    """Build the world device mesh in torchtitan layout.

    Mesh layout:
      - ``ep_size == 1``: ``(pp, dp_replicate, dp_shard, tp)``
      - ``ep_size  > 1``: ``(pp, dp_replicate, dp_shard_mod_ep, dp_shard_in_ep, tp)``
        where ``dp_shard = dp_shard_mod_ep * dp_shard_in_ep`` and
        ``dp_shard_in_ep == ep_size``.

    EP overlays a contiguous slice of ``dp_shard`` — the EP ranks are a subset
    of the DP ranks rather than a separate axis. Size-1 dims are kept in the
    mesh so dim lookups by name always succeed.

    The mesh is pre-flattened with the following composite dim names so that
    callers can access them directly:
      - ``"dp"``         : dp_replicate × dp_shard
      - ``"dp_shard"``   : dp_shard_mod_ep × dp_shard_in_ep (only when ep > 1)
      - ``"dp_mod_ep"``  : dp_replicate × dp_shard_mod_ep   (only when ep > 1)

    Use the ``get_*_mesh`` helpers below rather than indexing the mesh by name
    directly.
    """
    pp = max(getattr(training_args, "pp_size", 1), 1)
    tp = max(getattr(training_args, "tp_size", 1), 1)
    ep = max(getattr(training_args, "ep_size", 1), 1)

    dp_total = world_size // (pp * tp)
    if pp * tp * dp_total != world_size:
        raise ValueError(
            f"world_size={world_size} not divisible by pp*tp={pp*tp} (pp={pp}, tp={tp})"
        )

    dp_replicate = getattr(training_args, "dp_replicate_size", -1)
    dp_shard = getattr(training_args, "dp_shard_size", -1)
    explicit_replicate = dp_replicate is not None and dp_replicate > 0
    explicit_shard = dp_shard is not None and dp_shard > 0

    if explicit_replicate and explicit_shard:
        pass
    elif explicit_replicate:
        dp_shard = dp_total // dp_replicate
    elif explicit_shard:
        dp_replicate = dp_total // dp_shard
    else:
        # Infer from ``data_parallel`` legacy knob. EP > 1 forces FSDP because
        # routed-expert FSDP must run on the dp_shard sub-mesh orthogonal to EP.
        dp_mode = getattr(training_args, "data_parallel", "ddp")
        if ep > 1 or dp_mode == "fsdp":
            dp_shard, dp_replicate = dp_total, 1
        else:
            dp_shard, dp_replicate = 1, dp_total

    if dp_replicate * dp_shard != dp_total:
        raise ValueError(
            f"dp_replicate({dp_replicate}) * dp_shard({dp_shard}) != "
            f"dp_total({dp_total}) [world_size={world_size}, pp={pp}, tp={tp}]"
        )

    if ep > 1:
        if dp_shard % ep != 0:
            raise ValueError(
                f"ep_size={ep} must divide dp_shard={dp_shard}. "
                f"Either reduce ep_size, or set dp_shard_size to a multiple of ep_size."
            )
        dp_shard_mod_ep = dp_shard // ep
        dims = (pp, dp_replicate, dp_shard_mod_ep, ep, tp)
        names = ("pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "tp")
    else:
        dims = (pp, dp_replicate, dp_shard, tp)
        names = ("pp", "dp_replicate", "dp_shard", "tp")

    mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)

    # Pre-flatten composite DP dims and stash the resulting submeshes on the
    # root mesh. Avoids the deprecated "slice-from-root after flatten" pattern.
    flat = {}
    if ep > 1:
        flat["dp"]        = mesh[("dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep")]._flatten("dp")
        flat["dp_shard"]  = mesh[("dp_shard_mod_ep", "dp_shard_in_ep")]._flatten("dp_shard")
        flat["dp_mod_ep"] = mesh[("dp_replicate", "dp_shard_mod_ep")]._flatten("dp_mod_ep")
    else:
        flat["dp"] = mesh[("dp_replicate", "dp_shard")]._flatten("dp")
    mesh._flattened_submeshes = flat
    return mesh


def _flat(mesh, name):
    return mesh._flattened_submeshes[name]


def get_tp_mesh(mesh):
    return mesh["tp"]


def get_pp_mesh(mesh):
    return mesh["pp"]


def get_dp_mesh(mesh):
    """Flattened DP mesh = dp_replicate × dp_shard (× dp_shard_in_ep when ep>1)."""
    return _flat(mesh, "dp")


def get_dp_replicate_mesh(mesh):
    return mesh["dp_replicate"]


def get_dp_shard_mesh(mesh):
    """The pure-shard portion of DP. Includes the EP slice when ep > 1."""
    if "dp_shard" in mesh._flattened_submeshes:
        return _flat(mesh, "dp_shard")
    return mesh["dp_shard"]


def get_ep_mesh(mesh):
    """The EP submesh (a slice of dp_shard). ``None`` when ep_size == 1."""
    if "dp_shard_in_ep" in mesh.mesh_dim_names:
        return mesh["dp_shard_in_ep"]
    return None


def get_dp_mod_ep_mesh(mesh):
    """DP mesh with the EP slice factored out, for FSDP'ing routed experts.

    Equals the full DP mesh when ep_size == 1.
    """
    if "dp_mod_ep" in mesh._flattened_submeshes:
        return _flat(mesh, "dp_mod_ep")
    return _flat(mesh, "dp")


# Back-compat aliases: older code calls these. Prefer the ``_mesh`` names above.
get_tp_group = get_tp_mesh
get_pp_group = get_pp_mesh
get_dp_group = get_dp_mesh
get_ep_group = get_ep_mesh

def apply_ep(model, ep_mesh, tp_mesh=None):
    """Shard expert parameters across EP ranks and attach a TokenDispatcher.

    EP shards routed experts along ``num_experts``. When ``tp_mesh`` is provided
    (EP+TP), each rank additionally holds only its slice of
    ``moe_intermediate_size``; the partial down-projection is all-reduced
    across TP at the end of ``forward_ep``. The shared_expert is sharded by
    ``apply_tp`` separately.

    For pure EP (``tp_size == 1``) the expert parameters are wrapped as
    ``DTensor`` with ``Shard(0)`` placement on ``ep_mesh`` — autograd, FSDP
    composition (on the orthogonal ``dp_mod_ep`` submesh) and gradient
    reductions then follow the standard DTensor path. For EP+TP we keep the
    legacy raw-``nn.Parameter`` layout because the fused ``[gate; up]`` split
    is not expressible as a single ``Shard`` placement; the manual
    ``all_reduce`` shims in ``forward_ep`` handle TP partial sums.
    """
    ep_rank = ep_mesh.get_local_rank()
    ep_size = ep_mesh.size()
    tp_rank = tp_mesh.get_local_rank() if tp_mesh is not None else 0
    tp_size = tp_mesh.size() if tp_mesh is not None else 1

    lm = model.model.language_model
    for layer in lm.layers:
        moe = layer.mlp
        experts = moe.experts
        num_experts = experts.num_experts
        moe_inter = experts.intermediate_dim

        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
            )
        if tp_size > 1 and moe_inter % tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size={moe_inter} must be divisible by tp_size={tp_size}"
            )

        num_local = num_experts // ep_size
        e_start, e_end = ep_rank * num_local, (ep_rank + 1) * num_local
        local_inter = moe_inter // tp_size
        i_start, i_end = tp_rank * local_inter, (tp_rank + 1) * local_inter

        # If from_pretrained already reshaped the expert parameters on meta
        # (via _reshape_experts_for_ep_meta), the GPU tensors are already the
        # correct local size — skip the data-slicing step entirely.
        already_sliced = (experts.gate_up_proj.shape[0] == num_local)

        if not already_sliced:
            # gate_up_proj: [E, 2*I, H] → [E_local, 2*(I/tp), H]
            gate_up = experts.gate_up_proj.data[e_start:e_end]
            if tp_size > 1:
                gate_part = gate_up[:, :moe_inter, :][:, i_start:i_end, :]
                up_part = gate_up[:, moe_inter:, :][:, i_start:i_end, :]
                gate_up = torch.cat([gate_part, up_part], dim=1)
            experts.gate_up_proj = nn.Parameter(gate_up.contiguous())

            # down_proj: [E, H, I] → [E_local, H, I/tp]
            down = experts.down_proj.data[e_start:e_end, :, i_start:i_end]
            experts.down_proj = nn.Parameter(down.contiguous())

        if tp_size == 1:
            # Pure EP: wrap the locally-shaped expert tensors as DTensor on
            # ep_mesh with Shard(0). Subsequent FSDP wrapping on dp_mod_ep
            # composes the placements automatically.
            experts.gate_up_proj = nn.Parameter(
                DTensor.from_local(
                    experts.gate_up_proj.data, ep_mesh, (Shard(0),), run_check=False
                )
            )
            experts.down_proj = nn.Parameter(
                DTensor.from_local(
                    experts.down_proj.data, ep_mesh, (Shard(0),), run_check=False
                )
            )

        # Build the TokenDispatcher and tag the experts module with the TP
        # mesh (encapsulated inside MoE so this function only handles param
        # sharding).
        moe.attach_ep(ep_mesh, tp_mesh=tp_mesh)

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
    submodule_name: str = "layers",
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to ``model.<submodule_name>``.

    The wrapper iterates ``model.get_submodule(submodule_name).named_children()``
    and replaces each child with a checkpointed version. Use
    ``submodule_name='layers'`` for the language model decoder stack and
    ``submodule_name='blocks'`` for the ViT, where activation memory scales
    with patch count and is the dominant cause of memory variance under
    variable-resolution image inputs.
    """
    # see: https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    if ac_config.enabled:

        if not ac_config.full: op_sac_save_list = _op_sac_save_list
        else: op_sac_save_list = set()

        layers = model.get_submodule(submodule_name)
        for layer_id, transformer_block in layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block,
                ac_config,
                base_fqn=f"{submodule_name}.{layer_id}",
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

def apply_fsdp(model_type, model, *, world_mesh, reshard_after_forward_policy='never',
               shard_visual: bool = False):
    """Apply FSDP. ``world_mesh`` is the full mesh built by :func:`get_mesh`.

    Qwen3.5 uses ``world_mesh`` directly (it needs ``dp`` and ``dp_mod_ep``
    submeshes). The dense Qwen3 / Qwen3-VL paths only need the flattened ``dp``
    mesh and read it via :func:`get_dp_mesh`.

    ``shard_visual`` controls whether the ViT blocks are FSDP-wrapped. Default
    ``False`` (visual replicated) — see ``Training.fsdp_visual`` for rationale.
    """
    if model_type == ModelType.Qwen3_5:
        apply_fsdp_qwen3_5(model, world_mesh, reshard_after_forward_policy,
                           shard_visual=shard_visual)
    elif model_type == ModelType.Qwen3_text:
        apply_fsdp_qwen3(model, get_dp_mesh(world_mesh), reshard_after_forward_policy)
    elif model_type == ModelType.Qwen3_vl:
        apply_fsdp_qwen3_vl(model, get_dp_mesh(world_mesh), reshard_after_forward_policy)

def apply_fsdp_qwen3_5(model, world_mesh, reshard_after_forward_policy='never',
                       *, shard_visual: bool = False):
    """FSDP for Qwen3.5 MoE.

    When EP > 1, expert params are already DTensor on ``ep_mesh``; FSDP'ing
    them on ``dp_mesh`` would all-gather across EP ranks holding *different*
    expert subsets. We instead wrap ``mlp.experts`` separately on
    ``dp_mod_ep_mesh`` (orthogonal to EP) so the resulting 2D placement
    ``(ep × dp_mod_ep)`` covers all DP ranks correctly. Non-expert params in
    each block stay on the full ``dp_mesh``.
    """
    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never" | "default":
            reshard_after_forward = False
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    dp_mesh = get_dp_mesh(world_mesh)
    dp_mod_ep_mesh = get_dp_mod_ep_mesh(world_mesh)
    ep_mesh = get_ep_mesh(world_mesh)
    has_ep = ep_mesh is not None and ep_mesh.size() > 1

    inner = model.model
    lm = inner.language_model

    for transformer_block in lm.layers:
        if has_ep and hasattr(transformer_block.mlp, "experts"):
            # Wrap expert submodule first on the EP-orthogonal mesh — fully_shard
            # on a parent module skips already-FSDP'd children.
            fully_shard(
                transformer_block.mlp.experts,
                mesh=dp_mod_ep_mesh,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(
            transformer_block,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
        )

    # Visual policy:
    #   shard_visual=True  → per-block FSDP unit (max memory savings, but each
    #     block keeps its activation alive until the unit's post-bwd hook fires;
    #     under variable patch counts this introduces per-step memory variance).
    #   shard_visual=False → no per-block wrapping. Visual params get claimed
    #     by the top-level wrap below and end up sharded as one bulk group;
    #     the all-gather/reshard happens at model boundary, not between
    #     individual visual blocks → constant activation footprint per step.
    if shard_visual and inner.visual is not None:
        for transformer_block in inner.visual.blocks:
            fully_shard(
                transformer_block,
                mesh=dp_mesh,
                reshard_after_forward=reshard_after_forward,
            )

    top_lm = [x for x in [lm.norm, lm.embed_tokens] if x is not None]
    if top_lm:
        fully_shard(
            top_lm,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # Top-level wrap: single FSDP root claims any params not yet wrapped
    # (lm_head, visual.* when shard_visual=False, etc.).
    fully_shard(model, mesh=dp_mesh)

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
            "mlp.shared_expert.gate_proj": colwise_parallel(),
            "mlp.shared_expert.down_proj": rowwise_parallel(output_layouts=Replicate()),
            "mlp.shared_expert.up_proj": colwise_parallel(),
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
