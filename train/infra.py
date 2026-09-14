from dataclasses import dataclass
from functools import partial

from train.config import ModelType
from train.logger import logger

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
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from torchao.float8 import convert_to_float8_training
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

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
    """
    Creates a 2D DeviceMesh based on tp_size and world_size.
    Always returns ('dp', 'tp').
    """
    tp_size = training_args.tp_size
    
    if world_size % tp_size != 0:
        raise ValueError(f"World size {world_size} is not divisible by TP size {tp_size}")

    dp_size = world_size // tp_size

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

def compile_model(
    model: torch.nn.Module,
    fsdp: bool = False,
    dynamic: bool = False,
    compile_gdn: str = "auto",
    block_mode: str = "default",
    head_mode: str = "max-autotune-no-cudagraphs",
    vision_mode: str = "dynamic",
):
    """Compile the decoder blocks.

    Two torch 2.14 bugs at the `torch.compile` + DTensor boundary shape this, both
    reproduced on two GPUs with `scripts/debug/debug_local.sh`:

    * **`dynamic=True` recurses forever.** Tracing a tensor-parallel module builds a
      chain of ~1274 dependent SymInt proxies and dies with `RecursionError` out of
      `torch/fx/experimental/proxy_tensor.py`. Raising `sys.setrecursionlimit` to
      50000 and running on a 1 GiB thread stack both fail to move it. It is not our
      code: a plain three-`nn.Linear` SwiGLU MLP under `ColwiseParallel`/
      `RowwiseParallel` reproduces it exactly, and the same MLP passes with
      `dynamic=False`. Hence static shapes here.

    * **GatedDeltaNet compiles only when FSDP will wrap the model.** With TP alone
      the linear-attention layers fail in the DTensor backward with
      `AttributeError: 'Tensor' object has no attribute '_local_tensor'`
      (`torch/distributed/tensor/_redistribute.py:2069` -- `NestedRedistribute`
      receiving a plain tensor as `grad_output`). Adding FSDP on top makes the same
      layers compile and run. Measured on a dp=2 x tp=2 mesh, 8 layers, T=4096:
      GDN skipped 1133.9 ms / 10.38 GiB, GDN compiled 1057.1 ms / 10.24 GiB.

      Hence the `fsdp` flag: `train_qwen.py` passes `data_parallel == 'fsdp'`, since
      compilation happens before sharding and this function cannot tell otherwise.

    `compile_gdn` overrides the FSDP heuristic ("auto" | "on" | "off"), to check
    whether a newer torch has fixed the TP-only case.

    `head_mode` is the `torch.compile` mode for the three modules that are
    compiled on their own -- `language_model.norm`, `lm_head` and `visual.merger`
    -- or "off" to leave them eager. `max-autotune-no-cudagraphs` benchmarks every
    Triton candidate for a [T, 4096] x [4096, 248320] GEMM at compile time, which
    is most of the several minutes before the first steady step; "default" hands
    it to cuBLAS instead. Before the per-block rewrite `lm_head` lived outside
    `model.model` and was never compiled at all, so "off" is the pre-regression
    behaviour.
    """
    inner = model.model

    if compile_gdn not in ("auto", "on", "off"):
        raise ValueError(f"compile_gdn must be auto/on/off, got {compile_gdn!r}")
    want_gdn = fsdp if compile_gdn == "auto" else compile_gdn == "on"

    compiled = skipped = 0
    for transformer_block in inner.language_model.layers:
        is_gdn = not hasattr(transformer_block, "self_attn")
        if is_gdn and not want_gdn:
            skipped += 1
            continue
        transformer_block.compile(dynamic=dynamic, fullgraph=False, mode=block_mode)
        compiled += 1

    logger.info(
        f"compiled {compiled} decoder blocks (dynamic={dynamic}, mode={block_mode}), "
        f"left {skipped} gated-delta-rule blocks eager "
        f"(compile_gdn={compile_gdn}, fsdp={fsdp})"
    )

    if vision_mode not in ("off", "static", "dynamic"):
        raise ValueError(f"compile_vision must be off/static/dynamic, got {vision_mode!r}")

    if vision_mode == "off":
        logger.info("vision blocks left eager (compile_vision=off)")
    else:
        # `dynamic=True` rather than inheriting the decoder's setting: the patch
        # count is the leading dimension here and it changes nearly every step.
        vis_dynamic = True if vision_mode == "dynamic" else dynamic
        for transformer_block in inner.visual.blocks:
            transformer_block.compile(
                dynamic=vis_dynamic, fullgraph=False, mode=block_mode
            )
        logger.info(
            f"compiled {len(inner.visual.blocks)} vision blocks "
            f"(dynamic={vis_dynamic}, mode={block_mode}, compile_vision={vision_mode})"
        )

    if head_mode == "off":
        logger.info("norm / lm_head / visual.merger left eager (compile_head_mode=off)")
        return

    inner.language_model.norm = torch.compile(
        inner.language_model.norm, dynamic=dynamic, fullgraph=False, mode=head_mode)
    model.lm_head = torch.compile(
        model.lm_head, dynamic=dynamic, fullgraph=False, mode=head_mode)
    if vision_mode != "off":
        # the merger sees the same varying patch count as the blocks
        inner.visual.merger = torch.compile(
            inner.visual.merger,
            dynamic=True if vision_mode == "dynamic" else dynamic,
            fullgraph=False,
            mode=head_mode,
        )
    logger.info(f"compiled norm / lm_head / visual.merger (mode={head_mode})")

def apply_fsdp(model_type, model, **kwargs):
    if model_type == ModelType.Qwen3_text:
        apply_fsdp_qwen3(model, **kwargs)
    elif model_type in (ModelType.Qwen3_vl, ModelType.Qwen3_5):
        # Qwen3.5 has the same module tree as Qwen3-VL -- `language_model.layers`,
        # `language_model.{norm,embed_tokens}`, `visual.{patch_embed,pos_embed,
        # blocks,merger,deepstack_merger_list}` and `lm_head` -- so the same
        # sharding applies. Before this branch existed `data_parallel = 'fsdp'`
        # fell off the end of this function and did nothing for Qwen3.5: no
        # error, and the trainer still logged "sharding/parallelism applied",
        # while every rank kept the whole model. At 9B that is ~37.6 GB of fp32
        # parameters plus as much again in gradients, which is why 16-node runs
        # OOM'd at ~86 GiB allocated regardless of seq_len or optimizer.
        apply_fsdp_qwen3_vl(model, **kwargs)
    else:
        raise NotImplementedError(
            f"apply_fsdp has no branch for {model_type}. Returning silently here "
            "leaves every rank holding the full model; fail loudly instead."
        )

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
