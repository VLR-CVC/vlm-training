"""S3 gate (TITAN_MIGRATION_v2.md): TP and TP+SP match TP=1 on Qwen3.5-9B.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   python -m torch.distributed.run --nproc_per_node=1 models/tests/test_titan_tp_parity.py run tp1
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 models/tests/test_titan_tp_parity.py run tp2
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 models/tests/test_titan_tp_parity.py run tp2sp
    python models/tests/test_titan_tp_parity.py compare

Every run builds the model the way the trainer does (meta -> sharding configs ->
Module.parallelize -> FSDP on the storage mesh -> to_empty -> HF load), feeds the
same packed multimodal row, and saves: text-position logits (gathered over the
vocab shards), the step loss, the global gradient norm, and per-parameter
gradient norms. `compare` checks TP=2 and TP=2+SP against TP=1.
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SNAPSHOT = os.environ.get("QWEN3_5_9B_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_9b")
IMAGE = Path(__file__).resolve().parents[2] / "test_images" / "horse.png"
OUT = Path(os.environ.get("TP_PARITY_OUT", "/data/151-2/users/tockier/tests/s3_tp_parity")) / (
    ("vision_only" if os.environ.get("VISION_ONLY") == "1"
     else "text_only" if os.environ.get("TEXT_ONLY") == "1" else "multimodal")
    + ("_fp32" if os.environ.get("PARAM_DTYPE") == "fp32" else "")
    + ("_compiled" if os.environ.get("COMPILE") == "1" else ""))
MODES = {"tp1": (1, False), "tp2": (2, False), "tp2sp": (2, True), "dp2tp2sp": (2, True), "tp4sp": (4, True)}
# replicated weights used inside local (kernel) regions: the old DTensor TP needed
# hand-written all-reduce hooks for exactly these
WATCH = ("q_norm.weight", "k_norm.weight", "attn.norm.weight", "A_log", "dt_bias")


def make_row():
    """[image+text doc | text doc | padding], length a multiple of 8, as the
    energon batch -> titan batch adapter produces it."""
    from PIL import Image
    from transformers import AutoProcessor

    from data.titan_batch import to_titan_batch

    proc = AutoProcessor.from_pretrained(SNAPSHOT)
    cfg = json.loads((Path(SNAPSHOT) / "config.json").read_text())
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A brown horse standing in a grassy field."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False)
    image = Image.open(IMAGE).convert("RGB").resize((448, 448))
    mm = proc(text=[text], images=[image], return_tensors="pt")
    doc2 = proc.tokenizer("The capital of France is Paris, and the capital of Germany is Berlin.",
                          return_tensors="pt").input_ids[0]
    text_only = os.environ.get("TEXT_ONLY") == "1"
    if text_only:
        # same structure, no image: TP exactness without bf16 ViT noise
        doc1 = proc.tokenizer(proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "Describe a horse in a field."}]},
             msgs[1]], tokenize=False), return_tensors="pt").input_ids[0]
    else:
        doc1 = mm["input_ids"][0]
    ids = torch.cat([doc1, doc2])
    n1, n2 = doc1.shape[0], doc2.shape[0]
    total = ((ids.shape[0] + 1) // 8 + 1) * 8
    pad = total - ids.shape[0]
    ids = torch.cat([ids, torch.full((pad,), proc.tokenizer.pad_token_id)])
    labels = ids.clone()
    labels[n1 + n2:] = -100
    labels[ids == cfg["image_token_id"]] = -100  # image pads are never supervised
    batch = {
        "input_ids": ids,
        "labels": labels,
        "cu_seqlens": torch.tensor([0, n1, n1 + n2, total], dtype=torch.int32),
    }
    if not text_only:
        batch["pixel_values"] = mm["pixel_values"]
        batch["image_grid_thw"] = mm["image_grid_thw"]
    out = to_titan_batch(batch, image_token_id=cfg["image_token_id"],
                         video_token_id=cfg["video_token_id"],
                         spatial_merge_size=cfg["vision_config"]["spatial_merge_size"])
    text_pos = (ids != cfg["image_token_id"]) & (torch.arange(total) < n1 + n2)
    return out, text_pos, cfg["image_token_id"]


def run(mode: str) -> None:
    import torch.distributed._functional_collectives as funcol

    from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
    from train.parallel.parallel_dims import ParallelDims
    from train.parallel.parallelize import parallelize_qwen3_5
    from train.parallel.spmd import set_current_spmd_mesh, set_spmd_meshes
    from train.titan_step import forward_backward

    tp, sp = MODES[mode]
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    pd = ParallelDims(dp_replicate=1, dp_shard=world // tp, cp=1, tp=tp, pp=1, ep=1, world_size=world)
    pd.build_mesh()

    model = build_meta(SNAPSHOT, seq_len=4096, tp=tp, enable_sp=sp)
    # PARAM_DTYPE=fp32: fp32 GEMMs (TF32 off), only the GDN kernel runs in bf16 --
    # removes shape-dependent bf16 rounding so TP=1 and TP=2 should agree closely
    dtype = torch.float32 if os.environ.get("PARAM_DTYPE") == "fp32" else torch.bfloat16
    for p in model.parameters():
        p.data = p.data.to(dtype)
    parallelize_qwen3_5(model, pd, mode="fsdp", compile=os.environ.get("COMPILE") == "1",
                        param_dtype=dtype, reduce_dtype=torch.float32)
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)

    row, text_pos, image_id = make_row()
    row = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in row.items()}
    if "pixel_values" in row:
        row["pixel_values"] = row["pixel_values"].to(dtype)

    # per-layer outputs for localisation (TP without SP keeps them replicated)
    captured = {}
    if os.environ.get("CAPTURE_LAYERS") == "1" and not sp:
        def hook(name):
            def fn(mod, args, out):
                captured[name] = out.detach().float().cpu()
            return fn
        for i, layer in model.layers.items():
            layer.register_forward_hook(hook(f"layer{int(i):02d}"))
            if int(i) in (0, 3):
                for sub in ("attention_norm", "attn", "ffn_norm", "feed_forward"):
                    getattr(layer, sub).register_forward_hook(hook(f"layer{int(i):02d}.{sub}"))
                if int(i) == 0:
                    for sub in ("in_proj_q", "in_proj_v", "in_proj_b", "norm", "out_proj"):
                        getattr(layer.attn, sub).register_forward_hook(hook(f"layer00.attn.{sub}"))
                    layer.attn.inner_gated_delta_net.register_forward_hook(hook("layer00.attn.inner"))
        if "pixel_values" in row:
            model.vision_encoder.register_forward_hook(hook("vision"))
        model.tok_embeddings.register_forward_hook(hook("embed"))

    # logits, full vocab gathered from the TP shards
    set_spmd_meshes(dense_mesh=pd.spmd_dense_mesh(), sparse_mesh=None)
    with torch.no_grad(), set_current_spmd_mesh(pd.spmd_dense_mesh()):
        b = dict(row)
        b.pop("num_valid_tokens")
        inputs, _, extra = model.preprocess_inputs(b, parallel_dims=pd)
        model._skip_lm_head = False
        logits = model(inputs, special_tokens={"image_id": image_id}, **extra)
        if tp > 1:
            logits = funcol.all_gather_tensor(logits.contiguous(), -1, pd.get_mesh("tp").get_group())
            logits = funcol.wait_tensor(logits) if hasattr(funcol, "wait_tensor") else logits
    logits = logits[text_pos.cuda()].float().cpu()
    if captured and torch.distributed.get_rank() == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(captured, OUT / f"{mode}_layers.pt")

    grad_in = {}
    if os.environ.get("CAPTURE_GRADS") == "1" and not sp:
        def bhook(name):
            def fn(mod, g_in, g_out):
                gi = g_in[0] if g_in and g_in[0] is not None else None
                go = g_out[0] if g_out and g_out[0] is not None else None
                grad_in[name] = (None if gi is None else gi.float().norm().item(),
                                 None if go is None else go.float().norm().item())
            return fn
        for i in range(32):
            layer = model.layers[str(i)]
            layer.register_full_backward_hook(bhook(f"layer{i:02d}"))
            for sub in ("attention_norm", "attn", "ffn_norm", "feed_forward"):
                getattr(layer, sub).register_full_backward_hook(bhook(f"layer{i:02d}.{sub}"))

    if os.environ.get("VISION_ONLY") == "1":
        # vision tower in isolation: loss = <vision_encoder(pixels), fixed random W>
        import spmd_types as _spmd

        with set_current_spmd_mesh(pd.spmd_dense_mesh()):
            b = dict(row)
            b.pop("num_valid_tokens")
            _, _, extra = model.preprocess_inputs(b, parallel_dims=pd)
            out = model.vision_encoder(extra["pixel_values"], grid_thw=extra["grid_thw"])
            w = torch.randn(out.shape, generator=torch.Generator().manual_seed(7)).to(out.device, out.dtype)
            loss = (out * w).sum()
            with _spmd.no_typecheck():
                loss.backward()
    else:
        loss, _, _ = forward_backward(
            model, [dict(row)], dp_group=pd.get_optional_mesh("batch"),
            special_tokens={"image_id": image_id}, loss_chunks=8, parallel_dims=pd,
        )
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    full = {n: (g.full_tensor() if hasattr(g, "full_tensor") else g).float() for n, g in grads.items()}
    gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in full.values()])).item()
    per_param = {n: torch.linalg.vector_norm(g).item() for n, g in full.items()}
    # every DP rank got the same row; the loss is local_sum / global_tokens, so its
    # SUM over DP is the per-token mean, comparable with tp1. Gradients are reduced
    # as a SUM of identical halves, so they compare directly too.
    batch_mesh = pd.get_optional_mesh("batch")
    if batch_mesh is not None and batch_mesh.size() > 1:
        loss = loss.detach().clone()
        torch.distributed.all_reduce(loss, group=batch_mesh.get_group())
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits, "loss": loss.item(), "grad_norm": gnorm, "per_param": per_param},
                   OUT / f"{mode}.pt")
        print(json.dumps({"mode": mode, "loss": loss.item(), "grad_norm": gnorm, "n_text": int(text_pos.sum())}))
        if grad_in:
            torch.save(grad_in, OUT / f"{mode}_grads.pt")
    torch.distributed.destroy_process_group()


def compare() -> None:
    """Gate on runs with PARAM_DTYPE=fp32 (fp32 GEMMs, TF32 off; only the GDN kernel
    is bf16). In bf16 the bottom GDN layers amplify shape-dependent GEMM rounding:
    at TP=1 alone, bf16 vs fp32 moves the gradient norm by 12%, so bf16 runs cannot
    separate a sharding bug from precision. Measured fp32 noise floor: grad norm
    1.2-1.6%, worst parameter 3.5%. The per-parameter-type median ratio is what
    catches systematic errors -- an exact 2x on one parameter type barely moves the
    global norm."""
    import statistics

    if os.environ.get("PARAM_DTYPE") != "fp32":
        print("[WARN] gates are calibrated for PARAM_DTYPE=fp32 runs")
    ref = torch.load(OUT / "tp1.pt")
    if os.environ.get("VISION_ONLY") == "1":
        # The vision encoder output is R@TP, so a loss computed on every rank sums
        # its gradient over TP: an exact tp-fold ratio on every parameter is correct.
        # Measured: 2.0000 on all 25 parameter types at tp=2.
        r = torch.load(OUT / "tp2.pt")["per_param"]
        ratios = [r[n] / ref["per_param"][n] for n in ref["per_param"]]
        worst = max(abs(x - 2.0) for x in ratios)
        # 1e-3 fit Qwen3.5 (7e-5). Qwen3-VL-2B with DeepStack (job 1856053): 411
        # grads in 2.0000-2.0011, median 5e-5, worst on scattered layer 0-9 norms and
        # biases (grad norm 2.5e5). A sharding bug is an exact factor, not 5e-4.
        ok = worst < 2e-3
        print(f"[{'PASS' if ok else 'FAIL'}] vision-only tp2 vs tp1: every grad ratio 2.0 within {worst:.1e}")
        assert ok
        return
    ok_all = True
    for mode in ("tp2", "tp2sp", "dp2tp2sp", "tp4sp"):
        if not (OUT / f"{mode}.pt").exists():
            print(f"[SKIP] {mode}: no result"); continue
        r = torch.load(OUT / f"{mode}.pt")
        d = (r["logits"] - ref["logits"]).abs()
        top1 = (r["logits"].argmax(-1) == ref["logits"].argmax(-1)).float().mean().item()
        rel_loss = abs(r["loss"] - ref["loss"]) / ref["loss"]
        # decoder-only norm, for the same reason vision params are excluded below
        dec = lambda pp: sum(v * v for n, v in pp.items() if not n.startswith("vision_encoder.")) ** 0.5  # noqa: E731
        rel_gn = abs(dec(r["per_param"]) - dec(ref["per_param"])) / dec(ref["per_param"])
        groups = {}
        for n, v in ref["per_param"].items():
            # vision gradients arrive only through image tokens in the bottom GDN
            # layers, which amplify rounding (0.55-1.28 spread in fp32); the tower's
            # TP is checked exactly by the VISION_ONLY run instead
            if n.startswith("vision_encoder."):
                continue
            key = ".".join(p for p in n.split(".") if not p.isdigit())
            groups.setdefault(key, []).append(r["per_param"][n] / max(v, 1e-12))
        worst_type, worst_ratio = max(
            ((k, statistics.median(v)) for k, v in groups.items()), key=lambda kv: abs(kv[1] - 1)
        )
        # loss gate 2e-3: the varlen attention kernel stays bf16 in these fp32 runs, and
        # Qwen3-VL has it in every layer (Qwen3.5: one in four). Qwen3-VL-2B, job
        # 1856053: eager TP modes 1.3-1.5e-3, the same modes compiled 1.9-5.4e-4, with
        # top-1 100% and every per-type grad ratio within 0.5% -- kernel rounding, not
        # layout. Qwen3.5-9B stays at 3.4-8.1e-4.
        ok = (top1 == 1.0 and d.mean().item() < 0.05 and rel_loss < 2e-3 and rel_gn < 3e-2
              and abs(worst_ratio - 1) < 5e-2)
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {mode} vs tp1: logits top1={top1:.4f} mean|d|={d.mean().item():.3e} "
              f"| loss rel {rel_loss:.1e} | grad norm rel {rel_gn:.1e} "
              f"| worst param-type median grad ratio {worst_ratio:.3f} ({worst_type})")
        watch = {w: statistics.median(v) for k, v in groups.items() for w in WATCH if k.endswith(w)}
        print("       replicated-in-local-region grad ratios: "
              + " ".join(f"{w}={r:.3f}" for w, r in sorted(watch.items())))
    assert ok_all


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else compare()
