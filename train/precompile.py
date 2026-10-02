import time

import torch

from train.logger import logger
from models.common.attention import _MIN_MAX_SEQLEN

# The vision axis only: the decoder axis is covered by the image-free specs
# `vlm_specs` appends, so a text-only entry here would just duplicate one.
DEFAULT_SPECS = (
    (1, 2560, 2),    # smallest observed image
    (1, 4096, 4),    # }- see below: two values inside the 2561-5119 gap
    (1, 4736, 6),    # }
    (4, 2560, 8),    # several small images, many documents
    (2, 8192, 16),   # mid patch count, and 16 docs -> the max_k=2048 bucket
    (4, 8192, 6),    # upper-mid patch count
    (1, 25088, 1),   # one large image, single document
    (2, 24576, 3),   # near the top of the range
)

DOC_BUCKETS = (1, 64, 128, 256, 512)

def _graph_key(documents: int, seq_len: int, longest: int | None) -> tuple[int, int]:
    """The `(cu_seq_q length, max_k)` pair this packing produces -- the two
    quantities the compiled decoder block guards on."""
    from models.common.attention import create_varlen_metadata_for_document

    md = create_varlen_metadata_for_document(
        _synthetic_positions(documents, seq_len, longest)
    )
    return int(md.cu_seq_q.numel()), int(md.max_k)


def text_specs(
    seq_len: int, seen: set[tuple[int, int]] | None = None
) -> tuple[tuple[int, int], ...]:
    """`(documents, longest_document)`"""
    seen = set() if seen is None else seen
    specs: list[tuple[int, int]] = []
    longest = _MIN_MAX_SEQLEN
    while longest <= seq_len:
        for docs in DOC_BUCKETS:
            if longest + docs - 1 > seq_len:
                continue
            key = _graph_key(docs, seq_len, longest)
            if key not in seen:
                seen.add(key)
                specs.append((docs, longest))
        longest *= 2
    return tuple(specs)

def _synthetic_positions(documents: int, seq_len: int, longest: int | None) -> torch.Tensor:
    assert documents >= 1, documents
    # a single document has no remainder to split, so it is the whole row
    head = seq_len if documents == 1 else (longest or 0)
    assert head + documents - 1 <= seq_len, (head, documents, seq_len)

    positions = torch.empty(seq_len, dtype=torch.long)
    if head:
        positions[:head] = torch.arange(head)
    rest = documents - (1 if head else 0)
    if rest > 0:
        bounds = [head + round(i * (seq_len - head) / rest) for i in range(rest + 1)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            positions[a:b] = torch.arange(b - a)
    assert positions[0] == 0
    return positions

def spec_fits(
    images: int, patches_per_image: int, *, seq_len: int, spatial_merge_unit: int
) -> bool:
    return images * patches_per_image // spatial_merge_unit + images < seq_len


def vlm_specs(
    seq_len: int, spatial_merge_unit: int, specs=DEFAULT_SPECS
) -> tuple[list[tuple[int, int, int, int | None]], list[tuple[int, int, int]]]:
    """`(images, patches_per_image, documents, longest)`"""
    fitting = [
        s
        for s in specs
        if spec_fits(s[0], s[1], seq_len=seq_len, spatial_merge_unit=spatial_merge_unit)
    ]
    covered = {_graph_key(s[2], seq_len, None) for s in fitting}
    out: list[tuple[int, int, int, int | None]] = [(*s, None) for s in fitting]
    out += [(0, 0, docs, longest) for docs, longest in text_specs(seq_len, covered)]
    return out, [s for s in specs if s not in fitting]


def _near_square(patches: int, spatial_merge_unit: int) -> tuple[int, int]:
    """Factor `patches` into (h, w) as close to square as the merge grid allows."""
    side = int(spatial_merge_unit**0.5)  # each of h, w must be a multiple of this
    best = (side, patches // side)
    for h in range(int(patches**0.5) // side * side, side - 1, -side):
        if patches % h == 0 and (patches // h) % side == 0:
            best = (h, patches // h)
            break
    return best

def synthetic_batch_text(
    documents: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    longest: int | None = None,
) -> dict:
    tokens = torch.randint(
        0, 1000, (seq_len,), generator=generator, dtype=torch.long
    )

    positions = _synthetic_positions(documents, seq_len, longest)
    labels = tokens.roll(-1)
    labels[-1] = -100

    batch = {
        "input": tokens,
        "labels": labels,
        "positions": positions,
        "mrope_positions": positions.unsqueeze(-1).expand(-1, 3).contiguous(),
        "num_valid_tokens": int((labels != -100).sum()),
    }

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, non_blocking=True)
    return batch

def synthetic_batch(
    images: int,
    patches_per_image: int,
    documents: int,
    *,
    seq_len: int,
    image_token_id: int,
    vocab_size: int,
    patch_dim: int,
    spatial_merge_unit: int,
    device: torch.device,
    generator: torch.Generator,
    longest: int | None = None,
) -> dict:
    if patches_per_image % spatial_merge_unit:
        raise ValueError(
            f"patches_per_image={patches_per_image} must be a multiple of "
            f"spatial_merge_unit={spatial_merge_unit}"
        )

    total_patches = images * patches_per_image
    per_image = patches_per_image // spatial_merge_unit
    n_image_tokens = total_patches // spatial_merge_unit
    # +images for the separators below
    if not spec_fits(
        images, patches_per_image, seq_len=seq_len, spatial_merge_unit=spatial_merge_unit
    ):
        raise ValueError(
            f"{n_image_tokens} image tokens do not fit seq_len={seq_len}"
        )

    tokens = torch.randint(
        0, 1000, (seq_len,), generator=generator, dtype=torch.long
    )
    at = 0
    for _ in range(images):
        tokens[at : at + per_image] = image_token_id
        at += per_image + 1

    positions = _synthetic_positions(documents, seq_len, longest)

    labels = tokens.roll(-1)
    labels[-1] = -100

    batch = {
        "input": tokens,
        "labels": labels,
        "positions": positions,
        "mrope_positions": positions.unsqueeze(-1).expand(-1, 3).contiguous(),
        "num_valid_tokens": int((labels != -100).sum()),
    }
    if images:
        h, w = _near_square(patches_per_image, spatial_merge_unit)
        batch["grid_thw"] = torch.tensor([[1, h, w]] * images, dtype=torch.long)
        batch["pixel_values"] = torch.randn(
            total_patches, patch_dim, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


def precompile(trainer, specs=DEFAULT_SPECS) -> None:
    """Run `specs` through forward+backward, then drop the gradients.

    `trainer` is the `Trainer`; this reads its model, device and config only.
    """
    from train.step import forward_backward

    hf = trainer.hf_config

    vocab_size = hf.get("vocab_size") or hf["text_config"]["vocab_size"]
    seq_len = trainer.data_args.seq_len
    vis = hf.get("vision_config")

    if not vis:
        logger.warning("precompile: text-only setup")
        specs = text_specs(seq_len)
    else:
        patch_dim = vis["in_channels"] * vis["temporal_patch_size"] * vis["patch_size"] ** 2
        merge_unit = vis["spatial_merge_size"] ** 2

        n_given = len(specs)
        specs, skipped = vlm_specs(seq_len, merge_unit, specs)
        if skipped and trainer.if_log_rank():
            logger.warning(
                f"precompile: skipping {len(skipped)}/{n_given} spec(s) whose image "
                f"tokens exceed seq_len={seq_len}: {skipped}"
            )

    # fixed seed
    gen = torch.Generator().manual_seed(1234)

    t0 = time.perf_counter()
    for i, spec in enumerate(specs):
        if vis:
            # (images, patches_per_image, documents, longest_document)
            images, patches, docs, longest = spec
            label = f"images={images} patches={images * patches} docs={docs} longest={longest}"
            batch = synthetic_batch(
                images,
                patches,
                docs,
                longest=longest,
                seq_len=seq_len,
                image_token_id=trainer.image_token_id,
                vocab_size=vocab_size,
                patch_dim=patch_dim,
                spatial_merge_unit=merge_unit,
                device=trainer.device,
                generator=gen,
            )
        else:
            docs, longest = spec
            label = f"docs={docs} longest={longest}"
            batch = synthetic_batch_text(
                seq_len=seq_len,
                documents=docs,
                longest=longest,
                device=trainer.device,
                generator=gen,
            )
        s = time.perf_counter()
        forward_backward(
            trainer.model,
            [batch],
            dp_group=trainer.dp_mesh,
            special_tokens={"image_id": trainer.image_token_id},
            ddp=trainer.training_args.data_parallel == "ddp",
            loss_chunks=trainer.training_args.loss_chunks,
            compile_loss=trainer.training_args.compile,
            parallel_dims=trainer.parallel_dims,
        )
        trainer.model.zero_grad(set_to_none=True)
        if trainer.if_log_rank():
            logger.info(
                f"precompile {i + 1}/{len(specs)}: {label} "
                f"{time.perf_counter() - s:.1f}s"
            )

    torch.cuda.synchronize()
    if trainer.if_log_rank():
        logger.info(f"precompile: {len(specs)} batches in {time.perf_counter() - t0:.1f}s")
