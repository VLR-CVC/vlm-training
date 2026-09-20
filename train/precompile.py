import time

import torch

# the repo's own logger: `logging.getLogger(__name__)` has no handler here and
# propagate is off, so anything logged through it is silently dropped.
from train.logger import logger

# (images, patches_per_image, documents). Chosen to span the measured plotqa_cot
# range (patches 2,560-49,936 over 1-22 documents, `scripts/scaling/shape_histogram.py`)
# rather than to enumerate it: dynamo specialises on the first size it sees and
# generalises on the second, so two well-separated values per axis buy most of what
# a long list would, and every batch exercises the full and the SP-sharded shape at
# once. The text-only entry is first because it is the cheapest graph to get wrong.
DEFAULT_SPECS = (
    (0, 0, 4),       # text only
    (1, 2560, 2),    # smallest observed image
    (1, 4096, 4),    # }- see below: two values inside the 2561-5119 gap
    (1, 4736, 6),    # }
    (4, 2560, 8),    # several small images, many documents
    (2, 8192, 16),   # mid patch count, and 16 docs -> the max_k=2048 bucket
    (4, 8192, 6),    # upper-mid patch count
    (1, 25088, 1),   # one large image, single document
    (2, 24576, 3),   # near the top of the range
)

def _near_square(patches: int, spatial_merge_unit: int) -> tuple[int, int]:
    """Factor `patches` into (h, w) as close to square as the merge grid allows."""
    side = int(spatial_merge_unit**0.5)  # each of h, w must be a multiple of this
    best = (side, patches // side)
    for h in range(int(patches**0.5) // side * side, side - 1, -side):
        if patches % h == 0 and (patches // h) % side == 0:
            best = (h, patches // h)
            break
    return best

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
) -> dict:
    """One packed micro-batch with the same keys and dtypes the collator produces."""
    if patches_per_image % spatial_merge_unit:
        raise ValueError(
            f"patches_per_image={patches_per_image} must be a multiple of "
            f"spatial_merge_unit={spatial_merge_unit}"
        )

    total_patches = images * patches_per_image
    per_image = patches_per_image // spatial_merge_unit
    n_image_tokens = total_patches // spatial_merge_unit
    # +images for the separators below
    if n_image_tokens + images >= seq_len:
        raise ValueError(
            f"{n_image_tokens} image tokens do not fit seq_len={seq_len}"
        )

    # NOT randint(0, vocab_size): that draws `image_token_id` by chance and every
    # stray draw becomes another placeholder run, which `get_vision_positions`
    # rejects. Token *values* never affect a graph shape, so a low range is free.
    tokens = torch.randint(
        0, 1000, (seq_len,), generator=generator, dtype=torch.long
    )
    # `models/common/multimodal.py:47` requires exactly one contiguous placeholder
    # run per visual item, so the images cannot share a single block: one
    # separator token between them is what makes the runs distinct.
    at = 0
    for _ in range(images):
        tokens[at : at + per_image] = image_token_id
        at += per_image + 1

    # Documents are equal slices; `positions` restarting at 0 is what the model
    # derives the varlen offsets from, so this is what makes the doc count real.
    positions = torch.empty(seq_len, dtype=torch.long)
    bounds = [round(i * seq_len / documents) for i in range(documents + 1)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        positions[a:b] = torch.arange(b - a)

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
    vis = hf.get("vision_config")
    if not vis:
        logger.warning("precompile: no vision_config, skipping")
        return

    patch_dim = vis["in_channels"] * vis["temporal_patch_size"] * vis["patch_size"] ** 2
    merge_unit = vis["spatial_merge_size"] ** 2
    vocab_size = hf.get("vocab_size") or hf["text_config"]["vocab_size"]

    # fixed seed
    gen = torch.Generator().manual_seed(1234)

    t0 = time.perf_counter()
    for i, (images, patches, docs) in enumerate(specs):
        batch = synthetic_batch(
            images,
            patches,
            docs,
            seq_len=trainer.data_args.seq_len,
            image_token_id=trainer.image_token_id,
            vocab_size=vocab_size,
            patch_dim=patch_dim,
            spatial_merge_unit=merge_unit,
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
                f"precompile {i + 1}/{len(specs)}: images={images} "
                f"patches={images * patches} docs={docs} "
                f"{time.perf_counter() - s:.1f}s"
            )

    torch.cuda.synchronize()
    if trainer.if_log_rank():
        logger.info(f"precompile: {len(specs)} batches in {time.perf_counter() - t0:.1f}s")
