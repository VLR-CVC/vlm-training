"""What shapes does the dataloader actually produce? (CPU only, no GPUs, no SLURM.)

    python scripts/scaling/shape_histogram.py <config.toml> [batches] [dump.txt]

Every distinct shape that reaches a compiled region costs one `torch.compile` graph
on the rank that sees it first, and every rank blocks at the next collective while
that compile runs -- so the union over ranks is what a run's warmup costs
(`BENCHMARKS.md`). This prints the shape distributions that drive it, so buckets can
be sized before any masking code is written.

Reported per micro-batch:

* **documents per row** -- sets `len(cu_seq_q)`, which is guarded directly
  (`tensor 'args[N].cu_seq_q' size mismatch`)
* **longest document**, raw and after the power-of-two rounding
  `create_varlen_metadata_for_document` already applies -- that rounding is why
  `max_k` is NOT one of the unbounded axes, and is the model for the rest
* **vision patches** -- guarded as `Q_LEN` in `create_block_mask` and as the ViT
  input size; the one with no bucketing today

The dataloader is built exactly as `train/train_qwen.py` builds it, at
`world_size=1`: a single rank sees the whole stream rather than a 1/N shard, so
these are the shapes the *job* produces, not one rank's share.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from transformers import AutoProcessor

from data.task_encoder_factory import build_task_encoder
from train.config import Config
from train.config_manager import ConfigManager


def summarise(name: str, values: list[int], top: int = 8) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    counts = collections.Counter(values)
    v = sorted(values)
    n = len(v)
    print(f"\n{name}: n={n} distinct={len(counts)} "
          f"min={v[0]:,} p50={v[n // 2]:,} p90={v[int(n * 0.9)]:,} max={v[-1]:,}")
    print("  most common:", ", ".join(f"{k:,}x{c}" for k, c in counts.most_common(top)))
    for bucket in (256, 1024, 4096):
        print(f"  distinct after rounding up to a multiple of {bucket:,}: "
              f"{len({-(-x // bucket) * bucket for x in values})}")
    print(f"  distinct after rounding up to a power of two: "
          f"{len({1 if x <= 1 else 1 << (x - 1).bit_length() for x in values})}")


def main() -> None:
    cfg_path = sys.argv[1]
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    cfg = ConfigManager(Config).parse_args([f"--config={cfg_path}"])

    processor = AutoProcessor.from_pretrained(cfg.training.model_dir, max_pixels=1048576)
    import json
    hf_config = json.loads(Path(cfg.model.model_config).read_text())
    task_encoder, extra = build_task_encoder(
        cfg.data, processor,
        image_token_id=hf_config["image_token_id"],
        video_token_id=hf_config["video_token_id"],
        spatial_merge_size=hf_config["vision_config"]["spatial_merge_size"],
    )
    ds = get_train_dataset(
        cfg.data.data_path, batch_size=1, repeat=cfg.data.repeat,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        max_samples_per_sequence=cfg.data.max_samples_per_sequence,
        task_encoder=task_encoder,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        **extra,
    )

    docs, longest, patches, tokens = [], [], [], []
    for i, batch in enumerate(get_loader(ds)):
        if i >= n_batches:
            break
        positions = batch["positions"].squeeze(0)
        starts = (positions == 0).nonzero(as_tuple=True)[0]
        lengths = torch.diff(torch.cat([starts, torch.tensor([positions.numel()])]))
        docs.append(int(starts.numel()))
        longest.append(int(lengths.max()) if lengths.numel() else 0)
        tokens.append(int(positions.numel()))
        grid = batch.get("grid_thw")
        patches.append(int(grid.prod(-1).sum()) if grid is not None and grid.numel() else 0)

    dump = sys.argv[3] if len(sys.argv) > 3 else ""
    if dump:
        Path(dump).write_text(
            "tokens docs longest patches\n"
            + "\n".join(f"{t} {d} {l} {p}" for t, d, l, p in zip(tokens, docs, longest, patches))
            + "\n"
        )
        print(f"raw per-batch values -> {dump}")

    print(f"config {cfg_path}   dataset {cfg.data.data_path}   batches {len(docs)}")
    summarise("tokens per row", tokens)
    summarise("documents per row  (-> len(cu_seq_q) = this + 1)", docs)
    summarise("longest document   (-> max_k, already power-of-two rounded)", longest)
    summarise("vision patches     (-> Q_LEN, ViT input size; NOT bucketed today)", patches)


if __name__ == "__main__":
    main()
