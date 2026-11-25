"""Pull real micro-batches through the training data path, built from a training config
the way train_qwen.py's Trainer builds it (processor, chat template, task encoder, energon
dataset and loader with the config's workers), without a model or distributed setup: one
data rank's stream. Shows batch fill and load times before spending an allocation.

    python utils/test_metadataset_batch.py --batches 200 -- \\
        --config configs/jupiter/elliot/sft3k_repro_5k_32n.toml --data.rolling_packing true
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="after --: train_qwen.py arguments")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from data.task_encoder_factory import apply_chat_template, build_task_encoder
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset
    from models.qwen3_5.configs import resolve_model_config
    from train.config import Config
    from train.config_manager import ConfigManager

    train_args = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args
    config = ConfigManager(Config).parse_args(train_args)
    d, t = config.data, config.training

    processor_dir = t.model_dir if t.processor_dir == "NULL" else t.processor_dir
    processor = AutoProcessor.from_pretrained(processor_dir, max_pixels=d.max_pixels)
    apply_chat_template(processor, t.chat_template)

    config_path = resolve_model_config(config.model.model_config, t.model_dir, config.model.use_model_dir_config)
    hf_config = json.loads(Path(config_path).read_text())
    if hf_config.get("model_type") == "elliot_vlm":
        from models.elliot.configs import qwen_style_hf_config

        hf_config = qwen_style_hf_config(hf_config)
    vision_config = hf_config.get("vision_config")
    spatial_merge_size = vision_config["spatial_merge_size"] if vision_config else None

    tokens_per_microbatch = d.microbatch_tokens
    encoder, extra_ds_kwargs = build_task_encoder(
        d, processor,
        image_token_id=hf_config.get("image_token_id"),
        video_token_id=hf_config.get("video_token_id"),
        spatial_merge_size=spatial_merge_size,
    )
    print(f"{type(encoder).__name__}: {tokens_per_microbatch} tokens per micro-batch, "
          f"num_workers={d.num_workers} shuffle_buffer={d.shuffle_buffer_size} "
          f"packing_buffer={d.packing_buffer_size} max_pixels={d.max_pixels}", flush=True)

    ds = get_train_dataset(
        d.data_path, batch_size=1, repeat=d.repeat,
        shuffle_buffer_size=d.shuffle_buffer_size,
        max_samples_per_sequence=d.max_samples_per_sequence,
        task_encoder=encoder,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=d.num_workers),
        **extra_ds_kwargs,
    )
    loader = get_loader(ds)

    tokens_seen = assistant_tokens_seen = samples_seen = 0
    patches_seen = images_seen = 0
    batch_effs = []
    load_times = []          # steady-state only, first batch reported separately below
    it = iter(loader)
    t_fetch = time.time()
    first_batch_time = None
    t0 = time.time()
    for step in range(1, a.batches + 1):
        batch = next(it)
        this_load = time.time() - t_fetch
        if step == 1:
            first_batch_time = this_load
        else:
            load_times.append(this_load)
        n_tokens = int(batch["num_tokens"])
        n_loss = int(batch["num_valid_tokens"])
        n_samples = int(batch["num_samples"])
        shape = tuple(batch["input"].shape)
        grid = batch.get("grid_thw")
        n_images = int(grid.shape[0]) if grid is not None and grid.numel() else 0
        n_patches = int(grid.prod(-1).sum()) if grid is not None and grid.numel() else 0
        # train/batch_efficiency exactly: num_tokens / tokens_per_microbatch, not
        # num_tokens / actual-input-length (padding is itself a "document" in the packed
        # row, so len(input) == tokens_per_microbatch always; this is the real metric).
        batch_eff = 100 * n_tokens / tokens_per_microbatch

        tokens_seen += n_tokens
        assistant_tokens_seen += n_loss
        samples_seen += n_samples
        patches_seen += n_patches
        images_seen += n_images
        batch_effs.append(batch_eff)

        print(f"[{step:3d}] load={this_load * 1000:6.1f}ms input {shape} tokens={n_tokens:6d} "
              f"loss={n_loss:6d} ({100 * n_loss / max(n_tokens, 1):4.1f}%) "
              f"batch_eff={batch_eff:5.1f}% samples={n_samples:2d} images={n_images} "
              f"patches={n_patches:,}", flush=True)
        t_fetch = time.time()

    elapsed = time.time() - t0
    print(f"\n{a.batches} batches in {elapsed:.1f}s")
    print(f"tokens_seen={tokens_seen:,}  assistant_tokens_seen={assistant_tokens_seen:,} "
         f"({100 * assistant_tokens_seen / max(tokens_seen, 1):.1f}%)  num_samples={samples_seen:,}")
    print(f"mean tokens/batch={tokens_seen / a.batches:,.0f}  "
         f"(expected exactly {tokens_per_microbatch:,} if every batch is fully packed)")
    be = sorted(batch_effs)
    print(f"batch_efficiency: mean={sum(be) / len(be):.1f}%  p5={be[len(be) // 20]:.1f}%  "
          f"min={be[0]:.1f}%  max={be[-1]:.1f}%  under 90%: {sum(e < 90 for e in be)}/{len(be)}")
    print(f"images={images_seen:,}  vision_patches={patches_seen:,}  "
         f"(vision_tokens = patches / merge_size^2 = {patches_seen // ((spatial_merge_size or 1) ** 2):,})")

    print(f"\nfirst batch (incl. one-time dataset-open cost): {first_batch_time * 1000:.0f}ms")
    if load_times:
        lt = sorted(load_times)
        n = len(lt)
        p50 = lt[n // 2]
        p95 = lt[min(n - 1, int(n * 0.95))]
        print(f"steady-state load time ({n} batches): mean={sum(lt) / n * 1000:.1f}ms "
             f"p50={p50 * 1000:.1f}ms p95={p95 * 1000:.1f}ms max={lt[-1] * 1000:.1f}ms "
             f"min={lt[0] * 1000:.1f}ms")
        print(f"throughput: {n / sum(lt):.2f} micro-batches/s = {tokens_seen / elapsed:,.0f} tokens/s "
             f"(one data rank, no GPU compute overlapped: the most this rank's loader can supply)")


if __name__ == "__main__":
    main()
