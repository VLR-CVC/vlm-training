"""Pull real batches through the exact training path (PackedBatchEncoder, get_loader) from
a metadataset, at a given seq_len -- the same objects train_qwen.py's Trainer builds, just
without a model or distributed setup. Confirms the blend loads, every dataset's cooker
resolves, and shows what a 24K-token batch actually contains before spending a real
allocation on it.

    python utils/test_metadataset_batch.py \\
        --metadataset /gpfs/scratch/ehpc543/metadatasets/full_grouped.yaml \\
        --model-dir /path/to/Qwen3-VL-2B --seq-len 24576 --batches 20

Each printed line is one micro-batch (tokens_per_microbatch // seq_len rows joined into
one packed sequence, same as training): input_ids shape, tokens, loss (assistant) tokens,
batch efficiency (non-padding fraction of the microbatch's full token capacity -- exactly
train/batch_efficiency / the "batch_util" figure in train_qwen.py's console log, num_tokens
/ tokens_per_microbatch), images, and vision patches (grid_thw.prod(-1).sum(), the same
"patches" field train_qwen.py's QWEN_BATCH_LOG writes -- raw patch count *before* the
spatial-merge divide, so it is `spatial_merge_size**2` times the actual visual token count).
Running totals at the end are exactly train/tokens_seen, train/num_samples,
train/assistant_tokens_seen from wandb -- run for --batches long enough to sanity-check
against your own throughput expectations, not to reproduce a full run.

Per-batch load time (wall clock of one `next(it)` call) is printed and summarized, to
compare a metadataset with many blend entries (e.g. full_grouped.yaml's 120) against a
coarser one (e.g. 3: FineVision, plain onevision_instruct, one merged Nemotron dataset).
`num_workers=0` here, so this measures the raw per-sample Python cost of energon's
blend_epochized picking the next dataset and resolving its cooker on the SAME process
doing the timing -- not the stall time production sees. In the real run (`num_workers=2`
per the wandb config), that cost happens in background worker processes and is prefetched
while the GPU is busy with the previous micro-batch; it only shows up as a training stall
if it exceeds the per-step compute time (~0.9-1.4s/step at 256 GPUs, from job v8ojrckh's
log) *and* workers can't keep the prefetch queue full. First-batch time includes one-time
setup (opening every blended dataset's index.sqlite) and is reported separately from the
steady-state batches so it doesn't skew the comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadataset", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--chat-template", default="assets/chat_template_sft.jinja")
    ap.add_argument("--seq-len", type=int, default=24576)
    ap.add_argument("--tokens-per-microbatch", type=int, default=0, help="0 = one row (= seq_len)")
    ap.add_argument("--max-pixels", type=int, default=1048576, help="as train_qwen.py")
    ap.add_argument("--shuffle-buffer-size", type=int, default=100)
    ap.add_argument("--max-samples-per-sequence", type=int, default=100)
    ap.add_argument("--packing-buffer-size", type=int, default=100)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoConfig, AutoProcessor

    from data.energon_dataloader import PackedBatchEncoder
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    processor = AutoProcessor.from_pretrained(a.model_dir, max_pixels=a.max_pixels)
    if a.chat_template and a.chat_template != "NULL":
        with open(a.chat_template) as f:
            processor.chat_template = f.read()

    cfg = AutoConfig.from_pretrained(a.model_dir)
    image_token_id = getattr(cfg, "image_token_id", None)
    video_token_id = getattr(cfg, "video_token_id", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", None) if vision_cfg else None

    tokens_per_microbatch = a.tokens_per_microbatch or a.seq_len
    encoder = PackedBatchEncoder(
        processor, a.seq_len, tokens_per_microbatch,
        image_token_id=-1 if image_token_id is None else image_token_id,
        video_token_id=-1 if video_token_id is None else video_token_id,
        spatial_merge_size=spatial_merge_size or 1,
    )

    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=a.seed)
    ds = get_train_dataset(
        a.metadataset, batch_size=1, repeat=False,
        shuffle_buffer_size=a.shuffle_buffer_size,
        max_samples_per_sequence=a.max_samples_per_sequence,
        task_encoder=encoder, worker_config=worker_config,
        packing_buffer_size=a.packing_buffer_size,
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
    print(f"batch_efficiency: mean={sum(batch_effs) / len(batch_effs):.1f}%  "
         f"min={min(batch_effs):.1f}%  max={max(batch_effs):.1f}%")
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
        print(f"throughput: {n / sum(lt):.2f} batches/s = {tokens_seen / elapsed:,.0f} tokens/s "
             f"(single dataloader worker, no GPU compute overlapped -- a lower bound on what "
             f"num_workers>1 with prefetch achieves, not what production throughput will be)")
        # train_qwen.py logs ~0.9-1.4s/step at 256 GPUs (job v8ojrckh); flag if this
        # metadataset's own fetch cost is already in that neighborhood on a single worker.
        step_budget_s = 1.2
        if sum(lt) / n > step_budget_s:
            print(f"NOTE: mean load time exceeds a typical ~{step_budget_s:.1f}s/step compute "
                 f"budget -- with num_workers=2 prefetching, this may still hide behind compute, "
                 f"but it's worth comparing against a coarser metadataset (e.g. --metadataset "
                 f"pointed at a 3-entry FV+OV+Nemotron yaml) to see if dataset *count* is adding "
                 f"per-sample overhead, not just total data volume.")
        else:
            print(f"mean load time is well under a typical ~{step_budget_s:.1f}s/step compute "
                 f"budget -- at this dataset count, fetch cost should fully hide behind GPU "
                 f"compute with num_workers>=1 and not add stalls.")


if __name__ == "__main__":
    main()
