"""How clustered is an *optimal* QSA selection, on real weights and real data?

The benchmark's QSA arms ran with `random_init = true`, so the frozen indexer
held random weights and selected key blocks independently per query. That is the
worst case for a block-sparse kernel: the union over a 128-query tile covers
98.4% of causal tiles, so flex_attention runs dense and pays mask evaluation on
top.

The fix is not to guess at a better pattern. QSA's indexer is trained by
distillation against the dense attention distribution -- approximating dense
attention is its whole objective -- so *top-k over the real dense attention* is
the best any indexer for this model could produce. It bounds QSA from above.

This loads Qwen3.5-9B with its real weights, runs real packed batches, and for
every full-attention layer computes:

  oracle   top (budget/ratio) blocks by true summed attention mass
  random   the same count drawn independently per query   (what the ladder ran)
  local    the same count immediately preceding the query (pure locality prior)

and reports, per 128-query tile, how many key tiles the kernel would have to
visit. That union -- not per-query density -- is what decides the kernel cost.

    python oracle_mask.py <config.toml> [batches]
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

REPO = "/e/project1/open-sci-mm/ockier1/vlm-training"
sys.path.insert(0, REPO)

from megatron.energon import get_train_dataset, get_loader, WorkerConfig
from transformers import AutoProcessor, AutoTokenizer

from train.config import Config
from train.config_manager import ConfigManager
from train.utils import round_max_seqlen
from data.task_encoder_factory import build_task_encoder
from models.qwen3_5 import model as m5

TILE = 128          # flex_attention BlockMask tile width (_FLEX_BLOCK_SIZE)
BUDGET = 512        # indexer_budget
RATIO = 4           # indexer_compress_ratio
KEEP = BUDGET // RATIO

STATS: list[dict] = []


def tile_union(sel_blocks: torch.Tensor, q_lo: int, T: int) -> torch.Tensor:
    """sel_blocks (Q, KEEP) block ids -> bool (n_qtile, n_ktile) visited."""
    ntile = (T + TILE - 1) // TILE
    ktile = (sel_blocks * RATIO) // TILE
    Q = sel_blocks.shape[0]
    qtile = (torch.arange(Q, device=sel_blocks.device) + q_lo) // TILE
    grid = torch.zeros(ntile, ntile, dtype=torch.bool, device=sel_blocks.device)
    grid[qtile.unsqueeze(1).expand_as(ktile).reshape(-1), ktile.reshape(-1)] = True
    return grid


@torch.no_grad()
def capture(q, k, v, cu_seqlens, max_seqlen):
    """q,k,v are (T, H, D) post-rope. Recompute true attention to rank blocks."""
    T, H, D = q.shape
    HKV = k.shape[1]
    kk = k.repeat_interleave(H // HKV, dim=1) if HKV != H else k
    nblk = (T + RATIO - 1) // RATIO
    ntile = (T + TILE - 1) // TILE
    seg = torch.zeros(T, dtype=torch.long, device=q.device)
    seg[cu_seqlens[1:-1].long()] = 1
    seg = seg.cumsum(0)

    acc = {n: torch.zeros(ntile, ntile, dtype=torch.bool, device=q.device)
           for n in ("oracle", "random", "local")}
    pos = torch.arange(T, device=q.device)

    CH = 512
    for lo in range(0, T, CH):
        hi = min(lo + CH, T)
        qc = q[lo:hi].transpose(0, 1).float()                 # (H, c, D)
        sc = torch.matmul(qc, kk.transpose(0, 1).float().transpose(-1, -2))
        sc *= D ** -0.5
        bad = (seg[lo:hi].unsqueeze(1) != seg.unsqueeze(0)) | (pos.unsqueeze(0) > pos[lo:hi].unsqueeze(1))
        sc.masked_fill_(bad.unsqueeze(0), float("-inf"))
        p = sc.softmax(-1).sum(0)                              # (c, T) mass over heads
        p = torch.nan_to_num(p, 0.0)
        # pool token mass into blocks of RATIO, then take the top KEEP
        pad = nblk * RATIO - T
        pb = F.pad(p, (0, pad)).view(hi - lo, nblk, RATIO).sum(-1)
        n_avail = (pos[lo:hi] // RATIO + 1).clamp(min=1)
        kk_ = min(KEEP, nblk)
        oracle = pb.topk(kk_, dim=-1).indices
        rnd = (torch.rand(hi - lo, kk_, device=q.device) * n_avail.unsqueeze(1)).long()
        loc = ((pos[lo:hi] // RATIO).unsqueeze(1)
               - torch.arange(kk_, device=q.device).unsqueeze(0)).clamp(min=0)
        for n, s in (("oracle", oracle), ("random", rnd), ("local", loc)):
            acc[n] |= tile_union(s, lo, T)
        del sc, p, pb, bad

    causal = torch.tril(torch.ones(ntile, ntile, dtype=torch.bool, device=q.device))
    vis = causal.sum().item()
    STATS.append({n: (acc[n] & causal).sum().item() / vis for n in acc} | {"T": T, "ntile": ntile})


def main() -> None:
    cfg_path = sys.argv[1]
    nbatch = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfg = ConfigManager(Config).parse_args(["--config", cfg_path])

    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    model, _ = m5.Qwen3_5ForCausalLM.from_pretrained(cfg.training.model_dir)
    model = model.to("cuda", torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(cfg.training.model_dir)
    proc = AutoProcessor.from_pretrained(cfg.training.model_dir, max_pixels=1048576)
    enc, extra = build_task_encoder(cfg.data, tokenizer=tok, processor=proc)
    ds = get_train_dataset(
        cfg.data.data_path, batch_size=1, repeat=True,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        max_samples_per_sequence=cfg.data.max_samples_per_sequence,
        task_encoder=enc,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=1),
        **extra,
    )
    loader = iter(get_loader(ds))

    orig = m5.SelfAttention._run_varlen_attn

    @staticmethod
    def hooked(q, k, v, cu_seqlens, max_seqlen):
        capture(q, k, v, cu_seqlens, max_seqlen)
        return orig(q, k, v, cu_seqlens, max_seqlen)

    m5.SelfAttention._run_varlen_attn = hooked

    for i in range(nbatch):
        b = next(loader)
        b = b if isinstance(b, dict) else dict(vars(b))
        # Same normalization the trainer does before the model call
        # (train/train_qwen.py:535-556) -- the loader adds a leading batch dim
        # that the varlen contract does not expect.
        if b["cu_seqlens"].ndim > 1:
            b["cu_seqlens"] = b["cu_seqlens"].squeeze()
        grid = b.get("image_grid_thw")
        if grid is not None and grid.ndim > 2:
            grid = b["image_grid_thw"] = grid[0]
        cu = b["cu_seqlens"]
        kw = {"max_seqlen": round_max_seqlen(int((cu[1:] - cu[:-1]).max()))}
        if grid is not None and grid.numel():
            vis = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0])
            kw["vision_max_seqlen"] = round_max_seqlen(int(vis.max()))
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            v = b.get(k)
            if v is not None:
                if v.ndim > 2 and k == "pixel_values":
                    v = v[0]
                if k == "mm_token_type_ids" and v.ndim > 1:
                    v = v.squeeze(0)
                kw[k] = v.cuda()
        ids = b["input_ids"].view(1, -1).cuda()
        with torch.no_grad():
            model(input_ids=ids, attention_mask=cu.cuda().view(-1), **kw)
        torch.cuda.empty_cache()
        print(f"batch {i}: T={ids.shape[1]} docs={cu.numel()-1} "
              f"layers captured={len(STATS)}", flush=True)

    print(f"\n{'':12s}{'oracle':>10s}{'local':>10s}{'random':>10s}   (fraction of causal tiles visited)")
    for n in ("oracle", "local", "random"):
        vals = [s[n] for s in STATS]
        if n == "oracle":
            print(f"{'per-layer':12s}", end="")
            print("  ".join(f"{v*100:5.1f}" for v in vals[:12]))
    import statistics
    print(f"{'mean':12s}" + "".join(
        f"{100*statistics.mean(s[n] for s in STATS):9.1f}%" for n in ("oracle", "local", "random")))
    print(f"{'T':12s}{STATS[0]['T']}   tiles {STATS[0]['ntile']}   layers {len(STATS)}")


if __name__ == "__main__":
    main()
