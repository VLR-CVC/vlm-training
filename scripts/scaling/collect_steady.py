"""Post-warmup throughput per sweep point (TORCHTITAN_BENCHMARK.md §8 phase 4).

    python scripts/scaling/collect_steady.py logs_scaling/<sweep> [more sweeps...]

**The reported number is the mean over every step after the warmup phase ends.**
Not the mean over full-rate steps only: recompile stalls never stop entirely, and a
figure that drops them reports hardware the run does not have. Excluding warmup is
justified -- it is a fixed startup cost that amortises over a real training run --
but excluding the residual stalls is not, so they stay in.

Warmup ends at the last step where the stall rate in a rolling window is still above
`SETTLED`; after that point the run never returns to that rate. It is long and it
grows with world size: measured 2026-09-19 on Qwen3-VL-2B, ~400 steps at 32 nodes.
A 40-step benchmark therefore measures warmup and nothing else -- at 32 nodes the
same config read 3,704 tok/s/GPU over 40 steps and 8,373 over 500.

A step is stalled when it takes more than `FULL_RATE` x the run's fastest step.
Anchoring on the fastest step keeps that honest when most steps stall; the previous
`t < 3 * median` test in `collect_strong.sh` silently passed everything at 32 nodes,
where the median itself is a stalled step. Validated against job 1890083, where
per-rank TORCH_LOGS give ground truth: 27/36 stalled, 5.70s vs 1.33s.

`full_rate` and `stall%` are kept beside the reported mean as diagnostics: the gap
between them says how much is still being lost, rather than hiding it.
"""

import re
import statistics
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The `(N/gpu)` part is absent in logs written before commit c39d1a4. There `tps`
# is one dp replica's micro-batch over the step time, so per-GPU is `tps / tp`.
STEP = re.compile(
    r" - (\d+) - loss .*?tps ([\d.]+)(?: \((\d+)/gpu\))?.*?mfu ([\d.]+)% tflops ([\d.]+).*?"
    r"time ([\d.]+)s.*?mem [\d.]+/([\d.]+)G.*?nsamples (\d+)"
)
ACCUM = re.compile(r"tpi_multiplier=([\d.]+)")
TOKENS = re.compile(r"tokens_per_microbatch=(\d+)")
SEQ = re.compile(r"seq_len=(\d+)")
# Old logs print `mesh: DeviceMesh((dp=4, tp=4), ...)` with no replicate/shard split.
MESH = re.compile(r"dp=(\d+)(?: \(replicate=(\d+) shard=(-?\d+)\))?,? tp=(\d+)")

FULL_RATE = 1.5   # a step above this x the fastest step is stalled
WINDOW = 50       # rolling window for the stall rate, in steps
SETTLED = 0.10    # warm once the windowed stall rate stays below this


def warmup_end(stalled, window: int = WINDOW, settled: float = SETTLED) -> int | None:
    """Index of the first step after which the rolling stall rate never again
    exceeds `settled`. None when the run never settles within its length.

    Plain Python on purpose: this has to run on a login node, which has no numpy.
    Takes any sequence of bools, including a numpy array (steptime_plots imports it).
    """
    n = len(stalled)
    if n < window:
        return None
    hits = sum(1 for x in stalled[:window] if x)
    last_over = 0 if hits / window > settled else -1
    for i in range(1, n - window + 1):
        hits += int(bool(stalled[i + window - 1])) - int(bool(stalled[i - 1]))
        if hits / window > settled:
            last_over = i
    if last_over < 0:
        return 0
    end = last_over + window
    return end if end < n else None


def point(log: Path) -> dict | None:
    text = ANSI.sub("", log.read_text(errors="ignore"))
    mesh = MESH.search(text)
    tp = int(mesh.group(4)) if mesh else 1
    steps = [
        dict(step=int(g[0]), tps_gpu=int(g[2]) if g[2] else round(float(g[1]) / tp),
             mfu=float(g[3]), tflops=float(g[4]),
             time=float(g[5]), peak=float(g[6]), nsamples=int(g[7]))
        for g in (m.groups() for m in map(STEP.search, text.splitlines()) if m)
    ]
    if not steps:
        return None
    seq = int(m.group(1)) if (m := SEQ.search(text)) else 0
    tpm = int(m.group(1)) if (m := TOKENS.search(text)) else 0
    micro_tokens = tpm or seq
    accum = float(m.group(1)) if (m := ACCUM.search(text)) else 1.0
    dp = int(mesh.group(1)) if mesh else 0

    times = [s["time"] for s in steps]
    rates = [s["tps_gpu"] for s in steps]
    floor = min(times)
    stalled = [t > FULL_RATE * floor for t in times]
    end = warmup_end(stalled)

    tail = slice(end, None) if end is not None else slice(len(steps), None)
    full = [not x for x in stalled]
    pick = lambda xs, m: [x for x, keep in zip(xs, m) if keep]  # noqa: E731
    return dict(
        accum=accum, seq=seq, micro_tokens=micro_tokens,
        rows=micro_tokens // seq if seq else 0,
        dp=dp,
        ddp=int(mesh.group(2)) if mesh and mesh.group(2) else 0,
        fsdp=int(mesh.group(3)) if mesh and mesh.group(3) else 0,
        tp=tp,
        gbs_tokens=int(micro_tokens * dp * accum),
        gbs_samples=(statistics.fmean([s["nsamples"] for s in steps][slice(None)])
                     if steps else 0),
        n=len(steps),
        warmup=steps[end]["step"] if end is not None else None,
        # the reported figure: every step after warmup, residual stalls included
        reported=statistics.fmean(rates[tail]) if end is not None else float("nan"),
        reported_sd=(statistics.pstdev(rates[tail])
                     if end is not None and len(rates[tail]) > 1 else 0.0),
        tail_stall=(100 * statistics.fmean([float(x) for x in stalled[tail]])
                    if end is not None else float("nan")),
        full_rate=statistics.fmean(pick(rates, full)) if any(full) else float("nan"),
        step_full=statistics.fmean(pick(times, full)) if any(full) else float("nan"),
        step_tail=statistics.fmean(times[tail]) if end is not None else float("nan"),
        all_steps=statistics.fmean(rates),
        mfu=(statistics.fmean([s["mfu"] for s in steps][tail])
             if end is not None else float("nan")),
        tflops=(statistics.fmean([s["tflops"] for s in steps][tail])
                if end is not None else float("nan")),
        samples_tail=(statistics.fmean([s["nsamples"] for s in steps][tail])
                      if end is not None else float("nan")),
        peak=max(s["peak"] for s in steps),
    )


MD_HEADER = (
    "| Nodes | GPUs | GBS (samples) | GBS (tokens) | grad_acc | seq_len | MBS (rows) "
    "| DDP | FSDP | TP | tok/s/GPU | TFLOP/s/GPU | MFU | Description |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def markdown(dirs: list[str], desc: dict[str, str]) -> None:
    print(MD_HEADER)
    for d in dirs:
        root = Path(d)
        for p in sorted(root.glob("*n"), key=lambda q: int(q.name[:-1])):
            log = p / "log.out"
            if not log.exists() or (r := point(log)) is None:
                continue
            nodes = int(p.name[:-1])
            warm = r["warmup"] is not None
            tps = f"{r['reported']:,.0f}" if warm else "never warm"
            mfu = f"{r['mfu']:.1f}%" if warm else "--"
            tfl = f"{r['tflops']:.1f}" if warm else "--"
            samples = f"{r['samples_tail']:,.0f}" if warm else "--"
            print(f"| {nodes} | {nodes * 4} | {samples} | {r['gbs_tokens']:,} "
                  f"| {r['accum']:.0f} | {r['seq']:,} | {r['rows']} | {r['ddp']} | {r['fsdp']} "
                  f"| {r['tp']} | {tps} | {tfl} | {mfu} | {desc.get(root.name, root.name)} |")


def main(dirs: list[str]) -> None:
    for d in dirs:
        root = Path(d)
        pts = sorted(root.glob("*n"), key=lambda p: int(p.name[:-1]))
        if not pts:
            continue
        print(f"\n=== {root.name}   (reported = mean over steps after warmup)")
        print(f"{'nodes':>5} {'GPUs':>5} {'accum':>5} {'steps':>5} {'warmup':>7} "
              f"{'REPORTED/GPU':>18} {'stall%':>7} {'full_rate':>10} {'step_s':>7} "
              f"{'naive all':>10} {'peak':>7}")
        for p in pts:
            log = p / "log.out"
            if not log.exists() or (r := point(log)) is None:
                continue
            nodes = int(p.name[:-1])
            warm = r["warmup"] if r["warmup"] is not None else "never"
            rep = f"{r['reported']:,.0f} (sd {r['reported_sd']:,.0f})" if r["warmup"] else "-"
            print(f"{nodes:>5} {nodes * 4:>5} {r['accum']:>5.0f} {r['n']:>5} {str(warm):>7} "
                  f"{rep:>18} {r['tail_stall']:>6.1f}% {r['full_rate']:>10,.0f} "
                  f"{r['step_tail']:>7.2f} {r['all_steps']:>10,.0f} {r['peak']:>6.1f}G")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--markdown":
        # `--markdown <sweep>[=description] ...` emits the BENCHMARKS.md table
        pairs = [a.split("=", 1) for a in args[1:]]
        markdown([a[0] for a in pairs],
                 {Path(a[0]).name: (a[1] if len(a) > 1 else "") for a in pairs})
    else:
        main(args)
