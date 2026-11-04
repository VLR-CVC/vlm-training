#!/bin/bash
# Strong-scaling table: the global batch is fixed, so the number that matters is the
# wall time of one OPTIMIZER step. The trainer logs `time` per micro-batch, so that is
# multiplied by the accumulation depth (`tpi_multiplier`, read from the run's config
# line). Speedup and efficiency are against the smallest node count in the sweep.
#   scripts/scaling/collect_strong.sh logs_scaling/strong_20260918_101500
set -euo pipefail
dir="${1:?usage: collect_strong.sh <sweep dir>}"
python3 - "$dir" <<'PY'
import re, statistics, sys
from pathlib import Path
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r" - (\d+) - loss .*?tps ([\d.]+).*?mfu ([\d.]+)%.*?time ([\d.]+)s.*?mem [\d.]+/([\d.]+)G")
ACCUM = re.compile(r"tpi_multiplier=([\d.]+)")
TOKENS = re.compile(r"tokens_per_microbatch=(\d+)")
rows = []
for d in sorted(Path(sys.argv[1]).glob("*n"), key=lambda p: int(p.name[:-1])):
    log = d / "log.out"
    if not log.exists():
        continue
    text = ANSI.sub("", log.read_text(errors="ignore"))
    accum = float(ACCUM.search(text).group(1)) if ACCUM.search(text) else 1.0
    mbs = int(TOKENS.search(text).group(1)) if TOKENS.search(text) else 0
    steps = [(int(s), float(t), float(m), float(p))
             for s, _, m, t, p in (x.groups() for x in map(STEP.search, text.splitlines()) if x)]
    steps = [s for s in steps if s[0] > 10]          # warmup and the first compiles
    if not steps:
        continue
    times = sorted(t for _, t, _, _ in steps)
    fast = [t for t in times if t < 3 * times[len(times) // 2]]   # drop recompile stalls
    nodes, gpus = int(d.name[:-1]), int(d.name[:-1]) * 4
    rows.append(dict(nodes=nodes, gpus=gpus, accum=accum, mbs=mbs,
                     step=statistics.median(fast) * accum,
                     tps=mbs / statistics.median(fast),
                     mfu=statistics.median(m for _, _, m, _ in steps),
                     stalls=len(times) - len(fast), total=len(times),
                     peak=max(p for *_, p in steps)))
base = rows[0]
gbs = int(base["mbs"] * base["accum"] * base["gpus"])
print(f"global batch {gbs} tokens, micro-batch {base['mbs']} per GPU\n")
print(f"{'nodes':>5} {'GPUs':>5} {'accum':>6} {'step s':>8} {'speedup':>8} {'eff':>7} "
      f"{'tok/s/GPU':>10} {'mfu%':>6} {'stalls':>8} {'peak GiB':>9}")
for r in rows:
    speed = base["step"] / r["step"]
    ideal = r["nodes"] / base["nodes"]
    print(f"{r['nodes']:5d} {r['gpus']:5d} {r['accum']:6.0f} {r['step']:8.2f} {speed:7.2f}x "
          f"{speed / ideal * 100:6.1f}% {r['tps']:10.0f} {r['mfu']:6.1f} "
          f"{r['stalls']:4d}/{r['total']:<3d} {r['peak']:9.1f}")
PY
