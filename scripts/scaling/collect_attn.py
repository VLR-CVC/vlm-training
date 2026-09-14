"""Assemble the attention-benchmark table from logs_scaling/attnbench/*.out.

Median of the last 20 steps, not the mean: early steps are compilation, and the
QSA arms recompile `create_block_mask` every time the block grid widens, so a
mean over the whole run measures compiling rather than throughput.

`layers_ms` is the number this benchmark exists to move -- it is the decoder
stack, where the attention implementation lives. Step time also carries the
vision tower and the optimizer, which are identical across arms.
"""
import glob, os, re, statistics, sys

LOGS = sys.argv[1] if len(sys.argv) > 1 else \
    "/e/project1/open-sci-mm/ockier1/vlm-training/logs_scaling/attnbench"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW = re.compile(
    r"loss (\S+).*?mfu ([\d.]+)%.*?tflops ([\d.]+).*?time ([\d.]+)s"
    r".*?fwd ([\d.]+)s mem ([\d.]+)/([\d.]+)G")
LAYERS = re.compile(r"layers ([\d.]+)ms")
NAME = re.compile(r"(\d+)_(\w+?)_(clevr|plotqa)_(\d+)_(\d+)n\.out$")
ORDER = ["hybrid", "fullattn", "qsa512", "allqsa512"]

rows = []
for f in sorted(glob.glob(f"{LOGS}/*.out")):
    m = NAME.match(os.path.basename(f))
    if not m:
        continue
    job, variant, ds, sl, nodes = m.groups()
    body = ANSI.sub("", open(f, errors="ignore").read())
    steps = ROW.findall(body)
    layers = [float(x) for x in LAYERS.findall(body)]
    fail = [w for w in ("Traceback", "out of memory", "OutOfMemoryError",
                        "flex-attention unavailable", "srun: error", "CANCELLED")
            if w in body]
    nan = sum(1 for s in steps if s[0] == "nan")
    rec = dict(nodes=int(nodes), variant=variant, ds=ds, sl=int(sl), job=job,
               n=len(steps), fail=",".join(fail), nan=nan)
    if len(steps) >= 5:
        tail = steps[-20:]
        rec.update(step=statistics.median(float(s[3]) for s in tail),
                   mfu=statistics.median(float(s[1]) for s in tail),
                   tfs=statistics.median(float(s[2]) for s in tail),
                   mem=max(float(s[5]) for s in tail),
                   cap=float(tail[-1][6]),
                   layers=statistics.median(layers[-20:]) if layers else None)
    rows.append(rec)

rows.sort(key=lambda r: (r["ds"], r["sl"], r["nodes"],
                         ORDER.index(r["variant"]) if r["variant"] in ORDER else 9))
hdr = (f"{'nodes':>5s} {'variant':<10s} {'data':<7s} {'seq':>6s} {'job':>8s} "
       f"{'step_s':>7s} {'layer_ms':>8s} {'MFU%':>5s} {'TF/s':>6s} {'mem_G':>11s} {'n':>4s}  note")
print(hdr); print("-" * len(hdr))
base = {}
for r in rows:
    if "step" not in r:
        print(f"{r['nodes']:5d} {r['variant']:<10s} {r['ds']:<7s} {r['sl']:6d} {r['job']:>8s} "
              f"{'--':>7s} {'--':>8s} {'--':>5s} {'--':>6s} {'--':>11s} {r['n']:4d}  "
              f"{r['fail'] or 'too few steps'}")
        continue
    k = (r["variant"], r["ds"], r["sl"])
    base.setdefault(k, (r["nodes"], r["step"]))
    b_n, b_st = base[k]
    note = f"eff {100*b_st/r['step']:5.1f}% vs {b_n}n"
    if r["nan"]:
        note += f"  NAN x{r['nan']}"
    if r["fail"]:
        note += f"  [{r['fail']}]"
    print(f"{r['nodes']:5d} {r['variant']:<10s} {r['ds']:<7s} {r['sl']:6d} {r['job']:>8s} "
          f"{r['step']:7.3f} {r['layers'] or 0:8.1f} {r['mfu']:5.1f} {r['tfs']:6.1f} "
          f"{r['mem']:5.1f}/{r['cap']:5.1f} {r['n']:4d}  {note}")
