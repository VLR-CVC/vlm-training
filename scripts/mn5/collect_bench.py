"""Assemble the Qwen3-VL-2B sweep table from logs_bench/*.out.

Median of the last 20 of 40 steps. Early steps are compilation, and with
`compile_vision` in the sweep the compile cost is itself an arm-dependent
variable -- a mean over the whole run would measure compiling, not throughput.

`visual_ms` is reported next to step time because at 2B the vision tower is 47%
of forward, larger than the whole language model, so it is usually the number
that explains a difference between two arms.

One caveat carried from PERFORMANCE.md: `flops_per_token` is computed once from
config and seq_len, and `vision_flops` sizes the tower from seq_len rather than
the actual image count. MFU is therefore not comparable across arms that change
seq_len. Step time is.
"""
import glob, os, re, statistics, sys

LOGS = sys.argv[1] if len(sys.argv) > 1 else "logs_bench"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW = re.compile(
    r"loss (\S+).*?mfu ([\d.]+)%.*?tflops ([\d.]+).*?time ([\d.]+)s"
    r".*?fwd ([\d.]+)s mem ([\d.]+)/([\d.]+)G.*?nsamples (\d+).*?batch_util ([\d.]+)%")
SEC = re.compile(r"visual ([\d.]+)ms.*?(?:layers ([\d.]+)ms)?")
VISUAL = re.compile(r"visual ([\d.]+)ms")
LAYERS = re.compile(r"layers ([\d.]+)ms")
NAME = re.compile(r"(\d+)_(\w+)_(\d+)n\.out$")
ORDER = ["base", "vis_static", "vis_off", "seq4096", "seq8192", "seq12288",
         "opt_torchao_bf16", "opt_torchao_fp32", "opt_foreach_fp32",
         "fsdp", "nocompile", "ac05", "head_autotune"]

rows = []
for f in sorted(glob.glob(f"{LOGS}/*.out")):
    m = NAME.match(os.path.basename(f))
    if not m:
        continue
    job, arm, nodes = m.groups()
    body = ANSI.sub("", open(f, errors="ignore").read())
    steps = ROW.findall(body)
    vis = [float(x) for x in VISUAL.findall(body)]
    lay = [float(x) for x in LAYERS.findall(body)]
    fail = [w for w in ("Traceback", "out of memory", "OutOfMemoryError",
                        "srun: error", "CANCELLED", "unbound variable") if w in body]
    r = dict(arm=arm, nodes=int(nodes), job=job, n=len(steps), fail=",".join(fail))
    if len(steps) >= 5:
        t = steps[-20:]
        r.update(step=statistics.median(float(s[3]) for s in t),
                 mfu=statistics.median(float(s[1]) for s in t),
                 mem=max(float(s[5]) for s in t),
                 cap=float(t[-1][6]),
                 util=statistics.median(float(s[8]) for s in t),
                 nsamp=statistics.median(int(s[7]) for s in t),
                 visual=statistics.median(vis[-20:]) if vis else None,
                 layers=statistics.median(lay[-20:]) if lay else None,
                 nan=sum(1 for s in steps if s[0] == "nan"))
    rows.append(r)

rows.sort(key=lambda r: ORDER.index(r["arm"]) if r["arm"] in ORDER else 99)
hdr = (f"{'arm':<18s}{'job':>10s}{'step_s':>8s}{'visual':>8s}{'layers':>8s}"
       f"{'MFU%':>6s}{'mem_G':>12s}{'util%':>7s}{'smp':>5s}{'n':>4s}  vs base")
print(hdr); print("-" * len(hdr))
base = next((r.get("step") for r in rows if r["arm"] == "base" and "step" in r), None)
for r in rows:
    if "step" not in r:
        print(f"{r['arm']:<18s}{r['job']:>10s}{'--':>8s}{'--':>8s}{'--':>8s}{'--':>6s}"
              f"{'--':>12s}{'--':>7s}{'--':>5s}{r['n']:4d}  {r['fail'] or 'too few steps'}")
        continue
    rel = f"{100*(r['step']/base - 1):+5.1f}%" if base and r["arm"] != "base" else "  --  "
    note = rel + (f"  NAN x{r['nan']}" if r["nan"] else "") + (f"  [{r['fail']}]" if r["fail"] else "")
    print(f"{r['arm']:<18s}{r['job']:>10s}{r['step']:8.3f}"
          f"{r['visual'] or 0:8.1f}{r['layers'] or 0:8.1f}{r['mfu']:6.1f}"
          f"{r['mem']:6.1f}/{r['cap']:5.1f}{r['util']:7.1f}{r['nsamp']:5.0f}{r['n']:4d}  {note}")

print("\nMFU is not comparable across seq_len arms (flops_per_token is derived "
      "from config + seq_len). Compare step_s.")
