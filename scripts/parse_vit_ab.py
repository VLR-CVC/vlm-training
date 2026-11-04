"""Read the three vit_ab arm logs and report throughput on comparable steps.

    python scripts/parse_vit_ab.py [OUT_DIR]

`tps` alone is noise: a packed row is not always full (`batch_util` ran 57% to
100% in arm A) and a half-full row pushes half the tokens in nearly the same
time. `tflops` does not have that problem -- `batch_flops` bills the whole packed
row including padding, so it stayed at 59.1 against 59.5 across a 57.7%-fill step
and a 98.4%-fill one. It is the stable rate, so it leads here.

Both are still reported on >= MIN_UTIL fill only, and both are still needed:

  tflops/s  how fast the GPU retires modeled work. Comparable across arms as an
            EFFICIENCY figure only -- the two models do different FLOPs per token
            by construction (12 ViT blocks vs 4 extra decoder layers), so a
            higher rate does not by itself mean faster training.
  tps       tokens of training data per second. THIS is the throughput question.
            On full rows every arm moves ~seq_len tokens a step, so it is clean.

Medians, because a recompile spikes one step in twenty.

`mfu` is not reported: peak_tflops_per_gpu is hardcoded to 989.4 (H100 bf16) at
train_qwen.py:313 and these ran on a 44 GiB L40S.
"""

import pathlib
import re
import statistics
import sys

ARMS = ("vlm", "decoder", "text")
WARMUP = 10     # compile lands on step 1; the next few still settle
MIN_UTIL = 95.0

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r" - (\d+) - .*?tps (\d+) .*?tflops ([0-9.]+).*?time ([0-9.]+)s"
    r".*?data_pct ([0-9.]+)%.*?batch_util ([0-9.]+)%"
)


def arm_rows(log):
    text = ANSI.sub("", log.read_text(errors="replace"))
    return [
        dict(step=int(s), tps=int(tps), tflops=float(tf), time=float(t),
             data_pct=float(dp), util=float(bu))
        for s, tps, tf, t, dp, bu in STEP.findall(text)
    ]


def main(out):
    med = {}
    for arm in ARMS:
        log = out / f"{arm}.log"
        if not log.exists():
            print(f"{arm:8s} no log at {log}")
            continue
        rows = arm_rows(log)
        good = [r for r in rows[WARMUP:] if r["util"] >= MIN_UTIL]
        if not good:
            print(f"{arm:8s} {len(rows)} steps, none past warmup at >={MIN_UTIL:.0f}% fill")
            continue
        m = {k: statistics.median(r[k] for r in good)
             for k in ("tps", "time", "tflops", "data_pct", "util")}
        # modeled work in one step: exposes that the two models are not billed
        # the same FLOPs per token, which the rate alone hides
        m["tflop_per_step"] = m["tflops"] * m["time"]
        med[arm] = m
        print(f"{arm:8s} n={len(good):2d}/{len(rows)} tflops={m['tflops']:5.1f} "
              f"tps={m['tps']:7.0f} step={m['time']:.3f}s "
              f"TFLOP/step={m['tflop_per_step']:5.2f} "
              f"data={m['data_pct']:4.2f}% fill={m['util']:5.1f}%")

    def compare(label, x, y):
        print(f"\n{label}")
        for k, unit in (("tps", "tokens/s"), ("tflops", "TFLOP/s"), ("time", "s/step")):
            print(f"  {k:7s} {x[k]:8.2f} -> {y[k]:8.2f} {unit:9s} "
                  f"x{y[k] / x[k]:.3f} ({(y[k] / x[k] - 1) * 100:+.1f}%)")
        print(f"  modeled work per step: {x['tflop_per_step']:.2f} -> "
              f"{y['tflop_per_step']:.2f} TFLOP "
              f"({(y['tflop_per_step'] / x['tflop_per_step'] - 1) * 100:+.1f}%) "
              f"-- not the same model, so tflops/s is efficiency, tps is throughput")

    if "vlm" in med and "decoder" in med:
        compare("ViT removed. Same data, same tokens, same processor:",
                med["vlm"], med["decoder"])
    if "vlm" in med and "text" in med:
        compare("ViT AND image pipeline removed. DIFFERENT data -- read separately:",
                med["vlm"], med["text"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                      else "/data/151-2/users/tockier/profiling/vit_ab"))
