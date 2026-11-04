"""Draw the vision specs `train/precompile.py` warms up, as a standalone HTML page.

    python scripts/scaling/precompile_specs_plot.py [--seq-len N] [--out DIR]

`DEFAULT_SPECS` is `(images, patches_per_image, samples)`: one synthetic
forward+backward each before step 1, so those compiles land off the measured path.
The text-only entries `vlm_specs` appends are excluded -- they have no vision axis
and exist only to cover decoder graph keys the image specs miss.

The plot is the two guarded axes: total patches is the ViT input size, guarded as
`Q_LEN` in `create_block_mask`; the sample count sets `len(cu_seq_q)`, guarded
directly. Marker size is the number of images packed into the row, which is why two
specs can sit at the same patch count and still be different graphs for the ViT.

A spec is dropped by `vlm_specs` (`spec_fits`) and drawn in red for either of two
reasons: its image tokens (`patches / merge_unit`, plus one separator per image)
do not fit `seq_len`, or its total patch count exceeds `MAX_TOTAL_PATCHES`. The
second ceiling is the one that is not about the row: the ViT is replicated across
TP ranks, so a spec's patches are paid unsharded on every GPU.

The decoder-side key `(len(cu_seq_q), max_k)` is printed, not plotted: both halves
are bucketed by `create_varlen_metadata_for_document` -- sample count to a power of
two with a floor of 64, `max_k` to a power of two with a floor of 1024 -- so with
`DOC_BUCKETS` all <= 64 every spec here lands on `len(cu_seq_q) == 65`.

No matplotlib: the chart is hand-written SVG so this runs in the training env.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.precompile import DEFAULT_SPECS, _graph_key, spec_fits

FITS, DROPPED = "#0D47A1", "#A31515"
INK, GRID = "#1b1b1b", "#9e9e9e"
LEGEND_IMAGES = (1, 2, 4)
W, H, PAD = 660, 420, 60


def radius(images: int) -> float:
    """Marker area proportional to the image count, so 4 images reads as 4x of 1."""
    return 6.0 * math.sqrt(images)


def rows_for(seq_len: int, unit: int):
    out = []
    for images, patches, samples in DEFAULT_SPECS:
        total = images * patches
        tokens = total // unit + images  # one separator token per image
        fits = spec_fits(images, patches, seq_len=seq_len, spatial_merge_unit=unit)
        out.append({
            "images": images, "patches": patches, "samples": samples, "total": total,
            "tokens": tokens, "fits": fits,
            "key": _graph_key(samples, seq_len, None) if fits else None,
            "label": f"{images}x{patches:,}",
        })
    return out


def legend(x: float, y: float) -> str:
    """Boxed marker-size key, so the sample markers are not read as data points."""
    step, pad, h = 62, 16, 76
    w = step * (len(LEGEND_IMAGES) - 1) + 2 * pad + 24
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" '
        f'stroke="{INK}" stroke-width="1.1"/>',
        f'<text x="{x + pad}" y="{y + 19}" class="legend-title">'
        f"Legend &mdash; images in row</text>",
    ]
    for i, images in enumerate(LEGEND_IMAGES):
        cx = x + pad + 12 + i * step
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{y + 44}" r="{radius(images):.1f}" '
            f'fill="{FITS}" fill-opacity="0.3" stroke="{FITS}" stroke-width="1.6"/>'
            f'<text x="{cx:.1f}" y="{y + 68}" class="legend mid">{images}</text>'
        )
    return "".join(parts)


def scatter(rows) -> str:
    """Total patches (log x) against samples (log y), marker area by image count."""
    xs = [r["total"] for r in rows]
    ys = [r["samples"] for r in rows]
    x0, x1 = math.log2(min(xs)) - 0.35, math.log2(max(xs)) + 0.35
    y0, y1 = -0.45, math.log2(max(ys)) + 0.75

    def px(v):
        return PAD + (math.log2(v) - x0) / (x1 - x0) * (W - PAD - 26)

    def py(v):
        return H - PAD - (math.log2(v) - y0) / (y1 - y0) * (H - PAD - 30)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for t in (2048, 4096, 8192, 16384, 32768, 65536):
        if x0 <= math.log2(t) <= x1:
            parts.append(
                f'<line x1="{px(t):.1f}" y1="{H - PAD}" x2="{px(t):.1f}" y2="22" '
                f'stroke="{GRID}" stroke-width="0.9"/>'
                f'<text x="{px(t):.1f}" y="{H - PAD + 17}" class="tick mid">'
                f"{t // 1024}k</text>"
            )
    for d in (1, 2, 4, 8, 16, 32, 64):
        if d <= max(ys):
            parts.append(
                f'<line x1="{PAD}" y1="{py(d):.1f}" x2="{W - 26}" y2="{py(d):.1f}" '
                f'stroke="{GRID}" stroke-width="0.9"/>'
                f'<text x="{PAD - 9}" y="{py(d) + 4:.1f}" class="tick end">{d}</text>'
            )
    for r in rows:
        c = FITS if r["fits"] else DROPPED
        x, y, rad = px(r["total"]), py(r["samples"]), radius(r["images"])
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" '
            f'fill="{c}" fill-opacity="0.3" stroke="{c}" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y - rad - 7:.1f}" class="mark mid" fill="{c}">'
            f'{r["images"]} img &times; {r["patches"]:,}</text>'
        )

    # the top-left corner is empty -- few patches, many samples
    parts.append(legend(PAD + 18, 30))

    parts.append(
        f'<text x="{W / 2:.0f}" y="{H - 12}" class="axis mid">'
        f"total patches (images &times; patches per image)</text>"
        f'<text x="15" y="{H / 2:.0f}" class="axis mid" '
        f'transform="rotate(-90 15 {H / 2:.0f})">samples in the row</text></svg>'
    )
    return "".join(parts)


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:28px 16px 48px;background:#fff;color:#111;
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif}
main{max-width:720px;margin:0}
h1{font-size:20px;margin:0 0 4px}
p.sub{color:#2e2e2e;margin:0 0 20px;font-size:13.5px}
section{background:#fff;border:1px solid #9e9e9e;border-radius:10px;padding:16px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#111;
margin:0 0 10px}
svg.chart{width:100%;height:auto;overflow:visible}
.tick,.mark,.legend{fill:#1b1b1b;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}
.mark{font-size:10.5px}
.legend-title{fill:#111;font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace}
.axis{fill:#111;font-size:12px;font-weight:600}
.mid{text-anchor:middle}.end{text-anchor:end}
.cap{color:#2e2e2e;font-size:11.5px;margin:12px 0 0}
.cap b{color:#0D47A1;font-weight:600}.cap i{color:#A31515;font-style:normal;font-weight:600}
"""


def page(rows, seq_len: int, unit: int) -> str:
    from train.precompile import MAX_TOTAL_PATCHES as max_patches

    fitting = sum(r["fits"] for r in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Precompile Vision Specs</title><style>{CSS}</style></head>
<body><main>
<h1>AOT precompile: vision specs</h1>
<p class="sub">{len(rows)} predefined specs in <code>DEFAULT_SPECS</code>,
{fitting} fitting seq_len {seq_len:,} (spatial_merge_size {int(unit ** 0.5)}).
Text-only specs excluded.</p>
<section><h2>patches &times; samples</h2>{scatter(rows)}
<p class="cap">Marker <b>area</b> is the number of images packed into the row, so
the 4-image marker is four times the area of a 1-image one; label reads
<code>images &times; patches per image</code>. Both axes log<sub>2</sub>.
<b>Blue</b> is warmed; <i>red</i> is dropped by <code>vlm_specs</code> --
either its image tokens exceed seq_len {seq_len:,} or its total patches exceed
<code>MAX_TOTAL_PATCHES</code> ({max_patches:,}).</p></section>
</main></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=6144)
    p.add_argument("--merge-size", type=int, default=2, help="spatial_merge_size")
    p.add_argument("--out", default="/data/151-2/users/tockier/figures/precompile")
    args = p.parse_args()

    unit = args.merge_size**2
    rows = rows_for(args.seq_len, unit)
    for r in rows:
        key = f'cu_seq_q={r["key"][0]} max_k={r["key"][1]}' if r["key"] else "dropped"
        print(
            f'  {r["images"]:>2} img x {r["patches"]:>6,} patches = {r["total"]:>7,}  '
            f'{r["tokens"]:>6,} tokens ({r["tokens"] / args.seq_len:5.0%})  '
            f'{r["samples"]:>3} samples  {key}'
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"precompile_specs_seq{args.seq_len}.html"
    path.write_text(page(rows, args.seq_len, unit))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
