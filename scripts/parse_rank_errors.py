#!/usr/bin/env python3
"""What failed in a torchrun log directory, and on which rank.

    scripts/parse_rank_errors.py slurm_errors/46691090
    scripts/parse_rank_errors.py slurm_errors/46691090 --per-rank --full

`torchrun --log-dir X --redirects 2` writes one `X/<rdzv_id>_<hash>/attempt_0/
<local_rank>/stderr.log` per worker. The directory names carry the *local* rank
only -- the agent picks the hash at random and never writes the host or the
global rank into the tree. The global rank is in the log text instead: torch's
elastic error handler prefixes every line of the traceback that killed a worker
with `[rank111]:`. That prefix is the only place the two numbers meet, so it is
what this reads.

Tracebacks *without* the prefix are not the same thing: they are exceptions that
something caught and logged (energon logs a whole traceback per rejected sample
and keeps going), so they carry no rank and usually vastly outnumber the fatal
ones -- 180 of 256 files in job 46691090, against 5 rank-tagged. They are
reported separately because a flood of them is a data problem worth seeing even
though the run survived it. Either way every group prints a `path:line` of the
traceback itself, to open with `less +N` or an editor.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

RANK_TAG = re.compile(r"^\[rank(\d+)\]:\s?(.*)$")
FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
# The header closing a traceback: unindented, and the last dotted component
# names an exception. Anything looser matches the source lines echoed inside the
# traceback body, which are unindented too once the rank prefix is stripped.
EXC = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Failure|Timeout|Killed))"
    r"(?:: (.*))?$"
)


def numbered(text):
    """`(lineno, text)` for every logical line, numbered as the file is.

    Progress bars rewrite with a bare `\\r` and glue whatever comes next onto
    their own line, so a traceback can start mid-line and has to be split out --
    but the number stays that of the real line, or `less +N` and an editor jump
    land somewhere else.
    """
    return [
        (n, part)
        for n, line in enumerate(text.split("\n"), 1)
        for part in line.split("\r")
    ]


def blocks(lines):
    """Split a worker log into `(rank, [line, ...], first_lineno)` blocks.

    A rank-tagged block is every consecutive rank-prefixed line; an untagged one
    runs from `Traceback (most recent call last):` to the exception header that
    closes it.
    """
    out, cur, rank, at = [], [], None, 0
    for n, line in lines:
        m = RANK_TAG.match(line)
        if m:
            if rank is not None and int(m[1]) != rank:
                out.append((rank, cur, at))
                cur = []
            if not cur:
                at = n
            rank, line = int(m[1]), m[2]
            cur.append(line)
            continue
        if rank is not None:
            out.append((rank, cur, at))
            cur, rank = [], None
        if line.startswith("Traceback (most recent call last):"):
            cur, at = [line], n
        elif cur:
            cur.append(line)
            if EXC.match(line):
                out.append((None, cur, at))
                cur = []
    if cur:
        out.append((rank, cur, at))
    return out


def summarize(block, repo, trunc):
    """`(chain, frame)` -- every exception header in the block, root first (the
    order Python prints a `raise ... from ...` chain in), and the deepest frame
    that is in `repo` rather than in site-packages."""
    chain, frame = [], None
    for line in block:
        m = EXC.match(line)
        if m:
            msg = " ".join((m[2] or "").split())
            chain.append((m[1], msg if len(msg) <= trunc else msg[:trunc] + " ..."))
            continue
        f = FRAME.match(line)
        if f and repo in f[1]:
            frame = f"{Path(f[1]).relative_to(repo)}:{f[2]} in {f[3]}"
    return chain, frame


def ranges(ns):
    """`[1,2,3,7]` -> `1-3,7`."""
    ns, out, i = sorted(ns), [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}-{ns[j]}")
        i = j + 1
    return ",".join(out)


def report(title, groups, samples, per_rank):
    if not groups:
        return
    print(f"\n=== {title} ===")
    for key, occ in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        chain, frame = samples[key]
        occ = sorted(set(occ))
        ranked = [r for r, _ in occ if r is not None]
        head = f"ranks {ranges(ranked)}" if ranked else f"{len(occ)} hits"
        print(f"\n[{len(occ)}] {head}")
        # file order, with the closing header marked: for a `raise X from Y`
        # chain that is the exception that propagated, and for a DataLoader
        # `reraise` -- which prints the worker's own traceback *after* its own
        # header -- it is the original error. Both are the line you want.
        for i, (typ, msg) in enumerate(chain):
            label = "  -> " if i == len(chain) - 1 else "     "
            print(f"{label}{typ}: {msg}" if msg else f"{label}{typ}")
        if frame:
            print(f"  at  {frame}")
        # `path:line` of the traceback itself, so it can be opened and read
        for rank, loc in occ if per_rank else occ[:1]:
            tag = f"rank {rank}: " if rank is not None else ""
            print(f"  log {tag}{loc}")
        if not per_rank and len(occ) > 1:
            print(f"      (+{len(occ) - 1} more -- --per-rank lists them)")


SELFTEST = """\
Filling buffer:  9%|9| 45/500 [00:14<00:23]\rTraceback (most recent call last):
  File "/repo/data/cookers.py", line 268, in cooker_nemotron
    raise KeyError("json")
KeyError: 'json'
[rank7]: Traceback (most recent call last):
[rank7]:   File "/repo/train/train_qwen.py", line 508, in batch_generator
[rank7]:     batch = next(data_iter)
[rank7]: ValueError: Caught ValueError in DataLoader worker process 1.
W0927 21:48:24 some unrelated agent line
[rank7]: Original Traceback (most recent call last):
[rank7]: ValueError: bad sample 42
"""


def selftest():
    """The two shapes this has to get right: a progress bar that glues
    `Traceback` onto its own line with a bare `\\r` -- which must not shift the
    line numbers -- and a rank-tagged block whose real cause is printed *after*
    the wrapper's own header."""
    got = blocks(numbered(SELFTEST))
    assert [(r, at) for r, _, at in got] == [(None, 1), (7, 5), (7, 10)], got
    chain, frame = summarize(got[0][1], "/repo", 160)
    assert chain == [("KeyError", "'json'")], chain
    assert frame == "data/cookers.py:268 in cooker_nemotron", frame
    merged = [h for _, b, _ in got[1:] for h in summarize(b, "/repo", 160)[0]]
    assert [t for t, _ in merged] == ["ValueError", "ValueError"], merged
    assert merged[-1][1] == "bad sample 42", merged
    assert ranges([1, 2, 3, 7, 8, 20]) == "1-3,7-8,20"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log_dir", type=Path, nargs="?", help="slurm_errors/<jobid>")
    ap.add_argument("--selftest", action="store_true", help="check the parser, read nothing")
    ap.add_argument("--per-rank", action="store_true", help="list every rank and log file, not one example")
    ap.add_argument("--full", action="store_true", help="do not truncate messages")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.log_dir is None:
        ap.error("log_dir is required")

    # the checkout the job ran from: everything above the torchrun log dir
    repo = str(args.log_dir.resolve().parent.parent)
    trunc = 10**9 if args.full else 160

    files = sorted(args.log_dir.rglob("stderr.log"))
    if not files:
        raise SystemExit(f"no stderr.log under {args.log_dir}")

    fatal, caught = defaultdict(list), defaultdict(list)
    samples = {}
    for path in files:
        # A worker's fatal output is several rank-tagged sections -- energon
        # prints the wrapper and the sample it choked on separately -- so join
        # everything tagged with the same rank into one chain.
        merged, order = defaultdict(list), []
        # newline="": text mode otherwise translates the progress bars' bare
        # `\r` into `\n` before `numbered` can see it, and every line number
        # past the first bar comes out too high
        with open(path, errors="replace", newline="") as fh:  # not
            text = fh.read()  # Path.read_text(newline=) -- 3.13+ only
        for rank, block, at in blocks(numbered(text)):
            chain, frame = summarize(block, repo, trunc)
            if not chain:
                continue
            if rank is None:
                order.append((None, chain, frame, at))
            else:
                if rank not in merged:
                    order.append((rank, None, None, at))
                merged[rank].append((chain, frame))

        for rank, chain, frame, at in order:
            if rank is not None:
                parts = merged[rank]
                chain = [h for c, _ in parts for h in c]
                frame = next((f for _, f in reversed(parts) if f), None)
            # the numbers in a message are the payload, not the fault: 180
            # "aspect ratio ... got 382.0" lines are one problem, not 45
            key = (
                tuple((t, re.sub(r"\d+(?:\.\d+)?", "N", m)) for t, m in chain),
                frame,
            )
            samples.setdefault(key, (tuple(chain), frame))
            (caught if rank is None else fatal)[key].append((rank, f"{path}:{at}"))

    print(f"{len(files)} worker logs under {args.log_dir}")
    report("fatal (killed the worker; rank from the [rankN] tag)", fatal, samples, args.per_rank)
    report("caught and logged (run continued; no rank in the text)", caught, samples, args.per_rank)
    if not fatal and not caught:
        print("no tracebacks found")


if __name__ == "__main__":
    main()
