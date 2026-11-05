"""Split a one-file safetensors snapshot into Hub-sized shards.

Used to send a "big" model (4B>) to HugggingFace
"""

import json
import shutil
import struct
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

GB = 1024**3
SKIP_SUFFIXES = (".base-orig",)

def _header(path: Path) -> dict:
    """The safetensors header: {name: {dtype, shape, data_offsets}, ...}."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))

def _plan(header: dict, max_bytes: int) -> list[list[str]]:
    """Greedy fill, in header order, so a shard never exceeds max_bytes."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        nbytes = end - start
        if nbytes > max_bytes:
            raise SystemExit(
                f"tensor {name} is {nbytes / GB:.2f} GiB, over the "
                f"{max_bytes / GB:.2f} GiB shard size; raise max-shard-GB"
            )
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards

def reshard(src: Path, dst: Path, max_shard_gb: float = 5.0) -> None:
    source = src / "model.safetensors"
    if not source.is_file():
        raise SystemExit(f"no model.safetensors in {src}")

    header = _header(source)
    shards = _plan(header, int(max_shard_gb * GB))
    total = len(shards)
    dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    with safe_open(str(source), framework="pt", device="cpu") as fh:
        for i, names in enumerate(shards, start=1):
            name = f"model-{i:05d}-of-{total:05d}.safetensors"
            tensors = {k: fh.get_tensor(k) for k in names}
            save_file(tensors, str(dst / name), metadata={"format": "pt"})
            written = sum(t.numel() * t.element_size() for t in tensors.values())
            total_size += written
            weight_map.update({k: name for k in names})
            print(f"  {name}  {len(names):4d} tensors  {written / GB:6.2f} GiB")
            del tensors

    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )

    for f in sorted(src.iterdir()):
        if not f.is_file() or f.name.startswith("model.safetensors"):
            continue
        if f.name.endswith(SKIP_SUFFIXES):
            print(f"  skip {f.name}")
            continue
        shutil.copy2(f, dst / f.name)
        print(f"  copy {f.name}")

    print(
        f"{len(weight_map)} tensors, {total_size / GB:.2f} GiB "
        f"across {total} shards -> {dst}"
    )

if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 4:
        raise SystemExit(__doc__)
    reshard(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        float(sys.argv[3]) if len(sys.argv) > 3 else 5.0,
    )
