#!/usr/bin/env python3
"""Re-vendor Megatron's min-memory DSA kernels into `models/common/dsa/`.

`dsa_min_memory_triton.py` has no Megatron imports at all and is copied byte for
byte. `dsa_min_memory.py` imports five Megatron symbols, which
`models/common/dsa/_shim.py` supplies; this script rewrites that import block
and nothing else, so a diff against upstream stays readable.

    python scripts/vendor_dsa.py [--ref <sha>]

Bump `REF` when upstream moves. Every rewrite below must fire exactly once; if
one does not, the import block changed and the patch needs re-reading rather
than loosening.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# "Add the min-memory DSA attention kernels" (#7152), merged 2026-10-13.
REF = "1e3c0c62fbb79c26d083f506f7da46f5f7add49e"
BASE = "https://raw.githubusercontent.com/NVIDIA/Megatron-LM/{ref}/megatron/core/transformer/experimental_attention_variant/{name}"
FILES = ("dsa_min_memory_triton.py", "dsa_min_memory.py")

HEADER = '''# Vendored from NVIDIA/Megatron-LM {ref}:
# megatron/core/transformer/experimental_attention_variant/{name}
# Apache-2.0 with the NVIDIA source headers kept below. Re-vendor with
# `python scripts/vendor_dsa.py`; the only edit is the import block.
'''

REWRITES = {
    "from megatron.core.models.common.embeddings.rope_utils import _rotate_half": (
        "from models.common.dsa._shim import _rotate_half"
    ),
    "from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (": (
        "from models.common.dsa._shim import ("
    ),
    "from megatron.core.process_groups_config import ProcessGroupCollection": (
        "from models.common.dsa._shim import ProcessGroupCollection"
    ),
    "from megatron.core.transformer.experimental_attention_variant.dsa import "
    "rotate_activation": "from models.common.dsa._shim import rotate_activation",
    "from megatron.core.transformer.experimental_attention_variant."
    "dsa_min_memory_triton import (": "from models.common.dsa.dsa_min_memory_triton import (",
    "from megatron.core.utils import ensure_params_ready": (
        "from models.common.dsa._shim import ensure_params_ready"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / "models" / "common" / "dsa"
    for name in FILES:
        url = BASE.format(ref=a.ref, name=name)
        with urllib.request.urlopen(url) as fh:
            text = fh.read().decode()
        for old, new in REWRITES.items():
            if old not in text:
                continue
            assert text.count(old) == 1, f"{name}: {old!r} appears {text.count(old)} times"
            text = text.replace(old, new)
        assert "megatron" not in text, f"{name}: unhandled megatron reference remains"
        dst = out / name
        dst.write_text(HEADER.format(ref=a.ref, name=name) + "\n" + text)
        print(f"wrote {dst} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
