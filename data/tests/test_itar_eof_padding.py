"""energon must read a shard's last sample when its `.tar.idx` span runs over the
end-of-archive padding (the mt_*cap_wds shards). Builds such a shard, then reads its
last sample through energon's reader with and without the patch.

    python data/tests/test_itar_eof_padding.py
"""

import io
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from megatron.energon.epathlib import EPath  # noqa: E402
from megatron.energon.flavors.webdataset.itar import ITarSamplePointer  # noqa: E402
from megatron.energon.flavors.webdataset.itar_reader import ITarReader  # noqa: E402

original = ITarReader._get_item_by_sample_pointer
import data.energon_dataloader  # noqa: E402,F401  (applies the patch)


class Reader(ITarReader):
    """Just enough of a reader to call `_get_item_by_sample_pointer`."""
    def __init__(self, base, names):
        super().__init__(base_path=EPath(base), tar_filenames=names,
                         tar_filepaths=[EPath(base) / n for n in names], disable_cache=True)
    def __len__(self): return 0
    def __str__(self): return "Reader"
    def _get_itar_sample_pointer(self, *a, **k): raise NotImplementedError


def main():
    d = tempfile.mkdtemp()
    with tarfile.open(f"{d}/s.tar", "w", format=tarfile.USTAR_FORMAT) as t:
        for i in range(3):
            data = f'{{"i": {i}}}'.encode()
            info = tarfile.TarInfo(f"{i:04d}.json"); info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    size = Path(f"{d}/s.tar").stat().st_size
    # the faulty layout: last sample at 2048 spans to the end of the file (padding)
    ptr = ITarSamplePointer(tar_file_id=0, byte_offset=2048, byte_size=size - 2048)
    r = Reader(d, ["s.tar"])
    try:
        original(r, ptr, 2)
        raise AssertionError("unpatched reader should fail on this layout")
    except ValueError as e:
        assert "Unexpected end of tar file" in str(e)
    s = r._get_item_by_sample_pointer(ptr, 2)
    assert s["__key__"] == "0002" and s["json"] == b'{"i": 2}', s
    # a normal span still reads unchanged
    assert r._get_item_by_sample_pointer(ITarSamplePointer(tar_file_id=0, byte_offset=0, byte_size=1024), 0)["json"] == b'{"i": 0}'
    print("ok")


if __name__ == "__main__":
    main()
