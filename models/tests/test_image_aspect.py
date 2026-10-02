"""`cap_image_size` must hand the processor something `smart_resize` accepts.

`transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize` raises on
anything longer than 200:1, and the nemotron mix has enough panoramas to log 340
of those tracebacks in one 64-node run (job 46691090). Capping the long side
cannot help: the ratio does not change under a uniform scale.
"""

import pytest
from PIL import Image

from data.energon_dataloader import MAX_IMAGE_SIZE, cap_image_size

# ratios taken from the failures in job 46691090's logs
SIZES = [(10000, 20), (1024, 2), (1002, 5), (3, 1024), (4096, 12), (800, 600), (1, 1)]


@pytest.mark.parametrize("size", SIZES)
def test_output_is_within_both_limits(size):
    w, h = cap_image_size(Image.new("RGB", size)).size
    assert max(w, h) <= MAX_IMAGE_SIZE, (size, w, h)
    assert max(w, h) <= 200 * min(w, h), (size, w, h, max(w, h) / min(w, h))


@pytest.mark.parametrize("size", SIZES)
def test_padding_is_centred_and_minimal(size):
    """The image itself must survive: padding adds rows, it does not crop or
    stretch, and it adds no more than one row past the limit."""
    src = Image.new("RGB", size, (255, 0, 0))
    out = cap_image_size(src)
    w, h = out.size
    if max(size) <= MAX_IMAGE_SIZE and max(size) <= 200 * min(size):
        assert out is src  # nothing to do, and no needless copy
        return
    # the original pixels are still there, centred
    assert out.getpixel((w // 2, h // 2)) == (255, 0, 0)
    # one row narrower would be over the limit again -- no slack
    assert (min(w, h) - 1) * 200 < max(w, h), (w, h)


def test_no_op_image_is_not_copied():
    src = Image.new("RGB", (640, 480))
    assert cap_image_size(src) is src
    assert cap_image_size(None) is None
