"""`resume_optimizer=False` must still restore everything else.

The 35B-A3B checkpoints hold optimizer moments that cannot be loaded at all:
`adamw_sr` keeps exp_avg/exp_avg_sq as plain local tensors, so DCP recorded them
with `chunks=1` and no offsets -- all 256 ranks wrote one key. For the 20
unevenly-sharded params the shapes disagree and the load raises; dropping the
"optimizer" key is the way past it. This pins the two facts that rests on:
a partial state_dict loads, and a shape conflict really does raise.
"""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException


def _save(tmp_path, sd):
    dcp.save(sd, checkpoint_id=str(tmp_path))


def test_partial_state_dict_loads_without_the_optimizer_key(tmp_path):
    saved = {
        "model": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "step": 2274,
        "optimizer": torch.zeros(1, 1152),  # the bad shard, as adamw_sr wrote it
    }
    _save(tmp_path, saved)

    # resume_optimizer=False: "optimizer" is simply not handed to dcp.load
    target = {"model": torch.empty(3, 4), "step": 0}
    dcp.load(target, checkpoint_id=str(tmp_path))

    assert target["step"] == 2274
    torch.testing.assert_close(target["model"], saved["model"])


def test_optimizer_key_with_a_conflicting_shape_raises(tmp_path):
    _save(tmp_path, {"optimizer": torch.zeros(1, 1152)})

    # resume_optimizer=True against a 5-row local shard -- the production failure.
    # dcp wraps the planner's ValueError in a CheckpointException.
    target = {"optimizer": torch.empty(5, 1152)}
    with pytest.raises(CheckpointException, match="Size mismatch"):
        dcp.load(target, checkpoint_id=str(tmp_path))


def test_resume_optimizer_defaults_to_true():
    from train.config import Training

    assert Training().resume_optimizer is True, "default must not change resume behaviour"
