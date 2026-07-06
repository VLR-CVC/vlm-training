import os
import re

import torch
import torch.distributed.checkpoint

from train.logger import logger


def _checkpoint_dir(output_dir: str, step) -> str:
    return os.path.join(output_dir, f"checkpoint-step-{step}")


def _dataloader_state_path(output_dir: str, step, data_rank: int) -> str:
    return os.path.join(_checkpoint_dir(output_dir, step), f"dataloader_state_rank{data_rank}.pth")


def save_dataloader_state(
    output_dir: str, step, loader, data_rank: int, is_leader: bool, is_log_rank: bool
) -> None:
    """Save the energon dataloader state to data_rank"""
    if not is_leader:
        return
    checkpoint_dir = _checkpoint_dir(output_dir, step)
    try:
        # the distributed checkpoint save creates this dir, but a config with
        # save_steps not aligned to the DCP write (or DCP disabled) may not have
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = loader.save_state_rank()
        torch.save(state, _dataloader_state_path(output_dir, step, data_rank))
    except Exception as e:
        logger.info(f"data_rank: {data_rank}")
        logger.info(f"exception saving dataloader state: {e}")
    else:
        if is_log_rank:
            logger.info(f"dataloader state at step {step} saved.")


def load_dataloader_state(output_dir: str, step, loader, data_rank: int) -> None:
    """Restore the energon dataloader state for ``data_rank``"""
    path = _dataloader_state_path(output_dir, step, data_rank)
    if not os.path.exists(path):
        logger.info(f"no dataloader state at {path}; starting data stream from scratch")
        return
    try:
        state = torch.load(path, weights_only=False)
        loader.restore_state_rank(state)
    except Exception as e:
        logger.info(f"data_rank: {data_rank}")
        logger.info(f"exception restoring dataloader state: {e}")


def save_distributed_checkpoint(output_dir: str, step, state_dict: dict, rank: int, is_log_rank: bool) -> None:
    """Save ``state_dict`` for ``step``. Failures are logged, not raised, so a
    single bad rank does not abort the whole run."""
    checkpoint_dir = _checkpoint_dir(output_dir, step)
    try:
        logger.info(f"checkpointing at {checkpoint_dir}")
        torch.distributed.checkpoint.save(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )
    except Exception as e:
        logger.info(f"rank: {rank}")
        logger.info(f"exception during checkpointing: {e}")
    else:
        if is_log_rank:
            logger.info(f"checkpoint at step {step} saved.")


def load_distributed_checkpoint(output_dir: str, step_num, state_dict: dict, rank: int) -> dict | None:
    """Load ``step_num`` into ``state_dict`` in-place and return it, or ``None``
    on failure. Synchronizes all ranks before loading."""
    checkpoint_dir = _checkpoint_dir(output_dir, step_num)

    # we syncronize all of the processes
    torch.distributed.barrier()

    try:
        logger.info(f"checkpointing at {checkpoint_dir}")
        torch.distributed.checkpoint.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )
    except Exception as e:
        logger.info(f"rank: {rank}")
        logger.info(f"exception during checkpointing: {e}")
        return None
    return state_dict


def find_latest_checkpoint_step(output_dir: str) -> int | None:
    """Return the largest step number among ``checkpoint-step-*`` dirs, or
    ``None`` if there is nothing to resume from."""
    possible_steps = []
    for path in os.listdir(output_dir):
        match = re.search(r"(\d+\.?\d*)$", path)
        if not match:
            continue
        try:
            possible_steps.append(int(match.group(1)))
        except ValueError:
            pass
    return max(possible_steps) if possible_steps else None
