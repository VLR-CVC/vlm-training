import logging
import sys
import os
import warnings

class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"
    turquoise = "\033[38;2;54;234;195m"

logger = logging.getLogger("train_logger")


def redirect_rank_io(log_dir="logs"):
    """Send this rank's raw stdout/stderr to its own per-rank ``.err`` file.

    Used for energon `SkipSample`. send to stderr instead of stdout
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not os.environ.get("SLURM_JOB_ID") and world_size <= 1:
        return None

    rank = int(os.environ.get("RANK", "0"))
    job = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs(log_dir, exist_ok=True)

    real_stdout = os.fdopen(os.dup(1), "w", buffering=1)

    err_path = os.path.join(log_dir, f"rank{rank}_{job}.err")
    err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(err_fd, 1)  # fd 1 -> per-rank err file (this process + children)
    os.dup2(err_fd, 2)  # fd 2 -> per-rank err file
    os.close(err_fd)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    return real_stdout


def init_logger(stream=None):
    if logger.handlers:
        logger.handlers.clear()

    rank = int(os.environ.get("RANK", "0"))

    # fall back to the live sys.stdout when no redirect happened (local runs).
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Propagate false to avoid double logging if root logger is configured
    logger.propagate = False

    if rank == 0:
        logger.setLevel(logging.INFO)
        # Configure warnings for rank 0
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.addHandler(handler)
        warnings_logger.setLevel(logging.INFO)
        
    else:
        logger.setLevel(logging.ERROR) 
        
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)