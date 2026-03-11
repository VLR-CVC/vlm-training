import logging
import sys
import os
import warnings

class Color:
    red = "\033[91m"
    green = "\033[92m"
    blue = "\033[94m"
    reset = "\033[0m"

logger = logging.getLogger("train_logger")

def init_logger():
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    rank = int(os.environ.get("RANK", "0"))
    
    handler = logging.StreamHandler(sys.stdout)
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
        # Suppress info logs on non-zero ranks
        logger.setLevel(logging.ERROR) 
        
        # Suppress warnings on non-zero ranks
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)