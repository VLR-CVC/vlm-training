import json
import time
import os
import wandb
import sys
import datetime

# Directory for logs
LOG_DIR = os.path.join("local_logging", "logs")
# Global to store current log file path
CURRENT_LOG_FILE = None

# Capture original functions
_original_init = wandb.init
_original_log = wandb.log
_original_finish = wandb.finish

def _write_event(event_type, data):
    global CURRENT_LOG_FILE
    if not CURRENT_LOG_FILE:
        return

    try:
        entry = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        with open(CURRENT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Local Logger Error: {e}", file=sys.stderr)

def _local_init(project=None, entity=None, config=None, name=None, **kwargs):
    """
    Monkey-patched wandb.init
    """
    global CURRENT_LOG_FILE
    
    # Ensure directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Generate unique filename based on name and timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_name = str(name).replace("/", "_").replace(" ", "_") if name else "unnamed"
    filename = f"{sanitized_name}_{timestamp_str}.jsonl"
    CURRENT_LOG_FILE = os.path.join(LOG_DIR, filename)
    
    # Create the file
    with open(CURRENT_LOG_FILE, "w") as f:
        pass
        
    init_data = {
        "project": project,
        "entity": entity,
        "name": name,
        "config": config,
        "kwargs": str(kwargs) 
    }
    
    _write_event("init", init_data)
    
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Local Logger: Initialized run '{name}' in {CURRENT_LOG_FILE}")

    # Call original
    run = _original_init(project=project, entity=entity, config=config, name=name, **kwargs)
    
    # Re-patch wandb.log in case init overwrote it
    if wandb.log != _local_log:
        # print("Local Logger: Re-patching wandb.log after init")
        wandb.log = _local_log
        
    return run

def _local_log(data, step=None, commit=True, sync=True):
    """
    Monkey-patched wandb.log
    """
    log_data = {
        "step": step,
        **data
    }
    _write_event("log", log_data)

    # Call original
    try:
        if wandb.run is not None:
             wandb.run.log(data, step=step, commit=commit)
    except Exception:
        pass

def _local_finish(exit_code=None, quiet=None):
    """
    Monkey-patched wandb.finish
    """
    _write_event("finish", {"exit_code": exit_code})
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Local Logger: Run finished. Log saved to {CURRENT_LOG_FILE}")
    return _original_finish(exit_code=exit_code, quiet=quiet)

def patch_wandb():
    """
    Applies the monkey patch to wandb.
    """
    wandb.init = _local_init
    wandb.log = _local_log
    wandb.finish = _local_finish
    
    # Setup default log file if patch called before init? 
    # No, we wait for init to set the filename.
    # But user might log before init? (Unlikely for wandb but possible).
    # If so, they get no logs until init.
    
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Local logging patches applied. Logs directory -> {LOG_DIR}")