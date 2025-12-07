from dataclasses import dataclass

@dataclass
class Job:
    config = None
    resume_from  = None
    
    # Training
    lr: float = 1e-4
    batch_size: int = 32
