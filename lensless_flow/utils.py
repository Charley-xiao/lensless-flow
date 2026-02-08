import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
