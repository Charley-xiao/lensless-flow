import torch
from torch.utils.data import DataLoader
from lensless.utils.dataset import DiffuserCamMirflickrHF

def make_dataloader(split: str, downsample: int, flip_ud: bool, batch_size: int, num_workers: int):
    ds = DiffuserCamMirflickrHF(split=split, downsample=downsample, flip_ud=flip_ud)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                    num_workers=num_workers, pin_memory=True, drop_last=(split == "train"), 
                    persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None)
    return ds, dl
