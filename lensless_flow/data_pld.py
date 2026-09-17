"""Native 4x Parallel Lensless Dataset, RML measurements and registered RGB GT."""
from pathlib import Path
import re

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


def pld_split_ids(split):
    """Official numeric ordering, audited for the complete IDs 0..99,999."""
    name = str(split).lower()
    if name in {"train", "training"}:
        return range(5000, 100000)
    if name in {"val", "valid", "validation"}:
        return range(1000, 5000)
    if name == "test":
        return range(1000)
    raise ValueError(f"Unknown PLD split: {split}")


def _index_images(folder, camera):
    indexed = {}
    for path in folder.glob("*.tiff"):
        match = re.search(r"img_(\d+)_cam_(\d+)\.tiff$", path.name)
        if not match or int(match[2]) != camera:
            raise ValueError(f"Unexpected PLD filename: {path}")
        image_id = int(match[1])
        if image_id in indexed:
            raise ValueError(f"Duplicate PLD image ID: {image_id}")
        indexed[image_id] = path
    return indexed


def read_pld_rgb(path):
    image = tifffile.imread(path)
    if image.shape not in {(300, 480, 3), (300, 480, 4)} or image.dtype != np.uint8:
        raise ValueError(f"Expected native 4x uint8 RGB/RGBA TIFF, got {image.shape} {image.dtype}: {path}")
    # The fourth channel is alpha, not an additional measured color channel.
    return torch.from_numpy(np.ascontiguousarray(image[..., :3], dtype=np.float32) / 255.0)


class ParallelLenslessRMLDataset(Dataset):
    """Return HWC RGB pairs; no resizing, intensity stretching, flips, or rewarping.

    Incomplete copies are allowed only with explicit ``smoke_test=True``. IDs keep
    their original split even in these copies; missing files never shift a split.
    """
    def __init__(self, root, split="train", downsample=1, flip_ud=False,
                 flip_lr=False, max_samples=None, smoke_test=False):
        if float(downsample) != 1 or flip_ud or flip_lr:
            raise ValueError("PLD calibration requires native 4x files with downsample=1 and no flips")
        self.root = Path(root).expanduser()
        self.split = split
        self.psf = None
        self.smoke_test = bool(smoke_test)
        measurements = _index_images(self.root / "4x_rml", 1)
        targets = _index_images(self.root / "4x_undistorted_GT2RML", 2)
        if not measurements or measurements.keys() != targets.keys():
            missing_gt = sorted(measurements.keys() - targets.keys())[:5]
            missing_y = sorted(targets.keys() - measurements.keys())[:5]
            raise ValueError(f"PLD pairing failed: missing targets {missing_gt}, missing measurements {missing_y}")
        all_ids = set(measurements)
        if not all_ids.issubset(range(100000)):
            raise ValueError("PLD image IDs must be in 0..99999")
        if not self.smoke_test and all_ids != set(range(100000)):
            raise ValueError(f"Expected all 100000 PLD pairs, found {len(all_ids)}. Use smoke_test only for tiny test copies.")
        self.ids = [i for i in pld_split_ids(split) if i in all_ids]
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive")
            self.ids = self.ids[:int(max_samples)]
        if not self.ids:
            raise ValueError(f"No PLD pairs in split {split}")
        self.pairs = [(measurements[i], targets[i]) for i in self.ids]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        measurement, target = self.pairs[int(index)]
        return read_pld_rgb(measurement), read_pld_rgb(target)
