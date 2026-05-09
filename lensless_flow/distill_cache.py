import bisect
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


CACHE_VERSION = 1
SHARD_PREFIX = "shard_"
SHARD_SUFFIX = ".pt"
METADATA_JSON = "metadata.json"
METADATA_PT = "metadata.pt"


def list_cache_shards(cache_dir: str | os.PathLike) -> list[Path]:
    path = Path(cache_dir)
    shards = sorted(path.glob(f"{SHARD_PREFIX}*{SHARD_SUFFIX}"))
    if not shards:
        raise FileNotFoundError(f"No distillation cache shards found in {path}")
    return shards


def load_cache_metadata(cache_dir: str | os.PathLike) -> dict:
    metadata_path = Path(cache_dir) / METADATA_JSON
    if not metadata_path.is_file():
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_cache_metadata_pt(cache_dir: str | os.PathLike) -> dict:
    metadata_path = Path(cache_dir) / METADATA_PT
    if not metadata_path.is_file():
        return {}
    return torch.load(metadata_path, map_location="cpu")


class TeacherDistillCacheDataset(Dataset):
    """
    Lazy reader for teacher distillation cache shards.

    Each shard is a torch-saved dict with tensor fields:
      y, x_gt, z0, x_teacher, sample_id, repeat, latent_seed
    """

    def __init__(self, cache_dir: str | os.PathLike):
        self.cache_dir = Path(cache_dir)
        self.shards = list_cache_shards(self.cache_dir)
        self.metadata = load_cache_metadata(self.cache_dir)

        self._lengths: list[int] = []
        for shard_path in self.shards:
            shard = torch.load(shard_path, map_location="cpu")
            self._lengths.append(int(shard["y"].shape[0]))

        self._cum_lengths: list[int] = []
        total = 0
        for length in self._lengths:
            total += length
            self._cum_lengths.append(total)

        self._cached_shard_idx: int | None = None
        self._cached_shard: dict | None = None

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def _load_shard(self, shard_idx: int) -> dict:
        if self._cached_shard_idx != shard_idx:
            self._cached_shard = torch.load(self.shards[shard_idx], map_location="cpu")
            self._cached_shard_idx = shard_idx
        assert self._cached_shard is not None
        return self._cached_shard

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        shard_idx = bisect.bisect_right(self._cum_lengths, idx)
        prev = 0 if shard_idx == 0 else self._cum_lengths[shard_idx - 1]
        local_idx = idx - prev
        shard = self._load_shard(shard_idx)

        return {
            "y": shard["y"][local_idx].float(),
            "x_gt": shard["x_gt"][local_idx].float(),
            "z0": shard["z0"][local_idx].float(),
            "x_teacher": shard["x_teacher"][local_idx].float(),
            "sample_id": shard["sample_id"][local_idx].long(),
            "repeat": shard["repeat"][local_idx].long(),
            "latent_seed": shard["latent_seed"][local_idx].long(),
        }
