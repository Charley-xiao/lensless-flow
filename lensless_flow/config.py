from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf


def _normalize_cli_overrides(overrides: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in list(overrides or []):
        arg = str(raw).strip()
        if not arg or arg == "--":
            continue
        if arg.startswith("-"):
            raise ValueError(
                f"Unrecognized CLI argument '{arg}'. "
                "Use declared script flags with '--flag value', and use OmegaConf overrides like "
                "'train.batch_size=16 model.base_channels=48' for config values."
            )
        normalized.append(arg)
    return normalized


def merge_config(base_config: dict, overrides: Sequence[str] | None = None) -> dict:
    """
    Merge an in-memory mapping config with optional OmegaConf dotlist overrides.
    """

    base_cfg = OmegaConf.create(base_config)
    cli_overrides = _normalize_cli_overrides(overrides)
    if cli_overrides:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(cli_overrides))

    resolved = OmegaConf.to_container(base_cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected a mapping config, got {type(resolved).__name__}.")
    return resolved


def load_config(config_path: str | Path, overrides: Sequence[str] | None = None) -> dict:
    """
    Load a YAML config and merge optional OmegaConf dotlist overrides.

    Example:
        python -m scripts.train --config configs/a100_base.yaml train.batch_size=16 model.base_channels=48
    """

    base_cfg = OmegaConf.load(str(config_path))
    return merge_config(base_cfg, overrides)
