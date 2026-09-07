"""Detached pseudo-regions for wrapped RBC phase labels.

These are heuristic loss/metric masks, not annotated instance segmentations.
The circular local dispersion avoids mistaking a 0/1 phase branch cut for
large physical contrast. Morphology fills detected cell interiors.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi


def region_loss_config(cfg: dict) -> dict:
    return dict(cfg.get("cfm", {}).get("loss", {}).get("rbc_region", {}) or {})


def region_eval_enabled(cfg: dict) -> bool:
    return bool(cfg.get("eval", {}).get("rbc_regions", region_loss_config(cfg).get("enabled", False)))


def save_region_preview(path: str, examples: list[tuple[torch.Tensor, ...]]) -> None:
    """Save a fixed [0,1] scale validation grid; tensors are single CHW images."""
    from pathlib import Path
    from PIL import Image, ImageDraw

    if not examples:
        return
    height, width = examples[0][0].shape[-2:]
    header = 24
    canvas = Image.new("RGB", (4 * width, header + len(examples) * height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Hologram", "Phase target", "Flow reconstruction", "Target pseudo-region (red)")):
        draw.text((column * width + 6, 6), title, fill="black")
    for row, (hologram, target, prediction, mask) in enumerate(examples):
        arrays = [item.detach().float().cpu().numpy()[0] for item in (hologram, target, prediction, mask)]
        for column, values in enumerate(arrays[:3]):
            tile = np.repeat((np.clip(values, 0, 1)[..., None] * 255).round().astype(np.uint8), 3, axis=-1)
            canvas.paste(Image.fromarray(tile), (column * width, header + row * height))
        overlay = np.repeat(np.clip(arrays[1], 0, 1)[..., None], 3, axis=-1)
        alpha = 0.4 * arrays[3][..., None]
        overlay = overlay * (1 - alpha) + np.array([1, 0, 0]) * alpha
        canvas.paste(Image.fromarray((overlay * 255).round().astype(np.uint8)), (3 * width, header + row * height))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def rbc_region_mask(
    target: torch.Tensor,
    *,
    window_size: int = 15,
    dispersion_threshold: float = 0.08,
    closing_iterations: int = 5,
    min_component_area: int = 80,
    reflect_pad: int = 48,
    min_fraction: float = 0.01,
    max_fraction: float = 0.85,
) -> torch.Tensor:
    """Return a detached B1HW binary mask on the target's device.

    All geometry parameters are in pixels (defaults audited at 256x256).
    Compute 1-|local_mean exp(2*pi*i*target)|, threshold, close gaps, fill
    enclosed holes, and remove small components. Reflect padding helps close
    partial cells at crop boundaries. Coverage outside the specified range
    returns an empty mask so the region loss falls back to ordinary MSE.

    CPU morphology entails a device synchronization when given CUDA targets.
    It is intentionally independent of the prediction, noisy path state, and
    flow time. For OT coupling, pass the matched target, not the original order.
    """
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("RBC region masks require a B1HW phase tensor.")
    if target.shape[0] < 1 or min(target.shape[-2:]) < 2:
        raise ValueError("RBC region masks require a nonempty batch and H,W >= 2.")
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("window_size must be an odd integer >= 3.")
    if not 0 < dispersion_threshold < 1:
        raise ValueError("dispersion_threshold must be between zero and one.")
    if closing_iterations < 1 or min_component_area < 1 or reflect_pad < window_size // 2 + closing_iterations:
        raise ValueError("Require positive morphology sizes and sufficient reflect padding.")
    if not 0 <= min_fraction < max_fraction <= 1:
        raise ValueError("Require 0 <= min_fraction < max_fraction <= 1.")
    values = target.detach().float().cpu().numpy()[:, 0]
    if not np.isfinite(values).all():
        raise ValueError("Phase targets contain non-finite values.")
    masks = []
    for value in values:
        padded = np.pad(value, reflect_pad, mode="reflect")
        angle = 2 * np.pi * padded
        mean_cos = ndi.uniform_filter(np.cos(angle), size=window_size)
        mean_sin = ndi.uniform_filter(np.sin(angle), size=window_size)
        dispersion = 1 - np.hypot(mean_cos, mean_sin)
        closed = ndi.binary_closing(dispersion > dispersion_threshold, iterations=closing_iterations)
        filled = ndi.binary_fill_holes(closed)
        labels, _ = ndi.label(filled)
        areas = np.bincount(labels.ravel())
        keep = areas >= min_component_area
        keep[0] = False
        mask = keep[labels][reflect_pad:-reflect_pad, reflect_pad:-reflect_pad].copy()
        # Apply the area floor in the actual crop as well as the padded one.
        cropped_labels, _ = ndi.label(mask)
        cropped_keep = np.bincount(cropped_labels.ravel()) >= min_component_area
        cropped_keep[0] = False
        mask = cropped_keep[cropped_labels]
        if not min_fraction <= mask.mean() <= max_fraction:
            mask[:] = False
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)[:, None].astype(np.float32)).to(target.device)
