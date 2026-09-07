"""Audit target-derived RBC pseudo-regions on training images; does not train.

Run ``python -m scripts.audit_rbc_regions --config configs/rbc_hologram_unet64.yaml``.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lensless_flow.config import load_config
from lensless_flow.rbc_regions import rbc_region_mask, region_loss_config


def equal_error_shares(fraction: float, balance_mix: float, foreground_weight: float) -> dict:
    """Loss coefficient mass when all pixels have the same squared error."""
    mixed = ((1 - balance_mix) * fraction + balance_mix * foreground_weight
             if 0 < fraction < 1 else fraction)
    return {
        "ordinary_foreground_share": fraction,
        "ordinary_background_share": 1 - fraction,
        "mixed_foreground_share": mixed,
        "mixed_background_share": 1 - mixed,
    }


def save_overlays(path: Path, examples: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_row = 4
    rows = (len(examples) + per_row - 1) // per_row
    fig, axes = plt.subplots(rows, 2 * per_row, figsize=(20, 2.65 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for i, example in enumerate(examples):
        target, mask = example["target"], example["mask"]
        left, right = axes[i // per_row, 2 * (i % per_row): 2 * (i % per_row) + 2]
        left.imshow(target, cmap="gray", vmin=0, vmax=1)
        left.set_title(f'Training index {example["index"]}\nPhase label', fontsize=9)
        right.imshow(target, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*target.shape, 4), dtype=float)
        overlay[..., 0] = 1
        overlay[..., 3] = mask * 0.4
        right.imshow(overlay)
        right.set_title(f'Pseudo-region: {mask.mean():.1%}\nNot a segmentation label', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    if float(data_cfg.get("downsample", 1)) != 1:
        raise ValueError("This RBC mask audit requires data.downsample=1 (native 256x256).")
    if args.max_samples < 1:
        raise ValueError("max_samples must be positive.")
    options = region_loss_config(cfg)
    balance_mix = float(options.get("balance_mix", 0.5))
    foreground_weight = float(options.get("foreground_weight", 0.75))
    if not 0 <= balance_mix <= 1 or not 0 <= foreground_weight <= 1:
        raise ValueError("balance_mix and foreground_weight must be in [0, 1].")
    mask_options = {
        name: parameter.default
        for name, parameter in inspect.signature(rbc_region_mask).parameters.items()
        if name != "target"
    }
    mask_options.update(dict(options.get("mask", {}) or {}))
    root = Path(args.data_root or data_cfg.get("path", "E:/RBCs Holograms"))
    phase_dir = root / "Phase" / "Training"
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"Missing training phase directory: {phase_dir}")
    paths = sorted(p for p in phase_dir.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No training PNG targets in {phase_dir}")
    indices = np.linspace(0, len(paths) - 1, min(args.max_samples, len(paths)), dtype=int)
    targets = []
    for index in indices:
        with Image.open(paths[index]) as image:
            if image.size != (256, 256):
                raise ValueError(f"Expected a native 256x256 target: {paths[index]}")
            target = np.asarray(image.convert("L"), dtype=np.float32) / 255
        if data_cfg.get("phase_invert", False):
            target = 1 - target
        if data_cfg.get("flip_ud", False):
            target = np.flipud(target)
        if data_cfg.get("flip_lr", False):
            target = np.fliplr(target)
        targets.append(np.ascontiguousarray(target))
    tensor = torch.from_numpy(np.stack(targets)[:, None])
    start = time.perf_counter()
    masks = rbc_region_mask(tensor, **mask_options).numpy()[:, 0]
    seconds = time.perf_counter() - start
    examples, records = [], []
    for index, target, mask in zip(indices, targets, masks):
        fraction = float(mask.mean())
        record = {
            "index": int(index), "phase_path": str(paths[index]),
            "foreground_fraction": fraction, "uniform_fallback": fraction == 0 or fraction == 1,
            **equal_error_shares(fraction, balance_mix, foreground_weight),
        }
        if args.flat_baseline:
            error = (target - 0.5) ** 2
            for name, selection in (("global", np.ones_like(mask, dtype=bool)),
                                    ("foreground", mask.astype(bool)),
                                    ("background", ~mask.astype(bool))):
                mse = float(error[selection].mean()) if selection.any() else None
                record[f"flat_0_5_{name}_mse"] = mse
                record[f"flat_0_5_{name}_psnr"] = (
                    float(-10 * np.log10(max(mse, 1e-12))) if mse is not None else None)
        records.append(record)
        examples.append({"index": int(index), "target": target, "mask": mask})
    fractions = np.array([r["foreground_fraction"] for r in records])
    summary = {
        "config": str(Path(args.config)), "split": "Training", "dataset_targets": len(paths),
        "selection": "evenly spaced indices of sorted training phase filenames",
        "num_samples": len(records), "region_loss_enabled_in_config": bool(options.get("enabled", False)),
        "balance_mix": balance_mix,
        "foreground_weight": foreground_weight, "mask_options": mask_options,
        "mask_cpu_seconds": seconds, "mask_cpu_ms_per_image": 1000 * seconds / len(records),
        "foreground_fraction_mean": float(fractions.mean()),
        "foreground_fraction_min": float(fractions.min()),
        "foreground_fraction_max": float(fractions.max()),
        "uniform_fallback_count": sum(r["uniform_fallback"] for r in records),
        "mean_ordinary_foreground_share": float(np.mean([r["ordinary_foreground_share"] for r in records])),
        "mean_mixed_foreground_share": float(np.mean([r["mixed_foreground_share"] for r in records])),
        "interpretation": "Pseudo-mask visual audit, not segmentation accuracy or trained-model evaluation. "
                          "Loss shares assume equal squared error at every pixel; actual gradient shares can differ. "
                          "The illustrated mixture uses config values or candidate defaults (0.5, 0.75), "
                          "even if region loss is disabled in the supplied config.",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_overlays(out / "overlays.png", examples)
    for i, example in enumerate(examples):
        Image.fromarray((example["mask"] * 255).astype(np.uint8)).save(out / f"mask_{i:03d}.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f"Visual audit: {out / 'overlays.png'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rbc_hologram_unet64.yaml")
    parser.add_argument("--data_root", default=None, help="Optional dataset root override.")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--out_dir", default="outputs/rbc_region_audit")
    parser.add_argument("--flat_baseline", action="store_true", help="Report optional fixed 0.5 baseline by region.")
    main(parser.parse_args())
