import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.holography_admm import (
    AngularSpectrumPropagator,
    simulate_hologram_from_normalized_phase,
    vanilla_holography_admm,
)
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir


SUMMARY_FIELDS = (
    "method",
    "distance_um",
    "samples",
    "mse",
    "psnr",
    "ssim",
    "physical_forward_mse",
    "physical_forward_corr",
    "runtime_ms_per_sample",
)

PER_SAMPLE_FIELDS = (
    "sample_id",
    "batch_idx",
    "batch_offset",
    "method",
    "distance_um",
    "mse",
    "psnr",
    "ssim",
    "physical_forward_mse",
    "physical_forward_corr",
    "runtime_ms_per_sample",
)


def _data_loader_kwargs(cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}) or {})
    excluded = {"path", "split", "eval_split", "downsample", "flip_ud", "num_workers"}
    return {k: v for k, v in data_cfg.items() if k not in excluded}


def _resolve_device(requested: str, cfg: dict) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        requested = str(cfg.get("device", "cuda")).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _parse_float_list(text: str) -> list[float]:
    values = []
    for token in str(text).replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric distance.")
    return values


def _avg(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _write_csv(path: str | os.PathLike, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _per_sample_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    numerator = (a * b).mean(dim=1)
    denominator = a.square().mean(dim=1).sqrt() * b.square().mean(dim=1).sqrt()
    return numerator / denominator.clamp_min(float(eps))


def _metrics_per_sample(
    *,
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    physical_forward_mse: torch.Tensor | None,
    physical_forward_corr: torch.Tensor | None,
) -> list[dict[str, float]]:
    x_hat = x_hat.clamp(0.0, 1.0).float()
    x_gt = x_gt.clamp(0.0, 1.0).float()
    mse_values = (x_hat - x_gt).reshape(x_gt.shape[0], -1).pow(2).mean(dim=1)
    rows = []
    for idx in range(int(x_gt.shape[0])):
        rows.append(
            {
                "mse": float(mse_values[idx].item()),
                "psnr": float(psnr(x_hat[idx : idx + 1], x_gt[idx : idx + 1])),
                "ssim": float(ssim_torch(x_hat[idx : idx + 1], x_gt[idx : idx + 1]).item()),
                "physical_forward_mse": float("nan")
                if physical_forward_mse is None
                else float(physical_forward_mse[idx].item()),
                "physical_forward_corr": float("nan")
                if physical_forward_corr is None
                else float(physical_forward_corr[idx].item()),
            }
        )
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        distance = float(row["distance_um"])
        key_distance = None if not math.isfinite(distance) else distance
        key = (str(row["method"]), key_distance)
        grouped.setdefault(key, []).append(row)

    def sort_key(item):
        (method, distance), _rows = item
        order = 0 if method == "measurement" else 1
        distance_sort = float("-inf") if distance is None else float(distance)
        return order, distance_sort, method

    summary_rows = []
    for (method, distance), group in sorted(grouped.items(), key=sort_key):
        summary = {
            "method": method,
            "distance_um": float("nan") if distance is None else float(distance),
            "samples": len(group),
        }
        for field in SUMMARY_FIELDS:
            if field in summary:
                continue
            summary[field] = _avg([float(row[field]) for row in group])
        summary_rows.append(summary)
    return summary_rows


def _save_image_grid(
    *,
    path: str | os.PathLike,
    hologram: torch.Tensor,
    target: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    max_rows: int,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    def to_show(t: torch.Tensor, row: int):
        image = t[row].detach().float().cpu()
        if image.ndim == 3:
            image = image[0]
        return image.clamp(0, 1)

    row_count = min(int(max_rows), int(hologram.shape[0]))
    panels = [("hologram", hologram), ("target phase", target), *outputs.items()]
    fig, axes = plt.subplots(row_count, len(panels), figsize=(3.0 * len(panels), 3.0 * row_count), constrained_layout=True)
    if row_count == 1:
        axes = axes[None, :]

    for row_idx in range(row_count):
        for col_idx, (title, image) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(to_show(image, row_idx), cmap="gray", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                ax.set_title(title)
            ax.axis("off")

    ensure_dir(str(Path(path).parent))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _save_reconstruction_pngs(
    *,
    out_dir: str | os.PathLike,
    sample_start: int,
    hologram: torch.Tensor,
    target: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {"hologram": hologram, "target_phase": target, **outputs}
    for offset in range(int(hologram.shape[0])):
        for name, tensor in images.items():
            image = tensor[offset].detach().float().cpu()
            if image.ndim == 3:
                image = image[0]
            plt.imsave(
                out_dir / f"sample{sample_start + offset:05d}_{name}.png",
                image.clamp(0, 1).numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )


@torch.no_grad()
def main(args, cfg: dict) -> None:
    if str(cfg.get("data", {}).get("dataset", "")).strip().lower().replace("-", "_") not in {
        "rbc",
        "rbcs",
        "rbc_hologram",
        "rbc_holograms",
        "human_rbc",
        "human_rbc_hologram",
    }:
        raise ValueError("scripts.eval_rbc_admm is intended for data.dataset='rbc_hologram'.")

    torch.manual_seed(int(args.seed if args.seed is not None else cfg.get("seed", 123)))
    device = _resolve_device(args.device, cfg)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    dataset, dataloader = make_dataloader(
        split=args.split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
        **_data_loader_kwargs(cfg),
    )

    y0, _ = dataset[0]
    y0 = to_nchw(y0)
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))
    wavelength_m = float(args.wavelength_nm) * 1e-9
    pixel_size_m = float(args.pixel_size_um) * 1e-6
    distances_um = _parse_float_list(args.distance_um)
    propagators = {
        distance_um: AngularSpectrumPropagator.create(
            im_hw,
            pixel_size_m=pixel_size_m,
            wavelength_m=wavelength_m,
            distance_m=float(distance_um) * 1e-6,
            device=device,
        )
        for distance_um in distances_um
    }

    per_sample_rows: list[dict] = []
    first_grid_payload = None
    sample_id = 0
    total_batches = len(dataloader) if int(args.max_batches) < 0 else min(len(dataloader), int(args.max_batches))
    pbar = tqdm(dataloader, total=total_batches, desc="vanilla holography ADMM")
    for batch_idx, (hologram, target) in enumerate(pbar):
        if int(args.max_batches) >= 0 and batch_idx >= int(args.max_batches):
            break
        if int(args.max_samples) >= 0 and sample_id >= int(args.max_samples):
            break

        hologram = to_nchw(hologram).to(device=device, dtype=torch.float32)
        target = to_nchw(target).to(device=device, dtype=torch.float32)
        if int(args.max_samples) >= 0:
            remaining = int(args.max_samples) - sample_id
            if int(hologram.shape[0]) > remaining:
                hologram = hologram[:remaining]
                target = target[:remaining]

        outputs_for_grid: dict[str, torch.Tensor] = {}
        measurement_rows = _metrics_per_sample(
            x_hat=hologram,
            x_gt=target,
            physical_forward_mse=None,
            physical_forward_corr=None,
        )
        for batch_offset, metric_row in enumerate(measurement_rows):
            per_sample_rows.append(
                {
                    "sample_id": sample_id + batch_offset,
                    "batch_idx": batch_idx,
                    "batch_offset": batch_offset,
                    "method": "measurement",
                    "distance_um": float("nan"),
                    "runtime_ms_per_sample": 0.0,
                    **metric_row,
                }
            )

        batch_postfix = {}
        for distance_um, propagator in propagators.items():
            simulated_hologram = simulate_hologram_from_normalized_phase(
                target,
                propagator,
                phase_output_mode=args.output_mode,
                normalize=args.forward_normalize,
            ).clamp(0.0, 1.0)
            physical_forward_mse = F.mse_loss(
                simulated_hologram,
                hologram.clamp(0.0, 1.0),
                reduction="none",
            ).reshape(hologram.shape[0], -1).mean(dim=1)
            physical_forward_corr = _per_sample_corr(simulated_hologram, hologram)

            start_time = time.perf_counter()
            phase_hat = vanilla_holography_admm(
                hologram,
                propagator,
                iterations=int(args.iters),
                rho=float(args.rho),
                amplitude_mode=args.amplitude_mode,
                amplitude_prox=args.amplitude_prox,
                init=args.init,
                output_mode=args.output_mode,
            )
            runtime_ms = (time.perf_counter() - start_time) * 1000.0 / max(1, int(hologram.shape[0]))
            method = f"admm_{distance_um:g}um"
            outputs_for_grid[method] = phase_hat.detach().cpu()

            rows = _metrics_per_sample(
                x_hat=phase_hat,
                x_gt=target,
                physical_forward_mse=physical_forward_mse,
                physical_forward_corr=physical_forward_corr,
            )
            batch_postfix[method] = f"{_avg([row['psnr'] for row in rows]):.2f}"
            for batch_offset, metric_row in enumerate(rows):
                per_sample_rows.append(
                    {
                        "sample_id": sample_id + batch_offset,
                        "batch_idx": batch_idx,
                        "batch_offset": batch_offset,
                        "method": "vanilla_admm",
                        "distance_um": float(distance_um),
                        "runtime_ms_per_sample": float(runtime_ms),
                        **metric_row,
                    }
                )

        if first_grid_payload is None:
            first_grid_payload = (
                hologram.detach().cpu(),
                target.detach().cpu(),
                outputs_for_grid,
            )

        if args.save_reconstructions:
            _save_reconstruction_pngs(
                out_dir=out_dir / "reconstructions",
                sample_start=sample_id,
                hologram=hologram.detach().cpu(),
                target=target.detach().cpu(),
                outputs=outputs_for_grid,
            )

        sample_id += int(hologram.shape[0])
        pbar.set_postfix(batch_postfix)

    summary_rows = _summarize(per_sample_rows)
    summary_csv = out_dir / "rbc_vanilla_admm_summary.csv"
    per_sample_csv = out_dir / "rbc_vanilla_admm_per_sample.csv"
    metadata_json = out_dir / "rbc_vanilla_admm_metadata.json"
    image_grid = None
    if first_grid_payload is not None and not args.no_grid:
        grid_hologram, grid_target, grid_outputs = first_grid_payload
        image_grid = _save_image_grid(
            path=out_dir / "rbc_vanilla_admm_examples.png",
            hologram=grid_hologram,
            target=grid_target,
            outputs=grid_outputs,
            max_rows=int(args.grid_rows),
        )

    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_csv(per_sample_csv, PER_SAMPLE_FIELDS, per_sample_rows)
    best = max(
        [row for row in summary_rows if row["method"] == "vanilla_admm"],
        key=lambda row: float(row["ssim"]),
        default=None,
    )
    with open(metadata_json, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "data_path": cfg["data"].get("path", None),
                "eval_pairs": len(dataset),
                "eval_split": args.split,
                "evaluated_samples": int(sample_id),
                "device": str(device),
                "model": {
                    "type": "angular_spectrum_phase_only",
                    "wavelength_nm": float(args.wavelength_nm),
                    "pixel_size_um": float(args.pixel_size_um),
                    "distance_um": distances_um,
                    "relationship": "u0 = exp(i*phi), uz = P_z u0, y = |uz|^2",
                    "uses_paired_data_for_fitting": False,
                },
                "admm": {
                    "iterations": int(args.iters),
                    "rho": float(args.rho),
                    "amplitude_mode": args.amplitude_mode,
                    "amplitude_prox": args.amplitude_prox,
                    "init": args.init,
                    "output_mode": args.output_mode,
                    "forward_normalize_for_diagnostic": args.forward_normalize,
                },
                "best_by_ssim": best,
                "outputs": {
                    "summary_csv": str(summary_csv),
                    "per_sample_csv": str(per_sample_csv),
                    "image_grid": image_grid,
                },
            },
            f,
            indent=2,
        )

    print("\n========= RBC Vanilla Holography ADMM =========")
    print(f"data_path: {cfg['data'].get('path', None)}")
    print(f"eval_pairs: {len(dataset)} ({args.split})")
    print(f"evaluated_samples: {sample_id}")
    print(f"physical model: u0=exp(i*phi), uz=P_z u0, y=|uz|^2")
    print(f"wavelength_nm: {float(args.wavelength_nm):g}")
    print(f"pixel_size_um: {float(args.pixel_size_um):g}")
    print(f"distance_um: {', '.join(f'{d:g}' for d in distances_um)}")
    print("-----------------------------------------------")
    print("method        dist_um  samples  mse       psnr    ssim    fwd_mse   fwd_corr  ms/sample")
    for row in summary_rows:
        distance = row["distance_um"]
        distance_text = "n/a" if not math.isfinite(float(distance)) else f"{float(distance):g}"
        fwd_mse = row["physical_forward_mse"]
        fwd_corr = row["physical_forward_corr"]
        print(
            f"{row['method']:<12s} {distance_text:>7s} {int(row['samples']):>8d} "
            f"{float(row['mse']):.6f} {float(row['psnr']):>6.2f} "
            f"{float(row['ssim']):.4f} "
            f"{('nan' if not math.isfinite(float(fwd_mse)) else f'{float(fwd_mse):.6f}'):>9s} "
            f"{('nan' if not math.isfinite(float(fwd_corr)) else f'{float(fwd_corr):.4f}'):>8s} "
            f"{float(row['runtime_ms_per_sample']):>9.1f}"
        )
    print("-----------------------------------------------")
    if best is not None:
        print(f"best_ssim_distance_um: {float(best['distance_um']):g} ({float(best['ssim']):.4f})")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    if image_grid:
        print(f"image_grid: {image_grid}")
    print("===============================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate vanilla angular-spectrum ADMM on RBC hologram phase reconstruction.")
    ap.add_argument("--config", type=str, default=os.path.join("configs", "rbc_hologram.yaml"))
    ap.add_argument("--split", type=str, default=None, help="Evaluation split; defaults to data.eval_split from config.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--distance_um", type=str, default="50,100,200,500,1000")
    ap.add_argument("--wavelength_nm", type=float, default=532.0)
    ap.add_argument("--pixel_size_um", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--amplitude_mode", type=str, default="raw", choices=["raw", "mean_one", "max_one"])
    ap.add_argument("--amplitude_prox", type=str, default="soft", choices=["soft", "hard"])
    ap.add_argument("--init", type=str, default="backprop", choices=["zeros", "random", "backprop"])
    ap.add_argument("--output_mode", type=str, default="wrapped_pm_pi", choices=["wrapped_pm_pi", "wrapped_0_2pi"])
    ap.add_argument("--forward_normalize", type=str, default="max_one", choices=["raw", "mean_one", "max_one"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "rbc_vanilla_admm"))
    ap.add_argument("--no_grid", action="store_true")
    ap.add_argument("--grid_rows", type=int, default=3)
    ap.add_argument("--save_reconstructions", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    if args.split is None:
        args.split = cfg.get("data", {}).get("eval_split", "validation")
    main(args, cfg)
