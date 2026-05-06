import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.measurement_source import measurement_initialization
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir


SUMMARY_FIELDS = (
    "method",
    "steps",
    "samples",
    "runtime_ms_per_sample",
    "image_mse",
    "image_psnr",
    "image_ssim",
    "measurement_mse",
    "measurement_rmse",
    "init_min",
    "init_max",
    "init_mean",
    "init_std",
)
PER_SAMPLE_FIELDS = (
    "sample_id",
    "batch_idx",
    "batch_offset",
    "method",
    "steps",
    "runtime_ms_per_sample",
    "image_mse",
    "image_psnr",
    "image_ssim",
    "measurement_mse",
    "measurement_rmse",
    "init_min",
    "init_max",
    "init_mean",
    "init_std",
)


def _parse_int_list(text: str) -> list[int]:
    values = []
    for token in str(text).replace(",", " ").split():
        if token.strip():
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one ADMM step count.")
    return values


def _write_csv(path: str, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _sample_stats(x_init: torch.Tensor) -> dict[str, float]:
    x = x_init.detach().float()
    return {
        "init_min": float(x.min().item()),
        "init_max": float(x.max().item()),
        "init_mean": float(x.mean().item()),
        "init_std": float(x.std(unbiased=False).item()),
    }


def _row_metrics(
    *,
    x_init: torch.Tensor,
    x_gt: torch.Tensor,
    y: torch.Tensor,
    Hop: FFTLinearConvOperator,
) -> dict[str, float]:
    x_init_c = x_init.clamp(0.0, 1.0).float()
    x_gt_c = x_gt.clamp(0.0, 1.0).float()
    y_hat = Hop.forward(x_init_c)
    image_mse = float(F.mse_loss(x_init_c, x_gt_c).item())
    measurement_mse = float(F.mse_loss(y_hat.float(), y.float()).item())
    return {
        "image_mse": image_mse,
        "image_psnr": float(psnr(x_init_c, x_gt_c)),
        "image_ssim": float(ssim_torch(x_init_c, x_gt_c).item()),
        "measurement_mse": measurement_mse,
        "measurement_rmse": float(measurement_mse ** 0.5),
        **_sample_stats(x_init_c),
    }


def _summarize(per_sample_rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in per_sample_rows:
        key = (row["method"], int(row["steps"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (method, steps), rows in sorted(grouped.items(), key=lambda item: (item[0][0] != "adjoint", item[0][1])):
        summary = {
            "method": method,
            "steps": int(steps),
            "samples": len(rows),
        }
        for field in SUMMARY_FIELDS:
            if field in summary:
                continue
            summary[field] = _avg([float(row[field]) for row in rows])
        summary_rows.append(summary)
    return summary_rows


def _plot_summary(summary_rows: list[dict], out_path: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    admm_rows = sorted([row for row in summary_rows if row["method"] == "admm"], key=lambda row: int(row["steps"]))
    adjoint_rows = [row for row in summary_rows if row["method"] == "adjoint"]
    if not admm_rows:
        return None

    metric_specs = [
        ("image_psnr", "Image PSNR", "higher"),
        ("image_ssim", "Image SSIM", "higher"),
        ("image_mse", "Image MSE", "lower"),
        ("measurement_mse", "Measurement MSE", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    xs = [int(row["steps"]) for row in admm_rows]

    for ax, (metric, title, direction) in zip(axes, metric_specs):
        ys = [float(row[metric]) for row in admm_rows]
        ax.plot(xs, ys, marker="o", label="ADMM init")
        if adjoint_rows:
            baseline = float(adjoint_rows[0][metric])
            ax.axhline(baseline, linestyle="--", linewidth=1.5, color="gray", label="adjoint baseline")
        ax.set_title(f"{title} ({direction} is better)")
        ax.set_xlabel("ADMM outer steps")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_image_grid(
    *,
    path: str,
    y: torch.Tensor,
    x: torch.Tensor,
    inits: list[tuple[str, torch.Tensor]],
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    def to_show(t: torch.Tensor):
        img = t.detach().float().cpu()
        if img.ndim == 4:
            img = img[0]
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        if img.shape[0] == 1:
            return img[0]
        return img.permute(1, 2, 0).clamp(0, 1)

    panels = [("measurement y", y), ("ground truth x", x), *inits]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(to_show(image), cmap="gray" if int(image.shape[1]) == 1 else None)
        ax.set_title(title)
        ax.axis("off")

    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


@torch.no_grad()
def main(args, cfg: dict) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    steps_list = _parse_int_list(args.steps)

    dataset, dataloader = make_dataloader(
        split=args.split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
    )

    y0, _ = dataset[0]
    y0 = to_nchw(y0)
    psf = to_nchw(dataset.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(int(y0.shape[-2]), int(y0.shape[-1]))).to(device)

    ensure_dir(args.out_dir)
    per_sample_rows = []
    sample_id = 0
    image_grid_payload = None

    pbar = tqdm(dataloader, desc="ADMM init sweep")
    for batch_idx, (y_batch, x_batch) in enumerate(pbar):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break
        if args.max_samples >= 0 and sample_id >= args.max_samples:
            break

        y_batch = to_nchw(y_batch).to(device)
        x_batch = to_nchw(x_batch).to(device)
        if args.max_samples >= 0:
            remaining = int(args.max_samples) - sample_id
            if y_batch.shape[0] > remaining:
                y_batch = y_batch[:remaining]
                x_batch = x_batch[:remaining]

        methods = []
        start_time = time.perf_counter()
        adjoint_init = measurement_initialization(
            y=y_batch,
            H=Hop,
            method=args.start,
            normalize=args.start_normalize,
        )
        adjoint_runtime = (time.perf_counter() - start_time) * 1000.0 / max(1, y_batch.shape[0])
        methods.append(("adjoint", -1, adjoint_init, adjoint_runtime))

        for steps in steps_list:
            start_time = time.perf_counter()
            admm_init = measurement_initialization(
                y=y_batch,
                H=Hop,
                method="admm",
                normalize=args.normalize,
                admm_steps=int(steps),
                admm_inner_steps=args.inner_steps,
                admm_rho=args.rho,
                admm_step_size=args.step_size,
                admm_start=args.start,
                admm_start_normalize=args.start_normalize,
            )
            runtime = (time.perf_counter() - start_time) * 1000.0 / max(1, y_batch.shape[0])
            methods.append(("admm", int(steps), admm_init, runtime))

        if args.images and image_grid_payload is None:
            image_grid_payload = (
                y_batch[:1].detach().cpu(),
                x_batch[:1].detach().cpu(),
                [
                    (f"{method} {steps if steps >= 0 else ''}".strip(), x_init[:1].detach().cpu())
                    for method, steps, x_init, _runtime in methods
                ],
            )

        for batch_offset in range(y_batch.shape[0]):
            for method, steps, x_init, runtime in methods:
                metrics = _row_metrics(
                    x_init=x_init[batch_offset : batch_offset + 1],
                    x_gt=x_batch[batch_offset : batch_offset + 1],
                    y=y_batch[batch_offset : batch_offset + 1],
                    Hop=Hop,
                )
                per_sample_rows.append(
                    {
                        "sample_id": sample_id + batch_offset,
                        "batch_idx": batch_idx,
                        "batch_offset": batch_offset,
                        "method": method,
                        "steps": int(steps),
                        "runtime_ms_per_sample": float(runtime),
                        **metrics,
                    }
                )

        sample_id += int(y_batch.shape[0])

    summary_rows = _summarize(per_sample_rows)
    summary_csv = os.path.join(args.out_dir, "admm_init_quality_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "admm_init_quality_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "admm_init_quality_metadata.json")
    plot_path = os.path.join(args.out_dir, "admm_init_quality_curves.png")
    image_grid_path = os.path.join(args.out_dir, "admm_init_quality_example.png")

    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_csv(per_sample_csv, PER_SAMPLE_FIELDS, per_sample_rows)
    summary_plot = _plot_summary(summary_rows, plot_path)

    saved_image_grid = None
    if image_grid_payload is not None:
        y_grid, x_grid, inits_grid = image_grid_payload
        saved_image_grid = _save_image_grid(
            path=image_grid_path,
            y=y_grid,
            x=x_grid,
            inits=inits_grid,
        )

    with open(metadata_json, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "split": args.split,
                "steps": steps_list,
                "samples": sample_id,
                "inner_steps": int(args.inner_steps),
                "rho": float(args.rho),
                "step_size": float(args.step_size),
                "start": args.start,
                "start_normalize": args.start_normalize,
                "normalize": args.normalize,
                "summary_csv": summary_csv,
                "per_sample_csv": per_sample_csv,
                "summary_plot": summary_plot,
                "image_grid": saved_image_grid,
            },
            f,
            indent=2,
        )

    print("\n========= ADMM Init Quality =========")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    if summary_plot:
        print(f"plot: {summary_plot}")
    if saved_image_grid:
        print(f"image_grid: {saved_image_grid}")
    print("-------------------------------------")
    print("method   steps  img_mse   psnr    ssim    meas_mse  ms/sample")
    for row in summary_rows:
        print(
            f"{row['method']:<8s} {int(row['steps']):>5d} "
            f"{row['image_mse']:.6f} {row['image_psnr']:>6.2f} "
            f"{row['image_ssim']:.4f} {row['measurement_mse']:.6f} "
            f"{row['runtime_ms_per_sample']:>9.1f}"
        )
    print("=====================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join("configs", "base.yaml"))
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--steps", type=str, default="0,1,2,5,10,20")
    ap.add_argument("--inner_steps", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--step_size", type=float, default=0.001)
    ap.add_argument("--start", type=str, default="adjoint", choices=["adjoint", "measurement", "zeros"])
    ap.add_argument("--start_normalize", type=str, default="max")
    ap.add_argument("--normalize", type=str, default="clamp")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=16)
    ap.add_argument("--images", action="store_true", help="Save one visual grid of the tested initializers.")
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "admm_init_quality"))
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    main(args, cfg)
