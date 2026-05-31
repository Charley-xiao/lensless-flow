import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.measurement_source import normalize_measurement_init
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir


def parse_indices(text: str) -> list[int]:
    values = [token.strip() for token in str(text).replace(",", " ").split() if token.strip()]
    if not values:
        raise ValueError("Provide at least one sample index.")
    return [int(value) for value in values]


def tensor_to_image_array(x_bchw: torch.Tensor, normalize: bool = False):
    x = x_bchw[0].detach().float().cpu()
    if normalize:
        x = x - x.min()
        x = x / (x.max() + 1e-8)
    else:
        x = x.clamp(0, 1)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def save_image_only(x_bchw: torch.Tensor, out_path: str, normalize: bool = False, dpi: int = 180) -> None:
    img = tensor_to_image_array(x_bchw, normalize=normalize)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(x_hat: torch.Tensor, x_gt: torch.Tensor, y: torch.Tensor, H) -> dict[str, float]:
    x_hat_c = x_hat.clamp(0, 1).float()
    x_gt_c = x_gt.clamp(0, 1).float()
    residual = H.forward(x_hat_c) - y.float()
    return {
        "psnr": float(psnr(x_hat_c, x_gt_c)),
        "ssim": float(ssim_torch(x_hat_c, x_gt_c).item()),
        "mse": float((x_hat_c - x_gt_c).pow(2).mean().item()),
        "dc_rmse": float(residual.pow(2).mean().sqrt().item()),
    }


def grad_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def grad_adjoint(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(gx)
    out[..., :, :-1] -= gx[..., :, :-1]
    out[..., :, 1:] += gx[..., :, :-1]
    out[..., :-1, :] -= gy[..., :-1, :]
    out[..., 1:, :] += gy[..., :-1, :]
    return out


def isotropic_shrink(gx: torch.Tensor, gy: torch.Tensor, threshold: float, eps: float = 1e-12):
    norm = torch.sqrt(gx.square() + gy.square() + eps)
    scale = torch.clamp(1.0 - float(threshold) / norm, min=0.0)
    return scale * gx, scale * gy


def tv_norm(x: torch.Tensor) -> float:
    gx, gy = grad_forward(x.float())
    return float(torch.sqrt(gx.square() + gy.square() + 1e-12).mean().item())


def initial_x(y: torch.Tensor, H, start: str, normalize: str) -> torch.Tensor:
    key = str(start).lower()
    if key == "adjoint":
        x0 = H.adjoint(y.float()).to(dtype=y.dtype)
    elif key == "measurement":
        x0 = y.clone()
    elif key in {"zero", "zeros"}:
        x0 = torch.zeros_like(y)
    else:
        raise ValueError(f"Unknown start={start!r}")
    return normalize_measurement_init(x0, mode=normalize).float()


def suggested_x_step(H, rho_tv: float, rho_box: float, safety: float) -> float:
    data_lipschitz = 2.0 * float((H.otf.abs() ** 2).amax().item())
    tv_lipschitz = 8.0 * float(rho_tv)
    box_lipschitz = float(rho_box)
    return float(safety / max(data_lipschitz + tv_lipschitz + box_lipschitz, 1e-12))


@torch.no_grad()
def admm_tv_reconstruct(
    y: torch.Tensor,
    H,
    *,
    steps: int,
    inner_steps: int,
    tv_weight: float,
    rho_tv: float,
    rho_box: float,
    x_step: float,
    start: str,
    start_normalize: str,
    x_gt: torch.Tensor | None = None,
    log_every: int = 10,
) -> tuple[torch.Tensor, list[dict]]:
    x = initial_x(y, H, start=start, normalize=start_normalize)
    z = x.clamp(0, 1)
    ux = torch.zeros_like(x)
    px, py = grad_forward(x)
    upx = torch.zeros_like(px)
    upy = torch.zeros_like(py)
    history = []

    for k in range(int(steps)):
        for _ in range(int(inner_steps)):
            gx, gy = grad_forward(x)
            residual = H.forward(x) - y.float()
            data_grad = H.adjoint(residual).float()
            tv_grad = grad_adjoint(gx - px + upx, gy - py + upy)
            box_grad = x - z + ux
            full_grad = data_grad + float(rho_tv) * tv_grad + float(rho_box) * box_grad
            x = x - float(x_step) * full_grad

        gx, gy = grad_forward(x)
        px, py = isotropic_shrink(gx + upx, gy + upy, threshold=float(tv_weight) / float(rho_tv))
        z = (x + ux).clamp(0, 1)
        upx = upx + gx - px
        upy = upy + gy - py
        ux = ux + x - z

        if k == 0 or (k + 1) % max(1, int(log_every)) == 0 or k + 1 == int(steps):
            x_eval = z.clamp(0, 1)
            row = {
                "iter": k + 1,
                "data_rmse": float((H.forward(x_eval) - y.float()).pow(2).mean().sqrt().item()),
                "tv": tv_norm(x_eval),
                "x_min": float(x_eval.min().item()),
                "x_max": float(x_eval.max().item()),
            }
            if x_gt is not None:
                row.update(compute_metrics(x_eval, x_gt, y, H))
            history.append(row)

    return z.clamp(0, 1), history


def save_sample_images(
    *,
    idx: int,
    y: torch.Tensor,
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    out_dir: str,
    dpi: int,
) -> dict[str, str]:
    prefix = f"idx_{int(idx):05d}"
    paths = {
        "measurement": os.path.join(out_dir, f"{prefix}_y.png"),
        "reconstruction": os.path.join(out_dir, f"{prefix}_admm_tv.png"),
        "ground_truth": os.path.join(out_dir, f"{prefix}_gt.png"),
        "abs_error": os.path.join(out_dir, f"{prefix}_abs_error.png"),
    }
    save_image_only(y, paths["measurement"], normalize=True, dpi=dpi)
    save_image_only(x_hat, paths["reconstruction"], normalize=False, dpi=dpi)
    save_image_only(x_gt, paths["ground_truth"], normalize=False, dpi=dpi)
    save_image_only((x_hat.clamp(0, 1) - x_gt.clamp(0, 1)).abs(), paths["abs_error"], normalize=True, dpi=dpi)
    return paths


def save_history_plot(history_rows: list[dict], indices: list[int], out_path: str, dpi: int) -> None:
    if not history_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for idx in indices:
        rows = [row for row in history_rows if int(row["idx"]) == int(idx)]
        if not rows:
            continue
        xs = [row["iter"] for row in rows]
        axes[0].plot(xs, [row["psnr"] for row in rows], marker="o", label=f"idx {idx}")
        axes[1].plot(xs, [row["ssim"] for row in rows], marker="o", label=f"idx {idx}")
        axes[2].plot(xs, [row["data_rmse"] for row in rows], marker="o", label=f"idx {idx}")

    axes[0].set_title("PSNR vs ADMM iteration")
    axes[1].set_title("SSIM vs ADMM iteration")
    axes[2].set_title("data RMSE vs ADMM iteration")
    for ax in axes:
        ax.set_xlabel("ADMM iteration")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    cfg = load_config(args.config)
    indices = parse_indices(args.indices)
    device = torch.device(args.device if args.device else (cfg["device"] if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.out_dir)

    dataset, _ = make_dataloader(
        split=args.split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    y0, _ = dataset[indices[0]]
    y0 = to_nchw(y0)
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))
    psf = to_nchw(dataset.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    x_step = (
        suggested_x_step(Hop, args.rho_tv, args.rho_box, args.x_step_safety)
        if args.auto_x_step
        else float(args.x_step)
    )
    print(f"device={device}, split={args.split}, im_hw={im_hw}, indices={indices}")
    print(
        "ADMM-TV: "
        f"steps={args.steps}, inner_steps={args.inner_steps}, tv_weight={args.tv_weight}, "
        f"rho_tv={args.rho_tv}, rho_box={args.rho_box}, x_step={x_step:.3e}"
    )

    summary_rows = []
    history_rows = []
    image_rows = []

    for idx in tqdm(indices, desc="ADMM-TV samples"):
        y, x_gt = dataset[int(idx)]
        y = to_nchw(y).to(device)
        x_gt = to_nchw(x_gt).to(device)

        start_time = time.perf_counter()
        x_hat, history = admm_tv_reconstruct(
            y,
            Hop,
            steps=args.steps,
            inner_steps=args.inner_steps,
            tv_weight=args.tv_weight,
            rho_tv=args.rho_tv,
            rho_box=args.rho_box,
            x_step=x_step,
            start=args.start,
            start_normalize=args.start_normalize,
            x_gt=x_gt,
            log_every=args.log_every,
        )
        runtime_s = time.perf_counter() - start_time
        metrics = compute_metrics(x_hat, x_gt, y, Hop)
        summary_rows.append(
            {
                "idx": int(idx),
                "method": "admm_tv",
                "runtime_s": runtime_s,
                "steps": int(args.steps),
                "inner_steps": int(args.inner_steps),
                "tv_weight": float(args.tv_weight),
                "rho_tv": float(args.rho_tv),
                "rho_box": float(args.rho_box),
                "x_step": float(x_step),
                **metrics,
            }
        )

        for hist in history:
            history_rows.append({"idx": int(idx), **hist})

        if not args.no_save_images:
            paths = save_sample_images(
                idx=int(idx),
                y=y.detach().cpu(),
                x_hat=x_hat.detach().cpu(),
                x_gt=x_gt.detach().cpu(),
                out_dir=args.out_dir,
                dpi=args.dpi,
            )
            image_rows.append({"idx": int(idx), **paths})

    summary_csv = os.path.join(args.out_dir, "admm_tv_summary.csv")
    history_csv = os.path.join(args.out_dir, "admm_tv_history.csv")
    images_csv = os.path.join(args.out_dir, "admm_tv_images.csv")
    metadata_json = os.path.join(args.out_dir, "admm_tv_metadata.json")
    curves_png = os.path.join(args.out_dir, "admm_tv_curves.png")

    write_csv(summary_csv, summary_rows)
    write_csv(history_csv, history_rows)
    write_csv(images_csv, image_rows)
    if not args.no_curves:
        save_history_plot(history_rows, indices, curves_png, dpi=args.dpi)

    with open(metadata_json, "w") as f:
        json.dump(
            {
                "config": str(Path(args.config).resolve()),
                "split": args.split,
                "indices": [int(i) for i in indices],
                "out_dir": str(Path(args.out_dir).resolve()),
                "admm_steps": int(args.steps),
                "x_inner_steps": int(args.inner_steps),
                "tv_weight": float(args.tv_weight),
                "rho_tv": float(args.rho_tv),
                "rho_box": float(args.rho_box),
                "x_step": float(x_step),
                "auto_x_step": bool(args.auto_x_step),
                "x_step_safety": float(args.x_step_safety),
                "start": args.start,
                "start_normalize": args.start_normalize,
                "summary_csv": os.path.abspath(summary_csv),
                "history_csv": os.path.abspath(history_csv),
                "images_csv": os.path.abspath(images_csv),
                "curves_png": os.path.abspath(curves_png) if not args.no_curves else None,
            },
            f,
            indent=2,
        )

    print("\n========== ADMM-TV Summary ==========")
    print(f"summary_csv: {summary_csv}")
    print(f"history_csv: {history_csv}")
    if image_rows:
        print(f"images_csv: {images_csv}")
    print(f"metadata_json: {metadata_json}")
    if not args.no_curves:
        print(f"curves_png: {curves_png}")
    print("-------------------------------------")
    for row in summary_rows:
        print(
            f"idx {int(row['idx']):>5d} | "
            f"PSNR {row['psnr']:.2f} | SSIM {row['ssim']:.4f} | "
            f"MSE {row['mse']:.6f} | DC RMSE {row['dc_rmse']:.5f} | "
            f"{row['runtime_s']:.1f}s"
        )
    print("=====================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a non-training ADMM-TV lensless reconstruction baseline.")
    parser.add_argument("--config", type=str, default="configs/a100_base.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--indices", type=str, required=True, help='Sample indices, e.g. "0,1,2" or "0 1 2".')
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "admm_tv_eval"))
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--tv_weight", type=float, default=2.0e-3)
    parser.add_argument("--rho_tv", type=float, default=5.0e-2)
    parser.add_argument("--rho_box", type=float, default=2.0e-1)
    parser.add_argument("--x_step", type=float, default=1.0e-3)
    parser.add_argument("--auto_x_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x_step_safety", type=float, default=0.35)
    parser.add_argument("--start", type=str, default="adjoint", choices=["adjoint", "measurement", "zeros"])
    parser.add_argument(
        "--start_normalize",
        type=str,
        default="max",
        choices=["max", "minmax", "absmax", "clamp", "none"],
    )
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--no_save_images", action="store_true")
    parser.add_argument("--no_curves", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    main(parser.parse_args())
