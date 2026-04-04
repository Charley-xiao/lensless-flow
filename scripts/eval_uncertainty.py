import argparse
import csv
import json
import os

import torch
import yaml
from tqdm import tqdm

from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir
from scripts._paper_eval_utils import (
    avg,
    build_test_loader_and_operator,
    compute_metrics,
    effective_batch_size,
    load_flow_runner,
    load_unet_runner,
    maybe_import_pyplot,
    pairwise_mean_distance,
    parse_int_list,
    pearson_corr,
    run_with_latent_seed,
    spearman_corr,
    tensor_to_imshow,
)


METHOD_LABELS = {
    "v_prediction": "v-prediction",
    "x_prediction": "x-prediction",
    "unet": "baseline U-Net",
}


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _top_bottom_error_ratio(uncertainty_map: torch.Tensor, error_map: torch.Tensor, frac: float = 0.1) -> float:
    u = uncertainty_map.detach().float().reshape(-1)
    e = error_map.detach().float().reshape(-1)
    if u.numel() <= 1:
        return 1.0
    k = max(1, int(u.numel() * frac))
    order = torch.argsort(u)
    low = e[order[:k]].mean()
    high = e[order[-k:]].mean()
    return float(high.item() / (low.item() + 1e-8))


def _save_uncertainty_visual(
    method_name: str,
    sample_id: int,
    y: torch.Tensor,
    x: torch.Tensor,
    samples: torch.Tensor,
    x_mean: torch.Tensor,
    uncertainty_map: torch.Tensor,
    error_map: torch.Tensor,
    out_dir: str,
    max_show_samples: int,
) -> str | None:
    plt = maybe_import_pyplot()
    if plt is None:
        return None

    show_n = min(int(max_show_samples), samples.shape[0])
    cols = max(5, show_n)
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 5.8), constrained_layout=True)

    def _clear(ax):
        ax.axis("off")

    def _put(ax, img, title: str, cmap=None):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes.flat:
        _clear(ax)
    
    # vertically flip y, x, and x_mean
    y = torch.flip(y, dims=[2])
    x = torch.flip(x, dims=[2])
    x_mean = torch.flip(x_mean, dims=[2])
    uncertainty_map = torch.flip(uncertainty_map, dims=[-2])
    error_map = torch.flip(error_map, dims=[-2])

    _put(axes[0, 0], tensor_to_imshow(y), "Lensless y")
    _put(axes[0, 1], tensor_to_imshow(x), "Ground truth")
    _put(axes[0, 2], tensor_to_imshow(x_mean), "Sample mean")
    _put(axes[0, 3], uncertainty_map.detach().float().cpu().numpy(), "Predictive std", cmap="magma")
    _put(axes[0, 4], error_map.detach().float().cpu().numpy(), "Abs error", cmap="magma")

    for draw_idx in range(show_n):
        ax = axes[1, draw_idx]
        _put(ax, tensor_to_imshow(torch.flip(samples[draw_idx : draw_idx + 1], dims=[2])), f"Sample {draw_idx + 1}")

    # fig.suptitle(f"{METHOD_LABELS[method_name]} | idx={sample_id:05d}", fontsize=14)
    out_path = os.path.join(out_dir, f"uncertainty_{method_name}_idx_{sample_id:05d}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_summary(summary_rows: list[dict], out_path: str) -> str | None:
    plt = maybe_import_pyplot()
    if plt is None:
        return None

    methods = ["v_prediction", "x_prediction", "unet"]
    labels = [METHOD_LABELS[m] for m in methods if any(r["method"] == m for r in summary_rows)]
    rows_by_method = {row["method"]: row for row in summary_rows}
    methods = [m for m in methods if m in rows_by_method]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metric_specs = [
        ("sample_psnr", "Mean Sample PSNR"),
        ("best_psnr", "Best-of-N PSNR"),
        ("pairwise_l1", "Pairwise Diversity (L1)"),
        ("uncertainty_error_spearman", "Uncertainty/Error Spearman"),
    ]

    for ax, (key, title) in zip(axes.flat, metric_specs):
        values = [float(rows_by_method[m][key]) for m in methods]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main(args):
    with open(args.config, "r") as f:
        flow_cfg = yaml.safe_load(f)

    if args.unet_config:
        with open(args.unet_config, "r") as f:
            unet_cfg = yaml.safe_load(f)
    else:
        unet_cfg = flow_cfg

    device = torch.device(flow_cfg["device"] if torch.cuda.is_available() else "cpu")
    batch_size = effective_batch_size(args.batch_size)

    _, test_dl, _, Hop, img_channels, _ = build_test_loader_and_operator(
        cfg=flow_cfg,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    methods = {
        "v_prediction": load_flow_runner(
            cfg=flow_cfg,
            ckpt_path=args.v_ckpt,
            pred_type="vanilla",
            img_channels=img_channels,
            device=device,
            steps_override=args.steps,
            dc_steps_override=args.flow_dc_steps,
            dc_step_size_override=args.flow_dc_step_size,
            disable_physics_override=args.flow_disable_physics,
        ),
        "x_prediction": load_flow_runner(
            cfg=flow_cfg,
            ckpt_path=args.x_ckpt,
            pred_type="btb",
            img_channels=img_channels,
            device=device,
            steps_override=args.steps,
            dc_steps_override=args.flow_dc_steps,
            dc_step_size_override=args.flow_dc_step_size,
            disable_physics_override=args.flow_disable_physics,
        ),
        "unet": load_unet_runner(
            cfg=unet_cfg,
            ckpt_path=args.unet_ckpt,
            img_channels=img_channels,
            device=device,
        ),
    }

    ensure_dir(args.out_dir)
    visuals_dir = os.path.join(args.out_dir, "visuals")
    ensure_dir(visuals_dir)

    if args.viz_idxs:
        viz_ids = set(parse_int_list(args.viz_idxs))
    else:
        viz_ids = set(range(max(0, int(args.save_visuals))))

    summary_acc = {
        method_name: {
            "sample_psnr": [],
            "sample_ssim": [],
            "sample_mse": [],
            "sample_psnr_std": [],
            "sample_ssim_std": [],
            "sample_mse_std": [],
            "mean_psnr": [],
            "mean_ssim": [],
            "mean_mse": [],
            "best_psnr": [],
            "best_ssim": [],
            "best_mse": [],
            "oracle_psnr_gain": [],
            "oracle_ssim_gain": [],
            "oracle_mse_gain": [],
            "pairwise_l1": [],
            "pairwise_rmse": [],
            "predictive_std": [],
            "uncertainty_error_pearson": [],
            "uncertainty_error_spearman": [],
            "top10_error_ratio": [],
        }
        for method_name in methods
    }
    per_sample_rows: list[dict] = []
    saved_visuals: list[dict] = []

    sample_id = 0
    pbar = tqdm(test_dl, desc="uncertainty eval")
    for batch_idx, (y_batch, x_batch) in enumerate(pbar):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break

        y_batch = to_nchw(y_batch).to(device)
        x_batch = to_nchw(x_batch).to(device)

        for batch_offset in range(y_batch.shape[0]):
            if args.max_samples >= 0 and sample_id >= args.max_samples:
                break

            y = y_batch[batch_offset : batch_offset + 1]
            x = x_batch[batch_offset : batch_offset + 1]
            current_sample_id = sample_id
            sample_id += 1

            for method_idx, (method_name, runner) in enumerate(methods.items()):
                draws = []
                draw_metrics = []
                for draw_idx in range(args.num_samples):
                    latent_seed = (
                        args.seed
                        + current_sample_id * 100_003
                        + draw_idx * 7_919
                        + method_idx * 173
                    )
                    x_hat = run_with_latent_seed(runner=runner, y=y, Hop=Hop, latent_seed=latent_seed)
                    draws.append(x_hat)
                    draw_metrics.append(compute_metrics(x_hat, x))

                samples = torch.cat(draws, dim=0)
                x_mean = samples.mean(dim=0, keepdim=True)
                predictive_std_map = samples.float().std(dim=0, unbiased=False)
                uncertainty_map = predictive_std_map.mean(dim=0)
                error_map = (x_mean[0] - x[0]).abs().float().mean(dim=0)

                mean_metrics = compute_metrics(x_mean, x)
                sample_psnrs = [m["psnr"] for m in draw_metrics]
                sample_ssims = [m["ssim"] for m in draw_metrics]
                sample_mses = [m["mse"] for m in draw_metrics]

                mean_sample_psnr = avg(sample_psnrs)
                mean_sample_ssim = avg(sample_ssims)
                mean_sample_mse = avg(sample_mses)
                std_sample_psnr = float(torch.tensor(sample_psnrs).std(unbiased=False).item()) if len(sample_psnrs) > 1 else 0.0
                std_sample_ssim = float(torch.tensor(sample_ssims).std(unbiased=False).item()) if len(sample_ssims) > 1 else 0.0
                std_sample_mse = float(torch.tensor(sample_mses).std(unbiased=False).item()) if len(sample_mses) > 1 else 0.0

                best_psnr = max(sample_psnrs)
                best_ssim = max(sample_ssims)
                best_mse = min(sample_mses)

                pairwise_l1 = pairwise_mean_distance(samples, p=1)
                pairwise_rmse = pairwise_mean_distance(samples, p=2)
                predictive_std = float(predictive_std_map.mean().item())
                uncertainty_error_pearson = pearson_corr(uncertainty_map, error_map)
                uncertainty_error_spearman = spearman_corr(uncertainty_map, error_map)
                top10_error_ratio = _top_bottom_error_ratio(uncertainty_map, error_map, frac=0.1)

                row = {
                    "sample_id": current_sample_id,
                    "method": method_name,
                    "num_samples": args.num_samples,
                    "sample_psnr": mean_sample_psnr,
                    "sample_ssim": mean_sample_ssim,
                    "sample_mse": mean_sample_mse,
                    "sample_psnr_std": std_sample_psnr,
                    "sample_ssim_std": std_sample_ssim,
                    "sample_mse_std": std_sample_mse,
                    "mean_psnr": mean_metrics["psnr"],
                    "mean_ssim": mean_metrics["ssim"],
                    "mean_mse": mean_metrics["mse"],
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_mse": best_mse,
                    "oracle_psnr_gain": best_psnr - mean_sample_psnr,
                    "oracle_ssim_gain": best_ssim - mean_sample_ssim,
                    "oracle_mse_gain": mean_sample_mse - best_mse,
                    "pairwise_l1": pairwise_l1,
                    "pairwise_rmse": pairwise_rmse,
                    "predictive_std": predictive_std,
                    "uncertainty_error_pearson": uncertainty_error_pearson,
                    "uncertainty_error_spearman": uncertainty_error_spearman,
                    "top10_error_ratio": top10_error_ratio,
                    "batch_index": batch_idx,
                    "batch_offset": batch_offset,
                }
                per_sample_rows.append(row)

                for key in summary_acc[method_name]:
                    summary_acc[method_name][key].append(float(row[key]))

                if current_sample_id in viz_ids:
                    visual_path = _save_uncertainty_visual(
                        method_name=method_name,
                        sample_id=current_sample_id,
                        y=y,
                        x=x,
                        samples=samples,
                        x_mean=x_mean,
                        uncertainty_map=uncertainty_map,
                        error_map=error_map,
                        out_dir=visuals_dir,
                        max_show_samples=args.viz_num_samples,
                    )
                    if visual_path is not None:
                        saved_visuals.append(
                            {
                                "sample_id": current_sample_id,
                                "method": method_name,
                                "path": visual_path,
                            }
                        )

        if args.max_samples >= 0 and sample_id >= args.max_samples:
            break

    summary_rows = []
    for method_name, acc in summary_acc.items():
        summary_rows.append(
            {
                "method": method_name,
                "num_images": len(acc["sample_psnr"]),
                "num_samples_per_image": args.num_samples,
                "sample_psnr": avg(acc["sample_psnr"]),
                "sample_ssim": avg(acc["sample_ssim"]),
                "sample_mse": avg(acc["sample_mse"]),
                "sample_psnr_std": avg(acc["sample_psnr_std"]),
                "sample_ssim_std": avg(acc["sample_ssim_std"]),
                "sample_mse_std": avg(acc["sample_mse_std"]),
                "mean_psnr": avg(acc["mean_psnr"]),
                "mean_ssim": avg(acc["mean_ssim"]),
                "mean_mse": avg(acc["mean_mse"]),
                "best_psnr": avg(acc["best_psnr"]),
                "best_ssim": avg(acc["best_ssim"]),
                "best_mse": avg(acc["best_mse"]),
                "oracle_psnr_gain": avg(acc["oracle_psnr_gain"]),
                "oracle_ssim_gain": avg(acc["oracle_ssim_gain"]),
                "oracle_mse_gain": avg(acc["oracle_mse_gain"]),
                "pairwise_l1": avg(acc["pairwise_l1"]),
                "pairwise_rmse": avg(acc["pairwise_rmse"]),
                "predictive_std": avg(acc["predictive_std"]),
                "uncertainty_error_pearson": avg(acc["uncertainty_error_pearson"]),
                "uncertainty_error_spearman": avg(acc["uncertainty_error_spearman"]),
                "top10_error_ratio": avg(acc["top10_error_ratio"]),
            }
        )

    summary_rows.sort(key=lambda row: row["method"])
    per_sample_rows.sort(key=lambda row: (row["sample_id"], row["method"]))

    summary_csv = os.path.join(args.out_dir, "uncertainty_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "uncertainty_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "uncertainty_metadata.json")
    summary_plot = os.path.join(args.out_dir, "uncertainty_summary.png")

    _write_csv(
        summary_csv,
        summary_rows,
        [
            "method",
            "num_images",
            "num_samples_per_image",
            "sample_psnr",
            "sample_ssim",
            "sample_mse",
            "sample_psnr_std",
            "sample_ssim_std",
            "sample_mse_std",
            "mean_psnr",
            "mean_ssim",
            "mean_mse",
            "best_psnr",
            "best_ssim",
            "best_mse",
            "oracle_psnr_gain",
            "oracle_ssim_gain",
            "oracle_mse_gain",
            "pairwise_l1",
            "pairwise_rmse",
            "predictive_std",
            "uncertainty_error_pearson",
            "uncertainty_error_spearman",
            "top10_error_ratio",
        ],
    )
    _write_csv(
        per_sample_csv,
        per_sample_rows,
        [
            "sample_id",
            "method",
            "num_samples",
            "sample_psnr",
            "sample_ssim",
            "sample_mse",
            "sample_psnr_std",
            "sample_ssim_std",
            "sample_mse_std",
            "mean_psnr",
            "mean_ssim",
            "mean_mse",
            "best_psnr",
            "best_ssim",
            "best_mse",
            "oracle_psnr_gain",
            "oracle_ssim_gain",
            "oracle_mse_gain",
            "pairwise_l1",
            "pairwise_rmse",
            "predictive_std",
            "uncertainty_error_pearson",
            "uncertainty_error_spearman",
            "top10_error_ratio",
            "batch_index",
            "batch_offset",
        ],
    )

    plot_path = _plot_summary(summary_rows, summary_plot)

    with open(metadata_json, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "unet_config": os.path.abspath(args.unet_config) if args.unet_config else os.path.abspath(args.config),
                "v_ckpt": os.path.abspath(args.v_ckpt),
                "x_ckpt": os.path.abspath(args.x_ckpt),
                "unet_ckpt": os.path.abspath(args.unet_ckpt),
                "device": str(device),
                "batch_size": batch_size,
                "num_workers": args.num_workers,
                "max_batches": args.max_batches,
                "max_samples": args.max_samples,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "flow_steps": int(args.steps if args.steps is not None else flow_cfg["sample"]["steps"]),
                "flow_disable_physics": bool(
                    args.flow_disable_physics
                    if args.flow_disable_physics is not None
                    else flow_cfg.get("physics", {}).get("disable_in_eval", False)
                ),
                "flow_dc_steps": int(
                    args.flow_dc_steps
                    if args.flow_dc_steps is not None
                    else flow_cfg.get("physics", {}).get("dc_steps", 0)
                ),
                "flow_dc_step_size": float(
                    args.flow_dc_step_size
                    if args.flow_dc_step_size is not None
                    else flow_cfg.get("physics", {}).get("dc_step_size", 0.0)
                ),
                "visuals_saved": saved_visuals,
                "summary_plot": plot_path,
                "num_images_evaluated": sample_id,
            },
            f,
            indent=2,
        )

    print("\n========== Uncertainty Summary ==========")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    if plot_path is not None:
        print(f"summary_plot: {plot_path}")
    print("-----------------------------------------")
    for row in summary_rows:
        print(
            f"{row['method']:>12s} | "
            f"sample PSNR {row['sample_psnr']:.2f} | "
            f"mean PSNR {row['mean_psnr']:.2f} | "
            f"best PSNR {row['best_psnr']:.2f} | "
            f"diversity(L1) {row['pairwise_l1']:.5f} | "
            f"unc/error rho {row['uncertainty_error_spearman']:.3f}"
        )
    print("=========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Flow-model config.")
    ap.add_argument("--v_ckpt", type=str, required=True)
    ap.add_argument("--x_ckpt", type=str, required=True)
    ap.add_argument("--unet_ckpt", type=str, required=True)
    ap.add_argument("--unet_config", type=str, default=None)
    ap.add_argument("--num_samples", type=int, default=8, help="Number of reconstructions per measurement.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--flow_dc_steps", type=int, default=None)
    ap.add_argument("--flow_dc_step_size", type=float, default=None)
    ap.add_argument("--flow_disable_physics", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--save_visuals", type=int, default=6, help="If --viz_idxs is omitted, save visuals for the first N images.")
    ap.add_argument("--viz_idxs", type=str, default=None, help='Optional explicit sample ids, e.g. "0,3,7".')
    ap.add_argument("--viz_num_samples", type=int, default=6)
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "uncertainty"))
    main(ap.parse_args())
