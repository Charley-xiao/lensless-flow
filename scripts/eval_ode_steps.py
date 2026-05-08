import argparse
import csv
import json
import os
import time

import torch
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.measurement_source import source_sampler_kwargs_from_cfg, source_sigma0_from_cfg
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict, resolve_model_name
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir, set_seed


SUMMARY_FIELDS = (
    "steps",
    "num_evals",
    "psnr",
    "ssim",
    "lpips",
    "mse",
    "dc_rmse",
    "runtime_ms_per_sample",
)
PER_SAMPLE_FIELDS = (
    "sample_id",
    "repeat",
    "steps",
    "psnr",
    "ssim",
    "lpips",
    "mse",
    "dc_rmse",
    "runtime_ms_per_sample",
    "batch_idx",
    "batch_offset",
)
PLOT_METRICS = (
    ("psnr", "PSNR vs ODE Steps", "PSNR (dB)", "higher is better", "max"),
    ("ssim", "SSIM vs ODE Steps", "SSIM", "higher is better", "max"),
    ("lpips", "LPIPS vs ODE Steps", "LPIPS", "lower is better", "min"),
)


def _parse_int_list(text: str) -> list[int]:
    values = []
    for token in str(text).replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("At least one ODE step count is required.")
    return sorted(set(values))


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _write_csv(path: str, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _capture_rng_state():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def _restore_rng_state(cpu_state, cuda_states) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _run_with_seed(fn, seed: int):
    cpu_state, cuda_states = _capture_rng_state()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        return fn()
    finally:
        _restore_rng_state(cpu_state, cuda_states)


def _ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    channels = int(x.shape[1])
    if channels == 3:
        return x
    if channels == 1:
        return x.repeat(1, 3, 1, 1)
    if channels == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    if channels > 3:
        return x[:, :3]
    raise ValueError(f"Unsupported channel count for LPIPS: {channels}")


def _prepare_lpips_input(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_three_channels(x.clamp(0, 1).float())
    return x * 2.0 - 1.0


def _build_lpips_metric(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "ODE step sweep requires the `lpips` package. Install the project requirements first."
        ) from exc

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for param in metric.parameters():
        param.requires_grad_(False)
    return metric


def _load_model_and_metadata(cfg: dict, ckpt_path: str, img_channels: int, im_hw: tuple[int, int], device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model = build_flow_model(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=state,
    )
    load_checkpoint_state_dict(model, state)
    model.eval()

    pred_type = str(
        state.get("mode", cfg.get("train", {}).get("mode", "btb"))
        if isinstance(state, dict)
        else cfg.get("train", {}).get("mode", "btb")
    ).lower()
    if pred_type not in {"btb", "vanilla"}:
        raise ValueError(f"Unknown pred_type={pred_type}")

    matcher_name = normalize_flow_matcher_name(
        state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified"))
        if isinstance(state, dict)
        else cfg.get("cfm", {}).get("matcher", "rectified")
    )
    model_name = resolve_model_name(cfg, checkpoint_state=state if isinstance(state, dict) else None)
    return model, pred_type, matcher_name, model_name


def _compute_metrics(
    *,
    x_hat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    Hop: FFTLinearConvOperator,
    lpips_metric,
) -> dict[str, float]:
    x_hat_c = x_hat.clamp(0, 1).float()
    x_c = x.clamp(0, 1).float()
    mse = float((x_hat_c - x_c).pow(2).mean().item())
    residual = Hop.forward(x_hat.float()) - y.float()
    dc_rmse = float(residual.pow(2).mean().sqrt().item())
    lpips_value = float(lpips_metric(_prepare_lpips_input(x_hat_c), _prepare_lpips_input(x_c)).reshape(-1)[0].item())
    return {
        "psnr": float(psnr(x_hat_c, x_c)),
        "ssim": float(ssim_torch(x_hat_c, x_c).item()),
        "lpips": lpips_value,
        "mse": mse,
        "dc_rmse": dc_rmse,
    }


def _summarize(per_sample_rows: list[dict], steps_list: list[int]) -> list[dict]:
    summary_rows = []
    for steps in steps_list:
        rows = [row for row in per_sample_rows if int(row["steps"]) == int(steps)]
        summary = {
            "steps": int(steps),
            "num_evals": len(rows),
        }
        for field in SUMMARY_FIELDS:
            if field in summary:
                continue
            summary[field] = _avg([float(row[field]) for row in rows])
        summary_rows.append(summary)
    return summary_rows


def _best_by_metric(summary_rows: list[dict]) -> dict[str, dict]:
    rows = [row for row in summary_rows if int(row.get("num_evals", 0)) > 0]
    best_rows = {}
    for key, _, _, _, direction in PLOT_METRICS:
        if not rows:
            continue
        choose = min if direction == "min" else max
        best_rows[key] = choose(rows, key=lambda row: float(row[key]))
    return best_rows


def _plot_summary(summary_rows: list[dict], out_path: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    xs = [int(row["steps"]) for row in summary_rows]
    best_rows = _best_by_metric(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, (key, title, ylabel, note, _) in zip(axes, PLOT_METRICS):
        ys = [float(row[key]) for row in summary_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.2)
        best = best_rows.get(key)
        if best is not None:
            best_x = int(best["steps"])
            best_y = float(best[key])
            ax.scatter([best_x], [best_y], marker="*", s=120, color="tab:red", zorder=3)
            ax.annotate(
                f"best {best_x}",
                xy=(best_x, best_y),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=9,
                color="tab:red",
            )
        ax.set_title(f"{title}\n{note}")
        ax.set_xlabel("ODE steps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_individual(summary_rows: list[dict], out_dir: str) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    xs = [int(row["steps"]) for row in summary_rows]
    best_rows = _best_by_metric(summary_rows)
    paths = {}
    ensure_dir(out_dir)
    for key, title, ylabel, note, _ in PLOT_METRICS:
        ys = [float(row[key]) for row in summary_rows]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot(xs, ys, marker="o", linewidth=2.2)
        best = best_rows.get(key)
        if best is not None:
            best_x = int(best["steps"])
            best_y = float(best[key])
            ax.scatter([best_x], [best_y], marker="*", s=120, color="tab:red", zorder=3)
            ax.annotate(
                f"best {best_x}",
                xy=(best_x, best_y),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=9,
                color="tab:red",
            )
        ax.set_title(f"{title} ({note})")
        ax.set_xlabel("ODE steps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"ode_steps_{key}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[key] = path
    return paths


@torch.no_grad()
def main(args, cfg: dict) -> None:
    set_seed(int(args.seed))
    device = torch.device(args.device if args.device else (cfg["device"] if torch.cuda.is_available() else "cpu"))
    steps_list = _parse_int_list(args.steps)

    test_ds, test_dl = make_dataloader(
        split=args.split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
    )

    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))
    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    model, pred_type, matcher_name, model_name = _load_model_and_metadata(
        cfg=cfg,
        ckpt_path=args.ckpt,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
    )

    init_noise_std = source_sigma0_from_cfg(cfg)
    source_kwargs = source_sampler_kwargs_from_cfg(cfg)
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))
    disable_physics = bool(
        args.disable_physics if args.disable_physics is not None else cfg.get("physics", {}).get("disable_in_eval", False)
    )
    dc_steps = int(args.dc_steps if args.dc_steps is not None else cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(args.dc_step_size if args.dc_step_size is not None else cfg.get("physics", {}).get("dc_step_size", 0.0))

    lpips_metric = _build_lpips_metric(device)
    per_sample_rows = []
    sample_id = 0

    ensure_dir(args.out_dir)
    pbar = tqdm(test_dl, desc=f"ODE step sweep [{pred_type}, {matcher_name}]")
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

        for batch_offset in range(y_batch.shape[0]):
            y = y_batch[batch_offset : batch_offset + 1]
            x = x_batch[batch_offset : batch_offset + 1]
            current_sample_id = sample_id + batch_offset

            for repeat in range(args.repeats):
                latent_seed = int(args.seed + current_sample_id * 100_003 + repeat * 997)
                for steps in steps_list:
                    def _sample():
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start_time = time.perf_counter()
                        x_hat_inner = sample_with_physics_guidance(
                            model=model,
                            y=y,
                            H=Hop,
                            steps=int(steps),
                            dc_step=dc_step,
                            dc_steps=dc_steps,
                            init_noise_std=init_noise_std,
                            denom_min=denom_min,
                            clamp_x=False,
                            disable_physics=disable_physics,
                            pred_type=pred_type,
                            dc_mode="rgb",
                            solver=args.solver,
                            **source_kwargs,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        runtime_ms = (time.perf_counter() - start_time) * 1000.0
                        return x_hat_inner, runtime_ms

                    x_hat, runtime_ms = _run_with_seed(_sample, latent_seed)
                    metrics = _compute_metrics(
                        x_hat=x_hat,
                        x=x,
                        y=y,
                        Hop=Hop,
                        lpips_metric=lpips_metric,
                    )
                    per_sample_rows.append(
                        {
                            "sample_id": current_sample_id,
                            "repeat": repeat,
                            "steps": int(steps),
                            "runtime_ms_per_sample": float(runtime_ms),
                            "batch_idx": batch_idx,
                            "batch_offset": batch_offset,
                            **metrics,
                        }
                    )

        sample_id += int(y_batch.shape[0])
        latest = [row for row in per_sample_rows if int(row["steps"]) == steps_list[-1]]
        if latest:
            pbar.set_postfix(psnr=f"{_avg([float(row['psnr']) for row in latest]):.2f}")

    summary_rows = _summarize(per_sample_rows, steps_list)
    best_rows = _best_by_metric(summary_rows)
    summary_csv = os.path.join(args.out_dir, "ode_steps_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "ode_steps_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "ode_steps_metadata.json")
    combined_plot = os.path.join(args.out_dir, "ode_steps_curves.png")

    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_csv(per_sample_csv, PER_SAMPLE_FIELDS, per_sample_rows)
    combined_plot = _plot_summary(summary_rows, combined_plot)
    individual_plots = _plot_individual(summary_rows, args.out_dir)

    with open(metadata_json, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "ckpt": os.path.abspath(args.ckpt),
                "model_name": model_name,
                "pred_type": pred_type,
                "matcher": matcher_name,
                "split": args.split,
                "steps": steps_list,
                "samples": sample_id,
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "solver": args.solver,
                "disable_physics": disable_physics,
                "dc_steps": dc_steps,
                "dc_step": dc_step,
                "source": cfg.get("cfm", {}).get("source", {}),
                "summary_csv": summary_csv,
                "per_sample_csv": per_sample_csv,
                "combined_plot": combined_plot,
                "individual_plots": individual_plots,
                "best_steps": {
                    key: {
                        "steps": int(row["steps"]),
                        key: float(row[key]),
                    }
                    for key, row in best_rows.items()
                },
            },
            f,
            indent=2,
        )

    print("\n========== ODE Step Sweep ==========")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    if combined_plot:
        print(f"combined_plot: {combined_plot}")
    for key, path in individual_plots.items():
        print(f"{key}_plot: {path}")
    print("------------------------------------")
    print("steps  psnr    ssim    lpips    mse       dc_rmse   ms/sample")
    for row in summary_rows:
        print(
            f"{int(row['steps']):>5d} "
            f"{float(row['psnr']):>6.2f} "
            f"{float(row['ssim']):>7.4f} "
            f"{float(row['lpips']):>8.4f} "
            f"{float(row['mse']):>9.6f} "
            f"{float(row['dc_rmse']):>9.5f} "
            f"{float(row['runtime_ms_per_sample']):>9.1f}"
        )
    if best_rows:
        print("------------------------------------")
        print("Best steps by metric:")
        for key, row in best_rows.items():
            print(f"{key:>5s}: steps={int(row['steps'])}, {key}={float(row[key]):.4f}")
    print("====================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--steps", type=str, default="1,2,4,8,12,16,24,32,40,64")
    ap.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--disable_physics", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--dc_steps", type=int, default=None)
    ap.add_argument("--dc_step_size", type=float, default=None)
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "ode_steps"))
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    main(args, cfg)
