import argparse
import csv
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import (
    build_baseline_unet as build_baseline_unet_model,
    build_flow_model as build_flow_model_impl,
)
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir


def _parse_float_list(text: str) -> list[float]:
    text = text.replace(",", " ")
    return [float(x) for x in text.split() if x]


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _effective_batch_size(batch_size: int) -> int:
    return max(1, int(batch_size))


@torch.no_grad()
def _unet_forward_baseline(model, y: torch.Tensor) -> torch.Tensor:
    b = y.shape[0]
    x_t = torch.zeros_like(y)
    t = torch.zeros(b, device=y.device, dtype=y.dtype) if getattr(model, "use_time_conditioning", True) else None
    return model(x_t, y, t)


def _build_model(
    cfg,
    img_channels: int,
    im_hw: tuple[int, int],
    device: torch.device,
    checkpoint_state: dict | None = None,
):
    return build_flow_model_impl(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=checkpoint_state,
    )


def _build_baseline_model(
    cfg,
    img_channels: int,
    device: torch.device,
    checkpoint_state: dict | None = None,
):
    return build_baseline_unet_model(
        cfg=cfg,
        img_channels=img_channels,
        device=device,
        checkpoint_state=checkpoint_state,
    )


def _load_checkpoint_state(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device)


def _load_flow_runner(
    cfg,
    ckpt_path: str,
    pred_type: str,
    img_channels: int,
    im_hw: tuple[int, int],
    device: torch.device,
    steps_override: int | None,
    dc_steps_override: int | None,
    dc_step_size_override: float | None,
    disable_physics_override: bool | None,
):
    state = _load_checkpoint_state(ckpt_path, device=device)
    model = _build_model(
        cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=state,
    )
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    steps = int(steps_override if steps_override is not None else cfg["sample"]["steps"])
    dc_steps = int(
        dc_steps_override if dc_steps_override is not None else cfg.get("physics", {}).get("dc_steps", 0)
    )
    dc_step_size = float(
        dc_step_size_override
        if dc_step_size_override is not None
        else cfg.get("physics", {}).get("dc_step_size", 0.0)
    )
    disable_physics = bool(
        disable_physics_override
        if disable_physics_override is not None
        else cfg.get("physics", {}).get("disable_in_eval", False)
    )
    init_noise_std = float(cfg["sample"]["init_noise_std"])
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    @torch.no_grad()
    def _runner(y: torch.Tensor, Hop: FFTLinearConvOperator) -> torch.Tensor:
        return sample_with_physics_guidance(
            model=model,
            y=y,
            H=Hop,
            steps=steps,
            dc_step=dc_step_size,
            dc_steps=dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            clamp_x=False,
            disable_physics=disable_physics,
            pred_type=pred_type,
            dc_mode="rgb",
        )

    return _runner


def _load_unet_runner(cfg, ckpt_path: str, img_channels: int, device: torch.device):
    state = _load_checkpoint_state(ckpt_path, device=device)
    model = _build_baseline_model(cfg, img_channels=img_channels, device=device, checkpoint_state=state)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def _runner(y: torch.Tensor, _Hop: FFTLinearConvOperator) -> torch.Tensor:
        return _unet_forward_baseline(model, y)

    return _runner


def _compute_metrics(x_hat: torch.Tensor, x: torch.Tensor) -> dict[str, float]:
    x_hat_c = x_hat.clamp(0, 1).float()
    x_c = x.clamp(0, 1).float()
    return {
        "mse": float((x_hat_c - x_c).pow(2).mean().item()),
        "psnr": float(psnr(x_hat_c, x_c)),
        "ssim": float(ssim_torch(x_hat_c, x_c)),
    }


def _compute_metrics_per_sample(x_hat: torch.Tensor, x: torch.Tensor) -> list[dict[str, float]]:
    return [_compute_metrics(x_hat[i : i + 1], x[i : i + 1]) for i in range(x.shape[0])]


def _samplewise_rms(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return x.float().pow(2).mean(dim=dims, keepdim=True).sqrt()


def _samplewise_range(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    xmax = x.amax(dim=dims, keepdim=True)
    xmin = x.amin(dim=dims, keepdim=True)
    return (xmax - xmin).float()


def _perturb_measurement(
    y: torch.Tensor,
    noise_level: float,
    generator: torch.Generator,
    scale_mode: str,
    clamp_output: bool,
):
    if noise_level <= 0:
        zero = torch.zeros(y.shape[0], dtype=torch.float32, device=y.device)
        return y.clone(), zero

    if scale_mode == "rms":
        scale = _samplewise_rms(y).clamp_min(1e-8)
    elif scale_mode == "range":
        scale = _samplewise_range(y).clamp_min(1e-8)
    elif scale_mode == "absmax":
        dims = tuple(range(1, y.ndim))
        scale = y.abs().amax(dim=dims, keepdim=True).float().clamp_min(1e-8)
    elif scale_mode == "fixed":
        scale = torch.ones((y.shape[0],) + (1,) * (y.ndim - 1), device=y.device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")

    noise = torch.randn(
        y.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    noise = noise.to(y.device)
    noise = noise * (float(noise_level) * scale.float())
    y_pert = y.float() + noise
    if clamp_output:
        y_pert = y_pert.clamp(0.0, 1.0)

    actual_noise = (y_pert - y.float()).float()
    actual_rms = _samplewise_rms(actual_noise).reshape(-1)
    return y_pert.to(dtype=y.dtype), actual_rms


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _capture_rng_state():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def _restore_rng_state(cpu_state, cuda_states) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def _run_with_latent_seed(runner, y: torch.Tensor, Hop: FFTLinearConvOperator, latent_seed: int) -> torch.Tensor:
    cpu_state, cuda_states = _capture_rng_state()
    torch.manual_seed(int(latent_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(latent_seed))
    try:
        return runner(y, Hop)
    finally:
        _restore_rng_state(cpu_state, cuda_states)


def _plot_summary(summary_rows: list[dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping plot generation because matplotlib is not installed in this environment.")
        return

    methods = ["v_prediction", "x_prediction", "unet"]
    labels = {
        "v_prediction": "v-prediction",
        "x_prediction": "x-prediction",
        "unet": "baseline U-Net",
    }
    metric_titles = [
        ("noisy_ssim", "SSIM"),
        ("noisy_psnr", "PSNR"),
        ("noisy_mse", "MSE"),
        ("ssim_drop", "SSIM Drop"),
        ("psnr_drop", "PSNR Drop"),
        ("mse_increase", "MSE Increase"),
    ]

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["method"]].append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda row: float(row["noise_level"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.reshape(2, 3)

    for ax, (metric_key, title) in zip(axes.flat, metric_titles):
        for method in methods:
            rows = grouped.get(method, [])
            if not rows:
                continue
            xs = [float(row["noise_level"]) for row in rows]
            ys = [float(row[metric_key]) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=labels[method])
        ax.set_title(title)
        ax.set_xlabel("Noise level")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(args, flow_cfg, unet_cfg):
    device = torch.device(flow_cfg["device"] if torch.cuda.is_available() else "cpu")
    batch_size = _effective_batch_size(args.batch_size)

    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=flow_cfg["data"]["downsample"],
        flip_ud=flow_cfg["data"]["flip_ud"],
        batch_size=batch_size,
        num_workers=args.num_workers,
        path=flow_cfg["data"].get("path", None),
    )

    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))

    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    methods = {
        "v_prediction": _load_flow_runner(
            cfg=flow_cfg,
            ckpt_path=args.v_ckpt,
            pred_type="vanilla",
            img_channels=img_channels,
            im_hw=im_hw,
            device=device,
            steps_override=args.steps,
            dc_steps_override=args.flow_dc_steps,
            dc_step_size_override=args.flow_dc_step_size,
            disable_physics_override=args.flow_disable_physics,
        ),
        "x_prediction": _load_flow_runner(
            cfg=flow_cfg,
            ckpt_path=args.x_ckpt,
            pred_type="btb",
            img_channels=img_channels,
            im_hw=im_hw,
            device=device,
            steps_override=args.steps,
            dc_steps_override=args.flow_dc_steps,
            dc_step_size_override=args.flow_dc_step_size,
            disable_physics_override=args.flow_disable_physics,
        ),
        "unet": _load_unet_runner(
            cfg=unet_cfg,
            ckpt_path=args.unet_ckpt,
            img_channels=img_channels,
            device=device,
        ),
    }

    noise_levels = sorted(set(_parse_float_list(args.noise_levels)))
    if 0.0 not in noise_levels:
        noise_levels = [0.0] + noise_levels

    ensure_dir(args.out_dir)

    per_sample_rows: list[dict] = []
    summary_acc = {
        (method, level): {
            "clean_psnr": [],
            "clean_ssim": [],
            "clean_mse": [],
            "noisy_psnr": [],
            "noisy_ssim": [],
            "noisy_mse": [],
            "noise_rms": [],
        }
        for method in methods
        for level in noise_levels
    }

    total_batches = len(test_dl) if args.max_batches < 0 else min(len(test_dl), args.max_batches)
    sample_index = 0

    pbar = tqdm(test_dl, total=total_batches, desc="robustness eval")
    for batch_idx, (y, x) in enumerate(pbar):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break

        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)
        batch_n = y.shape[0]

        if args.max_samples >= 0:
            remaining = args.max_samples - sample_index
            if remaining <= 0:
                break
            if remaining < batch_n:
                y = y[:remaining]
                x = x[:remaining]
                batch_n = remaining

        batch_sample_ids = list(range(sample_index, sample_index + batch_n))
        sample_index += batch_n

        for repeat_idx in range(args.repeats):
            latent_seed = args.seed + batch_idx * 1_000_003 + repeat_idx * 97
            clean_outputs = {}
            clean_metrics_by_method = {}
            for method_name, runner in methods.items():
                clean_outputs[method_name] = _run_with_latent_seed(
                    runner=runner,
                    y=y,
                    Hop=Hop,
                    latent_seed=latent_seed,
                )
                clean_metrics_by_method[method_name] = _compute_metrics_per_sample(clean_outputs[method_name], x)

            for level_idx, noise_level in enumerate(noise_levels):
                seed = args.seed + batch_idx * 100_003 + level_idx * 10_007 + repeat_idx
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                y_pert, actual_noise_rms = _perturb_measurement(
                    y=y,
                    noise_level=noise_level,
                    generator=generator,
                    scale_mode=args.noise_scale,
                    clamp_output=not args.no_clamp_measurement,
                )

                for method_name, runner in methods.items():
                    if noise_level == 0.0:
                        x_hat = clean_outputs[method_name]
                    else:
                        x_hat = _run_with_latent_seed(
                            runner=runner,
                            y=y_pert,
                            Hop=Hop,
                            latent_seed=latent_seed,
                        )

                    noisy_metrics_list = _compute_metrics_per_sample(x_hat, x)
                    clean_metrics_list = clean_metrics_by_method[method_name]

                    key = (method_name, noise_level)
                    for sample_offset, sample_id in enumerate(batch_sample_ids):
                        clean_metrics = clean_metrics_list[sample_offset]
                        noisy_metrics = noisy_metrics_list[sample_offset]
                        noise_rms = float(actual_noise_rms[sample_offset].item())

                        summary_acc[key]["clean_psnr"].append(clean_metrics["psnr"])
                        summary_acc[key]["clean_ssim"].append(clean_metrics["ssim"])
                        summary_acc[key]["clean_mse"].append(clean_metrics["mse"])
                        summary_acc[key]["noisy_psnr"].append(noisy_metrics["psnr"])
                        summary_acc[key]["noisy_ssim"].append(noisy_metrics["ssim"])
                        summary_acc[key]["noisy_mse"].append(noisy_metrics["mse"])
                        summary_acc[key]["noise_rms"].append(noise_rms)

                        per_sample_rows.append(
                            {
                                "sample_id": sample_id,
                                "repeat": repeat_idx,
                                "noise_level": noise_level,
                                "noise_scale": args.noise_scale,
                                "method": method_name,
                                "clean_psnr": clean_metrics["psnr"],
                                "clean_ssim": clean_metrics["ssim"],
                                "clean_mse": clean_metrics["mse"],
                                "noisy_psnr": noisy_metrics["psnr"],
                                "noisy_ssim": noisy_metrics["ssim"],
                                "noisy_mse": noisy_metrics["mse"],
                                "psnr_drop": clean_metrics["psnr"] - noisy_metrics["psnr"],
                                "ssim_drop": clean_metrics["ssim"] - noisy_metrics["ssim"],
                                "mse_increase": noisy_metrics["mse"] - clean_metrics["mse"],
                                "actual_noise_rms": noise_rms,
                                "batch_index": batch_idx,
                                "batch_sample_offset": sample_offset,
                            }
                        )

        if args.max_samples >= 0 and sample_index >= args.max_samples:
            break

    summary_rows: list[dict] = []
    for method_name in methods:
        for noise_level in noise_levels:
            acc = summary_acc[(method_name, noise_level)]
            clean_psnr = _avg(acc["clean_psnr"])
            clean_ssim = _avg(acc["clean_ssim"])
            clean_mse = _avg(acc["clean_mse"])
            noisy_psnr = _avg(acc["noisy_psnr"])
            noisy_ssim = _avg(acc["noisy_ssim"])
            noisy_mse = _avg(acc["noisy_mse"])

            summary_rows.append(
                {
                    "method": method_name,
                    "noise_level": noise_level,
                    "noise_scale": args.noise_scale,
                    "actual_noise_rms": _avg(acc["noise_rms"]),
                    "clean_psnr": clean_psnr,
                    "clean_ssim": clean_ssim,
                    "clean_mse": clean_mse,
                    "noisy_psnr": noisy_psnr,
                    "noisy_ssim": noisy_ssim,
                    "noisy_mse": noisy_mse,
                    "psnr_drop": clean_psnr - noisy_psnr,
                    "ssim_drop": clean_ssim - noisy_ssim,
                    "mse_increase": noisy_mse - clean_mse,
                    "num_evals": len(acc["noisy_psnr"]),
                }
            )

    summary_rows.sort(key=lambda row: (row["method"], float(row["noise_level"])))
    per_sample_rows.sort(key=lambda row: (row["sample_id"], row["method"], float(row["noise_level"]), row["repeat"]))

    summary_csv = os.path.join(args.out_dir, "robustness_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "robustness_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "robustness_metadata.json")
    plot_path = os.path.join(args.out_dir, "robustness_curves.png")

    _write_csv(
        summary_csv,
        summary_rows,
        [
            "method",
            "noise_level",
            "noise_scale",
            "actual_noise_rms",
            "clean_psnr",
            "clean_ssim",
            "clean_mse",
            "noisy_psnr",
            "noisy_ssim",
            "noisy_mse",
            "psnr_drop",
            "ssim_drop",
            "mse_increase",
            "num_evals",
        ],
    )
    _write_csv(
        per_sample_csv,
        per_sample_rows,
        [
            "sample_id",
            "repeat",
            "noise_level",
            "noise_scale",
            "method",
            "clean_psnr",
            "clean_ssim",
            "clean_mse",
            "noisy_psnr",
            "noisy_ssim",
            "noisy_mse",
            "psnr_drop",
            "ssim_drop",
            "mse_increase",
            "actual_noise_rms",
            "batch_index",
            "batch_sample_offset",
        ],
    )

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
                "repeats": args.repeats,
                "seed": args.seed,
                "noise_scale": args.noise_scale,
                "noise_levels": noise_levels,
                "paired_latent_seed": True,
                "measurement_clamped": not args.no_clamp_measurement,
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
                "num_samples_evaluated": sample_index,
            },
            f,
            indent=2,
        )

    _plot_summary(summary_rows, plot_path)

    print("\n========== Robustness Summary ==========")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    print(f"plot: {plot_path}")
    print("----------------------------------------")
    for row in summary_rows:
        print(
            f"{row['method']:>12s} | noise={float(row['noise_level']):.4f} | "
            f"SSIM {row['clean_ssim']:.4f}->{row['noisy_ssim']:.4f} "
            f"(drop {row['ssim_drop']:.4f}) | "
            f"PSNR {row['clean_psnr']:.2f}->{row['noisy_psnr']:.2f} "
            f"(drop {row['psnr_drop']:.2f}) | "
            f"MSE {row['clean_mse']:.6f}->{row['noisy_mse']:.6f} "
            f"(+{row['mse_increase']:.6f})"
        )
    print("========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Flow-model config.")
    ap.add_argument("--v_ckpt", type=str, required=True, help="Velocity-prediction checkpoint.")
    ap.add_argument("--x_ckpt", type=str, required=True, help="Image-prediction checkpoint.")
    ap.add_argument("--unet_ckpt", type=str, required=True, help="Baseline U-Net checkpoint.")
    ap.add_argument("--unet_config", type=str, default=None, help="Optional baseline U-Net config.")
    ap.add_argument("--noise_levels", type=str, default="0.00,0.01,0.02,0.05,0.10")
    ap.add_argument("--noise_scale", type=str, default="rms", choices=["rms", "range", "absmax", "fixed"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=None, help="Optional override for flow ODE steps.")
    ap.add_argument("--flow_dc_steps", type=int, default=None)
    ap.add_argument("--flow_dc_step_size", type=float, default=None)
    ap.add_argument(
        "--flow_disable_physics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override physics guidance for flow models during robustness eval.",
    )
    ap.add_argument(
        "--no_clamp_measurement",
        action="store_true",
        help="Do not clamp perturbed measurements back to [0,1].",
    )
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "robustness"))
    args, overrides = ap.parse_known_args()
    flow_cfg = load_config(args.config, overrides)
    unet_cfg = load_config(args.unet_config, overrides) if args.unet_config else flow_cfg
    main(args, flow_cfg, unet_cfg)
