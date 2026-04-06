import argparse
import csv
import json
import os

import torch
import yaml
from tqdm import tqdm

from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir
from scripts._paper_eval_utils import (
    avg,
    build_test_loader_and_operator,
    capture_rng_state,
    compute_metrics,
    effective_batch_size,
    load_flow_runner,
    load_unet_runner,
    maybe_import_pyplot,
    parse_float_list,
    parse_int_list,
    restore_rng_state,
    run_with_latent_seed,
    samplewise_range,
    samplewise_rms,
    zero_fill_shift,
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


def _parse_corruption_list(text: str) -> list[str]:
    text = text.replace(",", " ")
    values = [x.strip() for x in text.split() if x.strip()]
    if "all" in values:
        return ["exposure_scale", "background_offset", "measurement_noise", "poisson_peak", "measurement_shift", "psf_shift"]
    return values


def _parse_poisson_levels(text: str) -> list[str | int]:
    values = []
    for token in text.replace(",", " ").split():
        token = token.strip().lower()
        if not token:
            continue
        if token in ["clean", "none", "inf", "infinite"]:
            values.append("clean")
        else:
            values.append(int(token))
    return values


def _shift_from_level(level: int, axis: str) -> tuple[int, int]:
    axis = axis.lower()
    if axis == "x":
        return 0, int(level)
    if axis == "y":
        return int(level), 0
    if axis == "diag":
        return int(level), int(level)
    raise ValueError(f"Unknown shift axis: {axis}")


def _poisson_corrupt(y: torch.Tensor, peak: int, seed: int, clamp_output: bool) -> torch.Tensor:
    cpu_state, cuda_states = capture_rng_state()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        y_cpu = y.detach().float().cpu().clamp_min(0.0)
        y_counts = y_cpu * float(peak)
        y_poisson = torch.poisson(y_counts) / float(peak)
        if clamp_output:
            y_poisson = y_poisson.clamp(0.0, 1.0)
        return y_poisson.to(y.device, dtype=y.dtype)
    finally:
        restore_rng_state(cpu_state, cuda_states)


def _measurement_noise_corrupt(
    y: torch.Tensor,
    noise_level: float,
    seed: int,
    scale_mode: str,
    clamp_output: bool,
) -> tuple[torch.Tensor, float]:
    if float(noise_level) <= 0.0:
        return y.clone(), 0.0

    cpu_state, cuda_states = capture_rng_state()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        y_float = y.detach().float()
        if scale_mode == "rms":
            scale = samplewise_rms(y_float).clamp_min(1e-8)
        elif scale_mode == "range":
            scale = samplewise_range(y_float).clamp_min(1e-8)
        elif scale_mode == "absmax":
            dims = tuple(range(1, y.ndim))
            scale = y_float.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
        elif scale_mode == "fixed":
            scale = torch.ones((y.shape[0],) + (1,) * (y.ndim - 1), device=y.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown measurement_noise_scale={scale_mode}")

        noise = torch.randn(y.shape, device="cpu", dtype=torch.float32).to(y.device)
        noise = noise * (float(noise_level) * scale.float())
        y_corr = y_float + noise
        if clamp_output:
            y_corr = y_corr.clamp(0.0, 1.0)

        actual_rms = float((y_corr - y_float).pow(2).mean().sqrt().item())
        return y_corr.to(dtype=y.dtype), actual_rms
    finally:
        restore_rng_state(cpu_state, cuda_states)


def _build_shifted_operator(psf: torch.Tensor, im_hw: tuple[int, int], shift_y: int, shift_x: int, device: torch.device):
    shifted_psf = zero_fill_shift(psf, shift_y=shift_y, shift_x=shift_x)
    return FFTLinearConvOperator(psf=shifted_psf.to(device), im_hw=im_hw).to(device), shifted_psf


def _corruption_levels(args) -> dict[str, list]:
    return {
        "exposure_scale": parse_float_list(args.exposure_levels),
        "background_offset": parse_float_list(args.offset_levels),
        "measurement_noise": parse_float_list(args.measurement_noise_levels),
        "poisson_peak": _parse_poisson_levels(args.poisson_peaks),
        "measurement_shift": parse_int_list(args.shift_levels),
        "psf_shift": parse_int_list(args.psf_shift_levels),
    }


def _apply_corruption(
    corruption: str,
    level,
    y: torch.Tensor,
    x: torch.Tensor,
    nominal_H: FFTLinearConvOperator,
    psf: torch.Tensor,
    im_hw: tuple[int, int],
    device: torch.device,
    seed: int,
    clamp_measurement: bool,
    shift_axis: str,
    measurement_noise_scale: str,
):
    y_corr = y.clone()
    Hop_corr = nominal_H
    level_label = str(level)

    if corruption == "exposure_scale":
        scale = float(level)
        y_corr = y.float() * scale
        if clamp_measurement:
            y_corr = y_corr.clamp(0.0, 1.0)
        actual_rms = float((y_corr - y.float()).pow(2).mean().sqrt().item())
        return y_corr.to(dtype=y.dtype), Hop_corr, level_label, actual_rms

    if corruption == "background_offset":
        offset = float(level)
        y_corr = y.float() + offset
        if clamp_measurement:
            y_corr = y_corr.clamp(0.0, 1.0)
        actual_rms = float((y_corr - y.float()).pow(2).mean().sqrt().item())
        return y_corr.to(dtype=y.dtype), Hop_corr, level_label, actual_rms

    if corruption == "measurement_noise":
        noise_level = float(level)
        y_corr, actual_rms = _measurement_noise_corrupt(
            y=y,
            noise_level=noise_level,
            seed=seed,
            scale_mode=measurement_noise_scale,
            clamp_output=clamp_measurement,
        )
        return y_corr, Hop_corr, level_label, actual_rms

    if corruption == "poisson_peak":
        if level == "clean":
            return y.clone(), Hop_corr, "clean", 0.0
        peak = int(level)
        y_corr = _poisson_corrupt(y=y, peak=peak, seed=seed, clamp_output=clamp_measurement)
        actual_rms = float((y_corr.float() - y.float()).pow(2).mean().sqrt().item())
        return y_corr, Hop_corr, str(peak), actual_rms

    if corruption == "measurement_shift":
        pixels = int(level)
        dy, dx = _shift_from_level(pixels, shift_axis)
        y_corr = zero_fill_shift(y, shift_y=dy, shift_x=dx)
        actual_rms = float((y_corr.float() - y.float()).pow(2).mean().sqrt().item())
        return y_corr, Hop_corr, f"{pixels}px", actual_rms

    if corruption == "psf_shift":
        pixels = int(level)
        dy, dx = _shift_from_level(pixels, shift_axis)
        Hop_corr, shifted_psf = _build_shifted_operator(psf=psf, im_hw=im_hw, shift_y=dy, shift_x=dx, device=device)
        if pixels == 0:
            return y.clone(), Hop_corr, "0px", 0.0
        nominal_meas = nominal_H.forward(x.float())
        shifted_meas = Hop_corr.forward(x.float())
        actual_rms = float((shifted_meas - nominal_meas).pow(2).mean().sqrt().item())
        return y.clone(), Hop_corr, f"{pixels}px", actual_rms

    raise ValueError(f"Unsupported corruption: {corruption}")


def _plot_summary(summary_rows: list[dict], out_path: str) -> str | None:
    plt = maybe_import_pyplot()
    if plt is None:
        return None

    corruptions = []
    for row in summary_rows:
        if row["corruption"] not in corruptions:
            corruptions.append(row["corruption"])

    fig, axes = plt.subplots(3, len(corruptions), figsize=(5 * len(corruptions), 12), constrained_layout=True)
    if len(corruptions) == 1:
        axes = axes.reshape(3, 1)

    for col, corruption in enumerate(corruptions):
        rows = [row for row in summary_rows if row["corruption"] == corruption]
        level_order = []
        for row in sorted(rows, key=lambda r: int(r["severity_rank"])):
            if row["level_label"] not in level_order:
                level_order.append(row["level_label"])

        for method_name in ["v_prediction", "x_prediction", "unet"]:
            method_rows = sorted(
                [row for row in rows if row["method"] == method_name],
                key=lambda r: int(r["severity_rank"]),
            )
            if not method_rows:
                continue
            xs = list(range(len(method_rows)))
            axes[0, col].plot(xs, [float(r["noisy_ssim"]) for r in method_rows], marker="o", label=METHOD_LABELS[method_name])
            axes[1, col].plot(xs, [float(r["noisy_psnr"]) for r in method_rows], marker="o", label=METHOD_LABELS[method_name])
            axes[2, col].plot(xs, [float(r["noisy_mse"]) for r in method_rows], marker="o", label=METHOD_LABELS[method_name])

        for row_idx, title in enumerate(["SSIM", "PSNR", "MSE"]):
            axes[row_idx, col].set_title(f"{corruption}: {title}")
            axes[row_idx, col].set_xticks(list(range(len(level_order))), level_order, rotation=25)
            axes[row_idx, col].grid(True, alpha=0.3)

    axes[0, 0].legend()
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

    _, test_dl, psf, Hop, img_channels, im_hw = build_test_loader_and_operator(
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

    corruption_levels = _corruption_levels(args)
    corruptions = _parse_corruption_list(args.corruptions)

    ensure_dir(args.out_dir)

    summary_acc = {
        (corruption, str(level), method_name): {
            "clean_psnr": [],
            "clean_ssim": [],
            "clean_mse": [],
            "noisy_psnr": [],
            "noisy_ssim": [],
            "noisy_mse": [],
            "psnr_drop": [],
            "ssim_drop": [],
            "mse_increase": [],
            "perturb_rms": [],
        }
        for corruption in corruptions
        for level in corruption_levels[corruption]
        for method_name in methods
    }
    per_sample_rows = []

    sample_id = 0
    pbar = tqdm(test_dl, desc="physical robustness")
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

            for repeat_idx in range(args.repeats):
                latent_seed = args.seed + current_sample_id * 100_003 + repeat_idx * 997
                clean_outputs = {}
                clean_metrics = {}
                for method_idx, (method_name, runner) in enumerate(methods.items()):
                    clean_outputs[method_name] = run_with_latent_seed(
                        runner=runner,
                        y=y,
                        Hop=Hop,
                        latent_seed=latent_seed + method_idx * 53,
                    )
                    clean_metrics[method_name] = compute_metrics(clean_outputs[method_name], x)

                for corruption in corruptions:
                    levels = corruption_levels[corruption]
                    for severity_rank, level in enumerate(levels):
                        corrupt_seed = args.seed + current_sample_id * 17_389 + repeat_idx * 409 + severity_rank * 31
                        y_corr, Hop_corr, level_label, perturb_rms = _apply_corruption(
                            corruption=corruption,
                            level=level,
                            y=y,
                            x=x,
                            nominal_H=Hop,
                            psf=psf,
                            im_hw=im_hw,
                            device=device,
                            seed=corrupt_seed,
                            clamp_measurement=not args.no_clamp_measurement,
                            shift_axis=args.shift_axis,
                            measurement_noise_scale=args.measurement_noise_scale,
                        )

                        for method_idx, (method_name, runner) in enumerate(methods.items()):
                            if (corruption != "psf_shift" and level_label in ["1.0", "0.0", "clean", "0px"]) or (
                                corruption == "psf_shift" and level_label == "0px"
                            ):
                                x_hat = clean_outputs[method_name]
                            else:
                                x_hat = run_with_latent_seed(
                                    runner=runner,
                                    y=y_corr,
                                    Hop=Hop_corr,
                                    latent_seed=latent_seed + method_idx * 53,
                                )

                            noisy_metrics = compute_metrics(x_hat, x)
                            base = clean_metrics[method_name]
                            key = (corruption, str(level), method_name)
                            summary_acc[key]["clean_psnr"].append(base["psnr"])
                            summary_acc[key]["clean_ssim"].append(base["ssim"])
                            summary_acc[key]["clean_mse"].append(base["mse"])
                            summary_acc[key]["noisy_psnr"].append(noisy_metrics["psnr"])
                            summary_acc[key]["noisy_ssim"].append(noisy_metrics["ssim"])
                            summary_acc[key]["noisy_mse"].append(noisy_metrics["mse"])
                            summary_acc[key]["psnr_drop"].append(base["psnr"] - noisy_metrics["psnr"])
                            summary_acc[key]["ssim_drop"].append(base["ssim"] - noisy_metrics["ssim"])
                            summary_acc[key]["mse_increase"].append(noisy_metrics["mse"] - base["mse"])
                            summary_acc[key]["perturb_rms"].append(float(perturb_rms))

                            per_sample_rows.append(
                                {
                                    "sample_id": current_sample_id,
                                    "repeat": repeat_idx,
                                    "method": method_name,
                                    "corruption": corruption,
                                    "level_value": str(level),
                                    "level_label": level_label,
                                    "severity_rank": severity_rank,
                                    "clean_psnr": base["psnr"],
                                    "clean_ssim": base["ssim"],
                                    "clean_mse": base["mse"],
                                    "noisy_psnr": noisy_metrics["psnr"],
                                    "noisy_ssim": noisy_metrics["ssim"],
                                    "noisy_mse": noisy_metrics["mse"],
                                    "psnr_drop": base["psnr"] - noisy_metrics["psnr"],
                                    "ssim_drop": base["ssim"] - noisy_metrics["ssim"],
                                    "mse_increase": noisy_metrics["mse"] - base["mse"],
                                    "perturb_rms": float(perturb_rms),
                                    "batch_index": batch_idx,
                                    "batch_offset": batch_offset,
                                }
                            )

        if args.max_samples >= 0 and sample_id >= args.max_samples:
            break

    summary_rows = []
    for corruption in corruptions:
        for severity_rank, level in enumerate(corruption_levels[corruption]):
            for method_name in methods:
                acc = summary_acc[(corruption, str(level), method_name)]
                level_label = str(level)
                if corruption == "poisson_peak" and level == "clean":
                    level_label = "clean"
                elif corruption in ["measurement_shift", "psf_shift"]:
                    level_label = f"{int(level)}px"

                summary_rows.append(
                    {
                        "corruption": corruption,
                        "level_value": str(level),
                        "level_label": level_label,
                        "severity_rank": severity_rank,
                        "method": method_name,
                        "num_evals": len(acc["noisy_psnr"]),
                        "perturb_rms": avg(acc["perturb_rms"]),
                        "clean_psnr": avg(acc["clean_psnr"]),
                        "clean_ssim": avg(acc["clean_ssim"]),
                        "clean_mse": avg(acc["clean_mse"]),
                        "noisy_psnr": avg(acc["noisy_psnr"]),
                        "noisy_ssim": avg(acc["noisy_ssim"]),
                        "noisy_mse": avg(acc["noisy_mse"]),
                        "psnr_drop": avg(acc["psnr_drop"]),
                        "ssim_drop": avg(acc["ssim_drop"]),
                        "mse_increase": avg(acc["mse_increase"]),
                    }
                )

    summary_rows.sort(key=lambda row: (row["corruption"], int(row["severity_rank"]), row["method"]))
    per_sample_rows.sort(key=lambda row: (row["sample_id"], row["corruption"], int(row["severity_rank"]), row["method"], row["repeat"]))

    summary_csv = os.path.join(args.out_dir, "physical_robustness_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "physical_robustness_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "physical_robustness_metadata.json")
    plot_path = os.path.join(args.out_dir, "physical_robustness_curves.png")

    _write_csv(
        summary_csv,
        summary_rows,
        [
            "corruption",
            "level_value",
            "level_label",
            "severity_rank",
            "method",
            "num_evals",
            "perturb_rms",
            "clean_psnr",
            "clean_ssim",
            "clean_mse",
            "noisy_psnr",
            "noisy_ssim",
            "noisy_mse",
            "psnr_drop",
            "ssim_drop",
            "mse_increase",
        ],
    )
    _write_csv(
        per_sample_csv,
        per_sample_rows,
        [
            "sample_id",
            "repeat",
            "method",
            "corruption",
            "level_value",
            "level_label",
            "severity_rank",
            "clean_psnr",
            "clean_ssim",
            "clean_mse",
            "noisy_psnr",
            "noisy_ssim",
            "noisy_mse",
            "psnr_drop",
            "ssim_drop",
            "mse_increase",
            "perturb_rms",
            "batch_index",
            "batch_offset",
        ],
    )

    summary_plot = _plot_summary(summary_rows, plot_path)

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
                "corruptions": corruptions,
                "corruption_levels": {k: [str(v) for v in vals] for k, vals in corruption_levels.items() if k in corruptions},
                "measurement_clamped": not args.no_clamp_measurement,
                "measurement_noise_scale": args.measurement_noise_scale,
                "shift_axis": args.shift_axis,
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
                "summary_plot": summary_plot,
                "num_images_evaluated": sample_id,
            },
            f,
            indent=2,
        )

    print("\n======= Physical Robustness Summary =======")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    if summary_plot is not None:
        print(f"plot: {summary_plot}")
    print("-------------------------------------------")
    for row in summary_rows:
        if int(row["severity_rank"]) == 0:
            continue
        print(
            f"{row['corruption']:>18s} | {row['level_label']:>8s} | {row['method']:>12s} | "
            f"SSIM {row['noisy_ssim']:.4f} | PSNR {row['noisy_psnr']:.2f} | MSE {row['noisy_mse']:.6f}"
        )
    print("===========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Flow-model config.")
    ap.add_argument("--v_ckpt", type=str, required=True)
    ap.add_argument("--x_ckpt", type=str, required=True)
    ap.add_argument("--unet_ckpt", type=str, required=True)
    ap.add_argument("--unet_config", type=str, default=None)
    ap.add_argument(
        "--corruptions",
        type=str,
        default="exposure_scale,background_offset,measurement_noise,poisson_peak,measurement_shift,psf_shift",
    )
    ap.add_argument("--exposure_levels", type=str, default="1.0,0.85,1.15,0.70,1.30")
    ap.add_argument("--offset_levels", type=str, default="0.0,0.01,0.03,0.05")
    ap.add_argument("--measurement_noise_levels", type=str, default="0.0,0.01,0.02,0.05")
    ap.add_argument(
        "--measurement_noise_scale",
        type=str,
        default="rms",
        choices=["rms", "range", "absmax", "fixed"],
        help="Scale for additive Gaussian noise on the lensless measurement.",
    )
    ap.add_argument("--poisson_peaks", type=str, default="clean,1024,256,64,16")
    ap.add_argument("--shift_levels", type=str, default="0,1,2,4")
    ap.add_argument("--psf_shift_levels", type=str, default="0,1,2,4")
    ap.add_argument("--shift_axis", type=str, default="diag", choices=["x", "y", "diag"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--flow_dc_steps", type=int, default=None)
    ap.add_argument("--flow_dc_step_size", type=float, default=None)
    ap.add_argument("--flow_disable_physics", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no_clamp_measurement", action="store_true")
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "physical_robustness"))
    main(ap.parse_args())
