import argparse
import copy
import csv
import json
import os

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lensless_flow.config import load_config, merge_config
from lensless_flow.data import make_dataloader
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict, resolve_model_name
from lensless_flow.model_unet import resolve_use_time_conditioning
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir, set_seed


SUPPORTED_METHODS = ("nearest", "bilinear", "bicubic", "lanczos4", "bicubic_unsharp")


def _parse_method_list(text: str) -> list[str]:
    values = []
    for token in str(text).replace(",", " ").split():
        key = token.strip().lower()
        if not key:
            continue
        if key not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported upsampling method '{token}'. Expected one of {SUPPORTED_METHODS}.")
        values.append(key)
    if not values:
        raise ValueError("At least one upsampling method must be requested.")
    return values


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
            "Upsampling evaluation requires the `lpips` package. Install the project requirements first."
        ) from exc

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for param in metric.parameters():
        param.requires_grad_(False)
    return metric


def _resolve_config(state: dict, config_path: str | None, overrides: list[str]) -> dict:
    ckpt_cfg = state.get("cfg")
    if not isinstance(ckpt_cfg, dict):
        if config_path is None:
            raise ValueError("Checkpoint does not contain a saved cfg, so --config is required.")
        return load_config(config_path, overrides)

    merged_cfg = ckpt_cfg
    if config_path is not None:
        file_cfg = load_config(config_path, overrides=None)
        merged_cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(ckpt_cfg), OmegaConf.create(file_cfg)),
            resolve=True,
        )
    return merge_config(merged_cfg, overrides)


def _cv2_resize_single(img_chw: torch.Tensor, size_hw: tuple[int, int], interpolation: int) -> torch.Tensor:
    img = img_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    resized = cv2.resize(img, dsize=(int(size_hw[1]), int(size_hw[0])), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)


def _unsharp_mask_single(img_chw: torch.Tensor, sigma: float = 1.0, amount: float = 0.5) -> torch.Tensor:
    img = img_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharpened = np.clip(img + float(amount) * (img - blurred), 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(sharpened)).permute(2, 0, 1)


def _upsample_batch(x: torch.Tensor, size_hw: tuple[int, int], method: str) -> torch.Tensor:
    method = str(method).lower()
    if method == "nearest":
        return F.interpolate(x, size=size_hw, mode="nearest")
    if method == "bilinear":
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    if method == "bicubic":
        return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)

    outputs = []
    if method == "lanczos4":
        for sample in x:
            outputs.append(_cv2_resize_single(sample, size_hw=size_hw, interpolation=cv2.INTER_LANCZOS4))
    elif method == "bicubic_unsharp":
        for sample in x:
            up = _cv2_resize_single(sample, size_hw=size_hw, interpolation=cv2.INTER_CUBIC)
            outputs.append(_unsharp_mask_single(up, sigma=1.0, amount=0.5))
    else:
        raise ValueError(f"Unknown upsampling method '{method}'")

    return torch.stack(outputs, dim=0).to(device=x.device, dtype=x.dtype)


def _build_runner(cfg, state, device: torch.device, img_channels: int, im_hw: tuple[int, int]):
    model = build_flow_model(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=state,
    )
    load_checkpoint_state_dict(model, state)

    pred_type = str(state.get("mode", cfg.get("train", {}).get("mode", "btb"))).lower()
    if pred_type not in ["btb", "vanilla"]:
        raise ValueError(f"Unknown pred_type '{pred_type}' in checkpoint/config.")
    matcher_name = normalize_flow_matcher_name(
        state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified"))
    )
    model_name = resolve_model_name(cfg, checkpoint_state=state)
    use_time_conditioning = resolve_use_time_conditioning(cfg, state)
    return model, pred_type, matcher_name, model_name, use_time_conditioning


@torch.no_grad()
def main(args, overrides: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)
    cfg = _resolve_config(state, args.config, overrides)

    if args.device is not None:
        cfg["device"] = str(args.device)
        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    set_seed(int(args.seed))

    low_ds, low_dl = make_dataloader(
        split=args.split,
        downsample=int(cfg["data"]["downsample"]),
        flip_ud=bool(cfg["data"]["flip_ud"]),
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
    )

    full_cfg = copy.deepcopy(cfg)
    full_cfg.setdefault("data", {})
    full_cfg["data"]["downsample"] = 1
    full_ds, full_dl = make_dataloader(
        split=args.split,
        downsample=1,
        flip_ud=bool(full_cfg["data"]["flip_ud"]),
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=full_cfg["data"].get("path", None),
    )

    if len(low_ds) != len(full_ds):
        raise RuntimeError(f"Dataset length mismatch: low={len(low_ds)} vs full={len(full_ds)}")

    y0, _ = low_ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    low_hw = (int(y0.shape[-2]), int(y0.shape[-1]))

    x_full0 = to_nchw(full_ds[0][1])
    target_hw = (int(x_full0.shape[-2]), int(x_full0.shape[-1]))

    psf = to_nchw(low_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=low_hw).to(device)

    model, pred_type, matcher_name, model_name, use_time_conditioning = _build_runner(
        cfg=cfg,
        state=state,
        device=device,
        img_channels=img_channels,
        im_hw=low_hw,
    )

    steps = int(args.steps if args.steps is not None else cfg["sample"]["steps"])
    init_noise_std = float(cfg["sample"].get("init_noise_std", 1.0))
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))
    disable_physics = bool(
        args.disable_physics
        if args.disable_physics is not None
        else cfg.get("physics", {}).get("disable_in_eval", False)
    )
    dc_steps = int(args.dc_steps if args.dc_steps is not None else cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(
        args.dc_step_size if args.dc_step_size is not None else cfg.get("physics", {}).get("dc_step_size", 0.0)
    )

    methods = _parse_method_list(args.methods)
    lpips_metric = _build_lpips_metric(device)
    stats = {method: {"psnr": [], "ssim": [], "lpips": []} for method in methods}
    per_sample_rows: list[dict] = []

    ensure_dir(args.out_dir)
    total_batches = min(len(low_dl), len(full_dl))
    if args.max_batches >= 0:
        total_batches = min(total_batches, int(args.max_batches))

    sample_count = 0
    pbar = tqdm(zip(low_dl, full_dl), total=total_batches, desc="upsampling eval")
    for batch_idx, (low_batch, full_batch) in enumerate(pbar):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break

        y_low, _x_low = low_batch
        _y_full, x_full = full_batch
        y_low = to_nchw(y_low).to(device)
        x_full = to_nchw(x_full).to(device)

        if args.max_samples >= 0:
            remaining = int(args.max_samples) - int(sample_count)
            if remaining <= 0:
                break
            if y_low.shape[0] > remaining:
                y_low = y_low[:remaining]
                x_full = x_full[:remaining]

        x_hat_low = sample_with_physics_guidance(
            model=model,
            y=y_low,
            H=Hop,
            steps=steps,
            dc_step=dc_step,
            dc_steps=dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            clamp_x=False,
            disable_physics=disable_physics,
            pred_type=pred_type,
            dc_mode="rgb",
        )

        x_full_c = x_full.clamp(0, 1)
        x_full_lpips = _prepare_lpips_input(x_full_c)
        batch_postfix = {}

        for method in methods:
            x_up = _upsample_batch(x_hat_low, size_hw=target_hw, method=method).clamp(0, 1)
            lpips_values = lpips_metric(_prepare_lpips_input(x_up), x_full_lpips).reshape(-1)

            batch_psnr = []
            for sample_idx in range(x_up.shape[0]):
                psnr_value = float(psnr(x_up[sample_idx : sample_idx + 1], x_full_c[sample_idx : sample_idx + 1]))
                ssim_value = float(
                    ssim_torch(x_up[sample_idx : sample_idx + 1], x_full_c[sample_idx : sample_idx + 1]).item()
                )
                lpips_value = float(lpips_values[sample_idx].item())

                stats[method]["psnr"].append(psnr_value)
                stats[method]["ssim"].append(ssim_value)
                stats[method]["lpips"].append(lpips_value)
                batch_psnr.append(psnr_value)

                per_sample_rows.append(
                    {
                        "sample_idx": int(sample_count + sample_idx),
                        "batch_idx": int(batch_idx),
                        "method": method,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "lpips": lpips_value,
                    }
                )

            batch_postfix[f"{method}_psnr"] = f"{_avg(batch_psnr):.2f}"

        pbar.set_postfix(batch_postfix)
        sample_count += int(y_low.shape[0])
        if args.max_samples >= 0 and sample_count >= args.max_samples:
            break

    summary_rows = []
    for method in methods:
        summary_rows.append(
            {
                "method": method,
                "num_samples": int(len(stats[method]["psnr"])),
                "psnr": _avg(stats[method]["psnr"]),
                "ssim": _avg(stats[method]["ssim"]),
                "lpips": _avg(stats[method]["lpips"]),
            }
        )
    summary_rows.sort(key=lambda row: (-float(row["psnr"]), float(row["lpips"]), -float(row["ssim"])))

    summary_csv = os.path.join(args.out_dir, "upsampling_summary.csv")
    per_sample_csv = os.path.join(args.out_dir, "upsampling_per_sample.csv")
    metadata_json = os.path.join(args.out_dir, "upsampling_metadata.json")

    _write_csv(summary_csv, summary_rows, ["method", "num_samples", "psnr", "ssim", "lpips"])
    _write_csv(per_sample_csv, per_sample_rows, ["sample_idx", "batch_idx", "method", "psnr", "ssim", "lpips"])

    with open(metadata_json, "w") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(args.ckpt),
                "config": os.path.abspath(args.config) if args.config else None,
                "split": args.split,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "max_batches": int(args.max_batches),
                "max_samples": int(args.max_samples),
                "methods": methods,
                "device": str(device),
                "model_name": model_name,
                "pred_type": pred_type,
                "matcher": matcher_name,
                "use_time_conditioning": bool(use_time_conditioning),
                "low_downsample": int(cfg["data"]["downsample"]),
                "low_hw": low_hw,
                "target_hw": target_hw,
                "steps": steps,
                "disable_physics": bool(disable_physics),
                "dc_steps": int(dc_steps),
                "dc_step_size": float(dc_step),
            },
            f,
            indent=2,
        )

    print("\n========== Upsampling Summary ==========")
    print(f"ckpt: {args.ckpt}")
    print(f"model: {model_name}")
    print(f"pred_type: {pred_type}")
    print(f"matcher: {matcher_name}")
    print(f"low_res: {low_hw} (downsample={cfg['data']['downsample']})")
    print(f"target_res: {target_hw}")
    print(f"steps: {steps}")
    print(f"summary_csv: {summary_csv}")
    print(f"per_sample_csv: {per_sample_csv}")
    print(f"metadata_json: {metadata_json}")
    print("----------------------------------------")
    for row in summary_rows:
        print(
            f"{row['method']:>16s} | "
            f"PSNR {row['psnr']:.3f} | "
            f"SSIM {row['ssim']:.6f} | "
            f"LPIPS {row['lpips']:.6f}"
        )
    print("========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Low-resolution flow-model checkpoint.")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file to merge on top of the checkpoint's saved cfg before applying CLI overrides.",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dc_steps", type=int, default=None)
    ap.add_argument("--dc_step_size", type=float, default=None)
    ap.add_argument("--disable_physics", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--methods", type=str, default="nearest,bilinear,bicubic,lanczos4,bicubic_unsharp")
    ap.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "upsampling_eval"))
    args, overrides = ap.parse_known_args()
    main(args, overrides)
