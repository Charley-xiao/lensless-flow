import argparse
import math
import os

from tqdm import tqdm

import torch
import torch.nn as nn

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import (
    build_baseline_unet as build_baseline_unet_model,
    build_flow_model as build_flow_model_impl,
    load_checkpoint_state_dict,
)
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def _avg(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _load_state_dict(model: nn.Module, state) -> None:
    load_checkpoint_state_dict(model, state)


def _build_unet(
    cfg,
    img_channels: int,
    im_hw: tuple[int, int],
    device: torch.device,
    checkpoint_state: dict | None = None,
) -> nn.Module:
    return build_flow_model_impl(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=checkpoint_state,
    )


def _build_baseline_unet(
    cfg,
    img_channels: int,
    device: torch.device,
    checkpoint_state: dict | None = None,
) -> nn.Module:
    return build_baseline_unet_model(
        cfg=cfg,
        img_channels=img_channels,
        device=device,
        checkpoint_state=checkpoint_state,
    )


@torch.no_grad()
def _unet_forward_baseline(model: nn.Module, y: torch.Tensor) -> torch.Tensor:
    b = y.shape[0]
    x_t = torch.zeros_like(y)
    t = torch.zeros(b, device=y.device, dtype=y.dtype) if getattr(model, "use_time_conditioning", True) else None
    return model(x_t, y, t)


def _resolve_default_unet_ckpt(unet_ckpt: str | None) -> str | None:
    if unet_ckpt:
        return unet_ckpt

    default_ckpt = os.path.join("checkpoints", "unet.pt")
    if os.path.isfile(default_ckpt):
        return default_ckpt
    return None


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
    raise ValueError(f"Unsupported channel count for LPIPS/FID: {channels}")


def _prepare_lpips_input(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_three_channels(x.clamp(0, 1).float())
    return x * 2.0 - 1.0


def _prepare_fid_input(x: torch.Tensor) -> torch.Tensor:
    return _ensure_three_channels(x.clamp(0, 1).float())


def _build_lpips_metric(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS evaluation requires the `lpips` package. Install the project requirements first."
        ) from exc

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for param in metric.parameters():
        param.requires_grad_(False)
    return metric


def _build_fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "FID evaluation requires `torchmetrics` and `torch-fidelity`. Install the project requirements first."
        ) from exc

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    metric.eval()
    return metric


def _per_sample_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (x_hat.float() - x.float()).reshape(x.shape[0], -1).pow(2).mean(dim=1)


def _per_sample_dc_rmse(Hop: FFTLinearConvOperator, x_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    residual = Hop.forward(x_hat.float()) - y.float()
    return residual.reshape(residual.shape[0], -1).pow(2).mean(dim=1).sqrt()


def _method_postfix_key(method_name: str) -> str:
    return "unet_psnr" if method_name == "baseline_unet" else "flow_psnr"


def _print_method_summary(
    method_name: str,
    method_cfg: dict,
    stats: dict[str, list[float]],
    fid_value: float,
) -> None:
    print(f"[{method_name}]")
    for line in method_cfg["summary_lines"]:
        print(line)
    print(f"PSNR avg: {_avg(stats['psnr']):.3f}")
    print(f"SSIM avg: {_avg(stats['ssim']):.6f}")
    print(f"LPIPS avg: {_avg(stats['lpips']):.6f}")
    print(f"MSE avg: {_avg(stats['mse']):.8f}")
    print(f"Data-consistency RMSE avg: {_avg(stats['dc_rmse']):.6f}")
    if math.isnan(fid_value):
        print("FID: n/a (need at least 2 evaluated samples)")
    else:
        print(f"FID: {fid_value:.6f}")
    print("----------------------------------")


@torch.no_grad()
def main(
    flow_cfg,
    flow_config_path: str,
    ckpt: str,
    max_batches: int | None,
    unet_ckpt: str | None,
    unet_cfg,
    unet_config_path: str | None,
    batch_size: int,
    num_workers: int,
):
    device = torch.device(flow_cfg["device"] if torch.cuda.is_available() else "cpu")

    flow_state = torch.load(ckpt, map_location=device)
    pred_type = str(
        flow_state.get("mode", flow_cfg.get("train", {}).get("mode", "btb"))
        if isinstance(flow_state, dict)
        else flow_cfg.get("train", {}).get("mode", "btb")
    ).lower()
    assert pred_type in ["btb", "vanilla"], f"Unknown pred_type={pred_type}"
    flow_matcher_name = normalize_flow_matcher_name(
        flow_state.get("matcher", flow_cfg.get("cfm", {}).get("matcher", "rectified"))
        if isinstance(flow_state, dict)
        else flow_cfg.get("cfm", {}).get("matcher", "rectified")
    )

    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=flow_cfg["data"]["downsample"],
        flip_ud=flow_cfg["data"]["flip_ud"],
        batch_size=batch_size,
        num_workers=num_workers,
        path=flow_cfg["data"].get("path", None),
    )

    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))

    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    flow_model = _build_unet(
        flow_cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=flow_state,
    )
    _load_state_dict(flow_model, flow_state)

    steps = int(flow_cfg["sample"]["steps"])
    init_noise_std = float(flow_cfg["sample"]["init_noise_std"])
    denom_min = float(flow_cfg.get("btb", {}).get("denom_min", 0.05))
    disable_physics = bool(flow_cfg.get("physics", {}).get("disable_in_eval", False))
    dc_steps = int(flow_cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(flow_cfg.get("physics", {}).get("dc_step_size", 0.0))
    dc_mode = "rgb"

    methods: dict[str, dict] = {
        "flow": {
            "runner": lambda y: sample_with_physics_guidance(
                model=flow_model,
                y=y,
                H=Hop,
                steps=steps,
                dc_step=dc_step,
                dc_steps=dc_steps,
                init_noise_std=init_noise_std,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=disable_physics,
                pred_type=pred_type,
                dc_mode=dc_mode,
            ),
            "summary_lines": [
                f"ckpt: {ckpt}",
                f"pred_type: {pred_type}",
                f"matcher: {flow_matcher_name}",
                f"steps: {steps}",
                f"DC: {'disabled' if disable_physics else (dc_mode + f' (dc_steps={dc_steps}, dc_step={dc_step})')}",
            ],
        }
    }

    if unet_ckpt is not None:
        baseline_state = torch.load(unet_ckpt, map_location=device)
        baseline_model = _build_baseline_unet(
            unet_cfg,
            img_channels=img_channels,
            device=device,
            checkpoint_state=baseline_state,
        )
        _load_state_dict(baseline_model, baseline_state)
        methods["baseline_unet"] = {
            "runner": lambda y: _unet_forward_baseline(baseline_model, y),
            "summary_lines": [
                f"ckpt: {unet_ckpt}",
                f"config: {unet_config_path if unet_config_path is not None else flow_config_path}",
                "pred_type: deterministic baseline U-Net",
                "matcher: n/a",
                "steps: 1 forward pass",
                "DC: none",
            ],
        }

    lpips_metric = _build_lpips_metric(device)
    fid_metrics = {name: _build_fid_metric(device) for name in methods}

    stats = {
        name: {"psnr": [], "ssim": [], "lpips": [], "mse": [], "dc_rmse": []}
        for name in methods
    }

    total_batches = len(test_dl) if max_batches is None else min(len(test_dl), max_batches)
    total_samples = 0

    pbar = tqdm(test_dl, total=total_batches, desc=f"eval [{', '.join(methods.keys())}]")
    for i, (y, x) in enumerate(pbar):
        if (max_batches is not None) and (i >= max_batches):
            break

        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)
        total_samples += int(x.shape[0])

        x_c = x.clamp(0, 1)
        x_lpips = _prepare_lpips_input(x_c)
        x_fid = _prepare_fid_input(x_c)

        batch_postfix = {}
        for method_name, method_cfg in methods.items():
            x_hat = method_cfg["runner"](y)
            x_hat_c = x_hat.clamp(0, 1)

            mse_values = _per_sample_mse(x_hat_c, x_c)
            psnr_values = [float(psnr(x_hat_c[j : j + 1], x_c[j : j + 1])) for j in range(x.shape[0])]
            ssim_values = [float(ssim_torch(x_hat_c[j : j + 1], x_c[j : j + 1])) for j in range(x.shape[0])]
            dc_values = _per_sample_dc_rmse(Hop, x_hat, y)
            lpips_values = lpips_metric(_prepare_lpips_input(x_hat_c), x_lpips).reshape(-1)

            stats[method_name]["mse"].extend(mse_values.detach().cpu().tolist())
            stats[method_name]["psnr"].extend(psnr_values)
            stats[method_name]["ssim"].extend(ssim_values)
            stats[method_name]["dc_rmse"].extend(dc_values.detach().cpu().tolist())
            stats[method_name]["lpips"].extend(lpips_values.detach().float().cpu().tolist())

            fid_metrics[method_name].update(x_fid, real=True)
            fid_metrics[method_name].update(_prepare_fid_input(x_hat_c), real=False)

            batch_postfix[_method_postfix_key(method_name)] = f"{_avg(psnr_values):.2f}"

        pbar.set_postfix(batch_postfix)

    print("\n========== Eval Summary ==========")
    print(f"samples_evaluated: {total_samples}")
    print(f"batch_size: {batch_size}")
    print("----------------------------------")
    for method_name, method_cfg in methods.items():
        fid_value = float("nan")
        if total_samples >= 2:
            fid_value = float(fid_metrics[method_name].compute().item())
        _print_method_summary(
            method_name=method_name,
            method_cfg=method_cfg,
            stats=stats[method_name],
            fid_value=fid_value,
        )
    print("==================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Flow-model checkpoint.")
    ap.add_argument(
        "--unet_ckpt",
        type=str,
        default=None,
        help="Optional baseline U-Net checkpoint. Defaults to checkpoints/unet.pt when available.",
    )
    ap.add_argument("--unet_config", type=str, default=None, help="Optional baseline U-Net config.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=200)
    args, overrides = ap.parse_known_args()
    flow_cfg = load_config(args.config, overrides)
    unet_cfg = load_config(args.unet_config, overrides) if args.unet_config is not None else flow_cfg

    max_batches = None if args.max_batches < 0 else args.max_batches
    resolved_unet_ckpt = _resolve_default_unet_ckpt(args.unet_ckpt)

    main(
        flow_cfg=flow_cfg,
        flow_config_path=args.config,
        ckpt=args.ckpt,
        max_batches=max_batches,
        unet_ckpt=resolved_unet_ckpt,
        unet_cfg=unet_cfg,
        unet_config_path=args.unet_config,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
    )
