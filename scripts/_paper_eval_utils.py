import math
from typing import Callable

import torch

from lensless_flow.data import make_dataloader
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import (
    build_baseline_unet as build_baseline_unet_model,
    build_flow_model as build_flow_model_impl,
)
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def parse_float_list(text: str) -> list[float]:
    text = text.replace(",", " ")
    return [float(x) for x in text.split() if x]


def parse_int_list(text: str) -> list[int]:
    text = text.replace(",", " ")
    return [int(x) for x in text.split() if x]


def effective_batch_size(batch_size: int) -> int:
    return max(1, int(batch_size))


def tensor_to_imshow(x_bchw: torch.Tensor):
    x = x_bchw[0].detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def build_test_loader_and_operator(cfg, batch_size: int, num_workers: int, device: torch.device):
    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=effective_batch_size(batch_size),
        num_workers=num_workers,
        path=cfg["data"].get("path", None),
    )

    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))

    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)
    return test_ds, test_dl, psf, Hop, img_channels, im_hw


def build_model(
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


def build_baseline_model(
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


def load_checkpoint_state(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device)


@torch.no_grad()
def unet_forward_baseline(model, y: torch.Tensor) -> torch.Tensor:
    b = y.shape[0]
    x_t = torch.zeros_like(y)
    t = torch.zeros(b, device=y.device, dtype=y.dtype) if getattr(model, "use_time_conditioning", True) else None
    return model(x_t, y, t)


def load_flow_runner(
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
) -> Callable[[torch.Tensor, FFTLinearConvOperator], torch.Tensor]:
    state = load_checkpoint_state(ckpt_path, device=device)
    model = build_model(
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


def load_unet_runner(cfg, ckpt_path: str, img_channels: int, device: torch.device):
    state = load_checkpoint_state(ckpt_path, device=device)
    model = build_baseline_model(cfg, img_channels=img_channels, device=device, checkpoint_state=state)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def _runner(y: torch.Tensor, _Hop: FFTLinearConvOperator) -> torch.Tensor:
        return unet_forward_baseline(model, y)

    return _runner


def compute_metrics(x_hat: torch.Tensor, x: torch.Tensor) -> dict[str, float]:
    x_hat_c = x_hat.clamp(0, 1).float()
    x_c = x.clamp(0, 1).float()
    return {
        "mse": float((x_hat_c - x_c).pow(2).mean().item()),
        "psnr": float(psnr(x_hat_c, x_c)),
        "ssim": float(ssim_torch(x_hat_c, x_c)),
    }


def compute_metrics_per_sample(x_hat: torch.Tensor, x: torch.Tensor) -> list[dict[str, float]]:
    return [compute_metrics(x_hat[i : i + 1], x[i : i + 1]) for i in range(x.shape[0])]


def samplewise_rms(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return x.float().pow(2).mean(dim=dims, keepdim=True).sqrt()


def samplewise_range(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    xmax = x.amax(dim=dims, keepdim=True)
    xmin = x.amin(dim=dims, keepdim=True)
    return (xmax - xmin).float()


def capture_rng_state():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def restore_rng_state(cpu_state, cuda_states) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def run_with_latent_seed(
    runner,
    y: torch.Tensor,
    Hop: FFTLinearConvOperator,
    latent_seed: int,
) -> torch.Tensor:
    cpu_state, cuda_states = capture_rng_state()
    torch.manual_seed(int(latent_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(latent_seed))
    try:
        return runner(y, Hop)
    finally:
        restore_rng_state(cpu_state, cuda_states)


def zero_fill_shift(x: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    out = torch.zeros_like(x)

    src_y0 = max(0, -shift_y)
    src_y1 = x.shape[-2] - max(0, shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

    src_x0 = max(0, -shift_x)
    src_x1 = x.shape[-1] - max(0, shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if (src_y1 > src_y0) and (src_x1 > src_x0):
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return out


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    xf = x.detach().float().reshape(-1)
    yf = y.detach().float().reshape(-1)
    if xf.numel() <= 1:
        return 0.0
    xf = xf - xf.mean()
    yf = yf - yf.mean()
    denom = xf.norm() * yf.norm()
    if float(denom.item()) <= eps:
        return 0.0
    return float((xf * yf).sum().item() / (denom.item() + eps))


def pairwise_mean_distance(samples: torch.Tensor, p: int = 1) -> float:
    num_samples = samples.shape[0]
    if num_samples <= 1:
        return 0.0

    total = 0.0
    count = 0
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            diff = (samples[i] - samples[j]).float()
            if p == 1:
                value = diff.abs().mean().item()
            elif p == 2:
                value = diff.pow(2).mean().sqrt().item()
            else:
                raise ValueError(f"Unsupported p={p}")
            total += float(value)
            count += 1
    return float(total / max(1, count))


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    return pearson_corr(rankdata(x.reshape(-1)), rankdata(y.reshape(-1)), eps=eps)


def maybe_import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
