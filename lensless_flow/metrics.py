import torch
import torch.nn.functional as F


def psnr(x_hat, x):
    mse = F.mse_loss(x_hat, x).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

# -------------------------
# SSIM (pure PyTorch)
# -------------------------
def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).contiguous()
    return kernel_2d


def ssim_map_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
                   data_range: float = 1.0, K1: float = 0.01, K2: float = 0.03, eps: float = 1e-12):
    """
    Compute SSIM for tensors x, y in [B,C,H,W], values assumed in [0, data_range].
    Returns: the SSIM map with shape [B,C,H,W].

    This is the standard SSIM (single-scale) computed with a Gaussian window.
    """
    assert x.ndim == 4 and y.ndim == 4, "x,y must be [B,C,H,W]"
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"

    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # make gaussian window
    if window_size % 2 == 0:
        window_size += 1  # ensure odd
    kernel = _gaussian_kernel(window_size, sigma, device, dtype)
    kernel = kernel.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    padding = window_size // 2

    # depthwise conv
    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # SSIM map
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)

    return ssim_map


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0, K1: float = 0.01, K2: float = 0.03, eps: float = 1e-12):
    """Return mean SSIM, preserving the original Gaussian-window calculation."""
    return ssim_map_torch(x, y, window_size, sigma, data_range, K1, K2, eps).mean()


def _validated_region_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Validate a detached [B,1,H,W] or [B,C,H,W] mask and expand channels."""
    if reference.ndim != 4 or any(size == 0 for size in reference.shape):
        raise ValueError("Region calculations require nonempty [B,C,H,W] tensors.")
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
        raise ValueError("mask must be a [B,1,H,W] or [B,C,H,W] tensor.")
    if (mask.shape[0] != reference.shape[0] or mask.shape[2:] != reference.shape[2:]
            or mask.shape[1] not in (1, reference.shape[1])):
        raise ValueError(f"mask shape {tuple(mask.shape)} cannot match {tuple(reference.shape)}.")
    mask = mask.detach().to(device=reference.device, dtype=torch.float32)
    if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
        raise ValueError("mask must contain finite values in [0,1].")
    return mask.expand_as(reference)


def region_image_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         window_size: int = 11, sigma: float = 1.5,
                         data_range: float = 1.0) -> dict[str, torch.Tensor]:
    """Return per-image foreground/background scores using a fixed soft mask.

    SSIM averages the full-image SSIM map at region centers; its windows retain
    their original neighboring pixels. Circular error assumes one full phase
    cycle spans ``data_range`` image units. Absent-region scores are NaN. PSNR
    uses the existing repository convention of 99 dB for MSE <= 1e-12.
    Inputs are not clamped or independently contrast-normalized here.
    """
    import math

    if pred.shape != target.shape or pred.device != target.device:
        raise ValueError("pred and target must have matching shapes and devices.")
    if not math.isfinite(float(data_range)) or float(data_range) <= 0:
        raise ValueError("data_range must be finite and positive.")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("sigma must be finite and positive.")
    mask = _validated_region_mask(mask, pred)
    pred, target = pred.float(), target.float()
    dims = (1, 2, 3)
    fg_count, bg_count = mask.sum(dims), (1.0 - mask).sum(dims)

    def regional_mean(values, weights, count):
        value = (values * weights).sum(dims) / count.clamp_min(1e-12)
        return torch.where(count > 0, value, torch.full_like(value, float("nan")))

    err2 = (pred - target).square()
    fg_mse = regional_mean(err2, mask, fg_count)
    bg_mse = regional_mean(err2, 1.0 - mask, bg_count)

    def psnr_from_mse(mse):
        result = 10.0 * torch.log10(float(data_range) ** 2 / mse.clamp_min(1e-12))
        return torch.where(mse <= 1e-12, torch.full_like(mse, 99.0), result)

    ssim_map = ssim_map_torch(pred, target, window_size, sigma, data_range)
    phase_delta = (2.0 * math.pi / float(data_range)) * (pred - target)
    circular_residual = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta))
    return {
        "rbc_psnr": psnr_from_mse(fg_mse),
        "background_psnr": psnr_from_mse(bg_mse),
        "rbc_ssim": regional_mean(ssim_map, mask, fg_count),
        "background_ssim": regional_mean(ssim_map, 1.0 - mask, bg_count),
        "rbc_circular_rmse_rad": regional_mean(circular_residual.square(), mask, fg_count).sqrt(),
        "rbc_fraction": mask.mean(dims),
        "rbc_mse": fg_mse,
        "background_mse": bg_mse,
    }
