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


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0, K1: float = 0.01, K2: float = 0.03, eps: float = 1e-12):
    """
    Compute SSIM for tensors x, y in [B,C,H,W], values assumed in [0, data_range].
    Returns: scalar mean SSIM over batch and channels.

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

    return ssim_map.mean()  # scalar