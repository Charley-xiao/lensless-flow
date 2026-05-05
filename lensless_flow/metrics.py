import torch
import piq


def psnr(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> float:
    return float(piq.psnr(x_hat, x, data_range=data_range, reduction="mean").item())


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    downsample: bool = False,
) -> torch.Tensor:
    return piq.ssim(
        x,
        y,
        kernel_size=window_size,
        kernel_sigma=sigma,
        data_range=data_range,
        reduction="mean",
        downsample=downsample,
        k1=K1,
        k2=K2,
    )


def ssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    eps: float = 1e-12,
    downsample: bool = False,
) -> torch.Tensor:
    del eps
    return ssim(
        x,
        y,
        window_size=window_size,
        sigma=sigma,
        data_range=data_range,
        K1=K1,
        K2=K2,
        downsample=downsample,
    )
