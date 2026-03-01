import torch
import torch.nn as nn
import torch.nn.functional as F

def _ifftshift2(x):
    # shift center -> (0,0) for last two dims
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(x, shifts=(-h // 2, -w // 2), dims=(-2, -1))

def psf_to_otf(psf: torch.Tensor, im_hw: tuple[int, int]) -> torch.Tensor:
    H, W = im_hw
    # normalize shape to [1,C,h,w]
    if psf.ndim == 2:
        psf = psf[None, None, ...]
    elif psf.ndim == 3:
        psf = psf[None, ...]  # [1,C,h,w] if psf was [C,h,w]
    elif psf.ndim == 4:
        pass
    else:
        raise ValueError(f"psf ndim={psf.ndim} not supported")

    B, C, h, w = psf.shape
    assert B == 1, "psf batch dimension should be 1"

    # pad to (H,W) at top-left then shift
    psf_pad = torch.zeros((1, C, H, W), device=psf.device, dtype=psf.dtype)
    psf_pad[..., :h, :w] = psf

    # IMPORTANT: center -> origin for FFT conv convention
    psf_pad = _ifftshift2(psf_pad)

    otf = torch.fft.fft2(psf_pad)  # complex [1,C,H,W]
    return otf


class FFTConvOperator(nn.Module):
    """
    Linear operator H: per-channel circular convolution with PSF.
    x: [B,C,H,W]
    """
    def __init__(self, psf: torch.Tensor, im_hw: tuple[int, int]):
        super().__init__()
        otf = psf_to_otf(psf, im_hw)
        self.register_buffer("otf", otf)
        self.dc_safety = 0.1  # optional safety factor for suggested DC step size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x)
        Y = X * self.otf  # broadcast over B
        return torch.fft.ifft2(Y).real

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        Y = torch.fft.fft2(y)
        X = Y * torch.conj(self.otf)
        return torch.fft.ifft2(X).real

def data_consistency_step(x: torch.Tensor, y: torch.Tensor, H: FFTConvOperator, step: float, iters: int = 1):
    for _ in range(iters):
        r = H.forward(x) - y
        grad = 2.0 * H.adjoint(r)
        x = x - step * grad
    return x
