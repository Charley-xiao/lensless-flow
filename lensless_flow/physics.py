import torch
import torch.nn as nn

def psf_to_otf(psf: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    psf: [1,C,h,w] real
    returns OTF: [1,C,H,W] complex for circular conv per channel
    """
    assert psf.ndim == 4, f"psf must be [1,C,h,w], got {psf.shape}"
    _, C, h, w = psf.shape
    H, W = out_hw

    psf_pad = torch.zeros((1, C, H, W), dtype=psf.dtype, device=psf.device)
    psf_pad[..., :h, :w] = psf
    psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))
    return torch.fft.fft2(psf_pad)  # complex [1,C,H,W]

class FFTConvOperator(nn.Module):
    """
    Linear operator H: per-channel circular convolution with PSF.
    x: [B,C,H,W]
    """
    def __init__(self, psf: torch.Tensor, im_hw: tuple[int, int]):
        super().__init__()
        otf = psf_to_otf(psf, im_hw)
        self.register_buffer("otf", otf)

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
