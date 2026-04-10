import torch
import torch.nn as nn


def _to_1chw(psf: torch.Tensor) -> torch.Tensor:
    """
    Normalize psf to shape [1,C,h,w].
    Accepts [h,w], [C,h,w], [1,C,h,w], [1,h,w], [h,w,C] etc (best-effort).
    """
    if psf.ndim == 2:
        psf = psf[None, None, ...]  # [1,1,h,w]
    elif psf.ndim == 3:
        # could be [C,h,w] or [h,w,C]
        if psf.shape[0] in (1, 3) and psf.shape[-1] not in (1, 3):
            psf = psf[None, ...]  # [1,C,h,w]
        elif psf.shape[-1] in (1, 3) and psf.shape[0] not in (1, 3):
            psf = psf.permute(2, 0, 1)[None, ...]  # [1,C,h,w]
        else:
            # ambiguous; assume [C,h,w]
            psf = psf[None, ...]
    elif psf.ndim == 4:
        # [B,C,h,w] but we expect B==1
        if psf.shape[0] != 1:
            raise ValueError(f"psf batch dim must be 1, got {psf.shape}")
    else:
        raise ValueError(f"Unsupported psf ndim={psf.ndim}, shape={tuple(psf.shape)}")

    return psf


def psf_to_otf_linear(psf_1chw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    Build OTF for LINEAR convolution via FFT on a padded grid of size out_hw=(Hf,Wf).
    psf_1chw: [1,C,h,w]
    Returns otf: [1,C,Hf,Wf] complex
    """
    psf_1chw = _to_1chw(psf_1chw)
    _, C, h, w = psf_1chw.shape
    Hf, Wf = out_hw
    if Hf < h or Wf < w:
        raise ValueError(f"out_hw too small for psf: out_hw={out_hw}, psf={(h,w)}")

    psf_pad = torch.zeros((1, C, Hf, Wf), device=psf_1chw.device, dtype=psf_1chw.dtype)
    top = (Hf - h) // 2
    left = (Wf - w) // 2
    psf_pad[..., top:top + h, left:left + w] = psf_1chw

    otf = torch.fft.fft2(psf_pad)  # complex
    return otf


class FFTLinearConvOperator(nn.Module):
    """
    Linear operator H: per-channel LINEAR convolution with PSF using centered
    FFT padding + crop, matching LenslessPiCam's spatial convention.

    forward: x [B,C,H,W] -> y [B,C,H,W] (same spatial size as x) by:
        1) center-embed x on the full linear-convolution grid
        2) FFT multiply by OTF
        3) IFFT + ifftshift back to image coordinates
        4) center-crop to HxW

    adjoint: y [B,C,H,W] -> x [B,C,H,W] (the exact adjoint of above forward)
    """
    def __init__(self, psf: torch.Tensor, im_hw: tuple[int, int]):
        super().__init__()
        psf = _to_1chw(psf)  # [1,C,h,w]
        self.register_buffer("psf", psf)
        self.dc_safety = 0.05

        self.im_hw = (int(im_hw[0]), int(im_hw[1]))
        _, C, h, w = psf.shape
        self.psf_hw = (h, w)

        H, W = self.im_hw
        Hf, Wf = H + h - 1, W + w - 1
        self.full_hw = (Hf, Wf)

        otf = psf_to_otf_linear(psf, (Hf, Wf))  # [1,C,Hf,Wf]
        self.register_buffer("otf", otf)

    def _center_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Hf, Wf = self.full_hw
        top = (Hf - H) // 2
        left = (Wf - W) // 2
        out = torch.zeros((B, C, Hf, Wf), device=x.device, dtype=x.dtype)
        out[..., top:top + H, left:left + W] = x
        return out

    def _center_crop(self, y_full: torch.Tensor) -> torch.Tensor:
        """
        Crop HxW from the centered full-grid convolution result.
        """
        B, C, Hf, Wf = y_full.shape
        H, W = self.im_hw
        top = (Hf - H) // 2
        left = (Wf - W) // 2
        return y_full[..., top:top + H, left:left + W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        X = torch.fft.fft2(self._center_embed(x))
        Y = X * self.otf  # broadcast over B
        y_full = torch.fft.ifft2(Y).real
        y_full = torch.fft.ifftshift(y_full, dim=(-2, -1))
        return self._center_crop(y_full)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of forward:
        - Center-embed y on the full grid
        - Apply convolution with the flipped kernel via conj(otf)
        - Undo the forward's ifftshift with fftshift
        - Center-crop back to image size
        """
        Y = torch.fft.fft2(self._center_embed(y))
        X = Y * torch.conj(self.otf)
        x_full = torch.fft.ifft2(X).real
        x_full = torch.fft.fftshift(x_full, dim=(-2, -1))
        return self._center_crop(x_full)
