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
    psf_pad[..., :h, :w] = psf_1chw

    # IMPORTANT: shift PSF center to (0,0) for proper convolution in FFT domain.
    # ifftshift moves the "center" of spatial kernel to the origin.
    psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))

    otf = torch.fft.fft2(psf_pad)  # complex
    return otf


class FFTLinearConvOperator(nn.Module):
    """
    Linear operator H: per-channel LINEAR convolution with PSF using FFT padding + crop.

    forward: x [B,C,H,W] -> y [B,C,H,W] (same spatial size as x) by:
        1) pad x to full size (H+h-1, W+w-1)
        2) FFT multiply by OTF
        3) IFFT -> y_full
        4) crop to HxW

    adjoint: y [B,C,H,W] -> x [B,C,H,W] (the adjoint of above forward)
    """
    def __init__(self, psf: torch.Tensor, im_hw: tuple[int, int]):
        super().__init__()
        psf = _to_1chw(psf)  # [1,C,h,w]
        self.register_buffer("psf", psf)

        self.im_hw = (int(im_hw[0]), int(im_hw[1]))
        _, C, h, w = psf.shape
        self.psf_hw = (h, w)

        H, W = self.im_hw
        Hf, Wf = H + h - 1, W + w - 1
        self.full_hw = (Hf, Wf)

        otf = psf_to_otf_linear(psf, (Hf, Wf))  # [1,C,Hf,Wf]
        self.register_buffer("otf", otf)

        # For adjoint of linear conv: use conjugate OTF, BUT cropping/padding must match.
        # We'll implement adjoint explicitly via FFT with conj(otf) and matched crop.
        # Also store crop indices.
        self._crop_top = (h - 1) // 2
        self._crop_left = (w - 1) // 2

    def _pad_to_full(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Hf, Wf = self.full_hw
        out = torch.zeros((B, C, Hf, Wf), device=x.device, dtype=x.dtype)
        out[..., :H, :W] = x
        return out

    def _crop_from_full(self, y_full: torch.Tensor) -> torch.Tensor:
        """
        Crop HxW from y_full (full conv result) to align with "same" output.
        We use a center-ish crop based on psf size.
        """
        B, C, Hf, Wf = y_full.shape
        H, W = self.im_hw
        top = self._crop_top
        left = self._crop_left
        return y_full[..., top:top + H, left:left + W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        X = torch.fft.fft2(self._pad_to_full(x))
        Y = X * self.otf  # broadcast over B
        y_full = torch.fft.ifft2(Y).real
        return self._crop_from_full(y_full)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of forward:
        - Embed y into full grid (inverse of crop): place it at crop region in a zero full image
        - Apply convolution with flipped kernel => in FFT domain multiply by conj(otf)
        - Then take the top-left HxW region corresponding to original x placement
        """
        B, C, H, W = y.shape
        Hf, Wf = self.full_hw

        # inverse crop embedding
        y_full = torch.zeros((B, C, Hf, Wf), device=y.device, dtype=y.dtype)
        top = self._crop_top
        left = self._crop_left
        y_full[..., top:top + H, left:left + W] = y

        Y = torch.fft.fft2(y_full)
        X = Y * torch.conj(self.otf)
        x_full = torch.fft.ifft2(X).real

        # inverse of forward padding: forward placed x at [:H,:W]
        return x_full[..., :H, :W]