import torch
import torch.nn as nn


def _center_pad(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    Adjoint of center crop for BCHW tensors.

    Places x in the center of a zero tensor with spatial size out_hw.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")

    B, C, h, w = x.shape
    H, W = int(out_hw[0]), int(out_hw[1])
    if H < h or W < w:
        raise ValueError(f"out_hw={out_hw} is smaller than input spatial size={(h, w)}")

    out = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)
    top = (H - h) // 2
    left = (W - w) // 2
    out[..., top:top + h, left:left + w] = x
    return out


def _center_crop(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Center crop BCHW tensor x to spatial size out_hw."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")

    H, W = x.shape[-2:]
    h, w = int(out_hw[0]), int(out_hw[1])
    if H < h or W < w:
        raise ValueError(f"Cannot crop spatial size={(H, W)} to larger out_hw={out_hw}")

    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top:top + h, left:left + w]


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


def _to_mask_1chw(mask: torch.Tensor, channels: int, im_hw: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    mask = _to_1chw(mask).to(dtype=dtype)
    _, mask_channels, H, W = mask.shape
    if (H, W) != im_hw:
        raise ValueError(f"mask spatial shape must be {im_hw}, got {(H, W)}")
    if mask_channels not in (1, channels):
        raise ValueError(f"mask channels must be 1 or {channels}, got {mask_channels}")
    if mask_channels == 1 and channels != 1:
        mask = mask.expand(1, channels, H, W)
    return mask.contiguous()


def psf_to_otf_linear(psf_1chw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    Build an FFT kernel on a padded grid of size out_hw=(Hp,Wp).

    The PSF is center-padded, then ifftshifted so its center becomes the
    zero-phase FFT origin, matching the reference DiffuserCam convention.

    psf_1chw: [1,C,h,w]
    Returns otf: [1,C,Hp,Wp] complex
    """
    psf_1chw = _to_1chw(psf_1chw)
    _, C, h, w = psf_1chw.shape
    Hp, Wp = out_hw
    if Hp < h or Wp < w:
        raise ValueError(f"out_hw too small for psf: out_hw={out_hw}, psf={(h,w)}")

    psf_pad = _center_pad(psf_1chw, (Hp, Wp))
    psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))

    otf = torch.fft.fft2(psf_pad)  # complex
    return otf


class FFTLinearConvOperator(nn.Module):
    """
    FFT implementation of the DiffuserCam linear convolution operator.

    Reference padded-space model:
        A(x_pad) = mask * crop(x_pad * psf)
        A*(y)    = conv_psf^*(pad(mask * y))

    For compatibility with the rest of this repo, ``forward`` accepts a
    sensor/image-space tensor [B,C,H,W], center-pads it into the padded object
    space, applies A, then normalizes the simulated measurement to [0,1].
    ``adjoint`` returns the exact adjoint of the unnormalized same-size
    linear operator, i.e. crop(A*(y)).

    Use ``forward_linear`` / ``forward_padded`` and ``adjoint`` /
    ``adjoint_padded`` for direct reference linear-operator calculations.
    """
    def __init__(
        self,
        psf: torch.Tensor,
        im_hw: tuple[int, int],
        mask: torch.Tensor | None = None,
        padded_hw: tuple[int, int] | None = None,
        normalize_output: bool = True,
        normalize_eps: float = 1e-12,
    ):
        super().__init__()
        psf = _to_1chw(psf)  # [1,C,h,w]
        self.register_buffer("psf", psf)
        self.dc_safety = 0.05
        self.normalize_output = bool(normalize_output)
        self.normalize_eps = float(normalize_eps)

        self.im_hw = (int(im_hw[0]), int(im_hw[1]))
        _, C, h, w = psf.shape
        self.psf_hw = (h, w)

        H, W = self.im_hw
        if padded_hw is None:
            padded_hw = (2 * H, 2 * W)
        self.padded_hw = (int(padded_hw[0]), int(padded_hw[1]))
        self.full_hw = self.padded_hw

        otf = psf_to_otf_linear(psf, self.padded_hw)  # [1,C,Hp,Wp]
        self.register_buffer("otf", otf)

        if mask is None:
            mask = torch.ones((1, C, H, W), device=psf.device, dtype=psf.dtype)
        else:
            mask = _to_mask_1chw(mask.to(device=psf.device), C, self.im_hw, psf.dtype)
        self.register_buffer("mask", mask)

    def _center_embed(self, x: torch.Tensor) -> torch.Tensor:
        return _center_pad(x, self.padded_hw)

    def _center_crop(self, y_full: torch.Tensor) -> torch.Tensor:
        return _center_crop(y_full, self.im_hw)

    def _max_normalize_and_clip(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scale each BCHW measurement by its own maximum value and clip to [0,1].
        """
        scale = y.amax(dim=(1, 2, 3), keepdim=True).clamp_min(self.normalize_eps)
        return (y / scale).clamp(0.0, 1.0)

    def forward_padded(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference A(x): padded/object space [B,C,Hp,Wp] -> sensor [B,C,H,W].
        """
        if x.shape[-2:] != self.padded_hw:
            raise ValueError(f"forward_padded expects spatial shape {self.padded_hw}, got {tuple(x.shape[-2:])}")
        X = torch.fft.fft2(x, dim=(-2, -1))
        Y = X * self.otf  # broadcast over B
        y_full = torch.fft.ifft2(Y, dim=(-2, -1)).real
        return self.mask * self._center_crop(y_full)

    def adjoint_padded(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reference A*(y): sensor [B,C,H,W] -> padded/object space [B,C,Hp,Wp].
        """
        if y.shape[-2:] != self.im_hw:
            raise ValueError(f"adjoint_padded expects spatial shape {self.im_hw}, got {tuple(y.shape[-2:])}")
        y_pad = self._center_embed(self.mask * y)
        Y = torch.fft.fft2(y_pad, dim=(-2, -1))
        X = Y * torch.conj(self.otf)
        return torch.fft.ifft2(X, dim=(-2, -1)).real

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalized same-size linear operator: sensor/image space -> sensor space."""
        if x.shape[-2:] != self.im_hw:
            raise ValueError(
                f"forward_linear expects spatial shape {self.im_hw}, got {tuple(x.shape[-2:])}. "
                "Use forward_padded for padded/object-space inputs."
            )
        return self.forward_padded(self._center_embed(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Practical same-size API: normalize H(x) to match image-like [0,1] measurements.
        y = self.forward_linear(x)
        if self.normalize_output:
            y = self._max_normalize_and_clip(y)
        return y

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Exact adjoint of the same-size ``forward_linear`` operator.
        """
        return self._center_crop(self.adjoint_padded(y))


# Backward-compatible name used by older analysis scripts.
FFTConvOperator = FFTLinearConvOperator
