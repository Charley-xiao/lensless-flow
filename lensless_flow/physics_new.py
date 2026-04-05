import torch
import torch.nn as nn

from .tensor_utils import to_nchw


def _to_1chw(psf: torch.Tensor) -> torch.Tensor:
    """
    Normalize a PSF tensor to shape [1,C,H,W].

    Accepted layouts include [H,W], [C,H,W], [H,W,C], and [1,C,H,W].
    """
    if psf.ndim == 2:
        psf = psf[None, None, ...]
    elif psf.ndim == 3:
        if psf.shape[0] in (1, 3) and psf.shape[-1] not in (1, 3):
            psf = psf[None, ...]
        elif psf.shape[-1] in (1, 3) and psf.shape[0] not in (1, 3):
            psf = psf.permute(2, 0, 1)[None, ...]
        else:
            psf = psf[None, ...]
    elif psf.ndim == 4:
        if psf.shape[0] != 1:
            raise ValueError(f"psf batch dim must be 1, got {tuple(psf.shape)}")
    else:
        raise ValueError(f"Unsupported psf ndim={psf.ndim}, shape={tuple(psf.shape)}")
    return psf


def _center_slices(
    big_hw: tuple[int, int],
    small_hw: tuple[int, int],
    offset_hw: tuple[int, int] = (0, 0),
) -> tuple[slice, slice]:
    big_h, big_w = int(big_hw[0]), int(big_hw[1])
    small_h, small_w = int(small_hw[0]), int(small_hw[1])
    off_y, off_x = int(offset_hw[0]), int(offset_hw[1])
    if small_h > big_h or small_w > big_w:
        raise ValueError(f"Cannot center size {small_hw} inside {big_hw}")

    top = (big_h - small_h) // 2 + off_y
    left = (big_w - small_w) // 2 + off_x
    if top < 0 or left < 0 or top + small_h > big_h or left + small_w > big_w:
        raise ValueError(
            f"Offset {offset_hw} makes centered crop/pad invalid: big_hw={big_hw}, small_hw={small_hw}"
        )
    return slice(top, top + small_h), slice(left, left + small_w)


def _pad_center_bchw(
    x: torch.Tensor,
    out_hw: tuple[int, int],
    offset_hw: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    batch, channels, in_h, in_w = x.shape
    rs, cs = _center_slices((out_h, out_w), (in_h, in_w), offset_hw=offset_hw)
    out = torch.zeros((batch, channels, out_h, out_w), device=x.device, dtype=x.dtype)
    out[..., rs, cs] = x
    return out


def _crop_center_bchw(
    x: torch.Tensor,
    out_hw: tuple[int, int],
    offset_hw: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    _, _, in_h, in_w = x.shape
    rs, cs = _center_slices((in_h, in_w), out_hw, offset_hw=offset_hw)
    return x[..., rs, cs]


def _coerce_gain_tensor(
    gain,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if gain is None:
        return torch.ones((1, channels, 1, 1), device=device, dtype=dtype)
    gain_tensor = torch.as_tensor(gain, device=device, dtype=dtype)
    if gain_tensor.ndim == 0:
        gain_tensor = gain_tensor.view(1, 1, 1, 1).expand(1, channels, 1, 1)
    elif gain_tensor.ndim == 1 and int(gain_tensor.shape[0]) == channels:
        gain_tensor = gain_tensor.view(1, channels, 1, 1)
    elif gain_tensor.shape == (1, channels, 1, 1):
        pass
    else:
        raise ValueError(f"Unsupported gain shape {tuple(gain_tensor.shape)} for {channels} channels")
    return gain_tensor.contiguous()


def psf_to_otf_centered(psf_1chw: torch.Tensor, padded_hw: tuple[int, int]) -> torch.Tensor:
    """
    Notebook-style PSF embedding:
    1. center-pad the PSF onto the padded/object grid
    2. ifftshift it so the PSF center moves to the FFT origin
    3. FFT to obtain the padded convolution kernel
    """
    psf_1chw = _to_1chw(psf_1chw)
    _, _, psf_h, psf_w = psf_1chw.shape
    pad_h, pad_w = int(padded_hw[0]), int(padded_hw[1])
    if pad_h < psf_h or pad_w < psf_w:
        raise ValueError(
            f"padded_hw={padded_hw} must be at least as large as PSF size {(psf_h, psf_w)}"
        )

    psf_pad = _pad_center_bchw(psf_1chw, (pad_h, pad_w))
    psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))
    return torch.fft.fft2(psf_pad)


def psf_to_otf_linear(psf_1chw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    Backward-compatible alias used elsewhere in the repo.
    """
    return psf_to_otf_centered(psf_1chw, out_hw)


class FFTLinearConvOperator(nn.Module):
    """
    FFT forward/adjoint pair following the assignment notebook convention.

    We keep the repo's public interface square:
        A(x) = crop_center( conv( pad_center(x), h ) )

    so callers can continue to pass `x` and `y` at image resolution, while the
    internals use a centered padded/object grid of size `padded_hw`.
    """

    def __init__(
        self,
        psf: torch.Tensor,
        im_hw: tuple[int, int],
        padded_hw: tuple[int, int] | None = None,
        crop_offset: tuple[int, int] = (0, 0),
        gain=None,
    ):
        super().__init__()
        psf = _to_1chw(psf)
        self.register_buffer("psf", psf)

        self.im_hw = (int(im_hw[0]), int(im_hw[1]))
        _, channels, psf_h, psf_w = psf.shape
        self.channels = int(channels)
        self.psf_hw = (int(psf_h), int(psf_w))
        self.crop_offset = (int(crop_offset[0]), int(crop_offset[1]))

        if padded_hw is None:
            padded_hw = (
                max(2 * self.im_hw[0], self.psf_hw[0]),
                max(2 * self.im_hw[1], self.psf_hw[1]),
            )
        self.padded_hw = (int(padded_hw[0]), int(padded_hw[1]))
        self.full_hw = self.padded_hw
        if self.padded_hw[0] < self.im_hw[0] or self.padded_hw[1] < self.im_hw[1]:
            raise ValueError(f"padded_hw={self.padded_hw} must be at least im_hw={self.im_hw}")

        base_otf = psf_to_otf_centered(psf, self.padded_hw)
        self.register_buffer("base_otf", base_otf)
        self.register_buffer(
            "gain",
            _coerce_gain_tensor(gain, channels=self.channels, device=psf.device, dtype=psf.dtype),
        )

        self.dc_safety = 0.05

    @property
    def otf(self) -> torch.Tensor:
        return self.base_otf * self.gain

    def set_gain(self, gain) -> torch.Tensor:
        gain_tensor = _coerce_gain_tensor(
            gain,
            channels=self.channels,
            device=self.psf.device,
            dtype=self.psf.dtype,
        )
        self.gain.copy_(gain_tensor)
        return self.gain

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return _pad_center_bchw(x, self.padded_hw)

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        return _crop_center_bchw(x, self.im_hw, offset_hw=self.crop_offset)

    def _fft_conv_padded(self, x_pad: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x_pad)
        return torch.fft.ifft2(X * otf).real

    def _forward_with_otf(self, x: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        if int(x.shape[1]) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {int(x.shape[1])}")
        if tuple(x.shape[-2:]) != self.im_hw:
            raise ValueError(f"Expected x spatial shape {self.im_hw}, got {tuple(x.shape[-2:])}")
        x_pad = self.pad(x)
        y_pad = self._fft_conv_padded(x_pad, otf)
        return self.crop(y_pad)

    def forward_unscaled(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_otf(x, self.base_otf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_otf(x, self.otf)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(y.shape)}")
        if int(y.shape[1]) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {int(y.shape[1])}")
        if tuple(y.shape[-2:]) != self.im_hw:
            raise ValueError(f"Expected y spatial shape {self.im_hw}, got {tuple(y.shape[-2:])}")
        y_pad = self.pad(y)
        x_pad = self._fft_conv_padded(y_pad, torch.conj(self.otf))
        return self.crop(x_pad)


@torch.no_grad()
def calibrate_operator_gain(
    operator: FFTLinearConvOperator,
    dataset,
    num_samples: int = 32,
    per_channel: bool = True,
    start_idx: int = 0,
    step: int = 1,
) -> torch.Tensor:
    """
    Fit a least-squares gain so H(x_gt) matches the dataset measurement scale.

    This is needed because the dataset PSF normalization is not on the same
    absolute scale as the normalized measurements.
    """
    if num_samples <= 0:
        return operator.gain.detach().clone()

    total = min(len(dataset), max(0, int(num_samples)))
    if total <= 0:
        return operator.gain.detach().clone()

    device = operator.psf.device
    if per_channel:
        num = torch.zeros((operator.channels,), device=device, dtype=torch.float32)
        den = torch.zeros((operator.channels,), device=device, dtype=torch.float32)
    else:
        num = torch.zeros((), device=device, dtype=torch.float32)
        den = torch.zeros((), device=device, dtype=torch.float32)

    idx = int(start_idx)
    used = 0
    while used < total and idx < len(dataset):
        y, x = dataset[idx]
        y = to_nchw(y).to(device=device, dtype=operator.psf.dtype)
        x = to_nchw(x).to(device=device, dtype=operator.psf.dtype)
        y_hat = operator.forward_unscaled(x).float()
        y = y.float()

        if per_channel:
            num += (y_hat * y).sum(dim=(0, 2, 3))
            den += (y_hat * y_hat).sum(dim=(0, 2, 3))
        else:
            num += (y_hat * y).sum()
            den += (y_hat * y_hat).sum()

        used += 1
        idx += max(1, int(step))

    gain = num / den.clamp_min(1e-12)
    operator.set_gain(gain)
    return operator.gain.detach().clone()


@torch.no_grad()
def configure_operator_gain(
    operator: FFTLinearConvOperator,
    cfg,
    dataset=None,
    verbose: bool = False,
) -> FFTLinearConvOperator:
    """
    Apply either an explicit gain from config or a data-driven calibration.
    """
    physics_cfg = cfg.get("physics", {}) if isinstance(cfg, dict) else {}

    explicit_gain = physics_cfg.get("gain", None)
    if explicit_gain is not None:
        operator.set_gain(explicit_gain)
        if verbose:
            print(f"[physics] Using configured operator gain: {operator.gain.flatten().tolist()}")
        return operator

    if dataset is not None and bool(physics_cfg.get("auto_gain", True)):
        gain = calibrate_operator_gain(
            operator,
            dataset=dataset,
            num_samples=int(physics_cfg.get("gain_calibration_samples", 32)),
            per_channel=bool(physics_cfg.get("gain_per_channel", True)),
            start_idx=int(physics_cfg.get("gain_calibration_start_idx", 0)),
            step=int(physics_cfg.get("gain_calibration_step", 1)),
        )
        if verbose:
            print(f"[physics] Calibrated operator gain: {gain.flatten().tolist()}")

    return operator
