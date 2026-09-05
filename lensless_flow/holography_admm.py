from __future__ import annotations

from dataclasses import dataclass

import torch


def _as_complex_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(phase, dtype=torch.float32), phase.float())


def measured_amplitude(
    intensity: torch.Tensor,
    *,
    mode: str = "raw",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert a normalized hologram image into the amplitude constraint used by
    phase-retrieval ADMM.

    ``raw`` uses the PNG values as intensity directly. ``mean_one`` divides
    each image by its own mean, which is common for background-normalized
    in-line holograms, but it is opt-in here so the solver stays vanilla.
    """
    if intensity.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(intensity.shape)}")

    key = str(mode).strip().lower().replace("-", "_")
    intensity = intensity.float().clamp_min(0.0)
    if key in {"raw", "none"}:
        scaled = intensity
    elif key in {"mean_one", "mean"}:
        scaled = intensity / intensity.mean(dim=(1, 2, 3), keepdim=True).clamp_min(float(eps))
    elif key in {"max_one", "max"}:
        scaled = intensity / intensity.amax(dim=(1, 2, 3), keepdim=True).clamp_min(float(eps))
    else:
        raise ValueError("amplitude mode must be 'raw', 'mean_one', or 'max_one'.")
    return scaled.clamp_min(0.0).sqrt()


@dataclass
class AngularSpectrumPropagator:
    transfer: torch.Tensor
    im_hw: tuple[int, int]
    pixel_size_m: float
    wavelength_m: float
    distance_m: float

    @classmethod
    def create(
        cls,
        im_hw: tuple[int, int],
        *,
        pixel_size_m: float,
        wavelength_m: float,
        distance_m: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "AngularSpectrumPropagator":
        if pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive.")
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive.")

        device = torch.device(device)
        height, width = int(im_hw[0]), int(im_hw[1])
        fy = torch.fft.fftfreq(height, d=float(pixel_size_m), device=device).to(dtype)
        fx = torch.fft.fftfreq(width, d=float(pixel_size_m), device=device).to(dtype)
        yy = fy.view(height, 1)
        xx = fx.view(1, width)
        wavelength = torch.as_tensor(float(wavelength_m), device=device, dtype=dtype)
        k = torch.as_tensor(2.0 * torch.pi / float(wavelength_m), device=device, dtype=dtype)
        root = 1.0 - (wavelength * xx).square() - (wavelength * yy).square()

        propagating = root >= 0.0
        phase = k * float(distance_m) * root.clamp_min(0.0).sqrt()
        decay = torch.exp(-k * abs(float(distance_m)) * (-root).clamp_min(0.0).sqrt())
        real = torch.where(propagating, torch.cos(phase), decay)
        imag = torch.where(propagating, torch.sin(phase), torch.zeros_like(phase))
        transfer = torch.complex(real, imag).view(1, 1, height, width)
        return cls(
            transfer=transfer,
            im_hw=(height, width),
            pixel_size_m=float(pixel_size_m),
            wavelength_m=float(wavelength_m),
            distance_m=float(distance_m),
        )

    @property
    def device(self) -> torch.device:
        return self.transfer.device

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        self._validate(field)
        return torch.fft.ifft2(torch.fft.fft2(field, dim=(-2, -1)) * self.transfer, dim=(-2, -1))

    def adjoint(self, field: torch.Tensor) -> torch.Tensor:
        self._validate(field)
        return torch.fft.ifft2(torch.fft.fft2(field, dim=(-2, -1)) * self.transfer.conj(), dim=(-2, -1))

    def _validate(self, field: torch.Tensor) -> None:
        if field.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(field.shape)}")
        if tuple(int(v) for v in field.shape[-2:]) != self.im_hw:
            raise ValueError(f"Expected HW={self.im_hw}, got {tuple(field.shape[-2:])}")
        if int(field.shape[1]) != 1:
            raise ValueError("The vanilla holography ADMM implementation expects one grayscale channel.")


def _amplitude_prox(
    value: torch.Tensor,
    measured_amp: torch.Tensor,
    *,
    rho: float,
    mode: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    angle_factor = value / value.abs().clamp_min(float(eps))
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"hard", "projection", "project"}:
        magnitude = measured_amp
    elif key in {"soft", "quadratic"}:
        magnitude = (measured_amp + float(rho) * value.abs()) / (1.0 + float(rho))
    else:
        raise ValueError("amplitude prox mode must be 'soft' or 'hard'.")
    return magnitude.to(value.dtype) * angle_factor


def _phase_to_normalized(phase: torch.Tensor, mode: str) -> torch.Tensor:
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"wrapped_pm_pi", "minus_pi_pi", "pm_pi"}:
        return ((phase + torch.pi) / (2.0 * torch.pi)).clamp(0.0, 1.0)
    if key in {"wrapped_0_2pi", "zero_2pi"}:
        return torch.remainder(phase, 2.0 * torch.pi) / (2.0 * torch.pi)
    if key in {"cosine", "cos"}:
        return (torch.cos(phase) + 1.0) * 0.5
    raise ValueError("phase output mode must be 'wrapped_pm_pi', 'wrapped_0_2pi', or 'cosine'.")


@torch.no_grad()
def vanilla_holography_admm(
    hologram_intensity: torch.Tensor,
    propagator: AngularSpectrumPropagator,
    *,
    iterations: int = 100,
    rho: float = 1.0,
    amplitude_mode: str = "raw",
    amplitude_prox: str = "soft",
    init: str = "zeros",
    output_mode: str = "wrapped_pm_pi",
    eps: float = 1e-8,
    return_phase_radians: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Vanilla single-plane phase-retrieval ADMM for an in-line hologram.

    It solves with the explicit model:

        object field  u0 = exp(i * phi)
        sensor field  uz = P_z u0
        measurement   y  ~= |uz|^2

    using a detector amplitude proximal step and a phase-only object projection.
    No paired target image is used by the solver.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    y = hologram_intensity.to(device=propagator.device, dtype=torch.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(y.shape)}")
    if tuple(int(v) for v in y.shape[-2:]) != propagator.im_hw:
        raise ValueError(f"Expected HW={propagator.im_hw}, got {tuple(y.shape[-2:])}")

    amp = measured_amplitude(y, mode=amplitude_mode, eps=eps)
    init_key = str(init).strip().lower().replace("-", "_")
    if init_key in {"zeros", "zero", "flat"}:
        phase = torch.zeros_like(y)
    elif init_key in {"random", "rand"}:
        phase = (torch.rand_like(y) - 0.5) * (2.0 * torch.pi)
    elif init_key in {"backprop", "back_propagation", "adjoint"}:
        detector_field = torch.complex(amp, torch.zeros_like(amp))
        phase = torch.angle(propagator.adjoint(detector_field))
    else:
        raise ValueError("init must be 'zeros', 'random', or 'backprop'.")

    object_field = _as_complex_phase(phase)
    dual = torch.zeros_like(object_field)
    for _ in range(int(iterations)):
        detector_prediction = propagator.forward(object_field)
        detector_field = _amplitude_prox(
            detector_prediction + dual,
            amp,
            rho=float(rho),
            mode=amplitude_prox,
            eps=eps,
        )
        object_update = propagator.adjoint(detector_field - dual)
        phase = torch.angle(object_update)
        object_field = _as_complex_phase(phase)
        dual = dual + detector_prediction - detector_field

    phase_norm = _phase_to_normalized(phase, output_mode)
    if return_phase_radians:
        return phase_norm, phase
    return phase_norm


@torch.no_grad()
def simulate_hologram_from_normalized_phase(
    phase_norm: torch.Tensor,
    propagator: AngularSpectrumPropagator,
    *,
    phase_output_mode: str = "wrapped_pm_pi",
    normalize: str = "max_one",
    eps: float = 1e-8,
) -> torch.Tensor:
    key = str(phase_output_mode).strip().lower().replace("-", "_")
    if key in {"wrapped_pm_pi", "minus_pi_pi", "pm_pi"}:
        phase = phase_norm.float().clamp(0.0, 1.0) * (2.0 * torch.pi) - torch.pi
    elif key in {"wrapped_0_2pi", "zero_2pi"}:
        phase = phase_norm.float().clamp(0.0, 1.0) * (2.0 * torch.pi)
    else:
        raise ValueError("phase_output_mode must be 'wrapped_pm_pi' or 'wrapped_0_2pi'.")

    sensor_field = propagator.forward(_as_complex_phase(phase.to(propagator.device)))
    intensity = sensor_field.abs().square().float()
    norm_key = str(normalize).strip().lower().replace("-", "_")
    if norm_key in {"none", "raw"}:
        return intensity
    if norm_key in {"max_one", "max"}:
        return intensity / intensity.amax(dim=(1, 2, 3), keepdim=True).clamp_min(float(eps))
    if norm_key in {"mean_one", "mean"}:
        return intensity / intensity.mean(dim=(1, 2, 3), keepdim=True).clamp_min(float(eps))
    raise ValueError("normalize must be 'raw', 'max_one', or 'mean_one'.")
