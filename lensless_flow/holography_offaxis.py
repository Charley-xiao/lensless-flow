"""Off-axis, image-plane digital holography (no learned transfer function).

I = |O + R|^2 = |O|^2 + |R|^2 + O R* + O* R,
where O is a complex object field and R is a tilted plane reference.
A separated Fourier order gives O R* up to conjugation. This module keeps
its measured amplitude and estimates only carrier tilt and phase piston.

The background compensation assumes a predominantly flat background. Its
conjugation rule assumes isolated cells have positive optical path delay.
Neither assumption establishes the original PNG label's phase origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from skimage.restoration import unwrap_phase


@dataclass
class OffAxisField:
    cross_field: np.ndarray
    carrier_yx: tuple[float, float]  # cycles per pixel; FFT-bin estimate
    bandwidth: float
    sideband_peak_fraction: float


def _image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 32:
        raise ValueError("Expected a grayscale HW image with both dimensions >= 32.")
    if not np.isfinite(value).all():
        raise ValueError("Image contains non-finite values.")
    return value


def _coordinates(shape: tuple[int, int]):
    h, w = shape
    y, x = np.mgrid[:h, :w]
    return (x - w / 2) / w, (y - h / 2) / h


def extract_offaxis_field(
    intensity: np.ndarray,
    *,
    bandwidth: float = 0.08,
    taper: float = 0.015,
    dc_exclusion: float = 0.16,
    carrier_yx: tuple[float, float] | None = None,
) -> OffAxisField:
    """Isolate one order using an analytic raised-cosine circular aperture.

    All frequency parameters use cycles/pixel. The detection window is used
    only to find the carrier; reconstruction uses the original intensities.
    No target or fitted Fourier coefficients are accepted. Constant gain or
    offset preserves phase for an ideally separated sideband. With an explicit
    fractional carrier, finite-crop DC leakage makes offset invariance only
    approximate. Cropped holograms have periodic FFT boundary artifacts.
    """
    y = _image(intensity)
    if not 0 < taper <= bandwidth < 0.5 or not 0 < dc_exclusion < 0.5:
        raise ValueError("Require 0 < taper <= bandwidth < .5 and 0 < dc_exclusion < .5.")
    if np.std(y) < 1e-10:
        raise ValueError("No detectable carrier in a constant hologram.")
    h, w = y.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    detection = np.fft.fft2((y - y.mean()) * np.outer(np.hanning(h), np.hanning(w)))
    search = (fy < 0) & (np.hypot(fy, fx) > dc_exclusion)
    power = abs(detection) ** 2
    if carrier_yx is None:
        peak = np.unravel_index(np.argmax(np.where(search, power, 0)), y.shape)
        cy, cx = float(fy[peak]), float(fx[peak])
    else:
        cy, cx = map(float, carrier_yx)
        if not np.isfinite([cy, cx]).all() or max(abs(cy), abs(cx)) >= 0.5:
            raise ValueError("Carrier frequencies must be finite and strictly inside Nyquist.")
        peak = (int(round(cy * h)) % h, int(round(cx * w)) % w)
    # Conservative separation: central intensity order can be twice as wide
    # as either cross order. Also prevent the aperture crossing Nyquist.
    available = min(np.hypot(cy, cx) / 3, 0.5 - abs(cy), 0.5 - abs(cx))
    if bandwidth >= available:
        raise ValueError(f"bandwidth={bandwidth:g} risks order overlap/aliasing; use < {available:.4f}.")
    yy, xx = np.mgrid[:h, :w]
    demodulated = y * np.exp(-2j * np.pi * (cy * yy + cx * xx))
    ramp = np.clip((bandwidth - np.hypot(fy, fx)) / taper, 0, 1)
    aperture = 0.5 - 0.5 * np.cos(np.pi * ramp)
    field = np.fft.ifft2(np.fft.fft2(demodulated) * aperture)
    fraction = float(power[peak] / max(float(power[search].sum()), 1e-20))
    return OffAxisField(field, (cy, cx), bandwidth, fraction)


def compensate_background(field: OffAxisField, *, border: int = 16) -> dict:
    """Input-only carrier refinement, background piston, and RBC sign rule.

    Maximize circular phase concentration with two tilt parameters. This
    avoids treating phase wraps as edges during carrier optimization. The
    final 2D unwrapping is used only for the positive-path-delay sign rule
    and for exporting relative unwrapped phase; it is not a measured label.
    """
    c = field.cross_field / np.maximum(abs(field.cross_field), 1e-12)
    h, w = c.shape
    if not 0 <= border < min(h, w) // 3:
        raise ValueError("border must be nonnegative and less than one third of the image size.")
    stop = -border if border else None
    sl = (slice(border, stop, 2), slice(border, stop, 2))
    xx, yy = _coordinates(c.shape)
    gx = np.angle(c[:, 1:] * c[:, :-1].conj())
    gy = np.angle(c[1:] * c[:-1].conj())
    initial = [np.median(gx[sl]) * w, np.median(gy[sl]) * h]
    features = np.stack([xx[sl].ravel(), yy[sl].ravel()], axis=1)
    values = c[sl].ravel()

    def objective(b):
        z = values * np.exp(-1j * (features @ b))
        mean = z.mean()
        grad = (-1j * z[:, None] * features).mean(0)
        return -abs(mean) ** 2, -2 * np.real(mean.conjugate() * grad)

    fit = minimize(objective, initial, jac=True, method="BFGS", options={"maxiter":100})
    phase_plane = fit.x[0] * xx + fit.x[1] * yy
    compensated = c * np.exp(-1j * phase_plane)
    piston = float(np.angle(compensated[sl].mean()))
    compensated *= np.exp(-1j * piston)
    unwrapped = unwrap_phase(np.angle(compensated), rng=0)
    unwrapped -= np.median(unwrapped[sl])
    centered = unwrapped[sl]
    skew = float(np.mean(centered ** 3))
    sign = 1 if skew >= 0 else -1
    positive_phase = sign * np.angle(compensated)
    # Empirical PNG convention evaluated on training pairs: negative of
    # positive-path-delay phase, with background near mid-gray. The piston
    # is a chosen gauge and cannot recover a parent frame's arbitrary offset.
    png = (np.angle(np.exp(-1j * positive_phase)) + np.pi) / (2 * np.pi)
    cy, cx = field.carrier_yx
    return {
        "phase_png": png.astype(np.float32),
        "relative_phase_rad": (sign * unwrapped).astype(np.float32),
        "cross_amplitude": abs(field.cross_field).astype(np.float32),
        "carrier_yx": [cy + float(fit.x[1]) / (2 * np.pi * h),
                       cx + float(fit.x[0]) / (2 * np.pi * w)],
        "background_concentration": float(abs(compensated[sl].mean())),
        "positive_path_sign": sign,
        "sign_skew": skew,
        "piston_rad": piston,
        "tilt_xy_rad_per_image": fit.x.tolist(),
        "optimizer_success": bool(fit.success),
    }


def phase_compatibility_diagnostic(
    field: OffAxisField, target_png: np.ndarray, *, border: int = 16
) -> dict:
    """LABEL-ASSISTED model test, never a deployment reconstruction.

    Fit conjugation and three affine phase parameters on a sparse grid of
    target pixels. Evaluate circular agreement on disjoint interior pixels.
    Fixed label coding is 2*pi*PNG. No spatial operator/scale is learned.
    The fitted PNG is exported solely to visualize model compatibility.
    """
    target = _image(target_png)
    if target.shape != field.cross_field.shape:
        raise ValueError("Hologram and target shapes must match.")
    if not 0 <= border < min(target.shape) // 3:
        raise ValueError("Invalid diagnostic border.")
    xx, yy = _coordinates(target.shape)
    stop = -border if border else None
    sl = (slice(border, stop, 4), slice(border, stop, 4))
    features = np.stack([xx[sl].ravel(), yy[sl].ravel()], axis=1)
    design = np.column_stack([features, np.ones(features.shape[0])])
    raw = np.angle(field.cross_field)
    best = None
    for sign in (-1, 1):
        difference = sign * raw - 2 * np.pi * target
        # Keep even the initializer independent of evaluation-label pixels.
        unwrapped = unwrap_phase(np.angle(np.exp(1j * difference[sl])), rng=0)
        initial = -np.linalg.lstsq(design, unwrapped.ravel(), rcond=None)[0][:2]
        base = difference[sl].ravel()

        def objective(b):
            z = np.exp(1j * (base + features @ b))
            mean = z.mean()
            grad = (1j * z[:, None] * features).mean(0)
            return -abs(mean) ** 2, -2 * np.real(mean.conjugate() * grad)

        fit = minimize(objective, initial, jac=True, method="BFGS", options={"maxiter":120})
        if best is None or fit.fun < best[0]:
            best = (fit.fun, sign, fit.x, bool(fit.success))
    _, sign, tilt, success = best
    phi = sign * raw + tilt[0] * xx + tilt[1] * yy
    piston = -float(np.angle(np.exp(1j * (phi[sl] - 2*np.pi*target[sl])).mean()))
    phi += piston
    diff = np.angle(np.exp(1j * (phi - 2*np.pi*target)))
    evaluation = np.zeros(target.shape, dtype=bool)
    evaluation[border:stop, border:stop] = True
    evaluation[sl] = False
    residual = diff[evaluation]
    return {
        "phase_png": (np.mod(phi, 2*np.pi)/(2*np.pi)).astype(np.float32),
        "coherence": float(abs(np.exp(1j*residual).mean())),
        "circular_rmse_rad": float(np.sqrt(np.mean(residual**2))),
        "sign": sign,
        "piston_rad": piston,
        "tilt_xy_rad_per_image": tilt.tolist(),
        "optimizer_success": success,
        "fit_pixels": int(features.shape[0]),
        "evaluation_pixels": int(evaluation.sum()),
    }
