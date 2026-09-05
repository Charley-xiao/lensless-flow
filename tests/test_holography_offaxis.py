"""Synthetic physical checks for analytic off-axis hologram reconstruction.

Run with ``python -m unittest discover -s tests -p test_holography_offaxis.py``.
Ground truth is a complex object/reference interference field, not an RBC
training pair. Interior checks exclude FFT crop-boundary artifacts.
"""
import unittest

import numpy as np

from lensless_flow.holography_offaxis import (
    compensate_background,
    extract_offaxis_field,
    phase_compatibility_diagnostic,
)


SIZE = 192
BORDER = 24
INTERIOR = np.s_[BORDER:-BORDER, BORDER:-BORDER]


def interference(phase, amplitude, carrier_yx):
    """Physical image-plane interference of object and tilted reference."""
    yy, xx = np.indices(phase.shape)
    cy, cx = carrier_yx
    obj = amplitude * np.exp(1j * phase)
    reference = np.exp(-2j * np.pi * (cy * yy + cx * xx))
    return np.abs(obj + reference) ** 2


def isolated_object():
    yy, xx = np.indices((SIZE, SIZE), dtype=float)
    x = (xx - SIZE / 2) / SIZE
    y = (yy - SIZE / 2) / SIZE
    phase = (
        2.0 * np.exp(-((x + 0.10) / 0.075) ** 2 - ((y - 0.06) / 0.10) ** 2)
        + 0.85 * np.exp(-((x - 0.16) / 0.055) ** 2 - ((y + 0.13) / 0.065) ** 2)
    )
    amplitude = 0.82 - 0.16 * np.exp(-(x / 0.14) ** 2 - (y / 0.12) ** 2)
    return phase, amplitude


def circular_rmse_up_to_piston(observed, expected):
    delta = observed[INTERIOR] - expected[INTERIOR]
    piston = np.angle(np.exp(1j * delta).mean())
    residual = np.angle(np.exp(1j * (delta - piston)))
    return float(np.sqrt(np.mean(residual ** 2)))


class OffAxisPhysicalTests(unittest.TestCase):
    def test_fractional_carrier_recovers_phase_and_amplitude(self):
        phase, amplitude = isolated_object()
        carrier = (-56.4 / SIZE, 35.2 / SIZE)
        intensity = interference(phase, amplitude, carrier)
        field = extract_offaxis_field(intensity, carrier_yx=carrier)

        self.assertLess(circular_rmse_up_to_piston(np.angle(field.cross_field), phase), 0.015)
        amplitude_rmse = np.sqrt(np.mean((np.abs(field.cross_field)[INTERIOR] - amplitude[INTERIOR]) ** 2))
        self.assertLess(amplitude_rmse, 0.012)

    def test_input_only_reconstruction_survives_flips_and_rotation(self):
        phase, amplitude = isolated_object()
        intensity = interference(phase, amplitude, (-56.4 / SIZE, 35.2 / SIZE))
        transforms = {"identity": lambda a: a, "flip_x": np.fliplr,
                      "flip_y": np.flipud, "rotate_90": np.rot90}
        for name, transform in transforms.items():
            with self.subTest(transform=name):
                field = extract_offaxis_field(transform(intensity))
                result = compensate_background(field, border=BORDER)
                self.assertLess(
                    circular_rmse_up_to_piston(result["relative_phase_rad"], transform(phase)),
                    0.08,
                )
                self.assertGreater(float(result["relative_phase_rad"][INTERIOR].max()), 1.7)

    def test_constant_gain_and_offset_preserve_phase(self):
        phase, amplitude = isolated_object()
        # An integer carrier makes the constant/DC order exactly orthogonal
        # to the sideband on this periodic finite grid.
        carrier = (-56 / SIZE, 35 / SIZE)
        intensity = interference(phase, amplitude, carrier)
        original = extract_offaxis_field(intensity, carrier_yx=carrier)
        changed = extract_offaxis_field(3.2 * intensity + 1.1, carrier_yx=carrier)
        self.assertLess(
            circular_rmse_up_to_piston(np.angle(changed.cross_field), np.angle(original.cross_field)),
            1e-10,
        )
        np.testing.assert_allclose(changed.cross_field, 3.2 * original.cross_field, atol=1e-11, rtol=1e-11)

    def test_diagnostic_distinguishes_matched_from_wrong_and_shifted_labels(self):
        yy, xx = np.indices((SIZE, SIZE), dtype=float)
        x, y = xx / SIZE, yy / SIZE
        phase = (1.1 * np.sin(4 * np.pi * x) + 0.8 * np.cos(6 * np.pi * y)
                 + 0.45 * np.sin(4 * np.pi * (x + y)))
        amplitude = 0.8 + 0.1 * np.cos(2 * np.pi * x)
        carrier = (-56.4 / SIZE, 35.2 / SIZE)
        field = extract_offaxis_field(interference(phase, amplitude, carrier))
        labels = np.mod(phase + 1.2, 2 * np.pi) / (2 * np.pi)
        matched = phase_compatibility_diagnostic(field, labels, border=BORDER)
        shifted = phase_compatibility_diagnostic(field, np.roll(labels, (31, 23), axis=(0, 1)), border=BORDER)
        wrong_phase = 1.5 * np.sin(6 * np.pi * x) + 1.2 * np.cos(4 * np.pi * y)
        wrong = phase_compatibility_diagnostic(field, np.mod(wrong_phase, 2 * np.pi) / (2 * np.pi), border=BORDER)

        self.assertGreater(matched["coherence"], 0.995)
        self.assertLess(matched["circular_rmse_rad"], 0.10)
        for result in (shifted, wrong):
            self.assertLess(result["coherence"], 0.85)
            self.assertGreater(result["circular_rmse_rad"], 0.5)
        # The diagnostic evaluates other interior pixels, not its fit grid.
        interior_pixels = (SIZE - 2 * BORDER) ** 2
        self.assertEqual(matched["fit_pixels"], ((SIZE - 2 * BORDER) // 4) ** 2)
        self.assertEqual(matched["fit_pixels"] + matched["evaluation_pixels"], interior_pixels)
        self.assertGreater(matched["evaluation_pixels"], 10 * matched["fit_pixels"])

    def test_invalid_images_and_frequency_parameters(self):
        phase, amplitude = isolated_object()
        intensity = interference(phase, amplitude, (-56 / SIZE, 35 / SIZE))
        for bad in (np.ones((SIZE, SIZE)), np.zeros((16, SIZE)),
                    np.zeros((SIZE, SIZE, 1)), np.full((SIZE, SIZE), np.nan),
                    np.full((SIZE, SIZE), np.inf)):
            with self.subTest(image_shape=bad.shape, first_value=bad.flat[0]):
                with self.assertRaises(ValueError):
                    extract_offaxis_field(bad)
        for params in ({"bandwidth": 0}, {"bandwidth": 0.5}, {"taper": 0},
                       {"taper": 0.09}, {"dc_exclusion": 0}, {"dc_exclusion": 0.5},
                       {"carrier_yx": (0, 0)}, {"carrier_yx": (0.5, 0.2)},
                       {"carrier_yx": (np.nan, 0.2)}, {"carrier_yx": (0.49, 0.2)},
                       {"bandwidth": 0.13, "carrier_yx": (-0.3, 0.2)}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    extract_offaxis_field(intensity, **params)

    def test_excluded_diagnostic_labels_cannot_change_fitted_prediction(self):
        phase, amplitude = isolated_object()
        field = extract_offaxis_field(
            interference(phase, amplitude, (-56.4 / SIZE, 35.2 / SIZE))
        )
        _, xx = np.indices(phase.shape)
        labels = np.mod(phase + 1.2, 2 * np.pi) / (2 * np.pi)
        # The pi/2-per-pixel ramp vanishes modulo 2*pi on every fourth
        # column. Full-image target-dependent unwrapping would nevertheless
        # use excluded labels to select a different affine phase alias.
        aliased_labels = np.mod(phase + 1.2 + (np.pi / 2) * xx, 2 * np.pi) / (2 * np.pi)
        fit_grid = np.s_[BORDER:-BORDER:4, BORDER:-BORDER:4]
        aliased_labels[fit_grid] = labels[fit_grid]
        original = phase_compatibility_diagnostic(field, labels, border=BORDER)
        changed = phase_compatibility_diagnostic(field, aliased_labels, border=BORDER)

        self.assertEqual(original["sign"], changed["sign"])
        np.testing.assert_allclose(original["tilt_xy_rad_per_image"], changed["tilt_xy_rad_per_image"], atol=1e-10, rtol=0)
        self.assertAlmostEqual(original["piston_rad"], changed["piston_rad"], places=10)
        np.testing.assert_array_equal(original["phase_png"], changed["phase_png"])
        self.assertGreater(changed["circular_rmse_rad"], original["circular_rmse_rad"] + 1.0)

    def test_invalid_diagnostic_shapes_and_borders(self):
        phase, amplitude = isolated_object()
        field = extract_offaxis_field(interference(phase, amplitude, (-56 / SIZE, 35 / SIZE)))
        with self.assertRaises(ValueError):
            phase_compatibility_diagnostic(field, np.zeros((SIZE + 1, SIZE)))
        for border in (-1, SIZE // 3):
            with self.subTest(border=border):
                with self.assertRaises(ValueError):
                    compensate_background(field, border=border)
                with self.assertRaises(ValueError):
                    phase_compatibility_diagnostic(field, phase, border=border)


if __name__ == "__main__":
    unittest.main()
