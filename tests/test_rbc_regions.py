"""Synthetic safeguards for detached, circular target-only RBC pseudo-regions."""
import unittest

import numpy as np
import torch

from lensless_flow.rbc_regions import rbc_region_mask, region_loss_config


def synthetic_cell(center=(64, 64), background=0.25, cell_value=0.65):
    yy, xx = np.indices((128, 128))
    radius = np.hypot(yy - center[0], xx - center[1])
    image = np.full(radius.shape, background, dtype=np.float32)
    image[radius <= 22] = cell_value
    image[radius < 9] = np.mod(cell_value + 0.42, 1)
    return torch.from_numpy(image)[None, None], radius


class RBCRegionTests(unittest.TestCase):
    def test_mask_is_detached_and_preserves_batch_shape_and_device(self):
        target, _ = synthetic_cell()
        target.requires_grad_(True)
        mask = rbc_region_mask(target)
        self.assertEqual(mask.shape, target.shape)
        self.assertEqual(mask.device, target.device)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertFalse(mask.requires_grad)
        self.assertIsNone(mask.grad_fn)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))

    def test_constant_and_smooth_wrapping_backgrounds_are_empty(self):
        ramp = torch.linspace(0.92, 1.08, 128)[None, None, None].expand(1, 1, 128, 128) % 1
        for target in (torch.zeros(1, 1, 128, 128), torch.full((1, 1, 128, 128), 0.73), ramp):
            with self.subTest(minimum=float(target.min()), maximum=float(target.max())):
                self.assertEqual(float(rbc_region_mask(target).sum()), 0)

    def test_bright_dark_and_wrapped_cell_interiors_are_included(self):
        for background, value in ((0.25, 0.65), (0.75, 0.35), (0.4, 0.95)):
            with self.subTest(background=background, value=value):
                target, radius = synthetic_cell(background=background, cell_value=value)
                mask = rbc_region_mask(target)[0, 0].numpy()
                self.assertGreater(float(mask[radius <= 20].mean()), 0.99)
                self.assertLess(float(mask[radius >= 38].mean()), 0.01)

    def test_inversion_and_phase_origin_do_not_change_region(self):
        target, _ = synthetic_cell()
        original = rbc_region_mask(target)
        for altered in (1 - target, (target + 0.137) % 1, (target + 0.81) % 1):
            changed = rbc_region_mask(altered)
            self.assertLess(float((changed != original).float().mean()), 0.002)

    def test_masks_follow_target_order_and_do_not_depend_on_other_batch_images(self):
        first, _ = synthetic_cell(center=(34, 40))
        second, _ = synthetic_cell(center=(91, 88), background=0.7, cell_value=0.2)
        targets = torch.cat((first, second))
        separate = torch.cat((rbc_region_mask(first), rbc_region_mask(second)))
        together = rbc_region_mask(targets)
        self.assertTrue(torch.equal(separate, together))
        permutation = torch.tensor([1, 0])
        self.assertTrue(torch.equal(rbc_region_mask(targets[permutation]), together[permutation]))

    def test_area_and_coverage_safeguards_fall_back_to_empty(self):
        target, _ = synthetic_cell()
        self.assertGreater(float(rbc_region_mask(target).sum()), 0)
        for options in ({"min_component_area": 1000000}, {"min_fraction": 0.5}, {"max_fraction": 0.05}):
            with self.subTest(options=options):
                self.assertEqual(float(rbc_region_mask(target, **options).sum()), 0)

    def test_invalid_shapes_values_and_parameters_fail(self):
        invalid = (torch.zeros(128, 128), torch.zeros(1, 2, 128, 128),
                   torch.zeros(0, 1, 128, 128), torch.zeros(1, 1, 1, 128),
                   torch.full((1, 1, 128, 128), float("nan")),
                   torch.full((1, 1, 128, 128), float("inf")))
        for target in invalid:
            with self.subTest(shape=target.shape):
                with self.assertRaises(ValueError):
                    rbc_region_mask(target)
        target, _ = synthetic_cell()
        for options in ({"window_size": 2}, {"window_size": 4},
                        {"dispersion_threshold": 0}, {"dispersion_threshold": 1},
                        {"closing_iterations": 0}, {"min_component_area": 0},
                        {"reflect_pad": 2}, {"min_fraction": -0.1},
                        {"max_fraction": 1.1}, {"min_fraction": 0.9, "max_fraction": 0.1}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    rbc_region_mask(target, **options)

    def test_region_config_missing_and_explicit_options(self):
        self.assertEqual(region_loss_config({}), {})
        options = {"enabled": True, "balance_mix": 0.5, "foreground_weight": 0.75}
        self.assertEqual(region_loss_config({"cfm": {"loss": {"rbc_region": options}}}), options)


if __name__ == "__main__":
    unittest.main()
