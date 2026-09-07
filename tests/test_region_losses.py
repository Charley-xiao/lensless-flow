"""Analytical checks for optional region weighting and regional image metrics."""
import math
import unittest

import torch
import torch.nn.functional as F

from lensless_flow.flow_matching import x0_from_xt_v
from lensless_flow.losses import cfm_loss, region_balanced_cfm_loss
from lensless_flow.metrics import region_image_metrics, ssim_map_torch, ssim_torch


class RegionLossTests(unittest.TestCase):
    def test_analytical_allocation_and_gradient_emphasis(self):
        pred = torch.ones(1, 1, 2, 5, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros_like(pred)
        mask[..., 0, 0] = 1
        loss, stats = region_balanced_cfm_loss(pred, target, mask)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(stats["ordinary_foreground_contribution"].item(), 0.1)
        self.assertAlmostEqual(stats["weighted_foreground_contribution"].item(), 0.425)
        loss.backward()
        self.assertAlmostEqual(pred.grad[..., 0, 0].item(), 0.85, places=6)
        self.assertAlmostEqual(pred.grad[..., 0, 1].item(), 0.1 + 0.25 / 9, places=6)

    def test_per_image_and_channel_normalization(self):
        # Equal foreground error in two channels with different foreground area
        # must give them equal influence after regional normalization.
        mask = torch.tensor([[[[1., 0., 0., 0.]], [[1., 1., 1., 0.]]]])
        pred = 2 * mask + (1 - mask)
        loss, stats = region_balanced_cfm_loss(pred, torch.zeros_like(pred), mask,
                                               foreground_weight=.75, balance_mix=1)
        self.assertAlmostEqual(loss.item(), .75 * 4 + .25 * 1)
        self.assertAlmostEqual(stats["foreground_mse"].item(), 4)
        self.assertAlmostEqual(stats["background_mse"].item(), 1)
        # Batch repetition must not change the objective.
        repeated, _ = region_balanced_cfm_loss(pred.repeat(3, 1, 1, 1),
                                               torch.zeros_like(pred).repeat(3, 1, 1, 1),
                                               mask.repeat(3, 1, 1, 1), balance_mix=1)
        torch.testing.assert_close(repeated, loss)

    def test_mask_is_detached_and_broadcasts_channels(self):
        pred = torch.ones(2, 3, 4, 5, requires_grad=True)
        mask = torch.full((2, 1, 4, 5), .2, requires_grad=True)
        loss, stats = region_balanced_cfm_loss(pred, torch.zeros_like(pred), mask)
        loss.backward()
        self.assertIsNone(mask.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(all(not value.requires_grad and value.ndim == 0 for value in stats.values()))

    def test_empty_full_fallback_and_absent_statistics(self):
        pred = torch.arange(8.).reshape(2, 1, 2, 2)
        target = torch.zeros_like(pred)
        for value in [0., 1.]:
            loss, stats = region_balanced_cfm_loss(pred, target, torch.full_like(pred, value))
            torch.testing.assert_close(loss, cfm_loss(pred, target))
            self.assertEqual(stats["valid_mask_fraction"].item(), 0)
            absent = "foreground_mse" if value == 0 else "background_mse"
            self.assertTrue(torch.isnan(stats[absent]))
            torch.testing.assert_close(stats["weighted_foreground_contribution"], loss.detach() * value)

    def test_invalid_and_valid_images_mix_without_discarding_samples(self):
        pred = torch.tensor([[[[2., 1.]]], [[[3., 4.]]]])
        mask = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        loss, stats = region_balanced_cfm_loss(pred, torch.zeros_like(pred), mask, balance_mix=1)
        expected = ((.75 * 4 + .25 * 1) + (9 + 16) / 2) / 2
        self.assertAlmostEqual(loss.item(), expected)
        self.assertEqual(stats["valid_mask_fraction"].item(), .5)

    def test_zero_mix_is_exact_legacy_loss_and_gradient(self):
        pred = torch.linspace(-.8, .9, 120, dtype=torch.float64).reshape(2, 3, 4, 5).requires_grad_()
        target = pred.detach().sin()
        mask = torch.zeros(2, 1, 4, 5)
        mask[..., :2, :] = 1
        old = F.mse_loss(pred, target)
        actual, _ = region_balanced_cfm_loss(pred, target, mask, balance_mix=0)
        self.assertTrue(torch.equal(actual, old))
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(old, pred)[0], rtol=0, atol=0)

    def test_noncontiguous_tensors(self):
        pred = torch.arange(48.).reshape(2, 2, 3, 4).transpose(-1, -2)
        target = torch.ones_like(pred)
        mask = (pred[:, :1] % 3 == 0).float()
        self.assertFalse(pred.is_contiguous())
        loss, stats = region_balanced_cfm_loss(pred, target, mask)
        reference, _ = region_balanced_cfm_loss(pred.contiguous(), target.contiguous(), mask.contiguous())
        torch.testing.assert_close(loss, reference)
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))

    def test_float32_reduction_and_near_endpoint_gradients(self):
        # A half-precision squared residual would overflow at this magnitude.
        pred = torch.full((1, 1, 2, 2), 500., dtype=torch.float16, requires_grad=True)
        mask = torch.tensor([[[[1., 0.], [0., 0.]]]])
        loss, _ = region_balanced_cfm_loss(pred, torch.zeros_like(pred), mask)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

        source = torch.full((2, 1, 2, 2), -.5)
        target = torch.full_like(source, .7)
        t = torch.tensor([1e-7, 1-1e-7])
        ti = t[:, None, None, None]
        xt = (1-ti)*source + ti*target
        vstar = target-source
        velocity = (vstar + .2).detach().requires_grad_()
        endpoint = x0_from_xt_v(xt, velocity, t)
        torch.testing.assert_close(endpoint-target, (1-ti)*(velocity-vstar), atol=1e-7, rtol=1e-5)
        region_loss, _ = region_balanced_cfm_loss(velocity, vstar, mask.expand(2, -1, -1, -1))
        (region_loss + F.mse_loss(endpoint, target)).backward()
        self.assertTrue(torch.isfinite(velocity.grad).all())

    def test_invalid_arguments_rejected(self):
        pred = torch.ones(1, 1, 2, 2)
        for kwargs in [{"foreground_weight": -1}, {"foreground_weight": float("nan")},
                       {"balance_mix": 1.1}, {"balance_mix": float("inf")}]:
            with self.assertRaises(ValueError):
                region_balanced_cfm_loss(pred, pred, pred, **kwargs)
        for mask in [torch.ones(2, 2), torch.ones(2, 1, 2, 2),
                     torch.full_like(pred, -.1), torch.full_like(pred, float("nan"))]:
            with self.assertRaises(ValueError):
                region_balanced_cfm_loss(pred, pred, mask)


class RegionMetricTests(unittest.TestCase):
    def test_ssim_preserves_known_legacy_values(self):
        pred = torch.linspace(0, 1, 2*2*13*17).reshape(2, 2, 13, 17)
        target = torch.flip(pred, [-1]) * .9 + .03
        # Recorded from HEAD's original ssim_torch before refactoring its map.
        for window, expected in [(5, .979926586151123), (6, .9809526801109314), (11, .9817365407943726)]:
            score = ssim_torch(pred, target, window_size=window)
            self.assertAlmostEqual(score.item(), expected, places=6)
            score_map = ssim_map_torch(pred, target, window_size=window)
            self.assertEqual(score_map.shape, pred.shape)
            self.assertTrue(torch.equal(score, score_map.mean()))

    def test_region_scores_use_original_ssim_neighborhoods(self):
        pred = torch.zeros(1, 1, 15, 15)
        target = torch.zeros_like(pred)
        target[..., 7, 7] = 1
        mask = torch.zeros_like(pred)
        mask[..., 7, 8] = 1  # Equal center pixels, unequal neighborhood.
        result = region_image_metrics(pred, target, mask)
        expected = ssim_map_torch(pred, target)[..., 7, 8]
        torch.testing.assert_close(result["rbc_ssim"], expected.flatten())
        self.assertLess(result["rbc_ssim"].item(), .9)
        self.assertEqual(result["rbc_mse"].item(), 0.)
        self.assertEqual(result["rbc_psnr"].item(), 99.)

    def test_circular_wrap_and_absent_regions(self):
        pred = torch.full((2, 1, 3, 4), .99)
        target = torch.full_like(pred, .01)
        mask = torch.cat((torch.ones(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)))
        result = region_image_metrics(pred, target, mask)
        self.assertAlmostEqual(result["rbc_circular_rmse_rad"][0].item(), 2 * math.pi * .02, places=5)
        self.assertAlmostEqual(result["rbc_mse"][0].item(), .98**2, places=6)
        self.assertTrue(torch.isnan(result["background_psnr"][0]))
        self.assertTrue(torch.isnan(result["rbc_psnr"][1]))
        self.assertTrue(torch.isnan(result["rbc_circular_rmse_rad"][1]))
        self.assertTrue(all(value.shape == (2,) for value in result.values()))

    def test_metrics_noncontiguous_multichannel_soft_mask(self):
        pred = torch.linspace(0, 1, 120).reshape(2, 3, 4, 5).transpose(-1, -2)
        target = pred * .8
        mask = torch.full((2, 1, 4, 5), .3).transpose(-1, -2)
        result = region_image_metrics(pred, target, mask, window_size=3)
        actual_mse = (pred-target).square().mean((1, 2, 3))
        torch.testing.assert_close(result["rbc_mse"], actual_mse)
        torch.testing.assert_close(result["background_mse"], actual_mse)
        torch.testing.assert_close(result["rbc_fraction"], torch.full((2,), .3))

    def test_invalid_metric_parameters(self):
        pred = torch.zeros(1, 1, 2, 2)
        for kwargs in [{"data_range": 0}, {"data_range": float("inf")},
                       {"sigma": 0}, {"window_size": 0}]:
            with self.assertRaises(ValueError):
                region_image_metrics(pred, pred, pred, **kwargs)


if __name__ == "__main__":
    unittest.main()
