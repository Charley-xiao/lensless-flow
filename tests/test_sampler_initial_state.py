import unittest

import torch

from lensless_flow.sampler import sample_with_physics_guidance
from scripts.benchmark_rbc_reconstruction import image_latents


class Velocity(torch.nn.Module):
    use_time_conditioning = True

    def forward(self, z, y, t):
        return 0.2 * z + y * t[:, None, None, None]


def sample(y, **kwargs):
    return sample_with_physics_guidance(Velocity(), y, H=None, steps=4,
        dc_step=0, dc_steps=0, disable_physics=True, clamp_x=False,
        pred_type='vanilla', solver='heun', **kwargs)


class InitialStateTests(unittest.TestCase):
    def test_injected_latent_equals_original_random_source(self):
        y = torch.ones(1, 1, 8, 8)
        torch.manual_seed(123)
        expected = sample(y)
        z = image_latents([0], (1,8,8), 'cpu', seed=123)
        actual = sample(y, initial_state=z)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_per_image_seeds_independent_of_batch_partition(self):
        full = image_latents([0,1,2], (1,8,8), 'cpu')
        parts = torch.cat([image_latents([0],(1,8,8),'cpu'), image_latents([1,2],(1,8,8),'cpu')])
        self.assertTrue(torch.equal(full, parts))
        y = torch.ones_like(full)
        together = sample(y, initial_state=full)
        apart = torch.cat([sample(y[i:i+1], initial_state=full[i:i+1]) for i in range(3)])
        self.assertTrue(torch.equal(together, apart))

    def test_explicit_state_unchanged_and_rng_not_consumed(self):
        y = torch.ones(2,1,8,8)
        z = torch.zeros_like(y)
        original = z.clone()
        rng = torch.random.get_rng_state()
        sample(y, initial_state=z)
        self.assertTrue(torch.equal(z, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_invalid_state_fails(self):
        y = torch.zeros(1,1,8,8)
        for z in (torch.zeros(1,1,7,8), torch.zeros_like(y, dtype=torch.float64), torch.full_like(y, float('nan'))):
            with self.assertRaises(ValueError):
                sample(y, initial_state=z)


if __name__ == '__main__':
    unittest.main()
