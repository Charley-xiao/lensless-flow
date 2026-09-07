import unittest

import torch

from lensless_flow.metrics import psnr, ssim_torch
from scripts.score_rbc_benchmark import pixel_metrics, aggregate


class BenchmarkMetricTests(unittest.TestCase):
    def test_per_image_global_scores_match_existing_metrics(self):
        torch.manual_seed(42)
        target = torch.rand(3,1,48,48)
        pred = (target * .8 + .02).clamp(0,1)
        mask = torch.zeros_like(target)
        mask[...,12:36,12:36] = 1
        result = pixel_metrics(pred,target,mask)
        self.assertTrue(all(value.shape == (3,) for value in result.values()))
        for i in range(3):
            self.assertAlmostEqual(result['psnr'][i].item(),psnr(pred[i:i+1],target[i:i+1]),places=5)
            self.assertAlmostEqual(result['ssim'][i].item(),ssim_torch(pred[i:i+1],target[i:i+1]).item(),places=6)

    def test_circular_wrap_and_missing_regions(self):
        target = torch.full((2,1,48,48),.01)
        pred = torch.full_like(target,.99)
        mask = torch.zeros_like(target)
        mask[0] = 1
        result = pixel_metrics(pred,target,mask)
        torch.testing.assert_close(result['circular_rmse_rad'],torch.full((2,),.04*torch.pi))
        self.assertTrue(torch.isnan(result['rbc_psnr'][1]))
        self.assertTrue(torch.isnan(result['background_psnr'][0]))
        self.assertTrue(torch.isnan(result['background_circular_rmse_rad'][0]))

    def test_aggregate_preserves_missing_score_counts(self):
        values=aggregate([{'index':0,'psnr':10.,'rbc_psnr':None,'runtime_ms':None},
                          {'index':1,'psnr':20.,'rbc_psnr':15.,'runtime_ms':None}])
        self.assertEqual(values['psnr']['mean'],15.)
        self.assertEqual(values['psnr']['n'],2)
        self.assertEqual(values['rbc_psnr']['n'],1)
        self.assertEqual(values['runtime_ms']['n'],0)
        self.assertIsNone(values['runtime_ms']['mean'])


if __name__=='__main__':
    unittest.main()
