"""Regression checks for direct-baseline validation and deployment checkpoints."""
import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lensless_flow.model_factory import build_baseline_unet, load_checkpoint_state_dict
from lensless_flow.model_unet import SimpleCondUNet
from scripts.train_unet_baseline import baseline_forward, eval_loop, save_checkpoint, save_eval_record


class IdentityBaseline(torch.nn.Module):
    use_time_conditioning = False

    def forward(self, auxiliary, hologram, time):
        assert time is None
        assert torch.count_nonzero(auxiliary) == 0
        return hologram


@pytest.mark.parametrize("training", [True, False])
def test_validation_averages_images_preserves_rng_and_model_mode(training):
    model = IdentityBaseline().train(training)
    levels = torch.tensor([0.1, 0.4, 0.9])
    images = levels[:, None, None, None].expand(3, 16, 16, 1).clone()
    dataset = TensorDataset(images, torch.zeros_like(images))
    state = torch.random.get_rng_state()
    scores = eval_loop(model, DataLoader(dataset, batch_size=2), torch.device("cpu"))
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.training == training
    assert scores["eval/samples"] == 3
    assert scores["eval/mse"] == pytest.approx(levels.square().mean().item(), abs=1e-6)
    assert scores["eval/psnr"] == pytest.approx((-20 * levels.log10()).mean().item(), abs=1e-5)
    singles = eval_loop(model, DataLoader(dataset, batch_size=1), torch.device("cpu"))
    assert scores == pytest.approx(singles, abs=1e-6)


def test_checkpoint_loads_as_deterministic_direct_model(tmp_path):
    cfg = {"model": {"base_channels": 8, "channel_mults": [1, 2], "num_res_blocks": 1}}
    model = SimpleCondUNet(img_channels=1, base_ch=8, channel_mults=(1, 2),
                          num_res_blocks=1, use_time_conditioning=False).eval()
    state = {"model": model.state_dict(), "cfg": cfg, "use_time_conditioning": False,
             "prediction_type": "direct_image", "training_loss": "mse", "epoch": 1}
    path = tmp_path / "best.pt"
    save_checkpoint(state, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = build_baseline_unet(cfg, img_channels=1, device="cpu", checkpoint_state=saved)
    load_checkpoint_state_dict(restored, saved)
    assert not restored.use_time_conditioning
    assert not (tmp_path / "best.pt.tmp").exists()
    y = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        expected = baseline_forward(model, y)
        torch.rand(7)  # Prediction must not depend on a sampled latent.
        actual = baseline_forward(restored, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_metric_record_handles_absent_regions(tmp_path):
    path = tmp_path / "nested" / "metrics.jsonl"
    save_eval_record({"train": {"metrics_path": str(path)}},
                     {"eval/rbc_ssim": math.nan, "eval/samples": 2}, 3, 20)
    assert json.loads(path.read_text()) == {
        "epoch": 3, "step": 20, "eval/rbc_ssim": None, "eval/samples": 2}
