import numpy as np
import pytest
import tifffile
import torch

from lensless_flow.data_pld import ParallelLenslessRMLDataset, pld_split_ids, read_pld_rgb
from lensless_flow.losses import spatial_cfm_loss
from lensless_flow import pld_protocol


def write_pair(root, image_id, value=128):
    image = np.full((300, 480, 4), value, dtype=np.uint8)
    image[..., 3] = 255
    for folder, name in [("4x_rml", f"4x_img_{image_id}_cam_1.tiff"),
                         ("4x_undistorted_GT2RML", f"warped_4x_undistorted_img_{image_id}_cam_2.tiff")]:
        (root / folder).mkdir(exist_ok=True)
        tifffile.imwrite(root / folder / name, image)


def test_official_splits_are_disjoint_and_complete():
    train, val, test = (set(pld_split_ids(s)) for s in ("train", "validation", "test"))
    assert [len(train), len(val), len(test)] == [95000, 4000, 1000]
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(range(100000))


def test_partial_smoke_preserves_numeric_split_and_rejects_missing_pair(tmp_path):
    for image_id in (10000, 5000, 1000, 0):
        write_pair(tmp_path, image_id)
    with pytest.raises(ValueError, match="Expected all 100000"):
        ParallelLenslessRMLDataset(tmp_path)
    dataset = ParallelLenslessRMLDataset(tmp_path, smoke_test=True)
    assert dataset.ids == [5000, 10000]
    assert ParallelLenslessRMLDataset(tmp_path, "validation", smoke_test=True).ids == [1000]
    image, target = dataset[0]
    assert image.shape == (300, 480, 3)
    torch.testing.assert_close(image, torch.full_like(image, 128/255))
    torch.testing.assert_close(image, target)
    dataset.pairs[0][1].unlink()
    with pytest.raises(ValueError, match="pairing failed"):
        ParallelLenslessRMLDataset(tmp_path, smoke_test=True)


def test_wrong_sampling_and_dtype_fail_instead_of_silent_normalization(tmp_path):
    with pytest.raises(ValueError, match="native 4x"):
        ParallelLenslessRMLDataset(tmp_path, downsample=4)
    path = tmp_path / "bad.tiff"
    tifffile.imwrite(path, np.zeros((300, 480, 3), np.uint16))
    with pytest.raises(ValueError, match="uint8"):
        read_pld_rgb(path)


def test_spatial_loss_has_positive_outside_gradient_and_correct_weights():
    prediction = torch.ones(2, 3, 4, 6, requires_grad=True)
    loss, stats = spatial_cfm_loss(prediction, torch.zeros_like(prediction), (1, 3, 2, 4), 0.5)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert (prediction.grad > 0).all()
    assert (prediction.grad[0, 0, 1, 2] / prediction.grad[0, 0, 0, 0]).item() == pytest.approx(7)
    assert not stats["crop_mse"].requires_grad
    with pytest.raises(ValueError, match="crop_mix"):
        spatial_cfm_loss(prediction, prediction, (1, 3, 2, 4), 1)


def test_geometry_identity_crop_and_warp_before_clamp(monkeypatch):
    monkeypatch.setattr(pld_protocol, "_gt_to_rml", lambda: torch.eye(3)[None])
    image = torch.rand(1, 3, 300, 480)
    p, g = pld_protocol.rml_evaluation_pair(image, image)
    assert p.shape == (1, 3, 214, 214)
    torch.testing.assert_close(p, image[..., 52:266, 129:343], atol=5e-5, rtol=0)
    torch.testing.assert_close(p, g)
    matrix = torch.eye(3)[None]
    matrix[0, 0, 2] = 0.5
    monkeypatch.setattr(pld_protocol, "_gt_to_rml", lambda: matrix)
    prediction = torch.zeros_like(image)
    prediction[..., 200] = -1
    prediction[..., 201] = 1
    actual, _ = pld_protocol.rml_evaluation_pair(prediction, image)
    prematurely_clamped, _ = pld_protocol.rml_evaluation_pair(prediction.clamp(0, 1), image)
    assert (actual - prematurely_clamped).abs().max() > 0.4


def test_rgb_native_unet_velocity_shape_and_backward():
    from lensless_flow.model_unet import SimpleCondUNet
    model = SimpleCondUNet(img_channels=3, base_ch=8, channel_mults=(1, 2, 4, 8), num_res_blocks=1)
    y = torch.rand(1, 3, 300, 480)
    output = model(torch.randn_like(y), y, torch.tensor([0.25]))
    assert output.shape == y.shape
    loss, _ = spatial_cfm_loss(output, y, pld_protocol.RML_IMAGER_CROP)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
