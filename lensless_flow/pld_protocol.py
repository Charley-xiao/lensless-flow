"""PLD 4x RML geometry: published imager loss crop and common GT evaluation crop."""
from functools import lru_cache
import json
from pathlib import Path

import torch

RML_IMAGER_CROP = (31, 270, 128, 367)
COMMON_GT_CROP = (52, 266, 129, 343)


def crop_bchw(image, bounds):
    if image.ndim != 4 or len(bounds) != 4 or any(int(v) != v for v in bounds):
        raise ValueError("Expected BCHW tensor and four integer crop bounds")
    top, bottom, left, right = map(int, bounds)
    if not (0 <= top < bottom <= image.shape[-2] and 0 <= left < right <= image.shape[-1]):
        raise ValueError(f"Crop {bounds} exceeds image {tuple(image.shape)}")
    return image[..., top:bottom, left:right]


@lru_cache(maxsize=1)
def _gt_to_rml():
    path = Path(__file__).parent / "calibration" / "pld_rml_4x.json"
    return torch.tensor(json.loads(path.read_text())["gt_to_rml"], dtype=torch.float32).reshape(1, 3, 3)


def rml_evaluation_pair(prediction, target):
    """Warp BOTH registered images to GT coordinates, then clamp and crop.

    Matches ConvRML infer.py geometry (bilinear, zero padding, align_corners=True).
    Deliberately do not clamp predictions before interpolation or renormalize them.
    """
    from kornia.geometry.transform import warp_perspective
    if prediction.shape != target.shape or prediction.shape[1:] != (3, 300, 480):
        raise ValueError("RML scoring requires matched Bx3x300x480 tensors")
    matrix = torch.linalg.inv(_gt_to_rml().to(prediction.device)).expand(prediction.shape[0], -1, -1)
    def transform(image):
        warped = warp_perspective(image.float(), matrix, (300, 480),
                                  mode="bilinear", padding_mode="zeros", align_corners=True)
        return crop_bchw(warped.clamp(0, 1), COMMON_GT_CROP)
    return transform(prediction), transform(target)


def save_rml_preview(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(rows), 3, figsize=(10, 2.6 * len(rows)), squeeze=False)
    for row, images in enumerate(rows):
        for column, tensor in enumerate(images):
            axes[row, column].imshow(tensor.float().clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
    for axis, title in zip(axes[0], ("RML measurement (full)", "Ground truth (common crop)", "Flow (common crop)")):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
