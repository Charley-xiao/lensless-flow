import argparse
import yaml
import os

import torch
import matplotlib.pyplot as plt

from lensless_flow.utils import ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTConvOperator
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def to_imshow(x_bchw: torch.Tensor):
    """
    Convert [B,C,H,W] -> HWC numpy in [0,1] for plotting.
    Works for C=1 or C=3.
    """
    x = x_bchw[0].detach().cpu()  # [C,H,W]
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()  # HW
    return x.permute(1, 2, 0).numpy()  # HWC


def main(cfg, idx: int, ckpt: str):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    test_ds, _ = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
    )

    # Load one sample and convert to NCHW
    y, x = test_ds[idx]
    y = to_nchw(y).to(device)  # [1,C,H,W]
    x = to_nchw(x).to(device)  # [1,C,H,W]

    # PSF and physics operator (also NCHW)
    psf = to_nchw(test_ds.psf).to(device)  # [1,C,h,w] or [1,C,H,W]
    H_img, W_img = y.shape[-2], y.shape[-1]
    Hop = FFTConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    C = y.shape[1]
    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    x_hat = sample_with_physics_guidance(
        model=model,
        y=y,
        H=Hop,
        steps=cfg["sample"]["steps"],
        dc_step=cfg["physics"]["dc_step_size"],
        dc_steps=cfg["physics"]["dc_steps"],
        init_noise_std=cfg["sample"]["init_noise_std"],
    )

    ensure_dir(cfg["sample"]["save_dir"])
    out_path = os.path.join(cfg["sample"]["save_dir"], f"sample_{idx}.png")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(to_imshow(y), cmap="gray" if C == 1 else None)
    axs[0].set_title("Lensless y"); axs[0].axis("off")

    axs[1].imshow(to_imshow(x), cmap="gray" if C == 1 else None)
    axs[1].set_title("GT x"); axs[1].axis("off")

    axs[2].imshow(to_imshow(x_hat), cmap="gray" if C == 1 else None)
    axs[2].set_title("CFM + Physics"); axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, idx=args.idx, ckpt=args.ckpt)
