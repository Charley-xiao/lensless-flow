import argparse
import yaml
import os
from math import ceil

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
    Convert [B,C,H,W] -> numpy for imshow:
      - if C==1: HW
      - if C==3: HWC
    """
    x = x_bchw[0].detach().float().cpu()  # [C,H,W]
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def _call_sampler(model, y, Hop, steps, cfg):
    """
    Call sample_with_physics_guidance in a way that stays compatible with
    whether your sampler supports disable_physics or not.
    """
    kwargs = dict(
        model=model,
        y=y,
        H=Hop,
        steps=int(steps),
        dc_step=cfg["physics"]["dc_step_size"],
        dc_steps=cfg["physics"]["dc_steps"],
        init_noise_std=cfg["sample"]["init_noise_std"],
    )

    # If your sampler has disable_physics (you added it earlier), pass it.
    # Otherwise, just call without it.
    try:
        return sample_with_physics_guidance(
            **kwargs,
            disable_physics=False if cfg["cfm"]["loss"]["physics_weight"] > 0 else True,
        )
    except TypeError:
        return sample_with_physics_guidance(**kwargs)


def main(cfg, idx: int, ckpt: str, steps_list, cols: int):
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
    x = to_nchw(x).to(device)

    # PSF + operator (NCHW)
    psf = to_nchw(test_ds.psf).to(device)
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

    # Generate reconstructions for each step count
    recons = []
    with torch.no_grad():
        for s in steps_list:
            x_hat = _call_sampler(model, y, Hop, steps=s, cfg=cfg)
            recons.append((s, x_hat))

    # --- Plot: first row shows y and GT; rest is recon grid ---
    ensure_dir(cfg["sample"]["save_dir"])
    out_path = os.path.join(cfg["sample"]["save_dir"], f"steps_grid_{idx}.png")

    n = len(recons)
    rows = ceil((n + 2) / cols)  # +2 for y and GT
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.reshape(rows, cols)

    def put(ax, img, title):
        ax.imshow(img, cmap="gray" if (img.ndim == 2) else None)
        ax.set_title(title)
        ax.axis("off")

    # Slot 0: measurement y
    put(axs[0, 0], to_imshow(y), "Lensless y")
    # Slot 1: GT x
    if cols > 1:
        put(axs[0, 1], to_imshow(x), "GT x")
    else:
        # if cols==1, move GT to next row
        put(axs[1, 0], to_imshow(x), "GT x")

    # Fill remaining slots with reconstructions
    slot = 2
    for s, x_hat in recons:
        r = slot // cols
        c = slot % cols
        put(axs[r, c], to_imshow(x_hat), f"CFM{' +Phys' if cfg['cfm']['loss']['physics_weight']>0 else ''} | steps={s}")
        slot += 1

    # Turn off any leftover axes
    for k in range(slot, rows * cols):
        r = k // cols
        c = k % cols
        axs[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, required=True)

    # Comma-separated list of step counts to compare
    ap.add_argument("--steps", type=str, default="5,10,15,20,30,50",
                    help="Comma-separated step counts, e.g. 5,10,20,50")

    # Grid layout
    ap.add_argument("--cols", type=int, default=4, help="Number of columns in the output grid")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    main(cfg, idx=args.idx, ckpt=args.ckpt, steps_list=steps_list, cols=args.cols)
