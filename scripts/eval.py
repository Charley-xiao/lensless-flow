import argparse
import yaml
from tqdm import tqdm

import torch

from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.metrics import psnr, ssim_torch


@torch.no_grad()
def main(cfg, ckpt: str, max_batches: int | None):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device)
    pred_type = str(state.get("mode", cfg.get("train", {}).get("mode", "btb"))).lower()
    assert pred_type in ["btb", "vanilla"], f"Unknown pred_type={pred_type}"

    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=8,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    # Determine C,H,W from one sample (after converting to NCHW)
    y0, _ = test_ds[0]
    y0 = to_nchw(y0)  # [1,C,H,W]
    C = y0.shape[1]
    H_img, W_img = y0.shape[-2], y0.shape[-1]

    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    model.load_state_dict(state["model"])
    model.eval()

    steps = int(cfg["sample"]["steps"])
    init_noise_std = float(cfg["sample"]["init_noise_std"])

    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    disable_physics = bool(cfg.get("physics", {}).get("disable_in_eval", False))
    dc_steps = int(cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))
    dc_mode = "rgb"

    psnr_list = []
    ssim_list = []
    dc_rmse_list = []
    mse_list = []

    pbar = tqdm(test_dl, desc=f"eval ({pred_type}, steps={steps}, DC={'off' if disable_physics else dc_mode})")
    for i, (y, x) in enumerate(pbar):
        if (max_batches is not None) and (i >= max_batches):
            break

        y = to_nchw(y).to(device)  # [B,C,H,W]
        x = to_nchw(x).to(device)  # [B,C,H,W]

        x_hat = sample_with_physics_guidance(
            model=model,
            y=y,
            H=Hop,
            steps=steps,
            dc_step=dc_step,
            dc_steps=dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            clamp_x=False,
            disable_physics=disable_physics,
            pred_type=pred_type,
            dc_mode=dc_mode,
        )

        # Clamp for image-quality metrics (assumes target roughly in [0,1])
        x_hat_c = x_hat.clamp(0, 1)
        x_c = x.clamp(0, 1)

        mse = float((x_hat_c.float() - x_c.float()).pow(2).mean().item())

        p = float(psnr(x_hat_c, x_c))
        s = float(ssim_torch(x_hat_c, x_c))

        # Data-consistency RMSE in measurement space (no clamp)
        dc_rmse = float((Hop.forward(x_hat.float()) - y.float()).pow(2).mean().sqrt().item())

        psnr_list.append(p)
        ssim_list.append(s)
        dc_rmse_list.append(dc_rmse)
        mse_list.append(mse)

        pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}", dc_rmse=f"{dc_rmse:.4f}", mse=f"{mse:.6f}")

    n = max(1, len(psnr_list))
    print("\n========== Eval Summary ==========")
    print(f"ckpt: {ckpt}")
    print(f"pred_type: {pred_type}")
    print(f"steps: {steps}")
    print(f"DC: {'disabled' if disable_physics else (dc_mode + f' (dc_steps={dc_steps}, dc_step={dc_step})')}")
    print("----------------------------------")
    print(f"PSNR avg: {sum(psnr_list)/n:.3f}")
    print(f"SSIM avg: {sum(ssim_list)/n:.6f}")
    print(f"MSE avg: {sum(mse_list)/n:.8f}")  # NEW
    print(f"Data-consistency RMSE avg: {sum(dc_rmse_list)/n:.6f}")
    print("==================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    max_batches = None if args.max_batches < 0 else args.max_batches
    main(cfg, ckpt=args.ckpt, max_batches=max_batches)