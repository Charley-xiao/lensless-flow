import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTConvOperator
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def psnr(x_hat, x):
    # assume x_hat, x in [0,1]
    mse = F.mse_loss(x_hat, x).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main(cfg, ckpt: str, max_batches: int):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
    )

    # Determine H,W,C from one sample (after converting to NCHW)
    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    C = y0.shape[1]
    H_img, W_img = y0.shape[-2], y0.shape[-1]

    # PSF operator (NCHW)
    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    psnr_list = []
    dc_err_list = []

    for i, (y, x) in enumerate(tqdm(test_dl)):
        if (max_batches is not None) and (i >= max_batches):
            break

        # Convert batched tensors to NCHW; your loader yields B×1×H×W×C
        y = to_nchw(y).to(device)  # [B,C,H,W]
        x = to_nchw(x).to(device)  # [B,C,H,W]

        x_hat = sample_with_physics_guidance(
            model=model,
            y=y,
            H=Hop,
            steps=cfg["sample"]["steps"],
            dc_step=cfg["physics"]["dc_step_size"],
            dc_steps=cfg["physics"]["dc_steps"],
            init_noise_std=cfg["sample"]["init_noise_std"],
        )

        # metrics (assuming dataset roughly normalized to [0,1])
        x_hat_c = x_hat.clamp(0, 1)
        x_c = x.clamp(0, 1)

        psnr_list.append(psnr(x_hat_c, x_c))

        dc_err = (Hop.forward(x_hat) - y).pow(2).mean().sqrt().item()
        dc_err_list.append(dc_err)

    print(f"PSNR avg: {sum(psnr_list)/len(psnr_list):.3f}")
    print(f"Data-consistency RMSE avg: {sum(dc_err_list)/len(dc_err_list):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, ckpt=args.ckpt, max_batches=args.max_batches)
