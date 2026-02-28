import argparse
import yaml
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lensless_flow.data import make_dataloader
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.physics import FFTConvOperator


def normalize_for_vis(x: torch.Tensor):
    """x: [1,C,H,W] or [C,H,W] -> numpy [H,W] or [H,W,3] in [0,1]"""
    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def center_of_mass_2d(img: torch.Tensor):
    """
    img: [H,W] nonnegative
    returns (cy, cx) in float pixels
    """
    img = img.detach().float()
    img = img - img.min()
    img = torch.clamp(img, min=0.0)
    s = img.sum() + 1e-12
    H, W = img.shape
    ys = torch.arange(H, device=img.device, dtype=img.dtype)
    xs = torch.arange(W, device=img.device, dtype=img.dtype)
    cy = (img.sum(dim=1) * ys).sum() / s
    cx = (img.sum(dim=0) * xs).sum() / s
    return float(cy.item()), float(cx.item())


def make_delta(B: int, C: int, H: int, W: int, device, dtype, cy=None, cx=None):
    """
    delta image with a single 1 at (cy,cx). Default center.
    returns [B,C,H,W]
    """
    if cy is None:
        cy = H // 2
    if cx is None:
        cx = W // 2
    x = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    x[:, :, cy, cx] = 1.0
    return x


@torch.no_grad()
def main(cfg, idx: int = 0, save_dir: str = "debug_psf"):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load data exactly like train.py
    # -------------------------
    train_ds, _ = make_dataloader(
        split="train",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
    )

    y, x = train_ds[idx]
    y = to_nchw(y).to(device)  # [1,C,H,W]
    x = to_nchw(x).to(device)

    psf = to_nchw(train_ds.psf.to(device))  # [1,C,h,w] (expected)
    C = y.shape[1]
    H, W = y.shape[-2], y.shape[-1]

    print("y shape:", tuple(y.shape))
    print("x shape:", tuple(x.shape))
    print("psf shape:", tuple(psf.shape))

    # -------------------------
    # Operator under test
    # -------------------------
    Hop = FFTConvOperator(psf=psf, im_hw=(H, W)).to(device)

    # -------------------------
    # 1) Impulse response test
    # -------------------------
    delta_center = make_delta(1, C, H, W, device=device, dtype=y.dtype, cy=H//2, cx=W//2)
    y_delta = Hop.forward(delta_center)  # should resemble PSF, centered at the delta location (mod circular)

    # estimate COM of the impulse response (first channel)
    cy_hat, cx_hat = center_of_mass_2d(y_delta[0, 0])
    print(f"[Impulse response COM] (cy,cx)=({cy_hat:.2f},{cx_hat:.2f}), expected approx ({H//2},{W//2})")

    # -------------------------
    # 2) Forward-model check on real data
    # -------------------------
    y_hat = Hop.forward(x)

    # metrics: relative error and shift-like pattern
    mse = F.mse_loss(y_hat.float(), y.float()).item()
    l1 = F.l1_loss(y_hat.float(), y.float()).item()
    rel = (y_hat.float() - y.float()).pow(2).mean().sqrt() / (y.float().pow(2).mean().sqrt() + 1e-12)
    rel = float(rel.item())

    print(f"[Forward check] MSE={mse:.6g}  L1={l1:.6g}  RelRMSE={rel:.6g}")

    # -------------------------
    # Visualize
    # -------------------------
    # pick channel 0 for easy viewing
    psf_vis = normalize_for_vis(psf[:, :1])  # [h,w]
    y_vis = normalize_for_vis(y[:, :1])
    x_vis = normalize_for_vis(x[:, :1])
    yhat_vis = normalize_for_vis(y_hat[:, :1])
    ydelta_vis = normalize_for_vis(y_delta[:, :1])

    diff_y = (y_hat[:, :1] - y[:, :1]).detach().float().cpu()
    diff_y = diff_y[0, 0].numpy()

    # also show magnitude spectrum of OTF (alignment hints)
    otf = Hop.otf  # [1,C,H,W] complex
    otf_mag = torch.abs(otf[0, 0]).detach().float().cpu()
    otf_mag = otf_mag / (otf_mag.max() + 1e-8)
    otf_mag = otf_mag.numpy()

    fig = plt.figure(figsize=(16, 10))
    axs = fig.subplots(2, 4)

    axs[0, 0].imshow(psf_vis, cmap="gray")
    axs[0, 0].set_title("PSF (as loaded)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(ydelta_vis, cmap="gray")
    axs[0, 1].set_title("Hop(delta @ center)\nImpulse response")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(y_vis, cmap="gray")
    axs[0, 2].set_title("Measured y (dataset)")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(yhat_vis, cmap="gray")
    axs[0, 3].set_title("Hop(x_gt) predicted y_hat")
    axs[0, 3].axis("off")

    axs[1, 0].imshow(x_vis, cmap="gray")
    axs[1, 0].set_title("GT x (lensed)")
    axs[1, 0].axis("off")

    im = axs[1, 1].imshow(diff_y, cmap="bwr")
    axs[1, 1].set_title("y_hat - y (channel 0)")
    axs[1, 1].axis("off")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    axs[1, 2].imshow(otf_mag, cmap="gray")
    axs[1, 2].set_title("|OTF| (channel 0)")
    axs[1, 2].axis("off")

    # show cross-correlation peak (rough shift detection)
    # correlate y_hat and y to detect circular shift
    a = y_hat[0, 0].detach().float()
    b = y[0, 0].detach().float()
    A = torch.fft.fft2(a)
    B = torch.fft.fft2(b)
    cc = torch.fft.ifft2(A * torch.conj(B)).real  # circular cross-correlation
    cc = cc / (cc.max() + 1e-8)
    cc_vis = cc.detach().cpu().numpy()
    peak = torch.argmax(cc).item()
    py = peak // W
    px = peak % W
    axs[1, 3].imshow(cc_vis, cmap="gray")
    axs[1, 3].set_title(f"circular xcorr peak at (y={py},x={px})")
    axs[1, 3].axis("off")

    out_path = os.path.join(save_dir, f"psf_align_idx{idx}_ds{cfg['data']['downsample']}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved visualization:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="debug_psf")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, idx=args.idx, save_dir=args.save_dir)