import argparse
import yaml
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lensless_flow.utils import ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.model_unet import SimpleCondUNet, resolve_baseline_use_time_conditioning
from lensless_flow.tensor_utils import to_nchw


def to_imshow(x_bchw: torch.Tensor):
    """[B,C,H,W] -> numpy for imshow, per-image min-max; C==1 -> HW, C==3 -> HWC."""
    x = x_bchw[0].detach().float().cpu()  # [C,H,W]
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def save_image_only(img_np, out_path: str):
    """
    Save image with:
      - no title
      - no axes/ticks/frames
      - minimal padding (tight, borderless)
    """
    h, w = img_np.shape[:2]
    dpi = 300
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire canvas
    ax.imshow(img_np, cmap="gray" if (img_np.ndim == 2) else None)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _parse_int_list(s: str) -> list[int]:
    s = s.strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x]


@torch.no_grad()
def unet_forward_baseline(model, y: torch.Tensor) -> torch.Tensor:
    """
    Baseline: deterministic reconstruction x_hat = f(y).

    We reuse SimpleCondUNet by feeding a zero x_t tensor for channel compatibility.
    New baseline checkpoints keep time conditioning disabled, but we still
    support older checkpoints that explicitly saved a time-conditioned setting.
    """
    b = y.shape[0]
    x_t = torch.zeros_like(y)          # dummy
    t = torch.zeros(b, device=y.device) if getattr(model, "use_time_conditioning", True) else None
    x_hat = model(x_t, y, t)
    return x_hat


def dc_refine(
    x_init: torch.Tensor,
    y: torch.Tensor,
    Hop: FFTLinearConvOperator,
    steps: int,
    step_size: float,
    clamp_01: bool = True,
):
    """
    Optional data-consistency refinement:
        x <- x - alpha * grad ||H(x) - y||_2^2

    This is a simple baseline; keep step_size small.
    """
    if steps <= 0 or step_size <= 0:
        return x_init

    x = x_init.detach().clone().requires_grad_(True)
    for _ in range(steps):
        y_hat = Hop(x)
        loss = F.mse_loss(y_hat, y)
        (grad,) = torch.autograd.grad(loss, x, create_graph=False)
        with torch.no_grad():
            x -= step_size * grad
            if clamp_01:
                x.clamp_(0.0, 1.0)
        x.requires_grad_(True)
    return x.detach()


def main(
    cfg,
    idxs: list[int],
    ckpt: str,
    seed: int | None,
    out_dir: str,
    dc_steps: int,
    dc_step_size: float,
):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Load dataset (test)
    test_ds, _ = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    # Infer H,W,C from first idx
    y0, _ = test_ds[idxs[0]]
    y0 = to_nchw(y0)
    H_img, W_img = int(y0.shape[-2]), int(y0.shape[-1])
    C = int(y0.shape[1])

    # Forward operator (only needed for optional DC refine)
    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    # Model
    state = torch.load(ckpt, map_location=device)
    use_time_conditioning = resolve_baseline_use_time_conditioning(state)
    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        use_time_conditioning=use_time_conditioning,
    ).to(device)

    # Load checkpoint
    if "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        # allow raw state_dict checkpoints
        model.load_state_dict(state)
    model.eval()

    # RNG control (mostly irrelevant for deterministic U-Net, but keep for completeness)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ensure_dir(out_dir)

    for idx in idxs:
        y, x = test_ds[idx]
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        with torch.no_grad():
            x_hat = unet_forward_baseline(model, y)

        # Optional DC refinement (U-Net + DC baseline)
        if dc_steps > 0 and dc_step_size > 0:
            x_hat = dc_refine(
                x_init=x_hat,
                y=y,
                Hop=Hop,
                steps=int(dc_steps),
                step_size=float(dc_step_size),
                clamp_01=True,
            )

        # Save triplets
        gt_path = os.path.join(out_dir, f"idx_{idx:05d}_gt.png")
        y_path = os.path.join(out_dir, f"idx_{idx:05d}_y.png")
        recon_path = os.path.join(
            out_dir,
            f"idx_{idx:05d}_unet.png" if dc_steps <= 0 else f"idx_{idx:05d}_unet_dc{dc_steps}.png",
        )

        save_image_only(to_imshow(x), gt_path)
        save_image_only(to_imshow(y), y_path)
        save_image_only(to_imshow(x_hat), recon_path)

        print(f"saved: {gt_path}")
        print(f"saved: {y_path}")
        print(f"saved: {recon_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--idxs", type=str, required=True, help='e.g. "0,3,7,12" or "0 3 7 12"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=None)

    # Optional DC refinement knobs (set to 0 to disable)
    ap.add_argument("--dc_steps", type=int, default=0)
    ap.add_argument("--dc_step_size", type=float, default=0.0)

    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    idxs = _parse_int_list(args.idxs)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = cfg.get("sample", {}).get("save_dir", "samples_unet_triplets")

    main(
        cfg=cfg,
        idxs=idxs,
        ckpt=args.ckpt,
        seed=args.seed,
        out_dir=out_dir,
        dc_steps=args.dc_steps,
        dc_step_size=args.dc_step_size,
    )
