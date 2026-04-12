import argparse
import yaml
import os

import torch
import matplotlib.pyplot as plt

from lensless_flow.utils import ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.model_factory import build_flow_model, resolve_model_name
from lensless_flow.model_unet import resolve_use_time_conditioning
from lensless_flow.sampler import sample_with_physics_guidance
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


def _infer_mode(state: dict, fallback: str = "btb") -> str:
    mode = str(state.get("mode", fallback)).lower()
    if mode not in ["btb", "vanilla"]:
        mode = fallback
    return mode


def _parse_int_list(s: str) -> list[int]:
    s = s.strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x]


def main(cfg, idxs: list[int], ckpt: str, steps: int, seed: int | None, disable_physics_override: str | None, out_dir: str):
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

    # Operator setup (infer H,W from first idx)
    y0, _ = test_ds[idxs[0]]
    y0 = to_nchw(y0)
    H_img, W_img = int(y0.shape[-2]), int(y0.shape[-1])

    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    # Model
    state = torch.load(ckpt, map_location=device)
    use_time_conditioning = resolve_use_time_conditioning(cfg, state)
    C = int(y0.shape[1])
    model_name = resolve_model_name(cfg, checkpoint_state=state)
    model = build_flow_model(
        cfg=cfg,
        img_channels=C,
        im_hw=(H_img, W_img),
        device=device,
        checkpoint_state=state,
    )

    # Load checkpoint + pred_type
    model.load_state_dict(state["model"])
    model.eval()
    pred_type = _infer_mode(state, fallback=str(cfg.get("train", {}).get("mode", "btb")).lower())
    if pred_type not in ["btb", "vanilla"]:
        raise ValueError(f"Unknown pred_type/mode in ckpt/cfg: {pred_type}")
    flow_matcher_name = normalize_flow_matcher_name(
        state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified"))
    )
    print(
        f"[sample_mult.py] Using model={model_name}, pred_type={pred_type}, "
        f"matcher={flow_matcher_name}, use_time_conditioning={use_time_conditioning}"
    )

    # Sampling settings
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))
    init_noise_std = float(cfg.get("sample", {}).get("init_noise_std", 1.0))

    disable_physics = bool(cfg.get("physics", {}).get("disable_in_eval", False))
    if disable_physics_override is not None:
        s = disable_physics_override.strip().lower()
        if s in ["1", "true", "yes", "y", "on"]:
            disable_physics = True
        elif s in ["0", "false", "no", "n", "off"]:
            disable_physics = False
        else:
            raise ValueError("--disable_physics must be true/false (or 1/0).")

    dc_steps = int(cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))

    # RNG control
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ensure_dir(out_dir)

    for idx in idxs:
        y, x = test_ds[idx]
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        # Re-seed per idx for reproducibility if desired
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            x_hat = sample_with_physics_guidance(
                model=model,
                y=y,
                H=Hop,
                steps=int(steps),
                dc_step=dc_step,
                dc_steps=dc_steps,
                init_noise_std=init_noise_std,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=disable_physics,
                pred_type=pred_type,
                dc_mode="rgb",
            )

        gt_path = os.path.join(out_dir, f"idx_{idx:05d}_gt.png")
        y_path = os.path.join(out_dir, f"idx_{idx:05d}_y.png")
        recon_path = os.path.join(
            out_dir,
            f"idx_{idx:05d}_recon_steps{int(steps)}_{pred_type}_{flow_matcher_name}.png",
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
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable_physics", type=str, default=None, help="Override cfg.physics.disable_in_eval.")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    idxs = _parse_int_list(args.idxs)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = cfg.get("sample", {}).get("save_dir", "samples_triplets")

    main(
        cfg=cfg,
        idxs=idxs,
        ckpt=args.ckpt,
        steps=args.steps,
        seed=args.seed,
        disable_physics_override=args.disable_physics,
        out_dir=out_dir,
    )
