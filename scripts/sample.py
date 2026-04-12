import argparse
import yaml
import os
from math import ceil

import torch
import matplotlib.pyplot as plt

from lensless_flow.utils import ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.model_factory import build_flow_model, resolve_model_name
from lensless_flow.model_unet import resolve_use_time_conditioning
from lensless_flow.flow_matching import normalize_flow_matcher_name
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
    x = x.permute(1, 2, 0).numpy()
    # rotate 180 degrees for better visualization (optional, depending on dataset)
    return x[::-1, ::-1]


def main(cfg, idx: int, ckpt: str, steps_list, cols: int, seed: int | None, disable_physics_override: str | None):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load dataset (test)
    # -------------------------
    test_ds, _ = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    # Load one sample and convert to NCHW
    y, x = test_ds[idx]
    y = to_nchw(y).to(device)  # [1,C,H,W]
    x = to_nchw(x).to(device)

    # -------------------------
    # PSF + operator
    # -------------------------
    psf = to_nchw(test_ds.psf).to(device)  # [1,C,h,w] or [1,C,H,W]
    H_img, W_img = y.shape[-2], y.shape[-1]
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)
    y_hat = Hop.forward(x)
    print("rmse:", ((y_hat - y)**2).mean().sqrt().item())

    # -------------------------
    # Model
    # -------------------------
    state = torch.load(ckpt, map_location=device)
    use_time_conditioning = resolve_use_time_conditioning(cfg, state)
    C = y.shape[1]
    model_name = resolve_model_name(cfg, checkpoint_state=state)
    model = build_flow_model(
        cfg=cfg,
        img_channels=C,
        im_hw=(H_img, W_img),
        device=device,
        checkpoint_state=state,
    )

    # -------------------------
    # Load checkpoint + decide pred_type
    # -------------------------
    model.load_state_dict(state["model"])
    model.eval()

    pred_type = str(state.get("mode", cfg.get("train", {}).get("mode", "btb"))).lower()
    if pred_type not in ["btb", "vanilla"]:
        raise ValueError(f"Unknown pred_type/mode in ckpt/cfg: {pred_type}")
    flow_matcher_name = normalize_flow_matcher_name(
        state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified"))
    )
    print(
        f"[sample.py] Using pred_type={pred_type}, matcher={flow_matcher_name} "
        f"(model={model_name}, ckpt.mode={state.get('mode', None)}, ckpt.matcher={state.get('matcher', None)}, "
        f"use_time_conditioning={use_time_conditioning})"
    )

    # -------------------------
    # Sampling settings
    # -------------------------
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))
    init_noise_std = float(cfg.get("sample", {}).get("init_noise_std", 1.0))

    # Decide physics usage for sampling
    # Default: follow cfg.physics.disable_in_eval
    disable_physics = bool(cfg.get("physics", {}).get("disable_in_eval", False))
    if disable_physics_override is not None:
        s = disable_physics_override.strip().lower()
        if s in ["1", "true", "yes", "y", "on"]:
            disable_physics = True
        elif s in ["0", "false", "no", "n", "off"]:
            disable_physics = False
        else:
            raise ValueError("--disable_physics must be true/false (or 1/0)")

    dc_steps = int(cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))

    print(f"[sample.py] disable_physics={disable_physics}, dc_steps={dc_steps}, dc_step={dc_step} (<=0 => auto if enabled)")

    # Fix randomness for fair step-count comparison (optional)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Run sampling for each step count
    # -------------------------
    recons = []
    with torch.no_grad():
        for s in steps_list:
            # Use the SAME initial noise for each step-count if seed is set:
            # reset RNG to keep initial z identical across runs (apples-to-apples)
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            x_hat = sample_with_physics_guidance(
                model=model,
                y=y,
                H=Hop,
                steps=int(s),
                dc_step=dc_step,
                dc_steps=dc_steps,
                init_noise_std=init_noise_std,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=disable_physics,
                pred_type=pred_type,
                dc_mode="rgb"
            )
            recons.append((s, x_hat))

    # -------------------------
    # Plot grid
    # -------------------------
    ensure_dir(cfg["sample"]["save_dir"])
    out_path = os.path.join(cfg["sample"]["save_dir"], f"steps_grid_{idx}_{pred_type}_{flow_matcher_name}.pdf")

    n = len(recons)
    # rows = ceil((n + 2) / cols)  # +2 for y and GT
    rows = ceil(n / cols)  # just recons
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # handle edge case rows/cols==1
    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif cols == 1:
        axs = [[a] for a in axs]
    else:
        axs = axs.reshape(rows, cols)

    def put(ax, img, title):
        ax.imshow(img, cmap="gray" if (img.ndim == 2) else None)
        ax.set_title(title)
        ax.axis("off")

    # Slot 0: measurement y
    # put(axs[0][0], to_imshow(y), "Lensless y")
    # # Slot 1: GT x
    # if cols > 1:
    #     put(axs[0][1], to_imshow(x), "GT x")
    # else:
    #     put(axs[1][0], to_imshow(x), "GT x")

    # Fill remaining slots with reconstructions
    slot = 0
    pred_type = "v-pred" if pred_type == "vanilla" else "x-pred"
    for s, x_hat in recons:
        r = slot // cols
        c = slot % cols
        title = f"CFM ({pred_type}, {flow_matcher_name}) | steps={s}"
        if not disable_physics and dc_steps > 0:
            title += f" | DC={dc_steps}"
        put(axs[r][c], to_imshow(x_hat), title)
        slot += 1

    # Turn off leftover axes
    for k in range(slot, rows * cols):
        r = k // cols
        c = k % cols
        axs[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument(
        "--steps",
        type=str,
        default="5,10,15,20,30,50",
        help="Comma-separated step counts, e.g. 5,10,20,50",
    )
    ap.add_argument("--cols", type=int, default=4, help="Number of columns in the output grid")

    # Reproducibility / control switches
    ap.add_argument("--seed", type=int, default=0, help="Fix RNG seed for sampling (same initial noise across step counts)")
    ap.add_argument(
        "--disable_physics",
        type=str,
        default=None,
        help="Override cfg.physics.disable_in_eval. Use true/false (or 1/0).",
    )

    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    main(
        cfg,
        idx=args.idx,
        ckpt=args.ckpt,
        steps_list=steps_list,
        cols=args.cols,
        seed=args.seed,
        disable_physics_override=args.disable_physics,
    )
