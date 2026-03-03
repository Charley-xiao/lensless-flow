import argparse
import yaml
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw


def _infer_mode(state: dict, fallback: str = "btb") -> str:
    mode = str(state.get("mode", fallback)).lower()
    if mode not in ["btb", "vanilla"]:
        mode = fallback
    return mode


def _parse_int_list(s: str) -> list[int]:
    # Accept "0,1,2" or "0 1 2" or "0:10:1" (start:stop:step, inclusive stop)
    s = s.strip()
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("Range form must be start:stop or start:stop:step")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError("step must be > 0")
        return list(range(start, stop + 1, step))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x]


@torch.no_grad()
def eval_avg_mse_for_dc_steps(
    *,
    cfg: dict,
    ckpt_path: str,
    steps: int,
    dc_steps_list: list[int],
    max_batches: int | None,
    force_mode: str | None = None,
) -> tuple[str, list[float]]:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)

    if force_mode is None:
        pred_type = _infer_mode(state, fallback=str(cfg.get("train", {}).get("mode", "btb")).lower())
    else:
        pred_type = str(force_mode).lower()
    assert pred_type in ["btb", "vanilla"], f"Unknown pred_type={pred_type}"

    # Data
    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=8,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    # Infer shapes
    y0, _ = test_ds[0]
    y0 = to_nchw(y0)
    C = int(y0.shape[1])
    H_img, W_img = int(y0.shape[-2]), int(y0.shape[-1])

    # Physics operator
    psf = to_nchw(test_ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    # Model
    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    init_noise_std = float(cfg["sample"]["init_noise_std"])
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    # Keep dc_step size fixed (from config)
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))
    dc_mode = "rgb"

    avg_mse = []
    for dc_steps in dc_steps_list:
        mse_list = []
        pbar = tqdm(
            test_dl,
            desc=f"MSE ({pred_type}, steps={steps}, dc_steps={dc_steps})",
            leave=False,
        )
        for i, (y, x) in enumerate(pbar):
            if (max_batches is not None) and (i >= max_batches):
                break

            y = to_nchw(y).to(device)
            x = to_nchw(x).to(device)

            x_hat = sample_with_physics_guidance(
                model=model,
                y=y,
                H=Hop,
                steps=int(steps),
                dc_step=dc_step,
                dc_steps=int(dc_steps),
                init_noise_std=init_noise_std,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=False,  # sweeping dc_steps
                pred_type=pred_type,
                dc_mode=dc_mode,
            )

            # Image-domain MSE between reconstruction and GT (clamped to [0,1], consistent with your other metrics)
            x_hat_c = x_hat.clamp(0, 1).float()
            x_c = x.clamp(0, 1).float()
            mse = float((x_hat_c - x_c).pow(2).mean().item())

            mse_list.append(mse)
            pbar.set_postfix(mse=f"{mse:.6e}")

        n = max(1, len(mse_list))
        avg_mse.append(sum(mse_list) / n)

    return pred_type, avg_mse


def _set_plot_style():
    # "More professional" without specifying explicit colors.
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--steps", type=int, required=True, help="Sampling steps (fixed)")
    ap.add_argument("--dc_steps", type=str, required=True, help='e.g. "0,1,2,3,5,8,10" or "0:10:1"')
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--out", type=str, default="mse_vs_dcsteps.png")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--force_mode", type=str, default=None, choices=[None, "vanilla", "btb"])
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis (often helpful for MSE)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dc_steps_list = _parse_int_list(args.dc_steps)
    max_batches = None if args.max_batches < 0 else args.max_batches

    mode, mse_vals = eval_avg_mse_for_dc_steps(
        cfg=cfg,
        ckpt_path=args.ckpt,
        steps=args.steps,
        dc_steps_list=dc_steps_list,
        max_batches=max_batches,
        force_mode=args.force_mode,
    )

    title = args.title
    if title is None:
        title = f"MSE vs dc_steps (mode={mode}, steps={args.steps})"

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(dc_steps_list, mse_vals, marker="o", linewidth=2.0, markersize=5)

    ax.set_xlabel("dc_steps")
    ax.set_ylabel("MSE (recon vs. GT, avg)")
    ax.set_title(title)

    ax.set_xticks(dc_steps_list)

    # annotate best point (lowest MSE)
    best_i = int(min(range(len(mse_vals)), key=lambda i: mse_vals[i]))
    ax.scatter([dc_steps_list[best_i]], [mse_vals[best_i]], zorder=3)
    ax.annotate(
        f"best: {mse_vals[best_i]:.2e}",
        xy=(dc_steps_list[best_i], mse_vals[best_i]),
        xytext=(8, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )

    if args.logy:
        ax.set_yscale("log")

    ax.legend([f"{mode} (ckpt)"], frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(args.out)

    print("\n========== MSE vs dc_steps ==========")
    print(f"ckpt: {args.ckpt}")
    print(f"mode: {mode}")
    print(f"sampling steps: {args.steps}")
    print(f"dc_step_size (from cfg): {float(cfg.get('physics', {}).get('dc_step_size', 0.0))}")
    print("-------------------------------------")
    for k, m in zip(dc_steps_list, mse_vals):
        print(f"dc_steps={k:>3d}  MSE={m:.8e}")
    print(f"\nSaved plot to: {args.out}")
    print("=====================================\n")


if __name__ == "__main__":
    main()