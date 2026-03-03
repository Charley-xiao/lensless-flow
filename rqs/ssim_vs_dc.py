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
from lensless_flow.metrics import ssim_torch


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
def eval_avg_ssim_for_dc_steps(
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
        batch_size=1,
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

    # Keep dc_step size fixed (from config) unless overridden on CLI
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))
    dc_mode = "rgb"

    avg_ssim = []
    for dc_steps in dc_steps_list:
        ssim_list = []
        pbar = tqdm(
            test_dl,
            desc=f"SSIM ({pred_type}, steps={steps}, dc_steps={dc_steps})",
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
                disable_physics=False,  # we are explicitly sweeping dc_steps
                pred_type=pred_type,
                dc_mode=dc_mode,
            )

            x_hat_c = x_hat.clamp(0, 1)
            x_c = x.clamp(0, 1)
            s = float(ssim_torch(x_hat_c, x_c))
            ssim_list.append(s)
            pbar.set_postfix(ssim=f"{s:.4f}")

        n = max(1, len(ssim_list))
        avg_ssim.append(sum(ssim_list) / n)

    return pred_type, avg_ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--steps", type=int, required=True, help="Sampling steps (fixed)")
    ap.add_argument("--dc_steps", type=str, required=True, help='e.g. "0,1,2,3,5,8,10" or "0:10:1"')
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--out", type=str, default="ssim_vs_dcsteps.png")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--force_mode", type=str, default=None, choices=[None, "vanilla", "btb"])
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dc_steps_list = _parse_int_list(args.dc_steps)
    max_batches = None if args.max_batches < 0 else args.max_batches

    mode, ssim_vals = eval_avg_ssim_for_dc_steps(
        cfg=cfg,
        ckpt_path=args.ckpt,
        steps=args.steps,
        dc_steps_list=dc_steps_list,
        max_batches=max_batches,
        force_mode=args.force_mode,
    )

    title = args.title
    if title is None:
        title = f"SSIM vs dc_steps (mode={mode}, steps={args.steps})"

    plt.figure()
    plt.plot(dc_steps_list, ssim_vals, marker="o")
    plt.xlabel("dc_steps")
    plt.ylabel("SSIM (avg over test subset)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print("\n========== SSIM vs dc_steps ==========")
    print(f"ckpt: {args.ckpt}")
    print(f"mode: {mode}")
    print(f"sampling steps: {args.steps}")
    print(f"dc_step_size (from cfg): {float(cfg.get('physics', {}).get('dc_step_size', 0.0))}")
    print("--------------------------------------")
    for k, s in zip(dc_steps_list, ssim_vals):
        print(f"dc_steps={k:>3d}  SSIM={s:.6f}")
    print(f"\nSaved plot to: {args.out}")
    print("======================================\n")


if __name__ == "__main__":
    main()