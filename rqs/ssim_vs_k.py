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


def _parse_steps(s: str) -> list[int]:
    # Accept "5,10,20" or "5 10 20" or "5:50:5" (start:stop:step, inclusive stop)
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
def eval_avg_ssim_for_steps(
    *,
    cfg: dict,
    ckpt_path: str,
    steps_list: list[int],
    max_batches: int | None,
    disable_physics: bool,
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
    y0 = to_nchw(y0)  # [1,C,H,W]
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

    # Use cfg physics params when enabled; otherwise dc_steps=0 is fine, but we also pass disable_physics=True
    dc_steps = int(cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(cfg.get("physics", {}).get("dc_step_size", 0.0))
    dc_mode = "rgb"

    avg_ssim = []
    for steps in steps_list:
        ssim_list = []

        pbar = tqdm(
            test_dl,
            desc=f"SSIM ({pred_type}, steps={steps}, physics={'off' if disable_physics else 'on'})",
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
                dc_steps=dc_steps,
                init_noise_std=init_noise_std,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=bool(disable_physics),
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
    ap.add_argument("--ckpt_vanilla", type=str, required=True)
    ap.add_argument("--ckpt_btb", type=str, required=True)
    ap.add_argument("--steps", type=str, required=True, help='e.g. "5,10,20,50" or "5:50:5"')
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--out", type=str, default="ssim_vs_steps_4curves.png")
    ap.add_argument("--title", type=str, default="SSIM vs Sampling Steps (with/without Physics Guidance)")

    # Optional overrides if your ckpt metadata doesn't store mode cleanly
    ap.add_argument("--force_mode_vanilla", type=str, default=None, choices=[None, "vanilla", "btb"])
    ap.add_argument("--force_mode_btb", type=str, default=None, choices=[None, "vanilla", "btb"])

    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    steps_list = _parse_steps(args.steps)
    max_batches = None if args.max_batches < 0 else args.max_batches

    # Vanilla: physics off/on
    mode_v_off, ssim_v_off = eval_avg_ssim_for_steps(
        cfg=cfg,
        ckpt_path=args.ckpt_vanilla,
        steps_list=steps_list,
        max_batches=max_batches,
        disable_physics=True,
        force_mode=args.force_mode_vanilla,
    )
    mode_v_on, ssim_v_on = eval_avg_ssim_for_steps(
        cfg=cfg,
        ckpt_path=args.ckpt_vanilla,
        steps_list=steps_list,
        max_batches=max_batches,
        disable_physics=False,
        force_mode=args.force_mode_vanilla,
    )

    # BTB: physics off/on
    mode_b_off, ssim_b_off = eval_avg_ssim_for_steps(
        cfg=cfg,
        ckpt_path=args.ckpt_btb,
        steps_list=steps_list,
        max_batches=max_batches,
        disable_physics=True,
        force_mode=args.force_mode_btb,
    )
    mode_b_on, ssim_b_on = eval_avg_ssim_for_steps(
        cfg=cfg,
        ckpt_path=args.ckpt_btb,
        steps_list=steps_list,
        max_batches=max_batches,
        disable_physics=False,
        force_mode=args.force_mode_btb,
    )

    # Plot (no explicit colors per your environment rules; matplotlib will choose defaults)
    plt.figure()
    plt.plot(steps_list, ssim_v_off, marker="o", label=f"vanilla ckpt ({mode_v_off}) - physics OFF")
    plt.plot(steps_list, ssim_v_on, marker="o", label=f"vanilla ckpt ({mode_v_on}) - physics ON")
    plt.plot(steps_list, ssim_b_off, marker="o", label=f"btb ckpt ({mode_b_off}) - physics OFF")
    plt.plot(steps_list, ssim_b_on, marker="o", label=f"btb ckpt ({mode_b_on}) - physics ON")

    plt.xlabel("Sampling steps")
    plt.ylabel("SSIM (avg over test subset)")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    # Print table
    print("\n========== SSIM vs Steps (4 curves) ==========")
    print(f"steps: {steps_list}")
    print(f"vanilla_ckpt: {args.ckpt_vanilla} (mode={mode_v_off})")
    print(f"btb_ckpt:     {args.ckpt_btb} (mode={mode_b_off})")
    print("physics ON uses cfg.physics.dc_steps and cfg.physics.dc_step_size")
    print("----------------------------------------------")
    for k, a, b, c, d in zip(steps_list, ssim_v_off, ssim_v_on, ssim_b_off, ssim_b_on):
        print(
            f"{k:>4d}  "
            f"V_off={a:.6f}  V_on={b:.6f}  "
            f"B_off={c:.6f}  B_on={d:.6f}"
        )
    print(f"\nSaved plot to: {args.out}")
    print("==============================================\n")


if __name__ == "__main__":
    main()