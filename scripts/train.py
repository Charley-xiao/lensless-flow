import argparse
import yaml
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import wandb

from lensless_flow.utils import set_seed, ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTConvOperator
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.flow_matching import sample_t, cfm_forward
from lensless_flow.losses import cfm_loss, physics_loss_from_v
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.sampler import sample_with_physics_guidance


def psnr(x_hat, x):
    mse = F.mse_loss(x_hat, x).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def chw_to_wandb_image(x_bchw: torch.Tensor):
    """[B,C,H,W] -> wandb.Image"""
    x = x_bchw[0].detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return wandb.Image(x[0].numpy())
    return wandb.Image(x.permute(1, 2, 0).numpy())


@torch.no_grad()
def quick_eval(model, Hop, test_dl, cfg, device, max_batches=20, denom_min=0.05, pred_type="btb"):
    """
    Quick eval using sampler.py (Heun) with selectable pred_type:
      - pred_type="vanilla": model outputs v directly
      - pred_type="btb": model outputs x_pred and sampler converts to v
    """
    model.eval()
    psnr_list = []
    dc_rmse_list = []

    for i, (y, x) in enumerate(test_dl):
        if i >= max_batches:
            break
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        x_hat = sample_with_physics_guidance(
            model=model,
            y=y,
            H=Hop,
            steps=cfg["sample"]["steps"],
            dc_step=cfg["physics"]["dc_step_size"],
            dc_steps=cfg["physics"]["dc_steps"],
            init_noise_std=cfg["sample"]["init_noise_std"],
            denom_min=denom_min,
            clamp_x=True,
            disable_physics=bool(cfg.get("physics", {}).get("disable_in_eval", False)),
            pred_type=pred_type,
        )

        x_hat_c = x_hat.clamp(0, 1)
        x_c = x.clamp(0, 1)

        psnr_list.append(psnr(x_hat_c, x_c))
        dc_err = (Hop.forward(x_hat.float()) - y.float()).pow(2).mean().sqrt().item()
        dc_rmse_list.append(dc_err)

    return {
        "eval/psnr": float(sum(psnr_list) / max(1, len(psnr_list))),
        "eval/dc_rmse": float(sum(dc_rmse_list) / max(1, len(dc_rmse_list))),
    }


def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Experiment mode (ablation switch)
    # -------------------------
    train_mode = str(cfg.get("train", {}).get("mode", "btb")).lower()
    assert train_mode in ["btb", "vanilla"], f"cfg.train.mode must be 'btb' or 'vanilla', got {train_mode}"
    pred_type = train_mode  # used consistently in eval/sampling logs

    # -------------------------
    # W&B init
    # -------------------------
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", True))
    if use_wandb:
        wandb.init(
            project=wb.get("project", "lensless-flow"),
            entity=wb.get("entity", None),
            name=wb.get("name", None),
            tags=wb.get("tags", [pred_type, "cfm", "lensless"]),
            config=cfg,
        )

    # -------------------------
    # Data
    # -------------------------
    train_ds, train_dl = make_dataloader(
        split="train",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # For quick eval + image logging (optional)
    eval_batches = int(wb.get("eval_batches", 0) or 0)
    log_images_every = int(wb.get("log_images_every", 0) or 0)
    test_ds, test_dl = None, None
    if eval_batches > 0 or log_images_every > 0:
        test_ds, test_dl = make_dataloader(
            split="test",
            downsample=cfg["data"]["downsample"],
            flip_ud=cfg["data"]["flip_ud"],
            batch_size=1,
            num_workers=0,
        )

    # -------------------------
    # PSF + operator
    # -------------------------
    psf = to_nchw(train_ds.psf.to(device))  # [1,C,h,w] or [1,C,H,W]
    y0, x0 = train_ds[0]
    y0 = to_nchw(y0)  # [1,C,H,W]
    x0 = to_nchw(x0)  # [1,C,H,W]
    C = y0.shape[1]
    H_img, W_img = y0.shape[-2], y0.shape[-1]
    Hop = FFTConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    # -------------------------
    # Model
    # -------------------------
    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    ensure_dir("checkpoints")

    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    # logging frequency
    log_every = int(wb.get("log_every", 50))
    global_step = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch} ({pred_type})")

        # epoch accumulators
        sum_loss = 0.0
        sum_loss_v = 0.0
        sum_loss_phys = 0.0
        n_batches = 0

        for y, x in pbar:
            y = to_nchw(y).to(device, non_blocking=True)
            x = to_nchw(x).to(device, non_blocking=True)

            b = x.shape[0]
            t = sample_t(b, cfg["cfm"]["t_min"], cfg["cfm"]["t_max"], device)

            x_t, v_star, _ = cfm_forward(
                x0=x,
                y=y,
                t=t,
                noise_std=cfg["sample"]["init_noise_std"],
            )

            # BTB denom (only used in BTB)
            den = (1.0 - t).clamp_min(denom_min).view(b, 1, 1, 1)

            with autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                out = model(x_t, y, t)

                if pred_type == "vanilla":
                    # Vanilla FM: network directly outputs v_pred
                    v_pred = out
                    x_pred = None
                else:
                    # BTB: network outputs x_pred and we derive v_pred
                    x_pred = out
                    v_pred = (x_pred - x_t) / den

                loss_v = cfm_loss(v_pred, v_star) * cfg["cfm"]["loss"]["v_weight"]

                loss_phys = torch.tensor(0.0, device=device)
                if cfg["cfm"]["loss"]["physics_weight"] > 0:
                    loss_phys = physics_loss_from_v(x_t, v_pred, t, y, Hop) * cfg["cfm"]["loss"]["physics_weight"]

                loss = loss_v + loss_phys

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # grad norm (after unscale)
            scaler.unscale_(opt)
            if cfg["train"]["grad_clip"] is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"]).item()
            else:
                total = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total += p.grad.detach().float().norm(2).item() ** 2
                grad_norm = total ** 0.5

            scaler.step(opt)
            scaler.update()

            lr = opt.param_groups[0]["lr"]

            # tqdm
            pbar.set_postfix(
                loss=float(loss),
                loss_v=float(loss_v),
                loss_phys=float(loss_phys),
                gnorm=float(grad_norm),
            )

            # epoch stats
            sum_loss += float(loss)
            sum_loss_v += float(loss_v)
            sum_loss_phys += float(loss_phys)
            n_batches += 1

            # rich step logging (throttled)
            if use_wandb and (global_step % log_every == 0):
                def rms(z: torch.Tensor):
                    return float(z.detach().float().pow(2).mean().sqrt().item())

                # nan checks
                nan_or_inf = (
                    torch.isnan(v_pred).any() or torch.isinf(v_pred).any()
                    or torch.isnan(loss).any() or torch.isinf(loss).any()
                )
                if pred_type == "btb" and x_pred is not None:
                    nan_or_inf = nan_or_inf or torch.isnan(x_pred).any() or torch.isinf(x_pred).any()

                log_dict = {
                    "train/loss": float(loss),
                    "train/loss_v": float(loss_v),
                    "train/loss_phys": float(loss_phys),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(lr),
                    "train/mode": 0.0 if pred_type == "vanilla" else 1.0,
                    "train/t_mean": float(t.mean().item()),
                    "train/t_std": float(t.std(unbiased=False).item()),
                    "diag/x_t_rms": rms(x_t),
                    "diag/v_pred_rms": rms(v_pred),
                    "diag/nan_or_inf": float(nan_or_inf),
                }
                if pred_type == "btb" and x_pred is not None:
                    log_dict["train/denom_min"] = float(denom_min)
                    log_dict["diag/x_pred_rms"] = rms(x_pred)

                wandb.log(log_dict, step=global_step)

            global_step += 1

        # epoch-level logging
        avg_loss = sum_loss / max(1, n_batches)
        avg_loss_v = sum_loss_v / max(1, n_batches)
        avg_loss_phys = sum_loss_phys / max(1, n_batches)

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch/train_loss": avg_loss,
                    "epoch/train_loss_v": avg_loss_v,
                    "epoch/train_loss_phys": avg_loss_phys,
                },
                step=global_step,
            )

        # optional quick eval (uses same pred_type)
        if eval_batches > 0 and test_dl is not None:
            metrics = quick_eval(
                model, Hop, test_dl, cfg, device,
                max_batches=eval_batches,
                denom_min=denom_min,
                pred_type=pred_type,
            )
            if use_wandb:
                wandb.log(metrics, step=global_step)
            else:
                print(metrics)

        # optional image logging
        if use_wandb and log_images_every > 0 and test_ds is not None and (epoch % log_images_every == 0):
            y_ex, x_ex = test_ds[0]
            y_ex = to_nchw(y_ex).to(device)
            x_ex = to_nchw(x_ex).to(device)

            x_hat_ex = sample_with_physics_guidance(
                model=model,
                y=y_ex,
                H=Hop,
                steps=cfg["sample"]["steps"],
                dc_step=cfg["physics"]["dc_step_size"],
                dc_steps=cfg["physics"]["dc_steps"],
                init_noise_std=cfg["sample"]["init_noise_std"],
                denom_min=denom_min,
                clamp_x=True,
                disable_physics=bool(cfg.get("physics", {}).get("disable_in_eval", False)),
                pred_type=pred_type,
            )

            wandb.log(
                {
                    "viz/lensless_y": chw_to_wandb_image(y_ex),
                    "viz/gt_x": chw_to_wandb_image(x_ex),
                    "viz/recon_xhat": chw_to_wandb_image(x_hat_ex),
                },
                step=global_step,
            )

        # checkpoint
        if epoch % cfg["train"]["save_every"] == 0:
            ckpt_path = f"checkpoints/cfm_lensless_{pred_type}_epoch{epoch}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "mode": pred_type}, ckpt_path)
            print("Saved:", ckpt_path)

            if use_wandb and bool(wb.get("log_artifacts", True)):
                artifact = wandb.Artifact(
                    name=f"cfm_lensless_{pred_type}",
                    type="model",
                    metadata={"epoch": epoch, "C": C, "H": H_img, "W": W_img, "mode": pred_type, "denom_min": denom_min},
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}", "latest"])

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
