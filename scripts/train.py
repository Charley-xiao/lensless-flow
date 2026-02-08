import argparse
import yaml
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # new AMP API

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


@torch.no_grad()
def quick_eval(model, Hop, test_dl, cfg, device, max_batches=20):
    model.eval()
    psnr_list = []
    dc_rmse_list = []
    loss_dc_list = []

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
            disable_physics=False if cfg["cfm"]["loss"]["physics_weight"] > 0 else True,  # disable physics in sampling if not used in training
        )

        x_hat_c = x_hat.clamp(0, 1)
        x_c = x.clamp(0, 1)

        psnr_list.append(psnr(x_hat_c, x_c))

        resid = Hop.forward(x_hat) - y
        dc_rmse = resid.pow(2).mean().sqrt().item()
        dc_rmse_list.append(dc_rmse)
        loss_dc_list.append(resid.pow(2).mean().item())

    return {
        "eval/psnr": float(sum(psnr_list) / max(1, len(psnr_list))),
        "eval/dc_rmse": float(sum(dc_rmse_list) / max(1, len(dc_rmse_list))),
        "eval/dc_mse": float(sum(loss_dc_list) / max(1, len(loss_dc_list))),
    }


def chw_to_wandb_image(x_bchw):
    """
    x_bchw: [B,C,H,W] -> wandb.Image in HWC (or HW)
    """
    x = x_bchw[0].detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return wandb.Image(x[0].numpy())
    return wandb.Image(x.permute(1, 2, 0).numpy())


def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

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
    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
    )

    # PSF + operator
    psf = to_nchw(train_ds.psf.to(device))  # [1,C,h,w] or [1,C,H,W]
    y0, x0 = train_ds[0]
    y0 = to_nchw(y0)  # [1,C,H,W]
    x0 = to_nchw(x0)
    C = y0.shape[1]
    H_img, W_img = y0.shape[-2], y0.shape[-1]
    Hop = FFTConvOperator(psf=psf, im_hw=(H_img, W_img)).to(device)

    # -------------------------
    # Model/opt
    # -------------------------
    model = SimpleCondUNet(
        img_channels=C,
        base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    ensure_dir("checkpoints")

    # -------------------------
    # W&B init
    # -------------------------
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", True))

    run = None
    if use_wandb:
        run = wandb.init(
            project=wb.get("project", "lensless-flow"),
            entity=wb.get("entity", None),
            name=wb.get("name", None),
            tags=wb.get("tags", None),
            config=cfg,  # saves full config
        )
        # Optional: watch gradients/params (can slow a bit)
        # wandb.watch(model, log="gradients", log_freq=200)

    global_step = 0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()

        # running averages for epoch logging
        sum_loss = 0.0
        sum_loss_v = 0.0
        sum_loss_phys = 0.0
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for y, x in pbar:
            y = to_nchw(y).to(device)
            x = to_nchw(x).to(device)

            b = x.shape[0]
            t = sample_t(b, cfg["cfm"]["t_min"], cfg["cfm"]["t_max"], device)

            x_t, v_star, _ = cfm_forward(x0=x, y=y, t=t, noise_std=cfg["sample"]["init_noise_std"])

            with autocast("cuda", enabled=(cfg["train"]["amp"] and device.type == "cuda")):
                v_pred = model(x_t, y, t)
                loss_v = cfm_loss(v_pred, v_star) * cfg["cfm"]["loss"]["v_weight"]

                loss_phys = torch.tensor(0.0, device=device)
                if cfg["cfm"]["loss"]["physics_weight"] > 0:
                    loss_phys = physics_loss_from_v(x_t, v_pred, t, y, Hop) * cfg["cfm"]["loss"]["physics_weight"]

                loss = loss_v + loss_phys

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # grad norm (after unscale)
            grad_norm = None
            if cfg["train"]["grad_clip"] is not None:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"]).item()
            else:
                scaler.unscale_(opt)
                total = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total += p.grad.detach().float().norm(2).item() ** 2
                grad_norm = total ** 0.5

            scaler.step(opt)
            scaler.update()

            # LR (first param group)
            lr = opt.param_groups[0]["lr"]

            # tqdm display
            pbar.set_postfix(
                loss=float(loss),
                loss_v=float(loss_v),
                loss_phys=float(loss_phys),
            )

            # accumulate epoch stats
            sum_loss += float(loss)
            sum_loss_v += float(loss_v)
            sum_loss_phys += float(loss_phys)
            n_batches += 1

            # per-step wandb logging (W&B uses step as x-axis by default) :contentReference[oaicite:2]{index=2}
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/loss_v": float(loss_v),
                        "train/loss_phys": float(loss_phys),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": float(lr),
                        "train/t_mean": float(t.mean().item()),
                        "train/t_std": float(t.std(unbiased=False).item()),
                    },
                    step=global_step,
                )

            global_step += 1

        # -------------------------
        # End-of-epoch logging
        # -------------------------
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

        # quick eval every epoch (configurable)
        eval_batches = int(wb.get("eval_batches", 0) or 0)
        if eval_batches > 0:
            metrics = quick_eval(model, Hop, test_dl, cfg, device, max_batches=eval_batches)
            if use_wandb:
                wandb.log(metrics, step=global_step)
            else:
                print(metrics)

        # log a few example images (y, x, x_hat) every N epochs
        log_img_every = int(wb.get("log_images_every", 0) or 0)
        if use_wandb and log_img_every > 0 and (epoch % log_img_every == 0):
            # grab one fixed example from test set
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
                disable_physics=False if cfg["cfm"]["loss"]["physics_weight"] > 0 else True,  # disable physics in sampling if not used in training
            )

            wandb.log(
                {
                    "viz/lensless_y": chw_to_wandb_image(y_ex),
                    "viz/gt_x": chw_to_wandb_image(x_ex),
                    "viz/recon_xhat": chw_to_wandb_image(x_hat_ex),
                },
                step=global_step,
            )

        # -------------------------
        # Checkpoint
        # -------------------------
        if epoch % cfg["train"]["save_every"] == 0:
            ckpt_path = f"checkpoints/cfm_lensless_epoch{epoch}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
            print("Saved:", ckpt_path)

            # Log as W&B Artifact (recommended way to track checkpoints) :contentReference[oaicite:3]{index=3}
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"cfm_lensless",
                    type="model",
                    metadata={"epoch": epoch, "img_channels": C, "H": H_img, "W": W_img},
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
