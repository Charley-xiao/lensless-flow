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


# -------------------------
# SSIM (pure PyTorch)
# -------------------------
def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).contiguous()
    return kernel_2d


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0, K1: float = 0.01, K2: float = 0.03, eps: float = 1e-12):
    """
    Compute SSIM for tensors x, y in [B,C,H,W], values assumed in [0, data_range].
    Returns: scalar mean SSIM over batch and channels.

    This is the standard SSIM (single-scale) computed with a Gaussian window.
    """
    assert x.ndim == 4 and y.ndim == 4, "x,y must be [B,C,H,W]"
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"

    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # make gaussian window
    if window_size % 2 == 0:
        window_size += 1  # ensure odd
    kernel = _gaussian_kernel(window_size, sigma, device, dtype)
    kernel = kernel.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    padding = window_size // 2

    # depthwise conv
    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # SSIM map
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)

    return ssim_map.mean()  # scalar


@torch.no_grad()
def quick_eval(model, Hop, test_dl, cfg, device, max_batches=20, denom_min=0.05, pred_type="btb"):
    """
    Validation:
      - Generate x_hat via sampler
      - Compute x_hat vs x metrics: L1, MSE, PSNR, SSIM
      - Also compute data-consistency RMSE in measurement space: ||H x_hat - y||_2
    """
    model.eval()

    l1_list = []
    mse_list = []
    psnr_list = []
    ssim_list = []
    dc_rmse_list = []

    # SSIM params (optional overrides)
    ssim_cfg = cfg.get("ssim", {})
    ws = int(ssim_cfg.get("window_size", 11))
    sigma = float(ssim_cfg.get("sigma", 1.5))
    data_range = float(ssim_cfg.get("data_range", 1.0))

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

        # metrics in [0,1]
        x_hat_c = x_hat.clamp(0, 1)
        x_c = x.clamp(0, 1)

        l1_list.append(F.l1_loss(x_hat_c, x_c).item())
        mse_list.append(F.mse_loss(x_hat_c, x_c).item())
        psnr_list.append(psnr(x_hat_c, x_c))

        # SSIM computed in float32 for numerical stability
        ssim_val = float(ssim_torch(x_hat_c.float(), x_c.float(), window_size=ws, sigma=sigma, data_range=data_range).item())
        ssim_list.append(ssim_val)

        dc_err = (Hop.forward(x_hat.float()) - y.float()).pow(2).mean().sqrt().item()
        dc_rmse_list.append(dc_err)

    def avg(lst):
        return float(sum(lst) / max(1, len(lst)))

    return {
        "eval/l1": avg(l1_list),
        "eval/mse": avg(mse_list),
        "eval/psnr": avg(psnr_list),
        "eval/ssim": avg(ssim_list),
        "eval/dc_rmse": avg(dc_rmse_list),
    }


def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    if cfg.get("is_a100", False) and device.type == "cuda":
        # A100-specific optimization: enable TF32 for matmul and convolution
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -------------------------
    # Experiment mode (ablation switch)
    # -------------------------
    train_mode = str(cfg.get("train", {}).get("mode", "btb")).lower()
    assert train_mode in ["btb", "vanilla"], f"cfg.train.mode must be 'btb' or 'vanilla', got {train_mode}"
    pred_type = train_mode

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
    psf = to_nchw(train_ds.psf.to(device))
    y0, x0 = train_ds[0]
    y0 = to_nchw(y0)
    x0 = to_nchw(x0)
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
    if cfg.get("compile", {}).get("enabled", True) and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    # -------------------------
    # ReduceLROnPlateau on eval/ssim (maximize)
    # -------------------------
    sched_cfg = cfg.get("sched", {})
    use_sched = bool(sched_cfg.get("enabled", False))
    sched_metric = str(sched_cfg.get("metric", "eval/ssim"))
    sched_mode = str(sched_cfg.get("mode", "max"))  # IMPORTANT: SSIM wants "max"

    scheduler = None
    if use_sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=sched_mode,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            cooldown=int(sched_cfg.get("cooldown", 0)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    ensure_dir("checkpoints")
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    log_every = int(wb.get("log_every", 50))
    global_step = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch} ({pred_type})")

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

            den = (1.0 - t).clamp_min(denom_min).view(b, 1, 1, 1)

            with autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                out = model(x_t, y, t)

                if pred_type == "vanilla":
                    v_pred = out
                    x_pred = None
                else:
                    x_pred = out
                    v_pred = (x_pred - x_t) / den

                loss_v = cfm_loss(v_pred, v_star) * cfg["cfm"]["loss"]["v_weight"]

                loss_phys = torch.tensor(0.0, device=device)
                if cfg["cfm"]["loss"]["physics_weight"] > 0:
                    loss_phys = physics_loss_from_v(x_t, v_pred, t, y, Hop) * cfg["cfm"]["loss"]["physics_weight"]

                loss = loss_v + loss_phys

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

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

            pbar.set_postfix(
                loss=float(loss),
                loss_v=float(loss_v),
                loss_phys=float(loss_phys),
                gnorm=float(grad_norm),
                lr=float(lr),
            )

            sum_loss += float(loss)
            sum_loss_v += float(loss_v)
            sum_loss_phys += float(loss_phys)
            n_batches += 1

            if use_wandb and (global_step % log_every == 0):
                def rms(z: torch.Tensor):
                    return float(z.detach().float().pow(2).mean().sqrt().item())

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
                    "epoch/lr": float(opt.param_groups[0]["lr"]),
                },
                step=global_step,
            )

        # -------------------------
        # Validation + LR scheduler step (monitor eval/ssim)
        # -------------------------
        if eval_batches > 0 and test_dl is not None:
            eval_metrics = quick_eval(
                model, Hop, test_dl, cfg, device,
                max_batches=eval_batches,
                denom_min=denom_min,
                pred_type=pred_type,
            )
            if use_wandb:
                wandb.log(eval_metrics, step=global_step)
            else:
                print(eval_metrics)

            if scheduler is not None:
                if sched_metric not in eval_metrics:
                    raise KeyError(
                        f"sched.metric='{sched_metric}' not found in eval_metrics keys: {list(eval_metrics.keys())}"
                    )
                scheduler.step(eval_metrics[sched_metric])

                if use_wandb:
                    wandb.log({"sched/lr_after": float(opt.param_groups[0]["lr"])}, step=global_step)

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