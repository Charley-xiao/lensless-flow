import argparse, os, yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import wandb

from lensless_flow.utils import set_seed, ensure_dir
from lensless_flow.data import make_dataloader
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.metrics import psnr, ssim_torch


@torch.no_grad()
def eval_loop(model, dl, device, max_batches=0):
    model.eval()
    psnrs, ssims, n = 0.0, 0.0, 0
    mses = []
    for i, (y, x) in enumerate(dl):
        if max_batches > 0 and i >= max_batches:
            break
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        b = y.shape[0]
        x_t = torch.zeros_like(y)
        t = torch.zeros(b, device=device)

        x_hat = model(x_t, y, t).clamp(0, 1)
        psnrs += psnr(x_hat, x)
        ssims += float(ssim_torch(x_hat, x).item())
        mses.append(F.mse_loss(x_hat, x).item())
        n += 1
    model.train()
    return {"eval/psnr": psnrs / max(n, 1), "eval/ssim": ssims / max(n, 1), "eval/mse": sum(mses) / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # data
    train_ds, train_dl = make_dataloader(
        split="train",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        path=cfg["data"].get("path", None),
    )
    test_ds, test_dl = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    # infer channels from dataset
    y0, x0 = train_ds[0]
    y0 = to_nchw(y0)
    C = y0.shape[1]

    # model: reuse your existing U-Net implementation
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

    # wandb
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", True))
    if use_wandb:
        wandb.init(
            project=wb.get("project", "lensless-flow"),
            entity=wb.get("entity", None),
            name=wb.get("name", "unet_baseline"),
            tags=wb.get("tags", ["unet", "baseline", "supervised"]),
            config=cfg,
        )

    ensure_dir(cfg["train"]["save_dir"])
    save_every = int(cfg["train"].get("save_every", 1))
    eval_every = int(cfg["train"].get("eval_every", 1))

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch} (unet baseline)")
        sum_loss, n = 0.0, 0

        for y, x in pbar:
            y = to_nchw(y).to(device, non_blocking=True)
            x = to_nchw(x).to(device, non_blocking=True)

            b = y.shape[0]
            x_t = torch.zeros_like(y)
            t = torch.zeros(b, device=device)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                x_hat = model(x_t, y, t)
                if cfg["train"].get("loss", "l1").lower() == "mse":
                    loss = F.mse_loss(x_hat, x)
                else:
                    loss = F.l1_loss(x_hat, x)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            sum_loss += float(loss.item())
            n += 1
            pbar.set_postfix(loss=sum_loss / n)

            if use_wandb:
                wandb.log({"train/loss": float(loss.item())}, commit=False)

        if (epoch % eval_every) == 0:
            metrics = eval_loop(model, test_dl, device, max_batches=int(cfg["train"].get("eval_batches", 0)))
            if use_wandb:
                wandb.log(metrics)
            print(metrics)

        if (epoch % save_every) == 0:
            ckpt = os.path.join(cfg["train"]["save_dir"], f"unet_baseline_epoch{epoch}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt)
            print("saved:", ckpt)


if __name__ == "__main__":
    main()