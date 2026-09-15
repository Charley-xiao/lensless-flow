"""Train a deterministic, supervised image-to-image U-Net baseline."""
import argparse
import json
import math
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import wandb

from lensless_flow.config import load_config
from lensless_flow.utils import set_seed, ensure_dir
from lensless_flow.data import make_dataloader, HumanRBCHologramDataset
from lensless_flow.model_unet import SimpleCondUNet
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.metrics import ssim_map_torch, region_image_metrics
from lensless_flow.rbc_regions import rbc_region_mask, save_region_preview


def data_loader_kwargs(cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}) or {})
    excluded = {"path", "split", "eval_split", "downsample", "flip_ud", "num_workers"}
    return {k: v for k, v in data_cfg.items() if k not in excluded}


def baseline_forward(model, y):
    """Zero auxiliary input preserves compatibility with existing U-Net weights."""
    t = y.new_zeros(y.shape[0]) if getattr(model, "use_time_conditioning", True) else None
    return model(torch.zeros_like(y), y, t)


def save_checkpoint(state, path):
    temporary = str(path) + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_eval_record(cfg, metrics, epoch, step):
    path = cfg["train"].get("metrics_path")
    if path:
        ensure_dir(os.path.dirname(path) or ".")
        record = {"epoch": epoch, "step": step, **metrics}
        record = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in record.items()}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


@torch.no_grad()
def eval_loop(model, dl, device, max_batches=0, cfg=None, epoch=None):
    """Average per-image scores, including an incomplete final batch."""
    cfg = cfg or {}
    eval_cfg = cfg.get("eval", {})
    roi = bool(eval_cfg.get("rbc_regions", False))
    ssim_cfg = cfg.get("ssim", {})
    ssim_kwargs = dict(window_size=int(ssim_cfg.get("window_size", 11)),
                       sigma=float(ssim_cfg.get("sigma", 1.5)), data_range=float(ssim_cfg.get("data_range", 1)))
    values, previews, count = {}, [], 0
    was_training = model.training
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        # Even deterministic DataLoader iteration consumes a Torch RNG seed.
        with torch.random.fork_rng(devices=devices):
            for i, (y, x) in enumerate(dl):
                if max_batches > 0 and i >= max_batches:
                    break
                y, x = to_nchw(y).to(device), to_nchw(x).to(device)
                raw = baseline_forward(model, y).float()
                if not torch.isfinite(raw).all():
                    raise FloatingPointError("Nonfinite baseline validation prediction.")
                pred, target = raw.clamp(0, 1), x.float().clamp(0, 1)
                mse = (pred - target).square().mean((1, 2, 3))
                scores = {"mse": mse, "l1": (pred - target).abs().mean((1, 2, 3)),
                          "psnr": torch.where(mse <= 1e-12, 99.0, -10 * torch.log10(mse.clamp_min(1e-12))),
                          "ssim": ssim_map_torch(pred, target, **ssim_kwargs).mean((1, 2, 3)),
                          "clipped_fraction": ((raw < 0) | (raw > 1)).float().mean((1, 2, 3))}
                if roi:
                    mask = rbc_region_mask(target, **eval_cfg.get("rbc_mask", {}))
                    scores.update(region_image_metrics(pred, target, mask, **ssim_kwargs))
                    if eval_cfg.get("preview_dir"):
                        for j in range(min(y.shape[0], 8 - len(previews))):
                            previews.append(tuple(item[j].cpu() for item in (y, target, pred, mask)))
                for key, batch_values in scores.items():
                    values.setdefault(key, []).extend(batch_values[torch.isfinite(batch_values)].cpu().tolist())
                count += y.shape[0]
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation loader contains no images.")
    metrics = {"eval/samples": count}
    for key, items in values.items():
        metrics[f"eval/{key}"] = sum(items) / len(items) if items else float("nan")
        metrics[f"eval/{key}_valid_samples"] = len(items)
    if previews:
        save_region_preview(os.path.join(eval_cfg["preview_dir"], f"epoch_{epoch if epoch is not None else 'eval'}.png"),
                            previews, prediction_title="Direct U-Net")
    return metrics


def main(cfg=None):
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True)
        args, overrides = ap.parse_known_args()
        cfg = load_config(args.config, overrides)
    train = cfg["train"]
    loss_name = str(train.get("loss", "l1")).lower()
    if loss_name not in {"mse", "l1"}:
        raise ValueError("Baseline train.loss must be 'mse' or 'l1'.")
    if train.get("init_checkpoint"):
        raise ValueError("This baseline trainer starts from random initialization.")
    if int(train["epochs"]) < 1 or int(train.get("save_every", 1)) < 1 or int(train.get("eval_every", 1)) < 1:
        raise ValueError("Epoch, save, and evaluation intervals must be positive.")
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.get("is_a100", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_ds, train_dl = make_dataloader(
        split=cfg["data"].get("split", "train"), downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"], batch_size=train["batch_size"], num_workers=cfg["data"]["num_workers"],
        path=cfg["data"].get("path"), **data_loader_kwargs(cfg))
    test_ds, test_dl = make_dataloader(
        split=cfg["data"].get("eval_split", "test"), downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"], batch_size=int(cfg.get("eval", {}).get("batch_size", train["batch_size"])),
        num_workers=0, path=cfg["data"].get("path"), **data_loader_kwargs(cfg))
    if not len(train_dl):
        raise ValueError("Training requires at least one complete batch.")
    y0 = to_nchw(train_ds[0][0])
    if cfg.get("eval", {}).get("rbc_regions", False) and (
        not isinstance(test_ds, HumanRBCHologramDataset) or tuple(y0.shape[1:]) != (1, 256, 256)
    ):
        raise ValueError("RBC pseudo-region monitoring requires the native 256x256 grayscale RBC dataset.")
    model = SimpleCondUNet(img_channels=y0.shape[1], base_ch=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]), num_res_blocks=cfg["model"]["num_res_blocks"],
        use_time_conditioning=False).to(device)
    print("Baseline: one forward pass, hologram -> phase; time conditioning disabled")
    print("Initialization: random weights; fresh optimizer; loss:", loss_name)
    print("Model params (registered):", sum(p.numel() for p in model.parameters()))
    print(f"Data: {len(train_ds)} train pairs, {len(test_ds)} validation pairs; {len(train_dl)} updates/epoch")
    if cfg.get("compile", {}).get("enabled", True) and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")
    opt = torch.optim.AdamW(model.parameters(), lr=train["lr"])
    amp = bool(train["amp"]) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp)
    sched = cfg.get("sched", {})
    scheduler = None
    if sched.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=sched.get("mode", "max"),
            factor=float(sched.get("factor", 0.5)), patience=int(sched.get("patience", 10)),
            threshold=float(sched.get("threshold", 1e-4)), cooldown=int(sched.get("cooldown", 0)),
            min_lr=float(sched.get("min_lr", 2e-5)))
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", True))
    if use_wandb:
        wandb.init(project=wb.get("project", "lensless-flow"), entity=wb.get("entity"),
                   name=wb.get("name", "unet_baseline"), tags=wb.get("tags", ["unet", "baseline", "supervised"]), config=cfg)
        wandb.run.summary["startup/initialized"] = True
    ensure_dir(train["save_dir"])
    save_every, eval_every = int(train.get("save_every", 1)), int(train.get("eval_every", 1))
    best_metrics = dict(train.get("best_checkpoints", {}))  # safe filename stem -> SSIM metric to maximize
    if any(not name.replace("_", "").isalnum() for name in best_metrics):
        raise ValueError("Best-checkpoint names must be alphanumeric/underscore stems.")
    best_scores = {}
    global_step = 0
    max_eval_batches = int(train.get("eval_batches", 0))
    if train.get("eval_at_start", False):
        metrics = eval_loop(model, test_dl, device, max_eval_batches, cfg, epoch=0)
        save_eval_record(cfg, metrics, 0, global_step)
        print("Initial evaluation:", metrics)
        if use_wandb:
            wandb.log({"epoch": 0, **metrics}, step=0)
    for epoch in range(1, train["epochs"] + 1):
        model.train()
        sum_loss, batches = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch} (direct U-Net, {loss_name})", mininterval=5)
        for y, x in pbar:
            y, x = to_nchw(y).to(device, non_blocking=True), to_nchw(x).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp):
                pred = baseline_forward(model, y)
                loss = F.mse_loss(pred, x) if loss_name == "mse" else F.l1_loss(pred, x)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite baseline training loss.")
            scaler.scale(loss).backward()
            grad_norm = None
            if float(train.get("grad_clip", 0)) > 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["grad_clip"]), error_if_nonfinite=not amp)
            scaler.step(opt)
            scaler.update()
            global_step += 1
            sum_loss += float(loss)
            batches += 1
            pbar.set_postfix(loss=sum_loss / batches, lr=opt.param_groups[0]["lr"])
            # The final step is logged with evaluation below, avoiding duplicate-step commits.
            if use_wandb and batches < len(train_dl) and global_step % int(wb.get("log_every", 50)) == 0:
                logs = {"train/loss": float(loss), "train/lr": float(opt.param_groups[0]["lr"]),
                        "train/epoch": epoch, "diag/nan_or_inf": 0}
                if grad_norm is not None:
                    logs["train/grad_norm"] = float(grad_norm)
                wandb.log(logs, step=global_step, commit=True)
        metrics, improved = {}, []
        if epoch % eval_every == 0 or epoch == train["epochs"]:
            metrics = eval_loop(model, test_dl, device, max_eval_batches, cfg, epoch=epoch)
            save_eval_record(cfg, metrics, epoch, global_step)
            print("Evaluation:", {"epoch": epoch, **metrics})
            if scheduler is not None:
                metric = float(metrics[sched.get("metric", "eval/ssim")])
                if math.isfinite(metric):
                    scheduler.step(metric)
            for name, key in best_metrics.items():
                score = float(metrics[key])
                if math.isfinite(score) and (name not in best_scores or score > best_scores[name]):
                    best_scores[name] = score
                    improved.append(name)
        if use_wandb:
            logs = {"epoch": epoch, "epoch/train_loss": sum_loss / batches,
                    "train/loss": float(loss), "train/lr": float(opt.param_groups[0]["lr"]),
                    "train/epoch": epoch, "diag/nan_or_inf": 0,
                    "sched/lr_after": float(opt.param_groups[0]["lr"]), **metrics}
            preview = os.path.join(cfg.get("eval", {}).get("preview_dir", ""), f"epoch_{epoch}.png")
            if metrics and wb.get("log_images_every", 0) and epoch % int(wb["log_images_every"]) == 0 and os.path.isfile(preview):
                logs["viz/validation_grid"] = wandb.Image(preview, caption="Hologram, phase target, direct U-Net, target pseudo-region; fixed [0,1] scale")
            wandb.log(logs, step=global_step, commit=True)
        periodic = epoch == 1 or epoch % save_every == 0 or epoch == train["epochs"]
        if periodic or improved:
            state = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "global_step": global_step,
                     "use_time_conditioning": False, "model_name": "unet", "mode": "supervised",
                     "prediction_type": "direct_image", "eval_metrics": metrics, "training_loss": loss_name}
            for name in improved:
                path = os.path.join(train["save_dir"], name + ".pt")
                save_checkpoint(state, path)
                print(f"Saved {path}: epoch {epoch}, {best_metrics[name]}={best_scores[name]:.6f}")
                if use_wandb:
                    wandb.run.summary[f"{name}/epoch"] = epoch
                    wandb.run.summary[f"{name}/metric"] = best_metrics[name]
                    wandb.run.summary[f"{name}/score"] = best_scores[name]
            if periodic:
                path = os.path.join(train["save_dir"], f"unet_baseline_epoch{epoch}.pt")
                save_checkpoint(state, path)
                print("Saved:", path)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
