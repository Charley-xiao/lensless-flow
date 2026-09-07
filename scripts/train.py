import argparse
import json
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import wandb

from lensless_flow.config import load_config
from lensless_flow.utils import set_seed, ensure_dir
from lensless_flow.data import make_dataloader, HumanRBCHologramDataset
from lensless_flow.physics import build_forward_operator_from_dataset
from lensless_flow.model_factory import build_flow_model, resolve_model_name, load_checkpoint_state_dict
from lensless_flow.model_unet import use_time_conditioning_from_cfg
from lensless_flow.flow_matching import (
    build_flow_matcher,
    normalize_flow_matcher_name,
    sample_flow_matching_training_batch,
    sample_t,
    x0_from_xt_v,
)
from lensless_flow.losses import cfm_loss, physics_loss_from_v, region_balanced_cfm_loss
from lensless_flow.rbc_regions import rbc_region_mask, region_loss_config, region_eval_enabled, save_region_preview
from lensless_flow.measurement_source import (
    source_mode_from_cfg,
    source_sampler_kwargs_from_cfg,
    source_sigma0_from_cfg,
)
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.metrics import ssim_torch, psnr, region_image_metrics


def chw_to_wandb_image(x_bchw: torch.Tensor):
    """[B,C,H,W] -> wandb.Image"""
    x = x_bchw[0].detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return wandb.Image(x[0].numpy())
    return wandb.Image(x.permute(1, 2, 0).numpy())


def data_loader_kwargs(cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}) or {})
    excluded = {"path", "split", "eval_split", "downsample", "flip_ud", "num_workers"}
    return {k: v for k, v in data_cfg.items() if k not in excluded}


def save_eval_record(cfg: dict, metrics: dict, epoch: int, step: int) -> None:
    """Optional durable evaluation history, including the warm-start baseline."""
    path = cfg.get("train", {}).get("metrics_path")
    if path:
        ensure_dir(os.path.dirname(path) or ".")
        record = {"epoch": epoch, "step": step, **metrics}
        record = {key: (value if not isinstance(value, float) or torch.isfinite(torch.tensor(value)) else None)
                  for key, value in record.items()}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def save_checkpoint_atomic(state: dict, path: str) -> None:
    """Keep the previous checkpoint intact until its replacement is complete."""
    temporary_path = path + ".tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def _eval_seed_for_batch(cfg: dict, batch_idx: int, offset: int = 0) -> int | None:
    eval_cfg = dict(cfg.get("eval", {}) or {})
    if not bool(eval_cfg.get("fixed_latent", True)):
        return None
    base_seed = eval_cfg.get("seed", cfg.get("seed", None))
    if base_seed is None:
        return None
    stride = int(eval_cfg.get("seed_stride", 100_003))
    return int(base_seed) + int(offset) + int(batch_idx) * stride


def _call_with_preserved_rng(fn, seed: int | None, device: torch.device):
    if seed is None:
        return fn()

    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    try:
        torch.manual_seed(int(seed))
        if cuda_states is not None:
            torch.cuda.manual_seed_all(int(seed))
        return fn()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def quick_eval(model, Hop, test_dl, cfg, device, max_batches=20, denom_min=0.05, pred_type="btb", epoch=None):
    """
    Validation:
      - Generate x_hat via sampler
      - Compute x_hat vs x metrics: L1, MSE, PSNR, SSIM
      - If H is available, compute data-consistency RMSE in measurement space.
    """
    model.eval()

    l1_list = []
    mse_list = []
    psnr_list = []
    ssim_list = []
    dc_rmse_list = []
    region_lists = {}
    roi_cfg = region_loss_config(cfg)
    previews = []
    preview_dir = cfg.get("eval", {}).get("preview_dir")

    # SSIM params (optional overrides)
    ssim_cfg = cfg.get("ssim", {})
    ws = int(ssim_cfg.get("window_size", 11))
    sigma = float(ssim_cfg.get("sigma", 1.5))
    data_range = float(ssim_cfg.get("data_range", 1.0))
    source_sigma0 = source_sigma0_from_cfg(cfg)
    source_kwargs = source_sampler_kwargs_from_cfg(cfg)

    for i, (y, x) in enumerate(test_dl):
        if i >= max_batches:
            break

        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        latent_seed = _eval_seed_for_batch(cfg, i)

        def _sample():
            return sample_with_physics_guidance(
                model=model,
                y=y,
                H=Hop,
                steps=cfg["sample"]["steps"],
                dc_step=cfg["physics"]["dc_step_size"],
                dc_steps=cfg["physics"]["dc_steps"],
                init_noise_std=source_sigma0,
                denom_min=denom_min,
                clamp_x=False,
                disable_physics=bool(cfg.get("physics", {}).get("disable_in_eval", False)),
                pred_type=pred_type,
                solver=str(cfg.get("sample", {}).get("solver", "heun")),
                **source_kwargs,
            )

        x_hat = _call_with_preserved_rng(
            _sample,
            seed=latent_seed,
            device=device,
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

        if region_eval_enabled(cfg):
            mask = rbc_region_mask(x_c, **roi_cfg.get("mask", {}))
            region_values = region_image_metrics(
                x_hat_c, x_c, mask, window_size=ws, sigma=sigma, data_range=data_range,
            )
            for key, values in region_values.items():
                region_lists.setdefault(key, []).extend(values[torch.isfinite(values)].detach().cpu().tolist())
            if preview_dir and len(previews) < 8:
                previews.append(tuple(item[0].detach().cpu() for item in (y, x_c, x_hat_c, mask)))

        if Hop is not None:
            dc_err = (Hop.forward(x_hat.float()) - y.float()).pow(2).mean().sqrt().item()
            dc_rmse_list.append(dc_err)

    def avg(lst):
        return float(sum(lst) / max(1, len(lst)))

    result = {
        "eval/l1": avg(l1_list),
        "eval/mse": avg(mse_list),
        "eval/psnr": avg(psnr_list),
        "eval/ssim": avg(ssim_list),
        "eval/dc_rmse": avg(dc_rmse_list) if dc_rmse_list else float("nan"),
    }
    for key, values in region_lists.items():
        result[f"eval/{key}"] = avg(values) if values else float("nan")
        if key in {"rbc_psnr", "background_psnr"}:
            result[f"eval/{key}_valid_samples"] = len(values)
    if previews:
        save_region_preview(os.path.join(preview_dir, f"epoch_{epoch if epoch is not None else 'eval'}.png"), previews)
    return result


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
    use_time_conditioning = use_time_conditioning_from_cfg(cfg)
    if pred_type == "btb" and not use_time_conditioning:
        raise ValueError(
            "model.use_time_conditioning=false is only supported for train.mode='vanilla'. "
            "x-prediction / 'btb' depends on t."
        )
    flow_matcher_name = normalize_flow_matcher_name(cfg.get("cfm", {}).get("matcher", "rectified"))
    flow_matcher = build_flow_matcher(flow_matcher_name)
    source_mode = source_mode_from_cfg(cfg)
    source_sigma0 = source_sigma0_from_cfg(cfg)
    source_kwargs = source_sampler_kwargs_from_cfg(cfg)
    cfm_cfg = cfg["cfm"]
    roi_cfg = region_loss_config(cfg)
    roi_enabled = bool(roi_cfg.get("enabled", False))
    if (roi_enabled or region_eval_enabled(cfg)) and float(cfg["data"].get("downsample", 1)) != 1:
        raise ValueError("RBC pseudo-region defaults require native sampling (data.downsample=1).")
    if roi_enabled:
        print("RBC region loss: detached target pseudo-masks; "
              f"balance_mix={roi_cfg.get('balance_mix', 0.5)}, "
              f"foreground_weight={roi_cfg.get('foreground_weight', 0.75)}")
    t_distribution = str(cfm_cfg.get("t_distribution", "uniform"))
    t_alpha = float(cfm_cfg.get("t_alpha", 0.5))
    t_beta = float(cfm_cfg.get("t_beta", 0.5))
    time_tag = "tcond" if use_time_conditioning else "notime"
    print(f"Time conditioning: {use_time_conditioning}")
    print(f"CFM source: {source_mode} (sigma0={source_sigma0})")
    if t_distribution.strip().lower().replace("-", "_") == "beta":
        print(
            f"CFM t sampler: beta(alpha={t_alpha}, beta={t_beta}) "
            f"mapped to [{cfm_cfg['t_min']}, {cfm_cfg['t_max']}]"
        )
    else:
        print(f"CFM t sampler: {t_distribution} on [{cfm_cfg['t_min']}, {cfm_cfg['t_max']}]")

    # -------------------------
    # W&B init
    # -------------------------
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", True))
    if use_wandb:
        wb_tags = list(wb.get("tags", []))
        for tag in [pred_type, flow_matcher_name, time_tag, f"src_{source_mode}", "cfm", "lensless"]:
            if tag not in wb_tags:
                wb_tags.append(tag)
        wandb.init(
            project=wb.get("project", "lensless-flow"),
            entity=wb.get("entity", None),
            name=wb.get("name", None),
            tags=wb_tags,
            config=cfg,
        )
        # This summary reaches W&B before the potentially lengthy initial eval,
        # providing a sync check without consuming a training history step.
        wandb.run.summary["startup/initialized"] = True

    # -------------------------
    # Data
    # -------------------------
    train_ds, train_dl = make_dataloader(
        split=cfg["data"].get("split", "train"),
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        path=cfg["data"].get("path", None),
        **data_loader_kwargs(cfg),
    )
    if (roi_enabled or region_eval_enabled(cfg)) and not isinstance(train_ds, HumanRBCHologramDataset):
        raise ValueError("RBC region mode is only audited for HumanRBCHologramDataset phase labels.")

    eval_batches = int(wb.get("eval_batches", 0) or 0)
    log_images_every = int(wb.get("log_images_every", 0) or 0)
    test_ds, test_dl = None, None
    if eval_batches > 0 or log_images_every > 0:
        test_ds, test_dl = make_dataloader(
            split=cfg["data"].get("eval_split", "test"),
            downsample=cfg["data"]["downsample"],
            flip_ud=cfg["data"]["flip_ud"],
            batch_size=1,
            num_workers=0,
            path=cfg["data"].get("path", None),
            **data_loader_kwargs(cfg),
        )

    # -------------------------
    # Optional PSF + operator
    # -------------------------
    y0, x0 = train_ds[0]
    y0 = to_nchw(y0)
    x0 = to_nchw(x0)
    if y0.shape[1:] != x0.shape[1:]:
        raise ValueError(f"Expected paired y/x tensors with matching CHW shape, got {tuple(y0.shape)} and {tuple(x0.shape)}")
    C = y0.shape[1]
    H_img, W_img = y0.shape[-2], y0.shape[-1]
    if (roi_enabled or region_eval_enabled(cfg)) and (C != 1 or (H_img, W_img) != (256, 256)):
        raise ValueError("RBC region mode expects the native 256x256 grayscale phase dataset.")
    Hop = build_forward_operator_from_dataset(train_ds, y0.to(device), device=device)
    if Hop is None:
        print("Forward operator: none (pure conditional flow)")
    else:
        print("Forward operator: PSF FFT convolution")

    if float(cfg.get("cfm", {}).get("loss", {}).get("physics_weight", 0.0)) > 0 and Hop is None:
        raise ValueError("cfm.loss.physics_weight > 0 requires a dataset with a known forward operator.")

    # -------------------------
    # Model
    # -------------------------
    model_name = resolve_model_name(cfg)
    init_checkpoint = cfg.get("train", {}).get("init_checkpoint")
    init_state = None
    if init_checkpoint:
        init_state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        saved_mode = init_state.get("mode", init_state.get("cfg", {}).get("train", {}).get("mode"))
        if saved_mode is not None and saved_mode != pred_type:
            raise ValueError("Fine-tuning checkpoint prediction mode must match train.mode.")
        if resolve_model_name(cfg, init_state) != model_name:
            raise ValueError("Fine-tuning checkpoint architecture must match model.name.")
    model = build_flow_model(
        cfg=cfg,
        img_channels=C,
        im_hw=(H_img, W_img),
        device=device,
        checkpoint_state=None,
    )
    if init_state is not None:
        saved_time = init_state.get("use_time_conditioning", init_state.get("cfg", {}).get("model", {}).get("use_time_conditioning"))
        if saved_time is not None and bool(saved_time) != use_time_conditioning:
            raise ValueError("Fine-tuning checkpoint time conditioning must match the model config.")
        load_checkpoint_state_dict(model, init_state)
        del init_state
        print("Initialized model weights from", init_checkpoint, "(fresh optimizer and epoch count)")
    else:
        print("Initialized model weights randomly (no checkpoint; fresh optimizer and epoch count)")
    print("Model:", model_name)
    print("Model params:", sum(p.numel() for p in model.parameters()))
    if cfg.get("compile", {}).get("enabled", True) and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    # -------------------------
    # ReduceLROnPlateau on the configured validation metric
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

    checkpoint_dir = str(cfg.get("train", {}).get("checkpoint_dir", "checkpoints"))
    ensure_dir(checkpoint_dir)
    best_metric = cfg.get("train", {}).get("save_best_metric")
    best_mode = str(cfg.get("train", {}).get("save_best_mode", "max"))
    best_score = None
    if best_metric:
        if best_mode not in {"min", "max"}:
            raise ValueError("train.save_best_mode must be 'min' or 'max'.")
        if eval_batches <= 0 or test_dl is None:
            raise ValueError("train.save_best_metric requires validation each epoch.")
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    log_every = int(wb.get("log_every", 50))
    global_step = 0
    save_since_epoch = float(cfg["train"].get("save_since_epoch", 0.0)) # if <1, interpreted as fraction of total epochs; if >=1, interpreted as absolute epoch number
    if save_since_epoch < 1.0:
        save_since_epoch = int(cfg["train"]["epochs"] * save_since_epoch)
    eval_metrics = {}

    if bool(cfg.get("train", {}).get("eval_at_start", False)):
        if eval_batches <= 0 or test_dl is None:
            raise ValueError("train.eval_at_start requires wandb.eval_batches > 0.")
        print("Evaluating initial weights on the fixed validation subset...")
        eval_metrics = quick_eval(
            model, Hop, test_dl, cfg, device, max_batches=eval_batches,
            denom_min=denom_min, pred_type=pred_type,
            epoch=0,
        )
        print("Initial evaluation:", eval_metrics)
        save_eval_record(cfg, eval_metrics, epoch=0, step=global_step)
        if use_wandb:
            wandb.log({"epoch": 0, **eval_metrics}, step=global_step)
        # DataLoader iteration can consume a Torch RNG seed even without shuffle.
        # Reset before training so enabling baseline evaluation cannot change
        # the training data order or source/time samples in paired experiments.
        set_seed(cfg["seed"])
        global_step += 1

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch} ({pred_type}, {flow_matcher_name}, {time_tag})")

        sum_loss = 0.0
        sum_loss_v = 0.0
        sum_loss_phys = 0.0
        n_batches = 0

        for y, x in pbar:
            y = to_nchw(y).to(device, non_blocking=True)
            x = to_nchw(x).to(device, non_blocking=True)

            b = x.shape[0]
            t = sample_t(
                b,
                cfm_cfg["t_min"],
                cfm_cfg["t_max"],
                device,
                distribution=t_distribution,
                alpha=t_alpha,
                beta=t_beta,
            )

            fm_batch = sample_flow_matching_training_batch(
                x_target=x,
                y_cond=y,
                t=t,
                flow_matcher=flow_matcher,
                noise_std=source_sigma0,
                H=Hop,
                **source_kwargs,
            )
            t = fm_batch.t
            x_t = fm_batch.x_t
            v_star = fm_batch.v_target
            y_cond = fm_batch.y_cond
            roi_mask = None
            if roi_enabled:
                # This identity follows the sampled coupling, including OT
                # reordering. Original x may be in a different batch order.
                matched_target = x0_from_xt_v(x_t, v_star, t).detach()
                roi_mask = rbc_region_mask(matched_target, **roi_cfg.get("mask", {}))

            den = (1.0 - t).clamp_min(denom_min).view(b, 1, 1, 1)

            with autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                out = model(x_t, y_cond, t if use_time_conditioning else None)

                if pred_type == "vanilla":
                    v_pred = out
                    x_pred = None
                else:
                    x_pred = out
                    v_pred = (x_pred - x_t) / den

                region_stats = {}
                if roi_mask is not None:
                    velocity_loss, region_stats = region_balanced_cfm_loss(
                        v_pred, v_star, roi_mask,
                        foreground_weight=float(roi_cfg.get("foreground_weight", 0.75)),
                        balance_mix=float(roi_cfg.get("balance_mix", 0.5)),
                    )
                else:
                    velocity_loss = cfm_loss(v_pred, v_star)
                loss_v = velocity_loss * cfg["cfm"]["loss"]["v_weight"]

                loss_phys = torch.tensor(0.0, device=device)
                if cfg["cfm"]["loss"]["physics_weight"] > 0:
                    loss_phys = physics_loss_from_v(x_t, v_pred, t, y_cond, Hop) * cfg["cfm"]["loss"]["physics_weight"]

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
                    "train/use_time_conditioning": float(use_time_conditioning),
                    "train/t_mean": float(t.mean().item()),
                    "train/t_std": float(t.std(unbiased=False).item()),
                    "diag/x_t_rms": rms(x_t),
                    "diag/y_cond_rms": rms(y_cond),
                    "diag/x_source_rms": rms(fm_batch.x_source) if fm_batch.x_source is not None else 0.0,
                    "diag/v_pred_rms": rms(v_pred),
                    "diag/nan_or_inf": float(nan_or_inf),
                }
                for key, value in region_stats.items():
                    if torch.isfinite(value):
                        log_dict[f"train/region_{key}"] = float(value)
                if fm_batch.x_init is not None:
                    log_dict["diag/x_init_rms"] = rms(fm_batch.x_init)
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
        # Validation + LR scheduler step
        # -------------------------
        if eval_batches > 0 and test_dl is not None:
            eval_metrics = quick_eval(
                model, Hop, test_dl, cfg, device,
                max_batches=eval_batches,
                denom_min=denom_min,
                pred_type=pred_type,
                epoch=epoch,
            )
            save_eval_record(cfg, eval_metrics, epoch=epoch, step=global_step)
            if use_wandb:
                wandb.log(eval_metrics, step=global_step)
            else:
                print(eval_metrics)

            if scheduler is not None:
                if sched_metric not in eval_metrics:
                    raise KeyError(
                        f"sched.metric='{sched_metric}' not found in eval_metrics keys: {list(eval_metrics.keys())}"
                    )
                # An absent pseudo-region has no score; it should not trigger
                # an LR reduction as if it were a failed reconstruction.
                if torch.isfinite(torch.tensor(eval_metrics[sched_metric])):
                    scheduler.step(eval_metrics[sched_metric])

                if use_wandb:
                    wandb.log({"sched/lr_after": float(opt.param_groups[0]["lr"])}, step=global_step)

        # optional image logging
        if use_wandb and log_images_every > 0 and test_ds is not None and (epoch % log_images_every == 0):
            y_ex, x_ex = test_ds[0]
            y_ex = to_nchw(y_ex).to(device)
            x_ex = to_nchw(x_ex).to(device)

            def _sample_viz():
                return sample_with_physics_guidance(
                    model=model,
                    y=y_ex,
                    H=Hop,
                    steps=cfg["sample"]["steps"],
                    dc_step=cfg["physics"]["dc_step_size"],
                    dc_steps=cfg["physics"]["dc_steps"],
                    init_noise_std=source_sigma0,
                    denom_min=denom_min,
                    clamp_x=False,
                    disable_physics=bool(cfg.get("physics", {}).get("disable_in_eval", False)),
                    pred_type=pred_type,
                    solver=str(cfg.get("sample", {}).get("solver", "heun")),
                    **source_kwargs,
                )

            x_hat_ex = _call_with_preserved_rng(
                _sample_viz,
                seed=_eval_seed_for_batch(cfg, 0, offset=9_000_000),
                device=device,
            )

            wandb.log(
                {
                    "viz/lensless_y": chw_to_wandb_image(y_ex),
                    "viz/gt_x": chw_to_wandb_image(x_ex),
                    "viz/recon_xhat": chw_to_wandb_image(x_hat_ex),
                },
                step=global_step,
            )

        # Preserve periodic/final weights and optionally the best validation model.
        save_periodic = (
            (epoch % cfg["train"]["save_every"] == 0 and epoch >= save_since_epoch)
            or epoch == 1 or epoch == cfg["train"]["epochs"]
        )
        save_best = False
        if best_metric:
            if best_metric not in eval_metrics:
                raise KeyError(f"train.save_best_metric='{best_metric}' not found in eval_metrics.")
            score = float(eval_metrics[best_metric])
            if torch.isfinite(torch.tensor(score)):
                save_best = best_score is None or (score > best_score if best_mode == "max" else score < best_score)
                if save_best:
                    best_score = score
        if save_periodic or save_best:
            checkpoint_state = {
                "model": model.state_dict(),
                "cfg": cfg,
                "mode": pred_type,
                "matcher": flow_matcher_name,
                "source_mode": source_mode,
                "source_sigma0": source_sigma0,
                "model_name": model_name,
                "use_time_conditioning": use_time_conditioning,
                "epoch": epoch,
                "global_step": global_step,
                "eval_metrics": eval_metrics,
            }
        if save_best:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint_atomic(checkpoint_state, best_path)
            print(f"Saved best: {best_path} (epoch {epoch}, {best_metric}={best_score:.6f})")
            if use_wandb:
                wandb.run.summary["best/epoch"] = epoch
                wandb.run.summary["best/metric"] = best_metric
                wandb.run.summary["best/score"] = best_score
        if save_periodic:
            ssim = eval_metrics.get("eval/ssim", 0.0)
            dataset_tag = str(cfg.get("data", {}).get("dataset", "lensless")).lower().replace("-", "_")
            ckpt_path = os.path.join(checkpoint_dir, f"cfm_{dataset_tag}_{pred_type}_{flow_matcher_name}_{time_tag}_epoch{epoch}_ssim{ssim:.4f}.pt")
            save_checkpoint_atomic(checkpoint_state, ckpt_path)
            print("Saved:", ckpt_path)

            if use_wandb and bool(wb.get("log_artifacts", True)):
                artifact = wandb.Artifact(
                    name=f"cfm_lensless_{pred_type}_{flow_matcher_name}_{time_tag}",
                    type="model",
                    metadata={
                        "epoch": epoch, 
                        "C": C, "H": H_img, "W": W_img, 
                        "mode": pred_type,
                        "matcher": flow_matcher_name,
                        "source_mode": source_mode,
                        "source_sigma0": source_sigma0,
                        "model_name": model_name,
                        "use_time_conditioning": use_time_conditioning,
                        "denom_min": denom_min,
                    },
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}", "latest"])

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    main(cfg)
