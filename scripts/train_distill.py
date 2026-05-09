import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.distill_cache import TeacherDistillCacheDataset, load_cache_metadata_pt
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict, resolve_model_name
from lensless_flow.model_unet import use_time_conditioning_from_cfg
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.utils import ensure_dir, set_seed


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _model_for_state_dict(model):
    return getattr(model, "_orig_mod", model)


def _model_uses_time_conditioning(model, cfg: dict) -> bool:
    wrapped = _model_for_state_dict(model)
    return bool(getattr(wrapped, "use_time_conditioning", use_time_conditioning_from_cfg(cfg)))


def _get_nested(cfg: dict, path: str, default):
    current = cfg
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _arg_or_cfg(value, cfg: dict, path: str, default):
    return value if value is not None else _get_nested(cfg, path, default)


def _load_init_state(path: str | None, device: torch.device):
    if not path:
        return None
    return torch.load(path, map_location=device)


def _infer_mode(state, cfg: dict) -> str:
    mode = str(
        state.get("mode", cfg.get("train", {}).get("mode", "vanilla"))
        if isinstance(state, dict)
        else cfg.get("train", {}).get("mode", "vanilla")
    ).lower()
    return mode


def _build_student(
    cfg: dict,
    init_ckpt: str | None,
    img_channels: int,
    im_hw: tuple[int, int],
    device: torch.device,
):
    init_state = _load_init_state(init_ckpt, device)
    if init_state is not None:
        init_mode = _infer_mode(init_state, cfg)
        if init_mode != "vanilla":
            raise ValueError(f"Distillation expects a vanilla v-prediction init checkpoint, got mode={init_mode}")

    model = build_flow_model(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=init_state if isinstance(init_state, dict) else None,
    )
    if init_state is not None:
        load_checkpoint_state_dict(model, init_state)
    return model, init_state


def _make_hop_if_needed(cache_dir: str, device: torch.device, im_hw: tuple[int, int], physics_weight: float):
    if physics_weight <= 0:
        return None
    metadata_pt = load_cache_metadata_pt(cache_dir)
    psf = metadata_pt.get("psf")
    if psf is None:
        raise FileNotFoundError(
            "distill.physics_weight > 0 requires metadata.pt with a cached PSF. "
            "Regenerate the teacher cache with scripts.cache_teacher_distill."
        )
    return FFTLinearConvOperator(psf=psf.float().to(device), im_hw=im_hw).to(device)


def _sample_distill_t(batch_size: int, cfg: dict, phase: str, one_step_prob: float, device: torch.device) -> torch.Tensor:
    phase = str(phase).lower()
    t_min = float(_get_nested(cfg, "cfm.t_min", 0.001))
    t_max = float(_get_nested(cfg, "cfm.t_max", 0.999))

    if phase == "one_step":
        return torch.zeros(batch_size, device=device)
    if phase == "reflow":
        return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    if phase == "mixed":
        t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
        mask = torch.rand(batch_size, device=device) < float(one_step_prob)
        t[mask] = 0.0
        return t
    raise ValueError(f"Unsupported distill phase '{phase}'. Use reflow, one_step, or mixed.")


def _prepare_save_cfg(cfg: dict, cache_metadata: dict, phase: str) -> dict:
    save_cfg = copy.deepcopy(cfg)
    save_cfg.setdefault("train", {})["mode"] = "vanilla"
    save_cfg.setdefault("sample", {})["steps"] = 1
    save_cfg.setdefault("sample", {})["solver"] = "euler"
    save_cfg.setdefault("cfm", {})["matcher"] = cache_metadata.get("teacher_matcher", "rectified")
    save_cfg.setdefault("cfm", {})["source"] = {
        "mode": "gaussian",
        "sigma0": float(cache_metadata.get("init_noise_std", 1.0)),
    }
    save_cfg.setdefault("physics", {})["disable_in_eval"] = bool(cache_metadata.get("disable_physics", True))
    save_cfg.setdefault("distill", {})["phase"] = phase
    return save_cfg


def _loss_weights(cfg: dict, args) -> dict[str, float]:
    return {
        "v": float(_arg_or_cfg(args.v_weight, cfg, "distill.v_weight", 1.0)),
        "teacher_l1": float(_arg_or_cfg(args.teacher_l1_weight, cfg, "distill.teacher_l1_weight", 0.25)),
        "gt_l1": float(_arg_or_cfg(args.gt_l1_weight, cfg, "distill.gt_l1_weight", 0.05)),
        "physics": float(_arg_or_cfg(args.physics_weight, cfg, "distill.physics_weight", 0.0)),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward_student(
    model,
    batch: dict,
    t: torch.Tensor,
    use_time_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = batch["y"]
    z0 = batch["z0"]
    x_teacher = batch["x_teacher"]
    t_img = t[:, None, None, None].to(dtype=z0.dtype)
    x_t = (1.0 - t_img) * z0 + t_img * x_teacher
    v_target = x_teacher - z0
    v_pred = model(x_t, y, t if use_time_conditioning else None)
    x_endpoint = x_t + (1.0 - t_img) * v_pred
    return v_pred, v_target, x_endpoint


def _compute_loss(
    *,
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    x_endpoint: torch.Tensor,
    batch: dict,
    Hop,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_v = F.mse_loss(v_pred.float(), v_target.float())
    loss_teacher_l1 = F.l1_loss(x_endpoint.float(), batch["x_teacher"].float())
    loss_gt_l1 = F.l1_loss(x_endpoint.float(), batch["x_gt"].float())

    loss_physics = torch.tensor(0.0, device=x_endpoint.device)
    if Hop is not None and weights["physics"] > 0:
        residual = Hop.forward(x_endpoint.float()) - batch["y"].float()
        loss_physics = residual.pow(2).mean()

    loss = (
        weights["v"] * loss_v
        + weights["teacher_l1"] * loss_teacher_l1
        + weights["gt_l1"] * loss_gt_l1
        + weights["physics"] * loss_physics
    )
    stats = {
        "loss": float(loss.detach().item()),
        "loss_v": float(loss_v.detach().item()),
        "loss_teacher_l1": float(loss_teacher_l1.detach().item()),
        "loss_gt_l1": float(loss_gt_l1.detach().item()),
        "loss_physics": float(loss_physics.detach().item()),
    }
    return loss, stats


@torch.no_grad()
def _quick_eval(model, eval_dl, cfg: dict, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    use_time_conditioning = _model_uses_time_conditioning(model, cfg)
    values = {
        "student_gt_psnr": [],
        "student_gt_ssim": [],
        "student_teacher_psnr": [],
        "student_teacher_mse": [],
    }
    for batch_idx, batch in enumerate(eval_dl):
        if batch_idx >= max_batches:
            break
        batch = _move_batch(batch, device)
        b = int(batch["y"].shape[0])
        t = torch.zeros(b, device=device)
        _, _, x_endpoint = _forward_student(model, batch, t, use_time_conditioning)
        x_student = x_endpoint.clamp(0, 1).float()
        x_gt = batch["x_gt"].clamp(0, 1).float()
        x_teacher = batch["x_teacher"].clamp(0, 1).float()
        values["student_gt_psnr"].append(float(psnr(x_student, x_gt)))
        values["student_gt_ssim"].append(float(ssim_torch(x_student, x_gt).item()))
        values["student_teacher_psnr"].append(float(psnr(x_student, x_teacher)))
        values["student_teacher_mse"].append(float(F.mse_loss(x_student, x_teacher).item()))

    model.train()
    return {key: _avg(item) for key, item in values.items()}


def _save_checkpoint(
    *,
    path: str,
    model,
    cfg: dict,
    cache_metadata: dict,
    init_ckpt: str | None,
    epoch: int,
    global_step: int,
    phase: str,
    metrics: dict[str, float],
    model_name: str,
    use_time_conditioning: bool,
) -> None:
    save_cfg = _prepare_save_cfg(cfg, cache_metadata, phase=phase)
    torch.save(
        {
            "model": _model_for_state_dict(model).state_dict(),
            "cfg": save_cfg,
            "mode": "vanilla",
            "matcher": cache_metadata.get("teacher_matcher", save_cfg.get("cfm", {}).get("matcher", "rectified")),
            "source_mode": "gaussian",
            "source_sigma0": float(cache_metadata.get("init_noise_std", 1.0)),
            "model_name": model_name,
            "use_time_conditioning": bool(use_time_conditioning),
            "distilled": True,
            "distill": {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "phase": phase,
                "teacher_ckpt": cache_metadata.get("teacher_ckpt"),
                "teacher_steps": cache_metadata.get("teacher_steps"),
                "teacher_solver": cache_metadata.get("teacher_solver"),
                "init_ckpt": os.path.abspath(init_ckpt) if init_ckpt else None,
                "metrics": metrics,
            },
        },
        path,
    )


def main(args, cfg: dict) -> None:
    set_seed(int(args.seed))
    device = torch.device(args.device if args.device else (cfg["device"] if torch.cuda.is_available() else "cpu"))
    if cfg.get("is_a100", False) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_mode = str(cfg.get("train", {}).get("mode", "vanilla")).lower()
    if train_mode != "vanilla":
        raise ValueError("One-step distillation expects cfg.train.mode='vanilla'.")

    dataset = TeacherDistillCacheDataset(args.cache_dir)
    if len(dataset) <= 0:
        raise RuntimeError(f"Distillation cache is empty: {args.cache_dir}")
    cache_metadata = dataset.metadata
    first = dataset[0]
    img_channels = int(first["y"].shape[0])
    im_hw = (int(first["y"].shape[-2]), int(first["y"].shape[-1]))

    batch_size = int(_arg_or_cfg(args.batch_size, cfg, "distill.batch_size", cfg.get("train", {}).get("batch_size", 1)))
    num_workers = int(_arg_or_cfg(args.num_workers, cfg, "distill.num_workers", cfg.get("data", {}).get("num_workers", 0)))
    train_dl = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_dl = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model, init_state = _build_student(
        cfg=cfg,
        init_ckpt=args.init_ckpt,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
    )
    use_time_conditioning = _model_uses_time_conditioning(model, cfg)
    model_name = resolve_model_name(cfg, checkpoint_state=init_state if isinstance(init_state, dict) else None)
    if bool(_get_nested(cfg, "compile.enabled", False)) and device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    lr = float(_arg_or_cfg(args.lr, cfg, "distill.lr", cfg.get("train", {}).get("lr", 2e-4)))
    weight_decay = float(_arg_or_cfg(args.weight_decay, cfg, "distill.weight_decay", 1e-4))
    epochs = int(_arg_or_cfg(args.epochs, cfg, "distill.epochs", cfg.get("train", {}).get("epochs", 20)))
    phase = str(_arg_or_cfg(args.phase, cfg, "distill.phase", "reflow")).lower()
    one_step_prob = float(_arg_or_cfg(args.one_step_prob, cfg, "distill.one_step_prob", 0.5))
    save_every = int(_arg_or_cfg(args.save_every, cfg, "distill.save_every", 1))
    eval_batches = int(_arg_or_cfg(args.eval_batches, cfg, "distill.eval_batches", 8))
    grad_clip = _arg_or_cfg(args.grad_clip, cfg, "distill.grad_clip", cfg.get("train", {}).get("grad_clip", 1.0))
    grad_clip = None if grad_clip is None else float(grad_clip)
    amp_enabled = bool(_arg_or_cfg(args.amp, cfg, "distill.amp", cfg.get("train", {}).get("amp", True))) and device.type == "cuda"
    weights = _loss_weights(cfg, args)
    Hop = _make_hop_if_needed(args.cache_dir, device=device, im_hw=im_hw, physics_weight=weights["physics"])

    ensure_dir(args.out_dir)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", False)) and not args.no_wandb
    if use_wandb:
        import wandb

        tags = list(wb.get("tags", []))
        for tag in ["distill", phase, "one_step", "vanilla"]:
            if tag not in tags:
                tags.append(tag)
        wandb.init(
            project=wb.get("project", "lensless-flow"),
            entity=wb.get("entity", None),
            name=wb.get("name", None),
            tags=tags,
            config=cfg,
        )
    else:
        wandb = None

    print("Distill dataset entries:", len(dataset))
    print("Model:", model_name)
    print("Init checkpoint:", args.init_ckpt)
    print("Phase:", phase)
    print("Loss weights:", weights)

    global_step = 0
    best_psnr = float("-inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, list[float]] = {
            "loss": [],
            "loss_v": [],
            "loss_teacher_l1": [],
            "loss_gt_l1": [],
            "loss_physics": [],
        }
        pbar = tqdm(train_dl, desc=f"distill epoch {epoch}/{epochs} [{phase}]")
        for batch in pbar:
            batch = _move_batch(batch, device)
            b = int(batch["y"].shape[0])
            t = _sample_distill_t(b, cfg, phase=phase, one_step_prob=one_step_prob, device=device)

            with autocast("cuda", enabled=amp_enabled):
                v_pred, v_target, x_endpoint = _forward_student(
                    model=model,
                    batch=batch,
                    t=t,
                    use_time_conditioning=use_time_conditioning,
                )
                loss, loss_stats = _compute_loss(
                    v_pred=v_pred,
                    v_target=v_target,
                    x_endpoint=x_endpoint,
                    batch=batch,
                    Hop=Hop,
                    weights=weights,
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            for key, value in loss_stats.items():
                running[key].append(value)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_stats['loss']:.4f}", v=f"{loss_stats['loss_v']:.4f}")
            if use_wandb and global_step % int(wb.get("log_every", 50)) == 0:
                wandb.log({f"train/{k}": _avg(v) for k, v in running.items()}, step=global_step)

        eval_metrics = _quick_eval(model, eval_dl, cfg, device, max_batches=max(1, eval_batches))
        epoch_stats = {f"epoch/{k}": _avg(v) for k, v in running.items()}
        print(
            f"epoch {epoch}: loss={epoch_stats['epoch/loss']:.5f}, "
            f"gt_psnr={eval_metrics['student_gt_psnr']:.3f}, "
            f"gt_ssim={eval_metrics['student_gt_ssim']:.4f}, "
            f"teacher_mse={eval_metrics['student_teacher_mse']:.6f}"
        )
        if use_wandb:
            wandb.log({**epoch_stats, **{f"eval/{k}": v for k, v in eval_metrics.items()}}, step=global_step)

        should_save = (save_every > 0 and epoch % save_every == 0) or epoch == 1 or epoch == epochs
        if should_save:
            ckpt_path = os.path.join(args.out_dir, f"distill_1step_epoch{epoch}_psnr{eval_metrics['student_gt_psnr']:.3f}.pt")
            _save_checkpoint(
                path=ckpt_path,
                model=model,
                cfg=cfg,
                cache_metadata=cache_metadata,
                init_ckpt=args.init_ckpt,
                epoch=epoch,
                global_step=global_step,
                phase=phase,
                metrics=eval_metrics,
                model_name=model_name,
                use_time_conditioning=use_time_conditioning,
            )
            latest_path = os.path.join(args.out_dir, "distill_1step_latest.pt")
            _save_checkpoint(
                path=latest_path,
                model=model,
                cfg=cfg,
                cache_metadata=cache_metadata,
                init_ckpt=args.init_ckpt,
                epoch=epoch,
                global_step=global_step,
                phase=phase,
                metrics=eval_metrics,
                model_name=model_name,
                use_time_conditioning=use_time_conditioning,
            )
            print("Saved:", ckpt_path)

        if eval_metrics["student_gt_psnr"] > best_psnr:
            best_psnr = float(eval_metrics["student_gt_psnr"])
            best_path = os.path.join(args.out_dir, "distill_1step_best.pt")
            _save_checkpoint(
                path=best_path,
                model=model,
                cfg=cfg,
                cache_metadata=cache_metadata,
                init_ckpt=args.init_ckpt,
                epoch=epoch,
                global_step=global_step,
                phase=phase,
                metrics=eval_metrics,
                model_name=model_name,
                use_time_conditioning=use_time_conditioning,
            )
            print("Saved best:", best_path)

    elapsed = time.perf_counter() - started
    if use_wandb:
        wandb.finish()
    print(f"Done. Best eval PSNR={best_psnr:.3f}. seconds={elapsed:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "distill_1step"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--phase", type=str, default=None, choices=["reflow", "one_step", "mixed"])
    ap.add_argument("--one_step_prob", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--eval_batches", type=int, default=None)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--v_weight", type=float, default=None)
    ap.add_argument("--teacher_l1_weight", type=float, default=None)
    ap.add_argument("--gt_l1_weight", type=float, default=None)
    ap.add_argument("--physics_weight", type=float, default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    main(args, cfg)
