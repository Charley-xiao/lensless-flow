import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.measurement_source import source_sampler_kwargs_from_cfg, source_sigma0_from_cfg
from lensless_flow.model_factory import build_flow_model, resolve_model_name
from lensless_flow.model_unet import resolve_use_time_conditioning
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir


def data_loader_kwargs(cfg: dict) -> dict:
    data_cfg = dict(cfg.get("data", {}) or {})
    excluded = {"path", "split", "eval_split", "downsample", "flip_ud", "num_workers"}
    return {k: v for k, v in data_cfg.items() if k not in excluded}


def save_gray_png(path: str | os.PathLike, x: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr_u16 = (arr * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(arr_u16, mode="I;16").save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Infer RBC phase maps from holograms with a trained pure-flow checkpoint.")
    ap.add_argument("--config", type=str, default="configs/rbc_hologram.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default=None, help="Defaults to data.eval_split from the config.")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--solver", choices=["heun", "euler"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_inputs", action="store_true")
    ap.add_argument("--save_targets", action="store_true")
    args, overrides = ap.parse_known_args()

    cfg = load_config(args.config, overrides)
    split = args.split if args.split is not None else cfg["data"].get("eval_split", "validation")
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(cfg["sample"]["save_dir"], "infer")
    ensure_dir(out_dir)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device)

    ds, dl = make_dataloader(
        split=split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
        **data_loader_kwargs(cfg),
    )

    y0, _ = ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))

    use_time_conditioning = resolve_use_time_conditioning(cfg, state)
    model_name = resolve_model_name(cfg, checkpoint_state=state)
    model = build_flow_model(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=state,
    )
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    pred_type = str(state.get("mode", cfg.get("train", {}).get("mode", "vanilla"))).lower()
    matcher = normalize_flow_matcher_name(state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified")))
    steps = int(args.steps if args.steps is not None else cfg["sample"].get("steps", 40))
    solver = str(args.solver if args.solver is not None else cfg["sample"].get("solver", "heun")).lower()
    init_noise_std = source_sigma0_from_cfg(cfg)
    source_kwargs = source_sampler_kwargs_from_cfg(cfg)
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(
        f"infer_rbc: samples={len(ds)}, model={model_name}, mode={pred_type}, "
        f"matcher={matcher}, time_conditioning={use_time_conditioning}, steps={steps}, solver={solver}"
    )

    saved = 0
    for batch_idx, (y, x) in enumerate(tqdm(dl, desc=f"infer {split}")):
        if args.max_samples >= 0 and saved >= args.max_samples:
            break
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        x_hat = sample_with_physics_guidance(
            model=model,
            y=y,
            H=None,
            steps=steps,
            dc_step=0.0,
            dc_steps=0,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            clamp_x=False,
            disable_physics=True,
            pred_type=pred_type,
            solver=solver,
            **source_kwargs,
        ).clamp(0, 1)

        for j in range(x_hat.shape[0]):
            if args.max_samples >= 0 and saved >= args.max_samples:
                break
            sample_idx = batch_idx * max(1, int(args.batch_size)) + j
            if hasattr(ds, "pairs"):
                name = Path(ds.pairs[sample_idx][0]).stem
            else:
                name = f"{sample_idx:06d}"

            save_gray_png(Path(out_dir) / "recon" / f"{name}.png", x_hat[j])
            if args.save_inputs:
                save_gray_png(Path(out_dir) / "hologram" / f"{name}.png", y[j])
            if args.save_targets:
                save_gray_png(Path(out_dir) / "phase_gt" / f"{name}.png", x[j])
            saved += 1

    print(f"Saved {saved} reconstruction(s) to {out_dir}")


if __name__ == "__main__":
    main()
