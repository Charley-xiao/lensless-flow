import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.distill_cache import CACHE_VERSION, METADATA_JSON, METADATA_PT, SHARD_PREFIX, SHARD_SUFFIX
from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.measurement_source import source_sampler_kwargs_from_cfg, source_sigma0_from_cfg
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict, resolve_model_name
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir, set_seed


def _capture_rng_state():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def _restore_rng_state(cpu_state, cuda_states) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _run_with_seed(fn, seed: int):
    cpu_state, cuda_states = _capture_rng_state()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        return fn()
    finally:
        _restore_rng_state(cpu_state, cuda_states)


def _save_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --save_dtype '{name}'")


def _prepare_output_dir(out_dir: str, overwrite: bool) -> None:
    ensure_dir(out_dir)
    existing = list(Path(out_dir).glob(f"{SHARD_PREFIX}*{SHARD_SUFFIX}"))
    existing += [Path(out_dir) / METADATA_JSON, Path(out_dir) / METADATA_PT]
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} already contains distillation cache files. "
            "Pass --overwrite to replace them."
        )
    for path in existing:
        path.unlink()


def _source_kwargs_from_args(cfg: dict, source_mode: str) -> dict:
    source_mode = str(source_mode).lower()
    if source_mode == "cfg":
        return source_sampler_kwargs_from_cfg(cfg)
    if source_mode == "gaussian":
        return {"source_mode": "gaussian"}
    if source_mode == "measurement_initialized":
        kwargs = source_sampler_kwargs_from_cfg(cfg)
        kwargs["source_mode"] = "measurement_initialized"
        return kwargs
    raise ValueError(f"Unsupported source mode: {source_mode}")


def _load_teacher(
    cfg: dict,
    ckpt: str,
    img_channels: int,
    im_hw: tuple[int, int],
    device: torch.device,
):
    state = torch.load(ckpt, map_location=device)
    model = build_flow_model(
        cfg=cfg,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
        checkpoint_state=state if isinstance(state, dict) else None,
    )
    load_checkpoint_state_dict(model, state)
    model.eval()

    pred_type = str(
        state.get("mode", cfg.get("train", {}).get("mode", "vanilla"))
        if isinstance(state, dict)
        else cfg.get("train", {}).get("mode", "vanilla")
    ).lower()
    if pred_type not in {"vanilla", "btb"}:
        raise ValueError(f"Unknown teacher pred_type={pred_type}")

    matcher = normalize_flow_matcher_name(
        state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified"))
        if isinstance(state, dict)
        else cfg.get("cfm", {}).get("matcher", "rectified")
    )
    model_name = resolve_model_name(cfg, checkpoint_state=state if isinstance(state, dict) else None)
    return model, pred_type, matcher, model_name


def _new_accumulator() -> dict:
    return {
        "sample_id": [],
        "repeat": [],
        "latent_seed": [],
        "y": [],
        "x_gt": [],
        "z0": [],
        "x_teacher": [],
    }


def _accumulator_len(acc: dict) -> int:
    return len(acc["sample_id"])


def _append_entry(
    acc: dict,
    *,
    sample_id: int,
    repeat: int,
    latent_seed: int,
    y: torch.Tensor,
    x_gt: torch.Tensor,
    z0: torch.Tensor,
    x_teacher: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    acc["sample_id"].append(int(sample_id))
    acc["repeat"].append(int(repeat))
    acc["latent_seed"].append(int(latent_seed))
    acc["y"].append(y.detach().cpu().to(dtype=dtype))
    acc["x_gt"].append(x_gt.detach().cpu().to(dtype=dtype))
    acc["z0"].append(z0.detach().cpu().to(dtype=dtype))
    acc["x_teacher"].append(x_teacher.detach().cpu().to(dtype=dtype))


def _flush_shard(acc: dict, out_dir: str, shard_idx: int) -> int:
    if _accumulator_len(acc) <= 0:
        return shard_idx

    shard = {
        "sample_id": torch.tensor(acc["sample_id"], dtype=torch.long),
        "repeat": torch.tensor(acc["repeat"], dtype=torch.long),
        "latent_seed": torch.tensor(acc["latent_seed"], dtype=torch.long),
        "y": torch.cat(acc["y"], dim=0),
        "x_gt": torch.cat(acc["x_gt"], dim=0),
        "z0": torch.cat(acc["z0"], dim=0),
        "x_teacher": torch.cat(acc["x_teacher"], dim=0),
        "cache_version": CACHE_VERSION,
    }
    path = os.path.join(out_dir, f"{SHARD_PREFIX}{shard_idx:05d}{SHARD_SUFFIX}")
    torch.save(shard, path)
    return shard_idx + 1


@torch.no_grad()
def main(args, cfg: dict) -> None:
    set_seed(int(args.seed))
    device = torch.device(args.device if args.device else (cfg["device"] if torch.cuda.is_available() else "cpu"))
    save_dtype = _save_dtype(args.save_dtype)
    _prepare_output_dir(args.out_dir, overwrite=bool(args.overwrite))

    ds, dl = make_dataloader(
        split=args.split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        path=cfg["data"].get("path", None),
    )

    y0, _ = ds[0]
    y0 = to_nchw(y0)
    img_channels = int(y0.shape[1])
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))
    psf = to_nchw(ds.psf).to(device)
    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    teacher, pred_type, matcher, model_name = _load_teacher(
        cfg=cfg,
        ckpt=args.teacher_ckpt,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
    )

    teacher_steps = int(args.teacher_steps if args.teacher_steps is not None else cfg["sample"]["steps"])
    teacher_solver = str(args.teacher_solver).lower()
    init_noise_std = float(args.init_noise_std if args.init_noise_std is not None else source_sigma0_from_cfg(cfg))
    source_kwargs = _source_kwargs_from_args(cfg, args.source_mode)
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))
    disable_physics = bool(
        args.disable_physics if args.disable_physics is not None else cfg.get("physics", {}).get("disable_in_eval", False)
    )
    dc_steps = int(args.dc_steps if args.dc_steps is not None else cfg.get("physics", {}).get("dc_steps", 0))
    dc_step = float(args.dc_step_size if args.dc_step_size is not None else cfg.get("physics", {}).get("dc_step_size", 0.0))

    acc = _new_accumulator()
    shard_idx = 0
    image_count = 0
    entry_count = 0
    started = time.perf_counter()
    total_batches = len(dl) if args.max_samples < 0 else min(len(dl), max(1, args.max_samples))
    pbar = tqdm(dl, total=total_batches, desc=f"cache teacher [{pred_type}, {teacher_steps} {teacher_solver}]")

    for batch_idx, (y_batch, x_batch) in enumerate(pbar):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break
        if args.max_samples >= 0 and image_count >= args.max_samples:
            break

        y_batch = to_nchw(y_batch).to(device)
        x_batch = to_nchw(x_batch).to(device)
        if args.max_samples >= 0:
            remaining = int(args.max_samples) - image_count
            y_batch = y_batch[:remaining]
            x_batch = x_batch[:remaining]

        for batch_offset in range(y_batch.shape[0]):
            sample_id = image_count
            y = y_batch[batch_offset : batch_offset + 1]
            x_gt = x_batch[batch_offset : batch_offset + 1]

            for repeat in range(int(args.seeds_per_sample)):
                latent_seed = int(args.seed + sample_id * 100_003 + repeat * 997)

                def _sample_teacher():
                    trajectory: list[dict] = []
                    x_teacher = sample_with_physics_guidance(
                        model=teacher,
                        y=y,
                        H=Hop,
                        steps=teacher_steps,
                        dc_step=dc_step,
                        dc_steps=dc_steps,
                        init_noise_std=init_noise_std,
                        denom_min=denom_min,
                        clamp_x=False,
                        disable_physics=disable_physics,
                        pred_type=pred_type,
                        dc_mode="rgb",
                        solver=teacher_solver,
                        trajectory=trajectory,
                        **source_kwargs,
                    )
                    if not trajectory:
                        raise RuntimeError("Teacher sampler did not record the initial latent.")
                    return trajectory[0]["state"], x_teacher

                z0, x_teacher = _run_with_seed(_sample_teacher, latent_seed)
                _append_entry(
                    acc,
                    sample_id=sample_id,
                    repeat=repeat,
                    latent_seed=latent_seed,
                    y=y,
                    x_gt=x_gt,
                    z0=z0,
                    x_teacher=x_teacher,
                    dtype=save_dtype,
                )
                entry_count += 1

                if _accumulator_len(acc) >= int(args.shard_size):
                    shard_idx = _flush_shard(acc, args.out_dir, shard_idx)
                    acc = _new_accumulator()

            image_count += 1
            pbar.set_postfix(images=image_count, entries=entry_count)

    shard_idx = _flush_shard(acc, args.out_dir, shard_idx)
    elapsed = time.perf_counter() - started
    if entry_count <= 0:
        raise RuntimeError("No teacher distillation entries were cached.")

    metadata = {
        "cache_version": CACHE_VERSION,
        "config": os.path.abspath(args.config),
        "teacher_ckpt": os.path.abspath(args.teacher_ckpt),
        "split": args.split,
        "image_count": int(image_count),
        "entry_count": int(entry_count),
        "seeds_per_sample": int(args.seeds_per_sample),
        "seed": int(args.seed),
        "teacher_steps": int(teacher_steps),
        "teacher_solver": teacher_solver,
        "teacher_pred_type": pred_type,
        "teacher_matcher": matcher,
        "teacher_model_name": model_name,
        "source_mode": str(args.source_mode),
        "source_kwargs": source_kwargs,
        "init_noise_std": float(init_noise_std),
        "disable_physics": bool(disable_physics),
        "dc_steps": int(dc_steps),
        "dc_step_size": float(dc_step),
        "img_channels": int(img_channels),
        "im_hw": [int(im_hw[0]), int(im_hw[1])],
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "num_shards": int(shard_idx),
        "seconds": float(elapsed),
    }

    with open(os.path.join(args.out_dir, METADATA_JSON), "w") as f:
        json.dump(metadata, f, indent=2)
    torch.save(
        {
            "metadata": metadata,
            "psf": psf.detach().cpu().to(dtype=save_dtype),
        },
        os.path.join(args.out_dir, METADATA_PT),
    )

    print("\n========== Teacher Distill Cache ==========")
    print(f"out_dir: {args.out_dir}")
    print(f"images: {image_count}")
    print(f"entries: {entry_count}")
    print(f"shards: {shard_idx}")
    print(f"teacher: {args.teacher_ckpt}")
    print(f"steps/solver: {teacher_steps}/{teacher_solver}")
    print(f"source: {args.source_mode} (sigma={init_noise_std})")
    print(f"seconds: {elapsed:.1f}")
    print("==========================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "distill_cache"))
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seeds_per_sample", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teacher_steps", type=int, default=None)
    ap.add_argument("--teacher_solver", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument("--source_mode", type=str, default="gaussian", choices=["gaussian", "measurement_initialized", "cfg"])
    ap.add_argument("--init_noise_std", type=float, default=None)
    ap.add_argument("--disable_physics", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--dc_steps", type=int, default=None)
    ap.add_argument("--dc_step_size", type=float, default=None)
    ap.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    main(args, cfg)
