import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import torch
import yaml

from lensless_flow.flow_matching import normalize_flow_matcher_name
from lensless_flow.metrics import psnr, ssim_torch
from lensless_flow.sampler import sample_with_physics_guidance
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir
from scripts._paper_eval_utils import (
    build_model,
    build_test_loader_and_operator,
    capture_rng_state,
    load_checkpoint_state,
    restore_rng_state,
)


def _parse_indices(indices_text: str | None, max_samples: int, dataset_len: int) -> list[int]:
    if indices_text:
        values = []
        for token in indices_text.replace(",", " ").split():
            idx = int(token)
            if idx < 0 or idx >= dataset_len:
                raise ValueError(f"Sample index {idx} is out of range for test set of size {dataset_len}.")
            values.append(idx)
        if not values:
            raise ValueError("No valid sample indices were provided.")
        return values

    count = dataset_len if max_samples < 0 else min(dataset_len, max(1, int(max_samples)))
    return list(range(count))


def _to_imshow(x_bchw: torch.Tensor):
    x = x_bchw[0].detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def _ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    channels = int(x.shape[1])
    if channels == 3:
        return x
    if channels == 1:
        return x.repeat(1, 3, 1, 1)
    if channels == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    if channels > 3:
        return x[:, :3]
    raise ValueError(f"Unsupported channel count for LPIPS: {channels}")


def _prepare_lpips_input(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_three_channels(x.clamp(0, 1).float())
    return x * 2.0 - 1.0


def _build_lpips_metric(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "Trajectory LPIPS comparison requires the `lpips` package. Install the project requirements first."
        ) from exc

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for param in metric.parameters():
        param.requires_grad_(False)
    return metric


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sample_with_seed(
    *,
    model,
    y: torch.Tensor,
    Hop,
    steps: int,
    dc_step: float,
    dc_steps: int,
    init_noise_std: float,
    denom_min: float,
    pred_type: str,
    disable_physics: bool,
    latent_seed: int,
):
    trajectory = []
    cpu_state, cuda_states = capture_rng_state()
    torch.manual_seed(int(latent_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(latent_seed))
    try:
        x_hat = sample_with_physics_guidance(
            model=model,
            y=y,
            H=Hop,
            steps=steps,
            dc_step=dc_step,
            dc_steps=dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            clamp_x=False,
            disable_physics=disable_physics,
            pred_type=pred_type,
            dc_mode="rgb",
            trajectory=trajectory,
        )
    finally:
        restore_rng_state(cpu_state, cuda_states)
    return x_hat, trajectory


def _compute_trajectory_rows(
    *,
    sample_idx: int,
    branch_name: str,
    trajectory: list[dict],
    x_gt: torch.Tensor,
    y_meas: torch.Tensor,
    Hop,
    lpips_metric,
) -> list[dict]:
    rows = []
    x_gt_clamped = x_gt.clamp(0, 1)
    x_gt_lpips = _prepare_lpips_input(x_gt_clamped)

    for entry in trajectory:
        state = entry["state"]
        state_clamped = state.clamp(0, 1)
        residual = Hop.forward(state.float()) - y_meas.float()
        rows.append(
            {
                "sample_idx": int(sample_idx),
                "branch": branch_name,
                "step": int(entry["step"]),
                "time": float(entry["time"]),
                "lpips": float(
                    lpips_metric(_prepare_lpips_input(state_clamped), x_gt_lpips).reshape(-1)[0].item()
                ),
                "psnr": float(psnr(state_clamped, x_gt_clamped)),
                "ssim": float(ssim_torch(state_clamped, x_gt_clamped)),
                "dc_rmse": float(residual.pow(2).mean().sqrt().item()),
            }
        )

    return rows


def _select_display_indices(num_states: int, max_frames: int) -> list[int]:
    if num_states <= 0:
        return []
    if max_frames <= 0 or max_frames >= num_states:
        return list(range(num_states))

    raw = torch.linspace(0, num_states - 1, steps=max_frames)
    selected = []
    for value in raw.tolist():
        idx = int(round(value))
        if not selected or selected[-1] != idx:
            selected.append(idx)
    if selected[-1] != num_states - 1:
        selected.append(num_states - 1)
    return selected


def _plot_sample_trajectories(
    *,
    sample_idx: int,
    y_meas: torch.Tensor,
    x_gt: torch.Tensor,
    no_physics_traj: list[dict],
    guided_traj: list[dict],
    no_physics_rows: list[dict],
    guided_rows: list[dict],
    frame_indices: list[int],
    out_path: str,
) -> None:
    cols = 2 + len(frame_indices)
    fig, axes = plt.subplots(2, cols, figsize=(3.0 * cols, 6.5), constrained_layout=True)

    row_lookup = {
        "No Physics": {int(row["step"]): row for row in no_physics_rows},
        "Physics Guidance": {int(row["step"]): row for row in guided_rows},
    }
    traj_lookup = {
        "No Physics": no_physics_traj,
        "Physics Guidance": guided_traj,
    }

    for row_idx, branch_label in enumerate(["No Physics", "Physics Guidance"]):
        axes[row_idx, 0].imshow(_to_imshow(y_meas), cmap="gray" if int(y_meas.shape[1]) == 1 else None)
        axes[row_idx, 0].set_ylabel(branch_label)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(_to_imshow(x_gt), cmap="gray" if int(x_gt.shape[1]) == 1 else None)
        axes[row_idx, 1].axis("off")

        if row_idx == 0:
            axes[row_idx, 0].set_title("Measurement")
            axes[row_idx, 1].set_title("Ground Truth")

        for col_offset, frame_idx in enumerate(frame_indices, start=2):
            traj_entry = traj_lookup[branch_label][frame_idx]
            step = int(traj_entry["step"])
            state = traj_entry["state"]
            step_metrics = row_lookup[branch_label][step]
            axes[row_idx, col_offset].imshow(
                _to_imshow(state),
                cmap="gray" if int(state.shape[1]) == 1 else None,
            )
            axes[row_idx, col_offset].axis("off")
            if row_idx == 0:
                axes[row_idx, col_offset].set_title(
                    f"step {step}\nLPIPS {step_metrics['lpips']:.3f}"
                )
            else:
                axes[row_idx, col_offset].set_title(
                    f"t={step_metrics['time']:.2f}\nDC {step_metrics['dc_rmse']:.3f}"
                )

    fig.suptitle(f"Guidance Trajectory Comparison | sample {sample_idx}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _aggregate_average_rows(per_step_rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in per_step_rows:
        key = (row["branch"], int(row["step"]))
        grouped.setdefault(key, []).append(row)

    average_rows = []
    for (branch, step), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        count = len(rows)
        lpips_mean = float(sum(float(r["lpips"]) for r in rows) / max(1, count))
        average_rows.append(
            {
                "branch": branch,
                "step": int(step),
                "time": float(sum(float(r["time"]) for r in rows) / max(1, count)),
                "num_samples": int(count),
                "lpips_mean": lpips_mean,
                "lpips_std": float(
                    (sum((float(r["lpips"]) - lpips_mean) ** 2 for r in rows) / max(1, count)) ** 0.5
                ),
                "psnr_mean": float(sum(float(r["psnr"]) for r in rows) / max(1, count)),
                "ssim_mean": float(sum(float(r["ssim"]) for r in rows) / max(1, count)),
                "dc_rmse_mean": float(sum(float(r["dc_rmse"]) for r in rows) / max(1, count)),
            }
        )
    return average_rows


def _plot_average_lpips(average_rows: list[dict], out_path: str) -> None:
    branches = ["No Physics", "Physics Guidance"]
    plt.figure(figsize=(8, 5))
    for branch in branches:
        rows = [row for row in average_rows if row["branch"] == branch]
        if not rows:
            continue
        xs = [int(row["step"]) for row in rows]
        ys = [float(row["lpips_mean"]) for row in rows]
        yerr = [float(row["lpips_std"]) for row in rows]
        plt.plot(xs, ys, marker="o", label=branch)
        lower = [max(0.0, y - e) for y, e in zip(ys, yerr)]
        upper = [y + e for y, e in zip(ys, yerr)]
        plt.fill_between(xs, lower, upper, alpha=0.15)

    plt.xlabel("Sampling step")
    plt.ylabel("LPIPS to ground truth")
    plt.title("Intermediate LPIPS Across Sampling Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_model_and_metadata(cfg, ckpt_path: str, img_channels: int, im_hw: tuple[int, int], device: torch.device):
    model = build_model(cfg, img_channels=img_channels, im_hw=im_hw, device=device)
    state = load_checkpoint_state(ckpt_path, device=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    pred_type = str(state.get("mode", cfg.get("train", {}).get("mode", "btb"))).lower()
    if pred_type not in ["btb", "vanilla"]:
        raise ValueError(f"Unknown pred_type/mode in ckpt/cfg: {pred_type}")
    matcher_name = normalize_flow_matcher_name(state.get("matcher", cfg.get("cfm", {}).get("matcher", "rectified")))
    return model, pred_type, matcher_name


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    test_ds, _, _, Hop, img_channels, im_hw = build_test_loader_and_operator(
        cfg=cfg,
        batch_size=1,
        num_workers=0,
        device=device,
    )

    sample_indices = _parse_indices(args.indices, args.max_samples, len(test_ds))
    model, pred_type, matcher_name = _load_model_and_metadata(
        cfg=cfg,
        ckpt_path=args.ckpt,
        img_channels=img_channels,
        im_hw=im_hw,
        device=device,
    )

    steps = int(args.steps if args.steps is not None else cfg["sample"]["steps"])
    init_noise_std = float(cfg["sample"]["init_noise_std"])
    denom_min = float(cfg.get("btb", {}).get("denom_min", 0.05))

    cfg_dc_steps = int(cfg.get("physics", {}).get("dc_steps", 0))
    guided_dc_steps = int(args.guided_dc_steps if args.guided_dc_steps is not None else (cfg_dc_steps if cfg_dc_steps > 0 else 1))
    if guided_dc_steps <= 0:
        raise ValueError("Guided comparison requires guided_dc_steps > 0.")
    guided_dc_step = float(
        args.guided_dc_step_size
        if args.guided_dc_step_size is not None
        else cfg.get("physics", {}).get("dc_step_size", 0.0)
    )

    ensure_dir(args.out_dir)
    lpips_metric = _build_lpips_metric(device)

    per_step_rows = []
    for sample_offset, sample_idx in enumerate(sample_indices):
        y, x = test_ds[sample_idx]
        y = to_nchw(y).to(device)
        x = to_nchw(x).to(device)

        latent_seed = int(args.seed + sample_offset * 100_003 + sample_idx * 53)
        _, no_physics_traj = _sample_with_seed(
            model=model,
            y=y,
            Hop=Hop,
            steps=steps,
            dc_step=guided_dc_step,
            dc_steps=guided_dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            pred_type=pred_type,
            disable_physics=True,
            latent_seed=latent_seed,
        )
        _, guided_traj = _sample_with_seed(
            model=model,
            y=y,
            Hop=Hop,
            steps=steps,
            dc_step=guided_dc_step,
            dc_steps=guided_dc_steps,
            init_noise_std=init_noise_std,
            denom_min=denom_min,
            pred_type=pred_type,
            disable_physics=False,
            latent_seed=latent_seed,
        )

        no_physics_rows = _compute_trajectory_rows(
            sample_idx=sample_idx,
            branch_name="No Physics",
            trajectory=no_physics_traj,
            x_gt=x,
            y_meas=y,
            Hop=Hop,
            lpips_metric=lpips_metric,
        )
        guided_rows = _compute_trajectory_rows(
            sample_idx=sample_idx,
            branch_name="Physics Guidance",
            trajectory=guided_traj,
            x_gt=x,
            y_meas=y,
            Hop=Hop,
            lpips_metric=lpips_metric,
        )
        per_step_rows.extend(no_physics_rows)
        per_step_rows.extend(guided_rows)

        frame_indices = _select_display_indices(len(guided_traj), args.frames)
        sample_plot_path = os.path.join(args.out_dir, f"trajectory_compare_idx{sample_idx}.png")
        _plot_sample_trajectories(
            sample_idx=sample_idx,
            y_meas=y,
            x_gt=x,
            no_physics_traj=no_physics_traj,
            guided_traj=guided_traj,
            no_physics_rows=no_physics_rows,
            guided_rows=guided_rows,
            frame_indices=frame_indices,
            out_path=sample_plot_path,
        )

    average_rows = _aggregate_average_rows(per_step_rows)

    per_step_csv = os.path.join(args.out_dir, "trajectory_lpips_per_step.csv")
    average_csv = os.path.join(args.out_dir, "trajectory_lpips_average.csv")
    curve_path = os.path.join(args.out_dir, "trajectory_lpips_curve.png")
    metadata_path = os.path.join(args.out_dir, "trajectory_lpips_metadata.json")

    _write_csv(
        per_step_csv,
        per_step_rows,
        ["sample_idx", "branch", "step", "time", "lpips", "psnr", "ssim", "dc_rmse"],
    )
    _write_csv(
        average_csv,
        average_rows,
        ["branch", "step", "time", "num_samples", "lpips_mean", "lpips_std", "psnr_mean", "ssim_mean", "dc_rmse_mean"],
    )
    _plot_average_lpips(average_rows, curve_path)

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "ckpt": os.path.abspath(args.ckpt),
                "device": str(device),
                "sample_indices": sample_indices,
                "seed": int(args.seed),
                "steps": int(steps),
                "frames": int(args.frames),
                "pred_type": pred_type,
                "matcher": matcher_name,
                "guided_dc_steps": guided_dc_steps,
                "guided_dc_step_size": guided_dc_step,
                "init_noise_std": init_noise_std,
                "denom_min": denom_min,
            },
            f,
            indent=2,
        )

    print("\n======= Guidance Trajectory Comparison =======")
    print(f"sample_count: {len(sample_indices)}")
    print(f"pred_type: {pred_type}")
    print(f"matcher: {matcher_name}")
    print(f"steps: {steps}")
    print(f"guided_dc_steps: {guided_dc_steps}")
    print(f"guided_dc_step_size: {guided_dc_step} (<=0 means auto-suggested in sampler)")
    print(f"per_step_csv: {per_step_csv}")
    print(f"average_csv: {average_csv}")
    print(f"curve_plot: {curve_path}")
    print(f"metadata_json: {metadata_path}")
    print("Average LPIPS snapshots:")
    for target_step in sorted({0, steps // 4, steps // 2, steps}):
        rows_here = [row for row in average_rows if int(row["step"]) == int(target_step)]
        if not rows_here:
            continue
        no_phys = next((row for row in rows_here if row["branch"] == "No Physics"), None)
        guided = next((row for row in rows_here if row["branch"] == "Physics Guidance"), None)
        if no_phys is None or guided is None:
            continue
        print(
            f"  step {target_step:>3d} | "
            f"No Physics LPIPS {no_phys['lpips_mean']:.4f} | "
            f"Physics Guidance LPIPS {guided['lpips_mean']:.4f} | "
            f"delta {no_phys['lpips_mean'] - guided['lpips_mean']:+.4f}"
        )
    print("=============================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--indices", type=str, default=None, help="Comma-separated test indices to visualize.")
    ap.add_argument("--max_samples", type=int, default=8, help="Used when --indices is omitted.")
    ap.add_argument("--steps", type=int, default=None, help="Override cfg.sample.steps.")
    ap.add_argument("--frames", type=int, default=6, help="Number of trajectory snapshots to show per sample.")
    ap.add_argument("--seed", type=int, default=0, help="Base latent seed shared across branches.")
    ap.add_argument("--guided_dc_steps", type=int, default=None, help="Override physics-guided dc_steps.")
    ap.add_argument("--guided_dc_step_size", type=float, default=None, help="Override physics-guided dc_step_size.")
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "guidance_trajectory_compare"))
    main(ap.parse_args())
