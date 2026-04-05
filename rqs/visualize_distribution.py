import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from tqdm import tqdm

from lensless_flow.data import make_dataloader
from lensless_flow.tensor_utils import to_nchw
from lensless_flow.utils import ensure_dir, set_seed


def flatten_sample(x) -> np.ndarray:
    return to_nchw(x).squeeze(0).detach().cpu().float().numpy().reshape(-1)


def sample_indices(total: int, max_samples: int, seed: int) -> np.ndarray:
    n = total if max_samples <= 0 else min(total, max_samples)
    rng = np.random.default_rng(seed)
    if n >= total:
        return np.arange(total)
    return np.sort(rng.choice(total, size=n, replace=False))


def collect_pairs(ds, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lensless_vecs = []
    lensed_vecs = []

    for idx in tqdm(indices, desc="collecting samples"):
        lensless, lensed = ds[int(idx)]
        lensless_vecs.append(flatten_sample(lensless))
        lensed_vecs.append(flatten_sample(lensed))

    return np.stack(lensless_vecs, axis=0), np.stack(lensed_vecs, axis=0)


def fit_kde_2d(coords: np.ndarray, seed: int):
    try:
        return gaussian_kde(coords.T)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        jitter = 1e-6 * rng.standard_normal(coords.shape)
        return gaussian_kde((coords + jitter).T)


def evaluate_kde_grid(coords: np.ndarray, seed: int, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kde = fit_kde_2d(coords, seed=seed)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.12 * span

    x_grid = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], grid_size)
    y_grid = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def add_distribution_panel(
    ax,
    coords: np.ndarray,
    color: str,
    cmap: str,
    title: str,
    xlabel: str,
    ylabel: str,
    seed: int,
    grid_size: int,
):
    xx, yy, zz = evaluate_kde_grid(coords, seed=seed, grid_size=grid_size)
    ax.contourf(xx, yy, zz, levels=8, cmap=cmap, alpha=0.4)
    ax.contour(xx, yy, zz, levels=6, colors=color, linewidths=1.0, alpha=0.9)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c=color, alpha=0.18, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def main(cfg, split: str, max_samples: int, seed: int, out_path: str | None, grid_size: int):
    set_seed(seed)

    ds, _ = make_dataloader(
        split=split,
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    indices = sample_indices(len(ds), max_samples=max_samples, seed=seed)
    lensless_vecs, lensed_vecs = collect_pairs(ds, indices)

    lensless_pca = PCA(n_components=2, random_state=seed)
    lensed_pca = PCA(n_components=2, random_state=seed)
    joint_pca = PCA(n_components=2, random_state=seed)

    lensless_coords = lensless_pca.fit_transform(lensless_vecs)
    lensed_coords = lensed_pca.fit_transform(lensed_vecs)

    joint_stack = np.concatenate([lensless_vecs, lensed_vecs], axis=0)
    joint_coords = joint_pca.fit_transform(joint_stack)
    joint_lensless = joint_coords[: len(indices)]
    joint_lensed = joint_coords[len(indices) :]

    if out_path is None:
        save_dir = cfg.get("sample", {}).get("save_dir", "outputs")
        ensure_dir(save_dir)
        out_path = os.path.join(save_dir, f"pca_kde_distribution_{split}_n{len(indices)}.png")
    else:
        ensure_dir(os.path.dirname(out_path) or ".")

    lensless_var = 100.0 * lensless_pca.explained_variance_ratio_.sum()
    lensed_var = 100.0 * lensed_pca.explained_variance_ratio_.sum()
    joint_var = 100.0 * joint_pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    add_distribution_panel(
        axes[0],
        coords=lensless_coords,
        color="#0f766e",
        cmap="BuGn",
        title=f"Lensless measurements\n2D PCA captures {lensless_var:.1f}% variance",
        xlabel="PC1",
        ylabel="PC2",
        seed=seed,
        grid_size=grid_size,
    )

    add_distribution_panel(
        axes[1],
        coords=lensed_coords,
        color="#c2410c",
        cmap="OrRd",
        title=f"Lensed images\n2D PCA captures {lensed_var:.1f}% variance",
        xlabel="PC1",
        ylabel="PC2",
        seed=seed + 1,
        grid_size=grid_size,
    )

    xx_y, yy_y, zz_y = evaluate_kde_grid(joint_lensless, seed=seed, grid_size=grid_size)
    xx_x, yy_x, zz_x = evaluate_kde_grid(joint_lensed, seed=seed + 1, grid_size=grid_size)
    axes[2].contour(xx_y, yy_y, zz_y, levels=6, colors="#0f766e", linewidths=1.2)
    axes[2].contour(xx_x, yy_x, zz_x, levels=6, colors="#c2410c", linewidths=1.2)
    axes[2].scatter(joint_lensless[:, 0], joint_lensless[:, 1], s=10, c="#0f766e", alpha=0.14, edgecolors="none", label="Lensless")
    axes[2].scatter(joint_lensed[:, 0], joint_lensed[:, 1], s=10, c="#c2410c", alpha=0.14, edgecolors="none", label="Lensed")
    axes[2].set_title(f"Shared PCA space\n2D PCA captures {joint_var:.1f}% variance")
    axes[2].set_xlabel("Joint PC1")
    axes[2].set_ylabel("Joint PC2")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=True, loc="best")

    fig.suptitle(f"PCA + KDE distribution view for {split} split ({len(indices)} paired samples)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved visualization to: {out_path}")
    print(f"Samples used: {len(indices)}")
    print(f"Lensless explained variance (PC1+PC2): {lensless_var:.2f}%")
    print(f"Lensed explained variance (PC1+PC2): {lensed_var:.2f}%")
    print(f"Joint explained variance (PC1+PC2): {joint_var:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default=None, help="Dataset split to visualize. Defaults to cfg.data.split.")
    ap.add_argument("--max_samples", type=int, default=1000, help="Number of paired samples to project. Use -1 for all.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--grid_size", type=int, default=160, help="Grid resolution for KDE contours.")
    ap.add_argument("--out", type=str, default=None, help="Optional output image path.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    split = args.split or cfg["data"].get("split", "train")
    main(
        cfg=cfg,
        split=split,
        max_samples=args.max_samples,
        seed=args.seed,
        out_path=args.out,
        grid_size=args.grid_size,
    )
