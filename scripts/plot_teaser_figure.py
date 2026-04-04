import argparse
import math
import os

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FCFBF7",
            "axes.facecolor": "#FCFBF7",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_png(path: str) -> np.ndarray:
    arr = mpimg.imread(path)
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(np.float32) / 255.0
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        arr = rgb * alpha + (1.0 - alpha)
    return np.clip(arr[..., :3], 0.0, 1.0)


def make_noise_image(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    noise = noise - noise.min()
    noise = noise / max(float(noise.max()), 1e-8)
    return noise


def gaussian_density(x: np.ndarray, y: np.ndarray, mean: tuple[float, float], cov: np.ndarray) -> np.ndarray:
    diff = np.stack([x - mean[0], y - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(cov)
    det_cov = float(np.linalg.det(cov))
    exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / (2.0 * math.pi * math.sqrt(det_cov))
    return norm * np.exp(-0.5 * exponent)


def draw_distribution(ax, center, covs, weights, base_color, line_color):
    x = np.linspace(0.5, 10.5, 500)
    y = np.linspace(0.6, 6.2, 360)
    xx, yy = np.meshgrid(x, y)
    density = np.zeros_like(xx)
    for weight, cov, mean_shift in zip(weights, covs["covs"], covs["means"]):
        mean = (center[0] + mean_shift[0], center[1] + mean_shift[1])
        density += weight * gaussian_density(xx, yy, mean=mean, cov=np.array(cov, dtype=np.float32))

    levels = np.linspace(density.max() * 0.12, density.max() * 0.92, 6)
    ax.contourf(xx, yy, density, levels=levels, cmap=mpl.colors.LinearSegmentedColormap.from_list("dist_fill", ["#FFFFFF", base_color]), alpha=0.42, zorder=1)
    ax.contour(xx, yy, density, levels=levels[1:], colors=[line_color], linewidths=1.6, alpha=0.72, zorder=2)


def add_thumbnail(ax, image: np.ndarray, point: tuple[float, float], offset: tuple[float, float], frame_color: str, zoom: float) -> None:
    oi = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(
        oi,
        point,
        xybox=offset,
        xycoords="data",
        boxcoords=("offset points"),
        frameon=True,
        bboxprops={"edgecolor": frame_color, "linewidth": 1.6, "boxstyle": "round,pad=0.16", "facecolor": "white"},
        pad=0.18,
        zorder=6,
    )
    ax.add_artist(ab)


def add_point(ax, point: tuple[float, float], color: str, label: str, label_offset: tuple[float, float]) -> None:
    ax.scatter([point[0]], [point[1]], s=62, color=color, edgecolors="white", linewidths=1.1, zorder=7)
    ax.text(point[0] + label_offset[0], point[1] + label_offset[1], label, fontsize=13.5, color=color, ha="center", va="center", zorder=8)


def add_connection(ax, start: tuple[float, float], end: tuple[float, float], curve: float, color: str) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color=color,
        alpha=0.92,
        zorder=4,
    )
    ax.add_patch(patch)


def add_dashed_connection(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    curve: float,
    color: str,
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.7,
        linestyle=(0, (4.2, 3.0)),
        color=color,
        alpha=0.82,
        zorder=3,
    )
    ax.add_patch(patch)
    if text is not None and text_xy is not None:
        ax.text(text_xy[0], text_xy[1], text, fontsize=11.0, color=color, ha="center", va="center")


def resolve_target_path(target_dir: str, sample_id: str, target_mode: str) -> str:
    if target_mode == "gt":
        path = os.path.join(target_dir, f"idx_{sample_id}_gt.png")
    else:
        path = os.path.join(target_dir, f"idx_{sample_id}_recon_steps40_btb.png")
        if not os.path.exists(path):
            fallback_dirs = [
                os.path.join("outputs", "samples_btb_phys"),
                os.path.join("outputs", "samples_btb_no_phys"),
            ]
            for fallback_dir in fallback_dirs:
                candidate = os.path.join(fallback_dir, f"idx_{sample_id}_recon_steps40_btb.png")
                if os.path.exists(candidate):
                    return candidate
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def build_figure(args):
    style()

    sample_ids = [sample_id.zfill(5) for sample_id in args.sample_ids]
    if len(sample_ids) != 2:
        raise ValueError("Please provide exactly two sample ids.")

    target_images = [load_png(resolve_target_path(args.target_dir, sample_id, args.target_mode)) for sample_id in sample_ids]
    measurement_images = [load_png(os.path.join(args.measurement_dir, f"idx_{sample_id}_y.png")) for sample_id in sample_ids]
    noise_images = [make_noise_image(target_images[idx].shape, seed=args.noise_seeds[idx]) for idx in range(2)]

    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0.55, 6.35)
    ax.axis("off")

    left_specs = {
        "covs": [
            [[0.42, 0.0], [0.0, 0.95]],
            [[0.28, 0.05], [0.05, 0.55]],
        ],
        "means": [
            (0.0, 0.0),
            (-0.15, 0.10),
        ],
    }
    top_right_specs = {
        "covs": [
            [[0.58, 0.18], [0.18, 0.34]],
            [[0.26, -0.06], [-0.06, 0.16]],
        ],
        "means": [
            (0.00, 0.00),
            (0.52, -0.20),
        ],
    }
    bottom_right_specs = {
        "covs": [
            [[0.62, -0.16], [-0.16, 0.38]],
            [[0.24, 0.05], [0.05, 0.17]],
        ],
        "means": [
            (0.00, 0.00),
            (-0.55, 0.18),
        ],
    }

    draw_distribution(
        ax,
        center=(2.35, 3.25),
        covs=left_specs,
        weights=[0.78, 0.22],
        base_color="#AFC9DC",
        line_color="#5D86A3",
    )
    draw_distribution(
        ax,
        center=(7.55, 4.05),
        covs=top_right_specs,
        weights=[0.72, 0.28],
        base_color="#BEE4DC",
        line_color="#2D8478",
    )
    draw_distribution(
        ax,
        center=(8.78, 2.20),
        covs=bottom_right_specs,
        weights=[0.74, 0.26],
        base_color="#F0D0B4",
        line_color="#B66B34",
    )

    # fig.text(0.055, 0.945, "Physics-Guided Conditional Flow Matching for Lensless Imaging", fontsize=21.0, fontweight="semibold", color="#141414", ha="left", va="top")
    # fig.text(
    #     0.055,
    #     0.895,
    #     "A conditional pairwise flow-matching view: the same Gaussian prior branches into distinct target distributions once the conditioning measurement changes.",
    #     fontsize=12.2,
    #     color="#4B4B4B",
    #     ha="left",
    #     va="top",
    # )

    ax.text(2.15, 5.72, r"Gaussian prior $p_0(z)$", fontsize=18.0, fontweight="semibold", color="#2E5F82", ha="center")
    ax.text(8.15, 5.72, r"Conditional target distributions", fontsize=18.0, fontweight="semibold", color="#7B4F2A", ha="center")
    ax.text(7.55, 4.92, r"$p_{\mathrm{data}}(x \mid y^{(1)})$", fontsize=14.8, color="#2D8478", ha="center")
    ax.text(8.88, 1.20, r"$p_{\mathrm{data}}(x \mid y^{(2)})$", fontsize=14.8, color="#B66B34", ha="center")

    left_points = [(2.08, 4.12), (2.68, 2.18)]
    right_points = [(7.60, 4.12), (8.92, 2.18)]
    measurement_points = [(5.45, 4.48), (6.20, 1.94)]
    left_labels = [r"$z_0^{(1)}$", r"$z_0^{(2)}$"]
    right_labels = [r"$x_1^{(1)}$", r"$x_1^{(2)}$"]
    measurement_labels = [r"$y^{(1)}$", r"$y^{(2)}$"]

    top_color = "#2D8478"
    bottom_color = "#B66B34"
    guide_gray = "#7C7C7C"

    add_connection(ax, left_points[0], right_points[0], curve=0.14, color=top_color)
    add_connection(ax, left_points[1], right_points[1], curve=-0.14, color=bottom_color)

    for idx in range(2):
        add_point(ax, left_points[idx], color="#2E5F82", label=left_labels[idx], label_offset=(0.0, 0.38 if idx == 0 else -0.42))
    add_point(ax, right_points[0], color=top_color, label=right_labels[0], label_offset=(0.0, 0.38))
    add_point(ax, right_points[1], color=bottom_color, label=right_labels[1], label_offset=(0.0, -0.42))

    add_thumbnail(ax, noise_images[0], left_points[0], offset=(-88, 26), frame_color="#7AA3BF", zoom=args.zoom)
    add_thumbnail(ax, noise_images[1], left_points[1], offset=(-118, -26), frame_color="#7AA3BF", zoom=args.zoom)
    add_thumbnail(ax, target_images[0], right_points[0], offset=(118, 26), frame_color=top_color, zoom=args.zoom)
    add_thumbnail(ax, target_images[1], right_points[1], offset=(118, -26), frame_color=bottom_color, zoom=args.zoom)
    add_thumbnail(ax, measurement_images[0], measurement_points[0], offset=(-30, 24), frame_color="#6F6F6F", zoom=args.zoom * 0.58)
    add_thumbnail(ax, measurement_images[1], measurement_points[1], offset=(-40, 0), frame_color="#6F6F6F", zoom=args.zoom * 0.58)

    add_point(ax, measurement_points[0], color=guide_gray, label=measurement_labels[0], label_offset=(0.3, 0.40))
    add_point(ax, measurement_points[1], color=guide_gray, label=measurement_labels[1], label_offset=(0.18, -0.42))

    add_dashed_connection(
        ax,
        right_points[0],
        measurement_points[0],
        curve=-0.24,
        color=guide_gray,
        # text=r"forward model $H_{\phi}$",
        # text_xy=(6.20, 5.00),
    )
    add_dashed_connection(
        ax,
        measurement_points[0],
        (6.05, 4.28),
        curve=0.20,
        color=top_color,
        text=r"physics guidance",
        text_xy=(5.15, 4.16),
    )
    add_dashed_connection(
        ax,
        right_points[1],
        measurement_points[1],
        curve=0.24,
        color=guide_gray,
    )
    add_dashed_connection(
        ax,
        measurement_points[1],
        (6.85, 2.05),
        curve=-0.18,
        color=bottom_color,
    )

    ax.text(5.30, 0.94, r"$z_t^{(i)} = (1-t)\,z_0^{(i)} + t\,x_1^{(i)},\quad v^{\star}(z_t^{(i)},t)=x_1^{(i)}-z_0^{(i)}$", fontsize=16.0, color="#2C2C2C", ha="center")
    ax.text(5.30, 0.62, r"physics guidance nudges the flow using $-\eta \nabla_z \|H_{\phi}(z_t^{(i)})-y^{(i)}\|_2^2$", fontsize=12.5, color="#6A6A6A", ha="center")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Draw a clean method-only teaser schematic for the lensless CFM paper.")
    parser.add_argument("--sample_ids", nargs=2, default=["00001", "00004"], help="Two sample ids used on the target side.")
    parser.add_argument("--target_mode", choices=["gt", "generated"], default="gt")
    parser.add_argument("--target_dir", type=str, default=os.path.join("outputs", "unet"))
    parser.add_argument("--measurement_dir", type=str, default=os.path.join("outputs", "unet"))
    parser.add_argument("--zoom", type=float, default=0.80)
    parser.add_argument("--noise_seeds", nargs=2, type=int, default=[7, 19])
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--output_base", type=str, default=os.path.join("outputs", "paper", "teaser_method_schematic"))
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.output_base))
    fig = build_figure(args)
    pdf_path = args.output_base + ".pdf"
    png_path = args.output_base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
