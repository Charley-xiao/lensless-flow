"""Create a compact single-column schematic for the many-to-one lensless map.

The figure illustrates that for a fixed measurement y = Hx, a rank-deficient
forward operator H can map an affine family of scenes x = x0 + alpha * n
to the same observation y.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


BG = "#f7f4ee"
INK = "#32343a"
MUTED = "#737780"
BLUE = "#3f6f9e"
BLUE_LIGHT = "#a9bfd7"
TEAL = "#4c9b90"
TEAL_LIGHT = "#b7ddd7"
ORANGE = "#bf7a3b"
ORANGE_LIGHT = "#f0d6bc"
GRAY_ARROW = "#8a8f97"
WHITE = "#ffffff"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": INK,
            "axes.edgecolor": BG,
        }
    )


def stroked_text(ax, *args, **kwargs):
    text = ax.text(*args, **kwargs)
    text.set_path_effects([pe.withStroke(linewidth=2.1, foreground=BG, alpha=0.95)])
    return text


def add_soft_background(ax) -> None:
    ax.add_patch(Ellipse((0.27, 0.57), 0.46, 0.74, facecolor=BLUE, edgecolor="none", alpha=0.06, zorder=0))
    ax.add_patch(Ellipse((0.82, 0.50), 0.30, 0.44, facecolor=TEAL, edgecolor="none", alpha=0.08, zorder=0))


def draw_scene_space(ax) -> dict[str, tuple[float, float]]:
    origin = (0.09, 0.18)
    u = (0.28, 0.00)
    v = (0.00, 0.47)
    w = (0.08, 0.11)

    A = origin
    B = (A[0] + u[0], A[1] + u[1])
    C = (B[0] + v[0], B[1] + v[1])
    D = (A[0] + v[0], A[1] + v[1])
    A2 = (A[0] + w[0], A[1] + w[1])
    B2 = (B[0] + w[0], B[1] + w[1])
    C2 = (C[0] + w[0], C[1] + w[1])
    D2 = (D[0] + w[0], D[1] + w[1])

    for poly, color, alpha in [
        ([A, B, C, D], WHITE, 0.78),
        ([D, C, C2, D2], BLUE_LIGHT, 0.18),
        ([B, C, C2, B2], BLUE_LIGHT, 0.13),
    ]:
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="none", alpha=alpha, zorder=1))

    edge_sets = [
        (A, B),
        (B, C),
        (C, D),
        (D, A),
        (A2, B2),
        (B2, C2),
        (C2, D2),
        (D2, A2),
        (A, A2),
        (B, B2),
        (C, C2),
        (D, D2),
    ]
    for p0, p1 in edge_sets:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BLUE, lw=1.0, alpha=0.55, zorder=2)

    stroked_text(ax, 0.11, 0.90, r"Scene space $x \in \mathbb{R}^{n}$", fontsize=8.1, color=BLUE, fontweight="semibold")
    # stroked_text(ax, A[0] + u[0] + 0.015, A[1] - 0.018, r"$x_1$", fontsize=6.6, color=MUTED)
    # stroked_text(ax, D[0] - 0.028, D[1] + 0.010, r"$x_2$", fontsize=6.6, color=MUTED)
    # stroked_text(ax, A2[0] - 0.006, A2[1] + 0.010, r"$x_3$", fontsize=6.6, color=MUTED)

    line_start = (0.12, 0.23)
    line_end = (0.41, 0.58)
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color=WHITE,
        lw=7.0,
        alpha=0.92,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color=ORANGE,
        lw=2.6,
        alpha=0.95,
        solid_capstyle="round",
        zorder=4,
    )

    dot_ts = [0.10, 0.24, 0.58, 0.90]
    for t in dot_ts:
        x = line_start[0] + t * (line_end[0] - line_start[0])
        y = line_start[1] + t * (line_end[1] - line_start[1])
        ax.add_patch(Circle((x, y), 0.006, facecolor=ORANGE, edgecolor="none", alpha=0.45, zorder=5))

    samples = {
        r"$x^{(1)}$": (0.17, 0.29),
        r"$x^{(2)}$": (0.26, 0.40),
        r"$x^{(3)}$": (0.35, 0.51),
    }
    offsets = {
        r"$x^{(1)}$": (-0.03, -0.06),
        r"$x^{(2)}$": (-0.045, 0.03),
        r"$x^{(3)}$": (0.01, 0.03),
    }
    for label, (x, y) in samples.items():
        ax.add_patch(Circle((x, y), 0.0175, facecolor=ORANGE, edgecolor=WHITE, lw=1.3, zorder=6))
        dx, dy = offsets[label]
        stroked_text(ax, x + dx, y + dy, label, fontsize=7.2, color=ORANGE, zorder=7)

    stroked_text(ax, 0.072, 0.8, r"Affine preimage of $y$", fontsize=6.5, color=BLUE, fontweight="semibold", zorder=8)
    stroked_text(ax, 0.072, 0.713, r"$x = x_{0} + \alpha n,\; n \in \mathcal{N}(H_\phi)$", fontsize=6.3, color=INK, zorder=8)
    ax.add_patch(
        FancyArrowPatch(
            posA=(0.23, 0.695),
            posB=(0.31, 0.54),
            arrowstyle="-|>",
            mutation_scale=9.0,
            lw=0.9,
            color=BLUE,
            connectionstyle="arc3,rad=-0.18",
            alpha=0.75,
            zorder=7,
        )
    )

    return samples


def draw_measurement_space(ax) -> tuple[float, float]:
    stroked_text(ax, 0.63, 0.90, r"Measurement $y \in \mathbb{R}^{m}$", fontsize=8.1, color=TEAL, fontweight="semibold")

    panel = FancyBboxPatch(
        (0.74, 0.30),
        0.18,
        0.34,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=WHITE,
        edgecolor=TEAL,
        lw=1.0,
        alpha=0.96,
        zorder=2,
    )
    ax.add_patch(panel)

    for gx in [0.785, 0.835, 0.885]:
        ax.plot([gx, gx], [0.33, 0.61], color=TEAL_LIGHT, lw=0.7, alpha=0.65, zorder=2)
    for gy in [0.36, 0.42, 0.48, 0.54, 0.60]:
        ax.plot([0.76, 0.90], [gy, gy], color=TEAL_LIGHT, lw=0.7, alpha=0.65, zorder=2)

    y_pt = (0.825, 0.47)
    ax.add_patch(Circle(y_pt, 0.051, facecolor=TEAL, edgecolor="none", alpha=0.08, zorder=3))
    ax.add_patch(Circle(y_pt, 0.028, facecolor=TEAL, edgecolor="none", alpha=0.12, zorder=4))
    ax.add_patch(Circle(y_pt, 0.0175, facecolor=TEAL, edgecolor=WHITE, lw=1.3, zorder=5))
    stroked_text(ax, y_pt[0] + 0.026, y_pt[1] + 0.028, r"$y$", fontsize=7.2, color=TEAL, zorder=6)
    stroked_text(ax, 0.725, 0.230, r"one observation", fontsize=6.1, color=MUTED, zorder=6)

    return y_pt


def draw_many_to_one_arrows(ax, samples: dict[str, tuple[float, float]], y_pt: tuple[float, float]) -> None:
    radii = {
        r"$x^{(1)}$": -0.18,
        r"$x^{(2)}$": -0.05,
        r"$x^{(3)}$": 0.09,
    }
    widths = {
        r"$x^{(1)}$": 1.1,
        r"$x^{(2)}$": 1.35,
        r"$x^{(3)}$": 1.1,
    }
    alphas = {
        r"$x^{(1)}$": 0.75,
        r"$x^{(2)}$": 0.92,
        r"$x^{(3)}$": 0.75,
    }
    for label, start in samples.items():
        ax.add_patch(
            FancyArrowPatch(
                posA=start,
                posB=y_pt,
                arrowstyle="-|>",
                mutation_scale=10.0,
                lw=widths[label],
                color=GRAY_ARROW,
                alpha=alphas[label],
                connectionstyle=f"arc3,rad={radii[label]}",
                zorder=4,
            )
        )

    stroked_text(ax, 0.52, 0.8, r"lossy $H_\phi$", fontsize=7.2, color=INK, fontweight="semibold", zorder=8)
    stroked_text(ax, 0.485, 0.731, r"$\mathrm{rank}(H_\phi) < n$", fontsize=6.9, color=MUTED, zorder=8)

    eq_box = FancyBboxPatch(
        (0.34, 0.062),
        0.445,
        0.080,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor=WHITE,
        edgecolor="#d9d3ca",
        lw=0.9,
        alpha=0.98,
        zorder=6,
    )
    ax.add_patch(eq_box)
    stroked_text(
        ax,
        0.357,
        0.087,
        r"$H_\phi x^{(1)} = H_\phi x^{(2)} = H_\phi x^{(3)} = y$",
        fontsize=7.0,
        color=INK,
        zorder=7,
    )


def build_figure(width: float, height: float):
    setup_style()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_soft_background(ax)
    samples = draw_scene_space(ax)
    y_pt = draw_measurement_space(ax)
    draw_many_to_one_arrows(ax, samples, y_pt)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/paper/lossy_forward_schematic_singlecol"),
        help="Path stem for exported PDF/PNG outputs.",
    )
    parser.add_argument("--width", type=float, default=3.35, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=1.95, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_stem = args.output_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(width=args.width, height=args.height)
    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
