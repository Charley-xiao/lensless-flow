import argparse
import csv
import os
from collections import defaultdict

from lensless_flow.utils import ensure_dir


METHOD_ORDER = ["unet", "v_prediction", "x_prediction"]
METHOD_LABELS = {
    "unet": "U-Net",
    "v_prediction": "CFM v-pred",
    "x_prediction": "CFM x-pred",
}
METHOD_COLORS = {
    "unet": "#6C757D",
    "v_prediction": "#2A6F97",
    "x_prediction": "#D96C47",
}


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    return float(value)


def _load_uncertainty(path: str) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    out = {}
    for row in rows:
        method = row["method"]
        out[method] = {k: (_to_float(v) if k != "method" else v) for k, v in row.items()}
    return out


def _load_background(path: str) -> dict[str, list[dict[str, float | str]]]:
    rows = _read_csv(path)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        if row["corruption"] != "background_offset":
            continue
        method = row["method"]
        casted = {}
        for k, v in row.items():
            if k in {"corruption", "level_label", "level_value", "method"}:
                casted[k] = v
            else:
                casted[k] = _to_float(v)
        grouped[method].append(casted)
    for method in grouped:
        grouped[method].sort(key=lambda r: int(r["severity_rank"]))
    return grouped


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBF8F2",
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": "#242424",
            "axes.titleweight": "semibold",
            "axes.titlesize": 15.5,
            "axes.labelsize": 13.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 11.8,
            "ytick.labelsize": 11.8,
            "grid.color": "#D9D2C3",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_progression(ax, uncertainty: dict[str, dict[str, float]]):
    categories = ["Single", "Mean", "Best-of-8"]
    keys = ["sample_psnr", "mean_psnr", "best_psnr"]
    x = list(range(len(categories)))

    for method in METHOD_ORDER:
        row = uncertainty.get(method)
        if row is None:
            continue
        y = [float(row[key]) for key in keys]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7.5,
            linewidth=2.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax.scatter(x, y, s=46, color=METHOD_COLORS[method], zorder=3)

    ax.set_xticks(x, categories)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Uncertainty:\nSingle, Mean, and Best-of-8")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")


def _plot_diversity_alignment(ax, uncertainty: dict[str, dict[str, float]]):
    for method in METHOD_ORDER:
        row = uncertainty.get(method)
        if row is None:
            continue
        x = float(row["pairwise_l1"])
        y = float(row["uncertainty_error_spearman"])
        ax.scatter(
            [x],
            [y],
            s=120,
            color=METHOD_COLORS[method],
            edgecolors="#2B2B2B",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            x + 0.0007 if method == "unet" else x - 0.013,
            y + 0.008 if method == "unet" else y - 0.01,
            METHOD_LABELS[method],
            fontsize=10.6,
            color="#222222",
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(r"Mean Pairwise $\ell_1$")
    ax.set_ylabel("Uncertainty-Error Spearman")
    ax.set_title("Uncertainty: Error Alignment")
    ax.grid(True)


def _plot_background_metric(ax, grouped, metric_key: str, ylabel: str, title: str):
    for method in METHOD_ORDER:
        rows = grouped.get(method, [])
        if not rows:
            continue
        xs = [float(r["level_value"]) for r in rows]
        ys = [float(r[metric_key]) for r in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax.scatter(xs, ys, s=44, color=METHOD_COLORS[method], zorder=3)

    ax.set_xticks([0.00, 0.01, 0.03, 0.05], ["0.00", "0.01", "0.03", "0.05"])
    ax.set_xlabel(r"Offset $\beta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")


def main(args):
    _style()
    import matplotlib.pyplot as plt

    uncertainty = _load_uncertainty(args.uncertainty_csv)
    background = _load_background(args.physical_csv)

    ensure_dir(os.path.dirname(args.output_base))

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.9), constrained_layout=True)

    _plot_progression(axes[0], uncertainty=uncertainty)
    _plot_diversity_alignment(axes[1], uncertainty=uncertainty)
    _plot_background_metric(
        axes[2],
        grouped=background,
        metric_key="noisy_psnr",
        ylabel="PSNR (dB)",
        title="Background Offset:\nPSNR",
    )
    _plot_background_metric(
        axes[3],
        grouped=background,
        metric_key="noisy_ssim",
        ylabel="SSIM",
        title="Background Offset:\nSSIM",
    )

    pdf_path = args.output_base + ".pdf"
    png_path = args.output_base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--uncertainty_csv",
        type=str,
        default=os.path.join("outputs", "uncertainty", "uncertainty_summary.csv"),
    )
    ap.add_argument(
        "--physical_csv",
        type=str,
        default=os.path.join("outputs", "physical_robustness", "physical_robustness_summary.csv"),
    )
    ap.add_argument(
        "--output_base",
        type=str,
        default=os.path.join("outputs", "paper", "uncertainty_background_pretty"),
    )
    ap.add_argument("--dpi", type=int, default=220)
    main(ap.parse_args())
