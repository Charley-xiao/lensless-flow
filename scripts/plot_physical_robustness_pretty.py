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


def _load_physical(path: str) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    rows = _read_csv(path)
    grouped: dict[str, dict[str, list[dict[str, float | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        corruption = row["corruption"]
        if corruption != "background_offset":
            continue
        method = row["method"]
        casted = {}
        for k, v in row.items():
            if k in {"corruption", "level_label", "level_value", "method"}:
                casted[k] = v
            else:
                casted[k] = _to_float(v)
        grouped[corruption][method].append(casted)

    for corruption in grouped:
        for method in grouped[corruption]:
            grouped[corruption][method].sort(key=lambda r: int(r["severity_rank"]))
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
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
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


def _plot_metric(ax, grouped, corruption: str, metric_key: str, ylabel: str, title: str):
    for method in METHOD_ORDER:
        rows = grouped.get(corruption, {}).get(method, [])
        if not rows:
            continue
        xs = [float(r["level_value"]) for r in rows]
        ys = [float(r[metric_key]) for r in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=6.5,
            linewidth=2.6,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax.scatter(xs, ys, s=40, color=METHOD_COLORS[method], zorder=3)

    if corruption == "background_offset":
        ax.set_xticks([0.00, 0.01, 0.03, 0.05], ["0.00", "0.01", "0.03", "0.05"])
        ax.set_xlabel(r"Offset $\beta$")
    else:
        ticks = [0.70, 0.85, 1.00, 1.15, 1.30]
        ax.set_xticks(ticks, ["0.70", "0.85", "1.00", "1.15", "1.30"])
        ax.set_xlabel(r"Scale $\alpha$")
        ax.axvline(1.0, color="#7A6E5F", linestyle="--", linewidth=1.1, alpha=0.85)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")


def main(args):
    _style()
    import matplotlib.pyplot as plt

    grouped = _load_physical(args.physical_csv)
    ensure_dir(os.path.dirname(args.output_base))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)

    _plot_metric(
        axes[0],
        grouped=grouped,
        corruption="background_offset",
        metric_key="noisy_psnr",
        ylabel="PSNR (dB)",
        title="Background Offset: PSNR",
    )
    _plot_metric(
        axes[1],
        grouped=grouped,
        corruption="background_offset",
        metric_key="noisy_ssim",
        ylabel="SSIM",
        title="Background Offset: SSIM",
    )

    axes[0].legend(loc="lower left", ncol=1)

    fig.suptitle(
        "Photometric Robustness Under Additive Background Offset",
        fontsize=15.5,
        fontweight="semibold",
        color="#1E1E1E",
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
        "--physical_csv",
        type=str,
        default=os.path.join("outputs", "physical_robustness", "physical_robustness_summary.csv"),
    )
    ap.add_argument(
        "--output_base",
        type=str,
        default=os.path.join("outputs", "paper", "physical_robustness_pretty"),
    )
    ap.add_argument("--dpi", type=int, default=220)
    main(ap.parse_args())
