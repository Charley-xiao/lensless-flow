import argparse
import csv
import json
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
CORRUPTION_LABELS = {
    "background_offset": "Background Offset",
    "measurement_noise": "Measurement Noise",
}


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _read_metadata(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _to_float(value: str) -> float:
    return float(value)


def _load_physical(path: str) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    rows = _read_csv(path)
    grouped: dict[str, dict[str, list[dict[str, float | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        corruption = row["corruption"]
        if corruption not in {"background_offset", "measurement_noise"}:
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
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": "#D9D2C3",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _metric_title(corruption: str, metric_name: str, metadata: dict) -> str:
    title = CORRUPTION_LABELS.get(corruption, corruption.replace("_", " ").title())
    if corruption == "measurement_noise":
        scale = str(metadata.get("measurement_noise_scale", "rms"))
        title = f"{title} ({scale})"
    return f"{title}: {metric_name}"


def _x_label(corruption: str) -> str:
    if corruption == "background_offset":
        return r"Offset $\beta$"
    if corruption == "measurement_noise":
        return "Noise level"
    return "Severity"


def _plot_metric(ax, grouped, corruption: str, metric_key: str, ylabel: str, metadata: dict):
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
            markersize=6.8,
            linewidth=2.7,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax.scatter(xs, ys, s=42, color=METHOD_COLORS[method], zorder=3)
        ax.set_xticks(xs, [f"{x:.2f}" for x in xs])

    ax.set_xlabel(_x_label(corruption))
    ax.set_ylabel(ylabel)
    ax.set_title(_metric_title(corruption, ylabel, metadata))
    ax.grid(True, axis="y")


def main(args):
    _style()
    import matplotlib.pyplot as plt

    grouped = _load_physical(args.physical_csv)
    metadata = _read_metadata(args.metadata_json)
    ensure_dir(os.path.dirname(args.output_base))

    requested = [x.strip() for x in args.corruptions.replace(",", " ").split() if x.strip()]
    corruptions = [c for c in requested if c in grouped]
    if not corruptions:
        raise ValueError(f"No requested corruptions found in {args.physical_csv}: {requested}")

    fig, axes = plt.subplots(2, len(corruptions), figsize=(5.2 * len(corruptions), 6.0), constrained_layout=True)
    if len(corruptions) == 1:
        axes = axes.reshape(2, 1)

    for col, corruption in enumerate(corruptions):
        _plot_metric(
            axes[0, col],
            grouped=grouped,
            corruption=corruption,
            metric_key="noisy_psnr",
            ylabel="PSNR (dB)",
            metadata=metadata,
        )
        _plot_metric(
            axes[1, col],
            grouped=grouped,
            corruption=corruption,
            metric_key="noisy_ssim",
            ylabel="SSIM",
            metadata=metadata,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02))

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
        "--metadata_json",
        type=str,
        default=os.path.join("outputs", "physical_robustness", "physical_robustness_metadata.json"),
    )
    ap.add_argument(
        "--corruptions",
        type=str,
        default="background_offset,measurement_noise",
    )
    ap.add_argument(
        "--output_base",
        type=str,
        default=os.path.join("outputs", "paper", "physical_robustness_pretty"),
    )
    ap.add_argument("--dpi", type=int, default=220)
    main(ap.parse_args())
