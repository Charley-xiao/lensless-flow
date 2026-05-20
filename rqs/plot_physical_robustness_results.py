import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lensless_flow.utils import ensure_dir


SUMMARY_SUFFIX = "_physical_robustness_summary.csv"
METADATA_SUFFIX = "_physical_robustness_metadata.json"
BUILTIN_METHOD_ORDER = ("v_prediction", "x_prediction", "unet")
DEFAULT_LABELS = {
    "v_prediction": "v-prediction",
    "x_prediction": "x-prediction",
    "unet": "baseline U-Net",
}
DEFAULT_COLORS = {
    "v_prediction": "#2A6F97",
    "x_prediction": "#D96C47",
    "unet": "#6C757D",
}
FALLBACK_COLORS = (
    "#2E7D32",
    "#7B2CBF",
    "#C99700",
    "#008C9E",
    "#C44569",
    "#5D737E",
    "#8F6A00",
    "#4D908E",
)
METRICS = (
    ("noisy_ssim", "SSIM", "higher is better"),
    ("noisy_psnr", "PSNR (dB)", "higher is better"),
    ("noisy_mse", "MSE", "lower is better"),
)


def _parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def _summary_path(results_dir: str, method: str) -> str:
    return os.path.join(results_dir, f"{method}{SUMMARY_SUFFIX}")


def _metadata_path(results_dir: str, method: str) -> str:
    return os.path.join(results_dir, f"{method}{METADATA_SUFFIX}")


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _read_metadata(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_all_present(results_dir: str, methods: list[str]) -> None:
    missing = [path for method in methods if not os.path.isfile(path := _summary_path(results_dir, method))]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Physical robustness plotting waits until every requested model has a result file. "
            f"Missing:\n  {joined}"
        )


def _discover_methods(results_dir: str) -> list[str]:
    root = Path(results_dir)
    methods = []
    for path in sorted(root.glob(f"*{SUMMARY_SUFFIX}")):
        method = path.name[: -len(SUMMARY_SUFFIX)]
        if method:
            methods.append(method)

    builtin = [method for method in BUILTIN_METHOD_ORDER if method in methods]
    extra = sorted(method for method in methods if method not in BUILTIN_METHOD_ORDER)
    discovered = builtin + extra
    if not discovered:
        raise FileNotFoundError(
            f"No physical robustness summaries found in {root}. "
            f"Expected files named <method>{SUMMARY_SUFFIX}."
        )
    return discovered


def _resolve_methods(results_dir: str, methods_arg: str) -> tuple[list[str], bool]:
    requested = str(methods_arg).strip()
    if not requested or requested.lower() in {"auto", "*", "all"}:
        return _discover_methods(results_dir), True
    return _parse_list(requested), False


def _load_results(
    results_dir: str,
    methods: list[str],
    require_all: bool,
) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    if require_all:
        _require_all_present(results_dir, methods)

    all_rows: list[dict] = []
    metadata_by_method: dict[str, dict] = {}
    labels: dict[str, str] = {}

    for method in methods:
        if not os.path.isfile(_summary_path(results_dir, method)):
            continue
        metadata = _read_metadata(_metadata_path(results_dir, method))
        method_rows = _read_csv(_summary_path(results_dir, method))
        row_label = method_rows[0].get("label") if method_rows else None
        metadata_by_method[method] = metadata
        labels[method] = metadata.get("label") or row_label or DEFAULT_LABELS.get(method, method)
        for row in method_rows:
            row = dict(row)
            row["plot_method"] = method
            row["plot_label"] = labels[method]
            all_rows.append(row)

    if not all_rows:
        raise ValueError(f"No rows loaded from {results_dir} for methods: {methods}")
    return all_rows, metadata_by_method, labels


def _sort_key(row: dict) -> tuple:
    return (row["corruption"], int(row["severity_rank"]), row["plot_method"])


def _group_rows(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["corruption"]][row["plot_method"]].append(row)
    for by_method in grouped.values():
        for method_rows in by_method.values():
            method_rows.sort(key=lambda row: int(row["severity_rank"]))
    return grouped


def _level_order(rows: list[dict]) -> list[str]:
    ordered = []
    for row in sorted(rows, key=lambda item: int(item["severity_rank"])):
        if row["level_label"] not in ordered:
            ordered.append(row["level_label"])
    return ordered


def _colors_for_methods(methods: list[str]) -> dict[str, str]:
    colors = {}
    fallback_idx = 0
    for method in methods:
        if method in DEFAULT_COLORS:
            colors[method] = DEFAULT_COLORS[method]
        else:
            colors[method] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    return colors


def _plot(rows: list[dict], methods: list[str], labels: dict[str, str], out_base: str, dpi: int) -> tuple[str, str]:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=_sort_key)
    grouped = _group_rows(rows)
    corruptions = list(grouped.keys())
    colors = _colors_for_methods(methods)
    if not corruptions:
        raise ValueError("No robustness rows to plot.")

    fig, axes = plt.subplots(
        len(METRICS),
        len(corruptions),
        figsize=(5.2 * len(corruptions), 10.5),
        constrained_layout=True,
    )
    if len(corruptions) == 1:
        axes = axes.reshape(len(METRICS), 1)

    for col, corruption in enumerate(corruptions):
        corruption_rows = [row for row in rows if row["corruption"] == corruption]
        levels = _level_order(corruption_rows)
        x_positions = list(range(len(levels)))
        for row_idx, (metric_key, ylabel, note) in enumerate(METRICS):
            ax = axes[row_idx, col]
            for method in methods:
                method_rows = grouped[corruption].get(method, [])
                if not method_rows:
                    continue
                xs = [levels.index(row["level_label"]) for row in method_rows]
                ys = [float(row[metric_key]) for row in method_rows]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2.2,
                    color=colors.get(method),
                    label=labels.get(method, method),
                )
            ax.set_title(f"{corruption}: {ylabel}\n{note}")
            ax.set_xticks(x_positions, levels, rotation=25)
            ax.set_xlabel("corruption level")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02))

    ensure_dir(os.path.dirname(out_base))
    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main(args) -> None:
    methods, discovered = _resolve_methods(args.results_dir, args.methods)
    rows, metadata_by_method, labels = _load_results(args.results_dir, methods, require_all=not discovered)
    ensure_dir(args.out_dir)

    combined_csv = os.path.join(args.out_dir, "physical_robustness_combined_summary.csv")
    fieldnames = list(rows[0].keys())
    _write_csv(combined_csv, rows, fieldnames)

    metadata_json = os.path.join(args.out_dir, "physical_robustness_combined_metadata.json")
    with open(metadata_json, "w") as f:
        json.dump(
            {
                "results_dir": os.path.abspath(args.results_dir),
                "methods": methods,
                "method_mode": "auto" if discovered else "explicit",
                "labels": labels,
                "source_metadata": metadata_by_method,
                "combined_csv": os.path.abspath(combined_csv),
            },
            f,
            indent=2,
        )

    out_base = os.path.join(args.out_dir, "physical_robustness_curves")
    png_path, pdf_path = _plot(rows, methods, labels, out_base=out_base, dpi=args.dpi)

    print("saved:", combined_csv)
    print("saved:", metadata_json)
    print("saved:", png_path)
    print("saved:", pdf_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=os.path.join("outputs", "physical_robustness"))
    ap.add_argument(
        "--methods",
        type=str,
        default="auto",
        help="Comma/space-separated result names to plot, or 'auto' to discover every summary file.",
    )
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "paper", "physical_robustness"))
    ap.add_argument("--dpi", type=int, default=220)
    main(ap.parse_args())
