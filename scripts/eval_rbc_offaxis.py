"""Input-only off-axis reconstruction and optional label-assisted model audit.

Examples:
  python -m scripts.eval_rbc_offaxis --max_samples 256 --diagnose_labels
  python -m scripts.eval_rbc_offaxis --hologram path/to/image.png
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lensless_flow.config import load_config
from lensless_flow.data import HumanRBCHologramDataset
from lensless_flow.holography_offaxis import (
    compensate_background,
    extract_offaxis_field,
    phase_compatibility_diagnostic,
)
from lensless_flow.metrics import psnr, ssim_torch


def metrics(pred: np.ndarray, target: np.ndarray, border: int) -> dict:
    p = torch.from_numpy(np.ascontiguousarray(pred)).float()[None, None]
    t = torch.from_numpy(np.ascontiguousarray(target)).float()[None, None]
    stop = -border if border else None
    interior_p = p[..., border:stop, border:stop]
    interior_t = t[..., border:stop, border:stop]
    return {
        "psnr": float(psnr(p, t)),
        "ssim": float(ssim_torch(p, t)),
        "interior_psnr": float(psnr(interior_p, interior_t)),
        "interior_ssim": float(ssim_torch(interior_p, interior_t)),
    }


def save_grid(path: Path, examples: list, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    diagnostic = "label_assisted_diagnostic" in examples[0]["images"]
    keys = ["hologram", "target", "offaxis_input_only"]
    if diagnostic:
        keys += ["label_assisted_diagnostic"]
    titles = ["Measured hologram", "Phase label", "Input-only / background gauge"]
    if diagnostic:
        titles += ["Label-assisted physics test\nNOT reconstruction performance"]
    fig, axes = plt.subplots(len(examples), len(keys), figsize=(3.1*len(keys), 3.1*len(examples)), squeeze=False, layout="constrained")
    for row, item in enumerate(examples):
        for col, (key, title) in enumerate(zip(keys, titles)):
            axes[row, col].imshow(item["images"][key], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(f'{split.capitalize()} index {item["index"]}')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_spectrum(path: Path, hologram: np.ndarray, field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    h, w = hologram.shape
    spectrum = np.fft.fftshift(np.fft.fft2((hologram-hologram.mean()) * np.outer(np.hanning(h),np.hanning(w))))
    fig, ax = plt.subplots(figsize=(6,5), layout="constrained")
    ax.imshow(np.log1p(abs(spectrum)), origin="lower", extent=(-.5,.5,-.5,.5), cmap="magma")
    cy, cx = field.carrier_yx
    ax.add_patch(Circle((cx,cy),field.bandwidth,fill=False,color="cyan",lw=1.5))
    ax.add_patch(Circle((-cx,-cy),field.bandwidth,fill=False,color="white",lw=1.5,linestyle="--"))
    ax.set(xlabel="Horizontal frequency (cycles/pixel)", ylabel="Vertical frequency (cycles/pixel)", title="Separated off-axis orders; cyan order is extracted")
    fig.savefig(path,dpi=140);plt.close(fig)


def main(args):
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.hologram:
        with Image.open(args.hologram) as im:
            image = np.asarray(im.convert("L"), dtype=np.float64)/255
        field = extract_offaxis_field(image, bandwidth=args.bandwidth)
        result = compensate_background(field, border=args.border)
        np.savez_compressed(out/"reconstruction.npz", phase_png=result["phase_png"], relative_phase_rad=result["relative_phase_rad"], cross_amplitude=result["cross_amplitude"])
        Image.fromarray(np.uint8(np.round(result["phase_png"]*255))).save(out/"phase.png")
        (out/"metadata.json").write_text(json.dumps({k:v for k,v in result.items() if not isinstance(v,np.ndarray)},indent=2),encoding="utf-8")
        save_spectrum(out/"spectrum.png",image,field)
        print(f"Saved input-only reconstruction to {out.resolve()}")
        return

    cfg = load_config(args.config)
    data = cfg["data"]
    if float(data.get("downsample",1)) != 1:
        raise ValueError("This audit expects original sampling; use a config with downsample: 1.")
    dataset = HumanRBCHologramDataset(
        args.data_path or data["path"], split=args.split,
        flip_ud=data.get("flip_ud",False), flip_lr=data.get("flip_lr",False),
        phase_invert=data.get("phase_invert",False), hologram_invert=data.get("hologram_invert",False),
    )
    count = len(dataset) if args.max_samples < 0 else min(args.max_samples,len(dataset))
    if count == 0:
        raise ValueError("max_samples must be positive or -1.")
    if args.selection == "first":
        indices = np.arange(count)
    elif args.selection == "even":
        indices = np.linspace(0,len(dataset)-1,count,dtype=int)
    else:
        indices = np.sort(np.random.default_rng(args.seed).choice(len(dataset),count,replace=False))
    rows, details, examples = [], [], []
    start = time.perf_counter()
    for number, index in enumerate(indices):
        hologram, target = [v.numpy()[...,0] for v in dataset[int(index)]]
        t0 = time.perf_counter()
        field = extract_offaxis_field(hologram,bandwidth=args.bandwidth)
        result = compensate_background(field,border=args.border)
        if data.get("phase_invert",False):
            result["phase_png"] = 1 - result["phase_png"]
        runtime = 1000*(time.perf_counter()-t0)
        predictions = {
            "measurement": hologram,
            "constant_0.5": np.full_like(target,.5),
            "offaxis_input_only": result["phase_png"],
        }
        item = {"index": int(index),"hologram":dataset.pairs[int(index)][0].name,
                "target":dataset.pairs[int(index)][1].name,
                "carrier_fft_yx":list(field.carrier_yx),
                "sideband_peak_fraction":field.sideband_peak_fraction,
                "input_only":{k:v for k,v in result.items() if not isinstance(v,np.ndarray)}}
        if args.diagnose_labels:
            diagnostic = phase_compatibility_diagnostic(field,target,border=args.border)
            predictions["label_assisted_diagnostic"] = diagnostic["phase_png"]
            item["label_assisted"] = {k:v for k,v in diagnostic.items() if not isinstance(v,np.ndarray)}
        for name, pred in predictions.items():
            row = {"index":int(index),"method":name, **metrics(pred,target,args.border)}
            row["runtime_ms"] = runtime if name == "offaxis_input_only" else None
            row["coherence"] = diagnostic["coherence"] if name == "label_assisted_diagnostic" else None
            row["circular_rmse_rad"] = diagnostic["circular_rmse_rad"] if name == "label_assisted_diagnostic" else None
            rows.append(row)
        details.append(item)
        if len(examples) < args.examples:
            examples.append({"index":int(index),"images":{"hologram":hologram,"target":target,**predictions}})
        if number == 0:
            save_spectrum(out/"spectrum.png",hologram,field)
        if args.save_reconstructions:
            np.savez_compressed(out/f"sample_{int(index):05d}.npz",phase_png=result["phase_png"],relative_phase_rad=result["relative_phase_rad"],cross_amplitude=result["cross_amplitude"])
        if (number+1)%32 == 0 or number+1 == count:
            print(f"Completed {number+1}/{count} ({time.perf_counter()-start:.1f}s)",flush=True)

    summary=[]
    for name in predictions:
        selected=[r for r in rows if r["method"]==name]
        item={"method":name,"samples":len(selected)}
        for key in ("psnr","ssim","interior_psnr","interior_ssim","runtime_ms","coherence","circular_rmse_rad"):
            values=[r[key] for r in selected if r[key] is not None]
            item[key]=float(np.mean(values)) if values else None
        summary.append(item)
    with (out/"per_sample.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata={"arguments":vars(args),"dataset_count":len(dataset),"indices":indices.tolist(),
              "data_preprocessing":{k:data.get(k,False) for k in ("flip_ud","flip_lr","phase_invert","hologram_invert")},
              "summary":summary,"details":details,
              "metric_notes":"Arithmetic mean per-image PSNR and repository Gaussian SSIM, data range 1; interior excludes border pixels.",
              "input_only_notes":"No labels enter reconstruction. Background flatness and positive cell optical path choose gauge/sign. PNG uses negative path convention and midpoint background, then applies config phase_invert if requested.",
              "diagnostic_notes":"LABEL-ASSISTED: sign and 3 affine phase parameters fitted per pair. Fixed 2*pi PNG coding. Circular metrics use interior pixels excluded from sparse parameter-fit grid. Image PSNR/SSIM include fitted pixels and are NOT deployment scores.",
              "independence_notes":"Original augmented split has same-acquisition and flipped-crop overlap; this is a supplied-split benchmark, not independent acquisition generalization."}
    (out/"results.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    if examples:
        save_grid(out/"examples.png",examples,args.split)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/rbc_hologram.yaml")
    p.add_argument("--data_path")
    p.add_argument("--hologram",help="Reconstruct one PNG without loading any paired labels.")
    p.add_argument("--split",default="validation",choices=["train","validation","test"])
    p.add_argument("--selection",default="random",choices=["first","random","even"])
    p.add_argument("--max_samples",type=int,default=256)
    p.add_argument("--seed",type=int,default=20260905)
    p.add_argument("--bandwidth",type=float,default=.08,help="Analytic sideband cutoff in cycles/pixel.")
    p.add_argument("--border",type=int,default=16)
    p.add_argument("--examples",type=int,default=4)
    p.add_argument("--diagnose_labels",action="store_true",help="Explicitly run and label the target-assisted physics diagnostic.")
    p.add_argument("--save_reconstructions",action="store_true")
    p.add_argument("--out_dir",default="outputs/rbc_offaxis")
    main(p.parse_args())
