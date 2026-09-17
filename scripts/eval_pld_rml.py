"""Evaluate an RML flow checkpoint on validation or test, using the common GT crop."""
import argparse
import json
from pathlib import Path

import torch

from lensless_flow.config import load_config
from lensless_flow.data import make_dataloader
from lensless_flow.model_factory import build_flow_model, load_checkpoint_state_dict
from scripts.train import data_loader_kwargs, quick_eval


def main(args):
    cfg = load_config(args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Architecture/sampling/loss geometry come from the trained checkpoint;
    # only the local dataset path and hardware are taken from the supplied config.
    saved_cfg = state["cfg"]
    saved_cfg["data"]["path"] = args.data_root or cfg["data"]["path"]
    saved_cfg["device"] = args.device or cfg["device"]
    cfg = saved_cfg
    if cfg["data"]["dataset"] != "pld_rml" or cfg["eval"]["protocol"] != "pld_rml_common_gt":
        raise ValueError("Expected an RML checkpoint with the common GT scoring protocol")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg["eval"]["preview_dir"] = str(out / "previews")
    cfg["data"]["max_samples"] = args.max_samples
    _, loader = make_dataloader(split=args.split, downsample=1, flip_ud=False, batch_size=1,
                                num_workers=0, path=cfg["data"]["path"], **data_loader_kwargs(cfg))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_flow_model(cfg, img_channels=3, im_hw=(300, 480), device=device, checkpoint_state=state)
    load_checkpoint_state_dict(model, state)
    metrics = quick_eval(model, None, loader, cfg, device, max_batches=len(loader),
                         pred_type=cfg["train"]["mode"], epoch=state.get("epoch"))
    metrics = {key: (value if torch.isfinite(torch.tensor(value)) else None) for key, value in metrics.items()}
    record = {"checkpoint": str(Path(args.checkpoint).resolve()), "epoch": state.get("epoch"),
              "split": args.split, "protocol": cfg["eval"]["protocol"],
              "subset": args.max_samples is not None, "ssim": "repository Gaussian 11x11 sigma=1.5, zero padding",
              "steps": cfg["sample"]["steps"], "solver": cfg["sample"]["solver"], **metrics}
    (out / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--data_root", help="Override only the dataset location")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--out_dir", default="outputs/pld_rml_evaluation")
    main(parser.parse_args())
