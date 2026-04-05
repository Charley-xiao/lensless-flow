import argparse

import torch
import yaml

from lensless_flow.data import make_dataloader
from lensless_flow.physics import FFTLinearConvOperator
from lensless_flow.tensor_utils import to_nchw


def _peak_location(x: torch.Tensor) -> tuple[int, int]:
    flat_idx = int(x.reshape(-1).argmax().item())
    width = int(x.shape[-1])
    return flat_idx // width, flat_idx % width


@torch.no_grad()
def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    test_ds, _ = make_dataloader(
        split="test",
        downsample=cfg["data"]["downsample"],
        flip_ud=cfg["data"]["flip_ud"],
        batch_size=1,
        num_workers=0,
        path=cfg["data"].get("path", None),
    )

    y0, _ = test_ds[0]
    y0 = to_nchw(y0).to(device)
    psf = to_nchw(test_ds.psf).to(device)
    im_hw = (int(y0.shape[-2]), int(y0.shape[-1]))
    channels = int(y0.shape[1])

    Hop = FFTLinearConvOperator(psf=psf, im_hw=im_hw).to(device)

    print("===== H Operator Check =====")
    print(f"device: {device}")
    print(f"im_hw: {im_hw}")
    # print(f"padded_hw: {Hop.padded_hw}")
    # print(f"psf_hw: {Hop.psf_hw}")
    # print(f"channels: {channels}")
    print("----------------------------")

    max_rel_err = 0.0
    for trial in range(int(args.trials)):
        seed = int(args.seed + trial)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x_rand = torch.randn((1, channels, *im_hw), device=device, dtype=y0.dtype)
        y_rand = torch.randn((1, channels, *im_hw), device=device, dtype=y0.dtype)

        Ax = Hop.forward(x_rand)
        Aty = Hop.adjoint(y_rand)

        lhs = torch.vdot(Ax.reshape(-1), y_rand.reshape(-1))
        rhs = torch.vdot(x_rand.reshape(-1), Aty.reshape(-1))
        rel_err = float((lhs - rhs).abs().item() / (lhs.abs().item() + 1e-12))
        max_rel_err = max(max_rel_err, rel_err)

        print(
            f"trial {trial:02d} | seed {seed:6d} | "
            f"<Ax,y>={lhs.item(): .6e} | <x,A*y>={rhs.item(): .6e} | rel_err={rel_err:.3e}"
        )

    delta = torch.zeros((1, channels, *im_hw), device=device, dtype=y0.dtype)
    center_y = im_hw[0] // 2
    center_x = im_hw[1] // 2
    delta[:, :, center_y, center_x] = 1.0
    y_delta = Hop.forward(delta)
    y_mag = y_delta.abs().sum(dim=1)
    peak_y, peak_x = _peak_location(y_mag[0])

    print("----------------------------")
    print(f"delta peak location in measurement: ({peak_y}, {peak_x})")
    print(f"measurement center: ({center_y}, {center_x})")
    print(f"max_relative_error: {max_rel_err:.3e}")
    print("============================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
