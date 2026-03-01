# Physics-Guided Conditional Flow Matching for Lensless Imaging

This project trains a Conditional Flow Matching (CFM) model to map lensless measurements (y) to lensed images (x),
and uses physics-guided sampling with a known PSF forward model H to enforce data consistency.

Key pieces:
- CFM training: learn $v_\theta(t, x_t, y)$ where $x_t = (1-t)\epsilon + t x$
- Sampling: integrate ODE with Euler steps + data-consistency gradient step using $H^T(Hx - y)$

Dataset:
- DiffuserCam MirFlickr (via LenslessPiCam HuggingFace loader)

> [!CAUTION]
> There's still something wrong with the PSF guidance. Need to revise the math and implementation.
> For the time being, set physics.dc_steps to 0 to disable the data-consistency steps during sampling.

## Install

> [!CAUTION]
> The code does not require Python 3.11 anymore. Python 3.12 also works.

### Locally

Install Python 3.11 first and then:

```bash
pip install -r requirements.txt
```

### On a cluster

```bash
conda create -n py312 python=3.12 -y
conda activate py312
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<...>
hf download bezzam/DiffuserCam-Lensless-Mirflickr-Dataset --repo-type dataset --local-dir <...>
# Then go to configs/a100_base.yaml and set data.path to the local path of the downloaded dataset,
# and set wandb.log_artifacts to false.
wandb login --relogin
wandb offline
```

## Train

```bash
python -m scripts.train --config configs/base.yaml
# On Google Colab, use the A100 config:
python -m scripts.train --config configs/a100_base.yaml
```

```bash
python -m scripts.test_psf_alignment --config configs/base.yaml --idx 0 --save_dir debug_psf
```

## Sample / Visualize

```bash
python -m scripts.sample --config configs/base.yaml --ckpt checkpoints/cfm_lensless_epoch10.pt --idx 0 --steps 5,10,20,30,50 --cols 4

```