# Physics-Guided Conditional Flow Matching for Lensless Imaging

This project trains a Conditional Flow Matching (CFM) model to map lensless measurements (y) to lensed images (x),
and uses physics-guided sampling with a known PSF forward model H to enforce data consistency.

Key pieces:
- CFM training: learn $v_\theta(t, x_t, y)$ where $x_t = (1-t)\epsilon + t x$
- Sampling: integrate ODE with Euler steps + data-consistency gradient step using $H^T(Hx - y)$

Dataset:
- DiffuserCam MirFlickr (via LenslessPiCam HuggingFace loader)

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m scripts.train --config configs/default.yaml
# On Google Colab, use the A100 config:
python -m scripts.train --config configs/a100_{base/phys}.yaml
```

## Sample / Visualize

```bash
python -m scripts.sample --config configs/default.yaml --idx 0
```