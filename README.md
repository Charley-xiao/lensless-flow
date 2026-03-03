# Physics-Guided Conditional Flow Matching for Lensless Imaging

This project trains a Conditional Flow Matching (CFM) model to map lensless measurements (y) to lensed images (x),
and uses physics-guided sampling with a known PSF forward model H to enforce data consistency.

Key pieces:
- CFM training: learn $v_\theta(t, x_t, y)$ where $x_t = (1-t)\epsilon + t x$
- Sampling: integrate ODE with Euler steps + data-consistency gradient step using $H^T(Hx - y)$

Dataset:
- DiffuserCam MirFlickr (via LenslessPiCam HuggingFace loader)

Image shape: torch.Size([1, 3, 67, 120])
Training set size: 24000
Testing set size: 999

Param count: 14M

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
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

## Sample / Visualize

```bash
python -m scripts.sample --config configs/base.yaml --ckpt checkpoints/cfm_lensless_epoch10.pt --idx 0 --steps 5,10,20,30,50 --cols 4
python -m scripts.eval --config configs/base.yaml --ckpt ...
```

## Experiments

1. Train vanilla v-prediction CFM and x-prediction CFM, sample with and without physics guidance, report SSIM, PSNR, and RMSE metrics.
2. Compare with baselines such as ADMM and supervised U-Net.
3. Visualize SSIM vs number of sampling steps for with and without physics guidance, for both v-prediction and x-prediction models.
4. Visualize the effect of the physics guidance by plotting the intermediate reconstructions at different sampling steps, with and without guidance.
5. Ablation: vary the number of steps of the physics guidance and see how it affects the reconstruction quality.

```bash
python -m rqs.ssim_vs_k --config configs/a100_base.yaml --ckpt_vanilla checkpoints/vanilla.pt --ckpt_btb checkpoints/btb.pt --steps 1,5,10,20,30,40,50,60 --max_batches -1 --out outputs/ssim_vs_k.png
python -m rqs.ssim_vs_dc --config configs/a100_base.yaml --ckpt checkpoints/vanilla.pt --steps 40 --dc_steps 0,1,2,3,5,8,10 --max_batches -1 --out outputs/ssim_vs_dcsteps.png
```

```
   1  V_off=0.056797  V_on=0.056797  B_off=0.062834  B_on=0.062595
   5  V_off=0.354060  V_on=0.354649  B_off=0.341806  B_on=0.341521
  10  V_off=0.539874  V_on=0.534933  B_off=0.488173  B_on=0.487254
  20  V_off=0.709512  V_on=0.705812  B_off=0.578574  B_on=0.578247
  30  V_off=0.770059  V_on=0.765754  B_off=0.783751  B_on=0.781745
  40  V_off=0.794702  V_on=0.789456  B_off=0.809604  B_on=0.807175
  50  V_off=0.806247  V_on=0.800135  B_off=0.828759  B_on=0.826137
  60  V_off=0.811477  V_on=0.804530  B_off=0.836306  B_on=0.832553
```