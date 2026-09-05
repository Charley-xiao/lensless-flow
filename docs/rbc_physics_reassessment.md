# RBC physical-model reassessment — 2026-09-05

The acquisition model was wrong. These images are consistent with **off-axis,
in-focus digital holographic microscopy**, and their separated interference
orders recover the labeled cell structure directly. An in-line defocus-distance
sweep cannot represent the tilted reference wave in this experiment.

## Acquisition and model

The dataset matches [Castañeda, Trujillo and Doblas (2024)](https://doi.org/10.1016/j.dib.2024.110424)
and [OSF 8P7BA](https://osf.io/8p7ba/): the archive name, folder structure and
29,491/7,373 pair counts agree. The paper describes telecentric Mach–Zehnder DHM,
a 532 nm source, 40×/0.65 NA objective, 200 mm tube lens and 5.86 µm detector
pixels. Phase reconstruction preceded the cropping of full 1920×1200 frames
into 256×256 patches and geometric augmentation. The local second split is
called `Validation`.

The appropriate image-plane model is

\[
I(x,y)=|O(x,y)+R(x,y)|^2,
\quad O=Ae^{i\phi},
\quad R=B e^{i(2\pi(f_xx+f_yy)+\delta)}.
\]

The spectrum contains a central intensity term and conjugate cross terms
`O R*` and `O* R`. Isolating one cross term yields a complex field containing
both amplitude and phase. The remaining reference contributes a phase ramp,
piston and conjugation convention. No defocus propagation is needed for this
in-focus baseline. Automatic carrier compensation is also the principle used
by the authors' [Trujillo method](https://doi.org/10.1364/AO.55.010299) and
[pyDHM phase-compensation routines](https://catrujilla.github.io/pyDHM/).

The first local validation hologram has a clear sideband near
`(fy, fx) = (-61, -75)/256` cycles/pixel. The observed diagonal fringes and
separated sidebands support this model independently of label fitting.

The old `exp(i*phi)` propagation model already contains its transmitted
uniform background. Merely adding another uniform reference to that complete
field would double count it; the missing component here is the experiment's
separate **tilted** reference. The old experiment also imposed unit object
amplitude against raw PNG intensity with mean well below one, and used a
different normalization in its forward diagnostic. Its dual update uses the
pre-update object residual. The legacy code is retained as found; historical
ADMM numbers are not a validated implementation of this acquisition model.

## Implementation and assumptions

`lensless_flow/holography_offaxis.py` provides analytic circular Fourier-order
extraction, input-only background compensation, and a separately named
label-assisted phase-compatibility diagnostic. `scripts/eval_rbc_offaxis.py`
supports supplied-split evaluation and standalone PNG inference. Existing RBC
flow training/inference code remains usable.

The aperture has a raised-cosine edge with width 0.015 cycles/pixel and a
cutoff of 0.08 cycles/pixel. This scalar cutoff was chosen from a preliminary
16-image training audit comparing 0.08, 0.11 and 0.14; it was frozen for the
validation experiments below. A conservative separation check rejects
apertures that risk order overlap or Nyquist crossing. No transfer-function
coefficients, neural weights, or paired-data image operators are fitted.

Input-only reconstruction estimates two carrier-tilt corrections by maximizing
circular phase concentration, sets a background phase origin, and uses the
skew of relative unwrapped phase to select a positive optical-path convention
for RBCs. The exported PNG uses the empirically observed negative-path label
convention with background at approximately 0.5. These are explicit priors;
dense crops, insufficient background or unwrapping errors can violate them.
The NPZ export retains relative unwrapped phase and cross-term amplitude.
That amplitude is proportional to `A*B`, not independently calibrated
absorption, and relative phase is not an absolute thickness measurement.

PNG-to-radian metadata were not found in the published dataset description.
Local evidence strongly supports a `2*pi*PNG` circular encoding: on 16
training pairs, allowing an extra scale parameter gave a scale of
1.0022 ± 0.0158 relative to that assumption, with negligible improvement in
circular agreement. This is an empirical check, not recovered export metadata.

## Evaluation

Metrics use the repository's Gaussian SSIM, data range 1, and arithmetic mean
of per-image PSNR. Interior metrics exclude a 16-pixel border. All input-only
predictions are computed without passing labels to the reconstruction API.

The first 32 validation images reproduce the old experiment's sample order:

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Raw measurement | 11.01 | 0.2334 |
| Constant 0.5 | 14.10 | 0.7115 |
| Historical ADMM, 100 µm | 13.10 | 0.1870 |
| Off-axis, input-only | **14.44** | **0.8194** |

The flat baseline exposes why the old ADMM scores were misleading: it does
better without reconstructing any cell. Off-axis demodulation provides a
substantial SSIM increase and visibly recovers cells, but only a modest raw
PSNR improvement over this baseline.

On 256 randomly selected validation pairs (seed 20260905), with the same
frozen cutoff:

| Method | PSNR (dB) | SSIM | Interior PSNR | Interior SSIM |
|---|---:|---:|---:|---:|
| Raw measurement | 9.57 | 0.1365 | 9.59 | 0.1400 |
| Constant 0.5 | 12.54 | 0.6440 | 12.57 | 0.6446 |
| Off-axis, input-only | **12.75** | **0.7406** | **13.04** | **0.7514** |

### Label-assisted model test — not reconstruction performance

To test whether the recovered cross-term phase explains the labels, the
diagnostic fits only conjugation and three affine phase parameters (piston,
x tilt, y tilt). All fitting, including its initializer, uses a sparse
every-fourth-pixel interior grid. Circular error is evaluated on the remaining
interior pixels. These are disjoint pixels within the same crop, not
independent acquisitions. Image PSNR/SSIM include fitting pixels and must
never be presented as input-only results.

| Label-assisted diagnostic | First 32 | Random 256 |
|---|---:|---:|
| Circular coherence, disjoint evaluation pixels | 0.9969 | 0.9953 |
| Circular RMSE, radians | 0.0779 | 0.0907 |
| Full PNG PSNR, dB | 23.41 | 21.89 |
| Full PNG SSIM | 0.9298 | 0.9177 |

This is strong evidence that the **off-axis cross-term phase** explains the
paired phase structure. The gap to input-only scores identifies carrier/gauge
estimation and crop-boundary effects as important remaining limitations.
It does not prove a calibrated phase-to-intensity simulator: object amplitude,
reference amplitude and detector response are not supplied with the labels.
No measurement prediction derived from those labels alone is claimed.

The black/white wrap boundaries are particularly sensitive to an offset:
small circular phase errors can move pixels across the PNG's 0/1 boundary.
Thus the higher label-assisted PNG score does not indicate that a trainable
operator was needed; it shows the importance of reproducing the label's
reference convention. Exact original phase origins cannot generally be
recovered from a crop without a reference convention or additional metadata.

## Data split limitation

A local filename audit found training siblings for **7,088/7,373 (96.1%)**
validation holograms after removing terminal `-A`/`-B` augmentation suffixes.
The first ten checked sibling cases match training inputs exactly under
flips. All validation files share apparent acquisition timestamps with the
training split (192 training acquisition groups). The complete 7,088 cases
were not all compared pixel by pixel.

Consequently these supplied-split numbers are useful for paired-data model
compatibility but do not establish independent-acquisition generalization.
Future flow and physics comparisons should split acquisition groups before
augmentation. This audit did not modify the dataset split or remote training.

## Reproduce and inspect

Run from the repository using the supplied py312 environment:

```powershell
$rbcPython = 'C:\Users\12181\miniconda3\envs\py312\python.exe'
& $rbcPython -m scripts.eval_rbc_offaxis --max_samples 32 --selection first --diagnose_labels --out_dir outputs/rbc_offaxis_first32
& $rbcPython -m scripts.eval_rbc_offaxis --max_samples 256 --selection random --diagnose_labels --out_dir outputs/rbc_offaxis_random256
& $rbcPython -m scripts.eval_rbc_offaxis --hologram 'E:\RBCs Holograms\Holograms\Validation\Image__2021-01-28__15-49-02_1-11-B.png' --out_dir outputs/rbc_offaxis_single
& $rbcPython -m unittest discover -s tests -p test_holography_offaxis.py -v
```

Local outputs (ignored by Git; evaluation commands regenerate them):

- [First-32 visual grid](../outputs/rbc_offaxis_first32/examples.png),
  [metrics and parameters](../outputs/rbc_offaxis_first32/results.json).
- [Random-256 visual grid](../outputs/rbc_offaxis_random256/examples.png),
  [metrics and parameters](../outputs/rbc_offaxis_random256/results.json).
- [Fourier-sideband view](../outputs/rbc_offaxis_first32/spectrum.png).
- [Training encoding audit](../outputs/rbc_phase_audit/audit.json).
- [Split-overlap audit](../outputs/rbc_offaxis_split_audit.json).

Seven synthetic physical tests pass, including fractional carriers, amplitude
modulation, rotations/flips, intensity gain/offset behavior, and mismatched
phase labels. A regression ensures changes to excluded diagnostic labels
cannot change the fitted prediction. On the synthetic wrong/shifted-label
controls, coherence falls below 0.48 while the matched case exceeds 0.9999.
Standalone inference and Python compilation were also checked.

The next useful physical calibration would use a blank-reference or original
full-frame recording to establish carrier and phase gauge. Until then, the
analytic complex field is a defensible physics baseline and potential input
to a reconstruction model; a guessed in-line propagation loss is not.
