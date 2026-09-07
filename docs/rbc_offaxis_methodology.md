# Analytic Phase Reconstruction from Off-Axis Red Blood Cell Holograms

## Image formation, Fourier demodulation, reference compensation, and experimental validation

**Implementation-aligned methodological manuscript**  
**Date:** September 5, 2026  
**Repository:** `D:/Courses/EECS298/flow`  
**Implementation:** `lensless_flow/holography_offaxis.py` and `scripts/eval_rbc_offaxis.py`

## Abstract

We describe the physical model and numerical procedure used to reconstruct phase images from the paired red blood cell (RBC) hologram dataset available in this repository. The acquisition is consistent with an off-axis, in-focus digital holographic microscope. In this geometry, a tilted reference beam interferes with the object field and moves its complex information into separated Fourier sidebands. Reconstruction therefore consists of isolating one sideband, shifting it to zero frequency, and estimating the remaining reference phase. We implement an analytic Fourier filter followed by a two-parameter carrier correction, a phase-origin convention, and a heuristic selection of the conjugate phase sign. No image-valued transfer function or neural reconstruction operator is learned. A second, explicitly label-assisted procedure tests whether the extracted phase agrees with the paired target after allowing only conjugation and an affine phase correction. On 256 selected validation pairs, input-only reconstruction achieves 12.75 dB PSNR and 0.7406 SSIM; the label-assisted diagnostic achieves 0.9953 circular coherence on pixels excluded from parameter fitting. These quantities address different questions and are reported separately. The results support the off-axis image-formation model while identifying reference conventions, finite cropping, and background assumptions as limitations of the current input-only reconstruction. We provide the mathematical derivation, the implemented objectives, pseudocode, evaluation definitions, and commands needed to reproduce the analysis.

**Keywords:** digital holographic microscopy; red blood cells; Fourier demodulation; phase reconstruction; carrier compensation; circular statistics.

## 1. Introduction

A grayscale hologram is not a grayscale phase image. Its pixel values measure optical intensity, whereas the desired phase map describes the delay of a complex optical field. Recovering one from the other requires a model of how the optical instrument encodes phase into intensity.

The key observation in this dataset is the diagonal fringe pattern extending across the image. These fringes arise from interference between the object field and a tilted reference field. Locally, the phase of the object shifts the fringes, while the amplitudes of the two fields affect their contrast. Fourier analysis separates this modulated information from the slowly varying intensity background. Once one modulation component has been isolated, its complex argument supplies phase information.

This manuscript explains the method that was actually implemented. It is an application of established off-axis holographic reconstruction principles, with a specific background-compensation objective and evaluation protocol. It is not a claim that Fourier sideband reconstruction is a new algorithm. Reference compensation in telecentric microscopy is described by [Trujillo et al. (2016)](https://doi.org/10.1364/AO.55.010299), and a broader collection of reconstruction approaches is available in [pyDHM](https://doi.org/10.1371/journal.pone.0275818). Our implementation does not invoke pyDHM's full-ROI or efficient-ROI search routines; the optimizer used here is derived explicitly below.

There are two distinct procedures throughout this document:

1. **Input-only reconstruction:** accepts a hologram and produces a phase image without access to its paired target.
2. **Label-assisted model diagnosis:** accepts the extracted complex field and the paired target, then determines whether a small reference correction makes their phases compatible.

The first procedure is usable on a new input image. The second is an explanatory experiment. A high score from the second procedure must not be interpreted as the reconstruction accuracy of the first.

## 2. Dataset and experimental setting

### 2.1. Acquisition evidence

The local archive matches the dataset described by [Castañeda, Trujillo, and Doblas (2024)](https://doi.org/10.1016/j.dib.2024.110424) and distributed through [OSF](https://osf.io/8p7ba/). The publication specifies an off-axis, telecentric Mach–Zehnder microscope, a 532 nm source, a 40×/0.65 NA objective, a 200 mm tube lens, and a camera with 5.86 µm pixels. Reconstruction was performed on 1920 × 1200 recordings before cropping and geometric augmentation. These acquisition details motivate the image-plane interference model below; they are not propagation parameters fitted by our method.

The local dataset contains 29,491 training pairs and 7,373 validation pairs. Both inputs and targets are 256 × 256 grayscale PNG images. We use the original spatial sampling, convert 8-bit values to floating point by division by 255, and pair files through the repository's existing filename-matching rule. The standard configuration has no image inversion or geometric flips applied by the loader.

The targets are computationally reconstructed phase images, not independent direct measurements of physical cell thickness. Agreement with these targets demonstrates consistency with their reconstruction and encoding conventions.

### 2.2. Notation

| Symbol | Meaning |
|---|---|
| $H,W$ | Image height and width in pixels |
| $\mathbf r=(x,y)$ | Pixel coordinates, with $x$ horizontal and $y$ vertical |
| $I[\mathbf r]$ | Recorded hologram intensity after division by 255 |
| $P[\mathbf r]\in[0,1]$ | Paired target PNG |
| $O=Ae^{i\phi}$ | Effective complex object field at the image plane |
| $R$ | Reference field |
| $\mathbf f_r$ | Reference carrier, in cycles per pixel |
| $\mathbf c$ | Carrier of the Fourier order selected by the algorithm |
| $C$ | Complex field after sideband extraction and coarse demodulation |
| $b,\tau$ | Low-pass cutoff radius and taper width, in cycles per pixel |
| $\operatorname{wrap}(u)$ | $\operatorname{Arg}(e^{iu})$, with principal values approximately in $[-\pi,\pi]$ |
| $\mathbf q$ | Image-centered coordinates normalized by image width and height |
| $\Omega$ | Interior image region after excluding a border |
| $S_2,S_4$ | Every-second-pixel and every-fourth-pixel grids inside $\Omega$ |

The code stores frequency pairs in the order `(fy, fx)` and NumPy image arrays are indexed as `[y, x]`. Equations below write functions in the coordinate order $(x,y)$ and use the vector order $(f_x,f_y)$. These are storage conventions, not different physical quantities.

## 3. Image-formation model

### 3.1. Interference of object and reference fields

Let the effective image-plane object field be

$$
O(\mathbf r)=A(\mathbf r)e^{i\phi(\mathbf r)},
\tag{1}
$$

and approximate the reference over one crop by a plane wave with constant amplitude $B$:

$$
R(\mathbf r)=B\exp\!\left[i\left(2\pi\mathbf f_r\cdot\mathbf r+\delta\right)\right].
\tag{2}
$$

Here, $\delta$ is a constant reference phase. The detector measures the squared magnitude of the sum:

$$
I(\mathbf r)=|O(\mathbf r)+R(\mathbf r)|^2
=A^2(\mathbf r)+B^2
+2A(\mathbf r)B\cos\!\left[\phi(\mathbf r)-2\pi\mathbf f_r\cdot\mathbf r-\delta\right].
\tag{3}
$$

Equation (3) explains the main visual features. The term $A^2+B^2$ supplies the intensity background. The cosine produces the carrier fringes. The object phase $\phi$ changes their local phase, so it is encoded in fringe displacement rather than directly in pixel brightness.

A more complete detector description could include a positive gain $g$, an offset $d$, and noise $\eta$:

$$
I_{\mathrm{PNG}}=g|O+R|^2+d+\eta.
\tag{4}
$$

We do not fit these quantities. For an ideally isolated sideband, a constant positive gain rescales its amplitude and a constant offset remains in the central order. Their values are consequently unnecessary for the phase of that sideband. Spatially varying illumination, nonlinear intensity encoding, clipping, and finite-crop leakage are more complicated and are not corrected by this argument.

The field $O$ is the field produced by the microscope at the image plane. The objective's coherent imaging response is already implicit in it. The method does not deconvolve the microscope pupil or establish that every reconstructed phase value is an undistorted specimen-plane optical path measurement.

### 3.2. Why an in-line propagation model was insufficient

The earlier experiment used a phase-only object and free-space propagation:

$$
U_0=e^{i\phi},\qquad U_z=\mathcal P_zU_0,\qquad I=|U_z|^2.
\tag{5}
$$

That model can describe an appropriate in-line propagation experiment, but it does not represent the separate tilted reference beam of Equation (2). Changing $z$ cannot supply the missing acquisition geometry.

There is also no requirement here that $A=1$. Sideband extraction retains the measured cross-term amplitude, allowing object amplitude variation. A different optimization algorithm applied to Equation (5) would not, by itself, correct the image-formation mismatch.

The present implementation does not estimate a defocus distance or use angular-spectrum propagation. This choice concerns the stated in-focus acquisition; propagation can be useful in other geometries or when numerical refocusing is actually required.

### 3.3. How one Fourier sideband contains a complex field

Define

$$
D=A^2+B^2,\qquad Q=ABe^{i(\phi-\delta)}.
$$

Then Equation (3) becomes

$$
I(\mathbf r)=D(\mathbf r)
+Q(\mathbf r)e^{-i2\pi\mathbf f_r\cdot\mathbf r}
+Q^*(\mathbf r)e^{+i2\pi\mathbf f_r\cdot\mathbf r}.
\tag{6}
$$

Using the Fourier transform convention

$$
\widehat u(\mathbf f)
=\sum_{x=0}^{W-1}\sum_{y=0}^{H-1}
u[x,y]e^{-i2\pi(f_xx+f_yy)},
\tag{7}
$$

the modulation factors translate the spectra:

$$
\widehat I(\mathbf f)
=\widehat D(\mathbf f)
+\widehat Q(\mathbf f+\mathbf f_r)
+\widehat{Q^*}(\mathbf f-\mathbf f_r).
\tag{8}
$$

Thus there is a central order and two conjugate sidebands. Selecting the order centered at $-\mathbf f_r$ and removing its carrier gives approximately $Q$. Selecting the order centered at $+\mathbf f_r$ gives approximately $Q^*$. Their phases have opposite signs:

$$
\operatorname{Arg}Q=\operatorname{wrap}(\phi-\delta),
\qquad
\operatorname{Arg}Q^*=\operatorname{wrap}(-\phi+\delta).
\tag{9}
$$

This is the central reason a single off-axis intensity image can reveal complex-field information: the known form of the reference separates the two complex modulation components in frequency. No iterative search over a full phase image is needed once a usable sideband has been isolated.

In intuitive terms, the reference shifts the object's information to a recognizable frequency location. Fourier filtering selects that information, and demodulation shifts it back.

## 4. Fourier sideband extraction

### 4.1. Coarse carrier detection

The discrete frequency grids are generated with `numpy.fft.fftfreq`. Their units are cycles per pixel; for $H=W=256$, adjacent frequency samples are separated by $1/256$ cycles per pixel.

Carrier detection uses a mean-subtracted, Hann-windowed hologram:

$$
I_{\mathrm{det}}[x,y]
=\left(I[x,y]-\overline I\right)h_W[x]h_H[y],
\qquad
S=|\mathcal F\{I_{\mathrm{det}}\}|^2,
\tag{10}
$$

where, for example, $h_W[x]=\tfrac12[1-\cos(2\pi x/(W-1))]$. Windowing reduces the effect of the rectangular crop boundary on peak detection.

The candidate set is

$$
\mathcal A=\left\{(f_x,f_y):f_y<0,\ \sqrt{f_x^2+f_y^2}>0.16\right\},
$$

and the selected carrier is the strongest Fourier bin in this set:

$$
\mathbf c=\underset{\mathbf f\in\mathcal A}{\operatorname{arg\,max}}\ S(\mathbf f).
\tag{11}
$$

The negative-$f_y$ restriction selects one of the two conjugate orders consistently in array coordinates. It does not guarantee a consistent physical phase sign after an image has been flipped or rotated. This sign ambiguity is handled separately in Section 5.4.

For the first local validation image, the selected carrier in code ordering is

$$
(c_y,c_x)=\left(-\frac{61}{256},-\frac{75}{256}\right)
\approx(-0.23828,-0.29297).
$$

![Measured spectrum and selected sideband](D:/Courses/EECS298/flow/outputs/rbc_offaxis_first32/spectrum.png)

*Figure 1. Fourier magnitude on a logarithmic display scale. The cyan circle marks the extracted order; the dashed circle marks its conjugate. The displayed spectrum is computed from the detection window, while reconstruction uses the original intensity array.*

The implementation can alternatively accept an explicit carrier. The experiments reported here use automatic detection. A constant image is rejected. The stored peak-power fraction is descriptive metadata; it is not a calibrated confidence score or a comprehensive rejection test for non-holographic inputs.

### 4.2. Demodulation and analytic low-pass filtering

After selecting $\mathbf c$, the original, unwindowed intensity is demodulated:

$$
I_{\mathrm{demod}}[x,y]
=I[x,y]\exp[-i2\pi(c_xx+c_yy)].
\tag{12}
$$

This moves the selected Fourier order to approximately zero frequency. A circular low-pass aperture then isolates it:

$$
C=\mathcal F^{-1}\!\left\{
H_b(\mathbf f)\,\mathcal F\{I_{\mathrm{demod}}\}
\right\}.
\tag{13}
$$

The aperture is completely specified by its cutoff $b$ and taper width $\tau$. Writing $\rho=\sqrt{f_x^2+f_y^2}$,

$$
H_b(\rho)=
\begin{cases}
1, & 0\leq\rho\leq b-\tau,\\
\dfrac12\left[1+\cos\!\left(\dfrac{\pi(\rho-b+\tau)}{\tau}\right)\right],
& b-\tau<\rho<b,\\
0, & \rho\geq b.
\end{cases}
\tag{14}
$$

The experiments use $b=0.08$ and $\tau=0.015$ cycles per pixel. Consequently, frequencies below 0.065 pass unchanged, while the band from 0.065 to 0.08 is smoothly attenuated. The taper reduces the ringing associated with a hard circular cutoff.

The filter is a fixed analytic function. We do not estimate a separate coefficient at each frequency. The cutoff was chosen through the preliminary training audit described in Section 8.1; it is the only data-selected filter parameter in the reported validation experiments.

The resulting $C$ approximates a band-limited version of either $Q$ or $Q^*$, with residual phase tilt if the coarse carrier is imperfect. In particular, $|C|$ estimates a filtered cross-term amplitude proportional to $AB$, not an independently calibrated object absorption map. The phase of a filtered complex field is also not, in general, the same as filtering the underlying phase image directly.

### 4.3. Separation checks and finite-crop effects

The code requires

$$
b<\min\!\left(
\frac{\|\mathbf c\|_2}{3},
\frac12-|c_x|,
\frac12-|c_y|
\right).
\tag{15}
$$

The first condition is motivated by support geometry: if the complex object field has bandwidth at most $b$, its intensity can occupy bandwidth up to $2b$. A sideband of radius $b$ then needs approximately $3b$ separation from the origin. The other two conditions keep the circular aperture inside the componentwise Nyquist limits.

This is a conservative geometric check, not a proof that unknown spectral tails do not overlap. It is only as justified as the assumed spectral support of the actual fields.

The FFT operates on a finite crop and implicitly treats it as periodic. A cell crossing a boundary can therefore create leakage and ringing. The current code does not mosaic neighboring crops, pad with a measured background, or apply flat-field correction. Windowing is used for detection only. Constant-offset invariance is exact for an isolated integer-bin carrier on an ideal periodic grid; explicitly supplied fractional carriers can spread an offset through finite-window leakage.

## 5. Input-only reference compensation

### 5.1. What remains after filtering

A useful approximate description of the extracted phase is

$$
\operatorname{Arg}C(\mathbf r)
\approx\operatorname{wrap}\!\left[
s\phi(\mathbf r)+\mathbf a_{\mathrm{res}}^T\mathbf q(\mathbf r)+\delta_{\mathrm{res}}
\right],\qquad s\in\{-1,+1\},
\tag{16}
$$

where

$$
\mathbf q(x,y)=
\begin{pmatrix}
(x-W/2)/W\\
(y-H/2)/H
\end{pmatrix}.
\tag{17}
$$

The two entries of $\mathbf a_{\mathrm{res}}$ describe a residual phase ramp in radians per image width or height. The constant $\delta_{\mathrm{res}}$ is a phase offset, often called a *piston*. The sign $s$ depends on which conjugate order was selected.

The image supplies phase structure, but interpreting a slowly varying phase ramp as specimen structure or reference error requires an additional assumption. We assume that much of the crop has approximately constant background phase. We do not explicitly segment background pixels. The optimizer described next operates on a uniformly sampled interior grid and uses that assumption implicitly.

### 5.2. Two-parameter carrier refinement

First normalize the complex field:

$$
c(\mathbf r)=\frac{C(\mathbf r)}{\max(|C(\mathbf r)|,10^{-12})}.
\tag{18}
$$

Apart from near-zero amplitudes, these values lie on the unit circle. This normalization makes the objective phase-based rather than amplitude-weighted. Low-amplitude pixels are not otherwise masked or assigned a noise model.

Let $\Omega$ exclude 16 pixels on each border. For a 256 × 256 image,

$$
\Omega=\{16,17,\ldots,239\}^2,
\qquad
S_2=\{16,18,\ldots,238\}^2.
$$

The fitting set contains $|S_2|=12{,}544$ pixels. For a candidate tilt $\mathbf a=(a_x,a_y)^T$, define

$$
m(\mathbf a)=\frac{1}{|S_2|}
\sum_{\mathbf r\in S_2}
c(\mathbf r)e^{-i\mathbf a^T\mathbf q(\mathbf r)}.
\tag{19}
$$

We estimate the tilt by maximizing the length of this mean phasor:

$$
\widehat{\mathbf a}
=\underset{\mathbf a\in\mathbb R^2}{\operatorname{arg\,min}}
\left[-|m(\mathbf a)|^2\right].
\tag{20}
$$

The interpretation is straightforward. A residual reference ramp rotates the phase progressively across the crop, causing complex vectors from different background locations to cancel. Removing the correct ramp makes these vectors point in more similar directions, increasing $|m|$. The use of complex exponentials makes the objective insensitive to arbitrary $2\pi$ wrapping jumps.

An equivalent interpretation is a unit-circle least-squares fit. For unit-magnitude $c$, eliminating a constant phase $p$ from

$$
\min_{\mathbf a,p}\frac{1}{|S_2|}
\sum_{\mathbf r\in S_2}
\left|c(\mathbf r)e^{-i\mathbf a^T\mathbf q(\mathbf r)}-e^{ip}\right|^2
$$

gives the profiled objective $2-2|m(\mathbf a)|$. This has the same minimizers as Equation (20). The method is therefore fitting an approximately planar background phase, expressed in circular rather than ordinary linear coordinates.

The numerical initialization uses adjacent-pixel phase differences:

$$
g_x[x,y]=\operatorname{Arg}\!\left(c[x+1,y]c^*[x,y]\right),
\qquad
g_y[x,y]=\operatorname{Arg}\!\left(c[x,y+1]c^*[x,y]\right),
$$

$$
a_x^{(0)}=W\,\operatorname{median}(g_x),
\qquad
a_y^{(0)}=H\,\operatorname{median}(g_y),
\tag{21}
$$

with the medians taken over the corresponding sampled interior difference arrays. BFGS then minimizes Equation (20), using the analytic gradient and a maximum of 100 iterations. It is a local optimization, with no guaranteed global optimum and no multi-start search.

For completeness, write $z_j=c_j e^{-i\mathbf a^T\mathbf q_j}$ and $m=N^{-1}\sum_jz_j$. Then

$$
\frac{\partial m}{\partial a_k}
=-\frac{i}{N}\sum_jq_{jk}z_j,
\qquad
\frac{\partial(-|m|^2)}{\partial a_k}
=-2\operatorname{Re}\!\left[
m^*\frac{\partial m}{\partial a_k}
\right].
\tag{22}
$$

The implementation stores the refined selected-order carrier as

$$
c_{x,\mathrm{ref}}=c_x+\frac{\widehat a_x}{2\pi W},
\qquad
c_{y,\mathrm{ref}}=c_y+\frac{\widehat a_y}{2\pi H}.
\tag{23}
$$

The correction is applied to the already filtered field. The code does not re-extract the Fourier order around this refined carrier.

### 5.3. Choosing a phase origin

After tilt correction, compute

$$
c_1(\mathbf r)=c(\mathbf r)e^{-i\widehat{\mathbf a}^T\mathbf q(\mathbf r)},
\qquad
\widehat p=\operatorname{Arg}\!\left[
\frac{1}{|S_2|}\sum_{\mathbf r\in S_2}c_1(\mathbf r)
\right],
$$

$$
c_0(\mathbf r)=c_1(\mathbf r)e^{-i\widehat p}.
\tag{24}
$$

This sets the circular mean phase of the selected interior pixels to zero. It is a phase-origin convention, not a measurement of the original reference beam's absolute phase.

Because cells participate in this mean, the background is only approximately centered at zero. A dense crop can bias the estimate. The stored `background_concentration` is $|m(\widehat{\mathbf a})|$; a high value indicates concentrated phase under this convention, not independently verified reconstruction accuracy.

### 5.4. Resolving conjugation with an RBC prior

The selected Fourier order can represent either the object phase or its negative. To choose a sign without the target, we unwrap

$$
u(\mathbf r)=\mathcal U\!\left[\operatorname{Arg}c_0(\mathbf r)\right],
\qquad
\widetilde u(\mathbf r)=u(\mathbf r)-\operatorname{median}_{S_2}u.
\tag{25}
$$

Here $\mathcal U$ is `skimage.restoration.unwrap_phase`, used with `rng=0` for reproducibility. The library's two-dimensional implementation is associated with reliability-guided phase unwrapping; see the [official documentation](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase) and [Herráez et al. (2002)](https://doi.org/10.1364/AO.41.007437).

We then compute the median-centered third moment

$$
\mu_3=\frac{1}{|S_2|}\sum_{\mathbf r\in S_2}\widetilde u(\mathbf r)^3,
\qquad
\widehat s=
\begin{cases}
+1,&\mu_3\geq0,\\
-1,&\mu_3<0.
\end{cases}
\tag{26}
$$

This statistic is not standardized statistical skewness: the code does not divide by a variance and centers on the median rather than the mean. Its purpose is simply to choose the sign with predominantly positive excursions relative to the assumed background.

The prior is motivated by cells producing positive relative optical path delay. It is a heuristic, not a proof of the correct conjugate order. Dense cellular content, unwrapping errors, or a sample with a different phase distribution can invalidate it. No anatomical segmentation or cell-specific shape model is used.

### 5.5. Exported phase arrays and PNG convention

The relative unwrapped output is

$$
\widehat\phi_{\mathrm{rel}}(\mathbf r)=\widehat s\,\widetilde u(\mathbf r).
\tag{27}
$$

For comparison with the PNG labels, the code uses the empirically adopted negative-path convention:

$$
\widehat P(\mathbf r)
=\frac{\operatorname{wrap}\!\left[-\widehat s\operatorname{Arg}c_0(\mathbf r)\right]+\pi}{2\pi}.
\tag{28}
$$

The negative sign in Equation (28) is a display/label convention. It does not mean that the physical RBC optical path is asserted to be negative. Phase near zero maps to approximately 0.5 in the output PNG.

Equations (27) and (28) deliberately describe the actual implementation rather than assuming they are identical representations. The unwrapped array has an additional median subtraction, while the PNG uses the circular-mean origin of Equation (24). Therefore, `phase_png` is **not** obtained by directly wrapping `relative_phase_rad`; the arrays can differ by a constant phase origin even when their spatial phase structure agrees.

When evaluation uses `phase_invert: true`, it also replaces the predicted PNG by $1-\widehat P$ to match the configured target preprocessing. The experiments in this manuscript use `phase_invert: false`.

### 5.6. Algorithm 1: input-only reconstruction

**Input:** one normalized hologram $I$; fixed $b=0.08$, $\tau=0.015$, and border width 16.  
**Output:** phase PNG $\widehat P$, relative unwrapped phase $\widehat\phi_{\mathrm{rel}}$, cross-term amplitude $|C|$, and parameter metadata.

1. Construct the mean-subtracted Hann-windowed detection image and locate $\mathbf c$ using Equation (11).
2. Check aperture separation and sampling using Equation (15).
3. Demodulate the original hologram and filter it using Equations (12)–(14), obtaining $C$.
4. Normalize $C$ to phase vectors using Equation (18).
5. Initialize the two tilt parameters from adjacent-pixel phase differences; optimize Equation (20).
6. Remove the estimated tilt and circular-mean piston using Equation (24).
7. Unwrap the compensated phase, subtract its sampled median, and choose the sign using Equation (26).
8. Export Equations (27) and (28), together with $|C|$ and metadata.

No target image enters any step of this algorithm. The aperture setting and PNG convention are fixed before validation. The per-image optimization estimates only nuisance reference parameters from that image.

## 6. Label-assisted phase-compatibility diagnostic

### 6.1. Purpose and allowed information

Input-only reconstruction can fail to match a label because the object structure is wrong, or because its phase origin, sign, or residual tilt differs from the label's convention. The diagnostic is designed to distinguish these possibilities.

It begins with the same extracted complex field $C$, before input-only background compensation. It then asks:

> Can the phase of this field explain the paired target if we allow only a conjugate sign and an affine reference phase?

The diagnostic does not fit an arbitrary image, spatial deformation, neural network, or frequency response. However, it **does** fit parameters to each target image. Its results therefore describe model compatibility, not deployment reconstruction accuracy.

The target is provisionally interpreted as a circular phase through

$$
\theta_P(\mathbf r)=2\pi P(\mathbf r).
\tag{29}
$$

Using $2\pi(P-0.5)$ instead would differ only by a constant piston, which the diagnostic already permits. The empirical support for the $2\pi$ scale is discussed in Section 8.1.

### 6.2. Sparse parameter-fitting grid

For a 256 × 256 image with border width 16,

$$
S_4=\{16,20,\ldots,236\}^2,\qquad |S_4|=56^2=3{,}136.
\tag{30}
$$

Only target values on $S_4$ are allowed to affect the fitted parameters. The circular evaluation set is

$$
E=\Omega\setminus S_4,\qquad
|E|=224^2-56^2=47{,}040.
\tag{31}
$$

The hologram itself is available in full for Fourier extraction. The restriction concerns target information, not measurement information.

### 6.3. Fitting conjugation, tilt, and piston

For each candidate sign $s\in\{-1,+1\}$, define

$$
d_s(\mathbf r)=s\operatorname{Arg}C(\mathbf r)-\theta_P(\mathbf r),
$$

$$
m_s(\mathbf a)=\frac{1}{|S_4|}
\sum_{\mathbf r\in S_4}
\exp\!\left[i\left(d_s(\mathbf r)+\mathbf a^T\mathbf q(\mathbf r)\right)\right].
\tag{32}
$$

The fitted tilt and sign satisfy

$$
(\widehat s_d,\widehat{\mathbf a}_d)
=\underset{s\in\{-1,+1\},\,\mathbf a\in\mathbb R^2}{\operatorname{arg\,max}}
|m_s(\mathbf a)|^2.
\tag{33}
$$

Notice that the tilt enters Equation (32) with a **positive** sign because this procedure adds a correction to an existing phase difference. The input-only objective in Equation (19) instead subtracts an estimated reference ramp. Keeping these conventions explicit avoids a sign error when comparing the equations to the source code.

For each $s$, initialization proceeds as follows. Wrap $d_s$ on $S_4$, unwrap the resulting 56 × 56 array, fit an affine plane to those samples by ordinary least squares, and negate the two fitted slope coefficients. All of these steps use only the sparse fitting-grid targets. Unwrapping a dense target-dependent image before subsampling would allow evaluation labels to influence the initializer; the current implementation avoids that leakage.

BFGS minimizes $-|m_s|^2$ for each sign, with an analytic gradient and a maximum of 120 iterations. If $z_j=\exp[i(d_{sj}+\mathbf a^T\mathbf q_j)]$, then

$$
\frac{\partial m_s}{\partial a_k}
=\frac{i}{|S_4|}\sum_jq_{jk}z_j,
\qquad
\frac{\partial(-|m_s|^2)}{\partial a_k}
=-2\operatorname{Re}\!\left[
m_s^*\frac{\partial m_s}{\partial a_k}
\right].
\tag{34}
$$

After selecting the best sign and tilt, the piston is determined from the fitting grid:

$$
\widehat p_d=-\operatorname{Arg}\!\left[m_{\widehat s_d}(\widehat{\mathbf a}_d)\right].
\tag{35}
$$

The diagnostic phase and image are

$$
\widehat\theta_d(\mathbf r)
=\widehat s_d\operatorname{Arg}C(\mathbf r)
+\widehat{\mathbf a}_d^T\mathbf q(\mathbf r)+\widehat p_d,
$$

$$
\widehat P_d(\mathbf r)
=\frac{\widehat\theta_d(\mathbf r)\bmod 2\pi}{2\pi}.
\tag{36}
$$

Thus only three continuous phase parameters and one discrete sign are fitted per pair. No phase scale parameter is fitted in the production diagnostic.

The optimizer is local and the slopes are not explicitly bounded. Sparse sampling can admit phase-ramp aliases. Smooth residual phase and the coarse carrier estimate motivate the initialization, but they do not establish global uniqueness. Optimizer termination flags are saved rather than treated as proof of a correct physical solution.

### 6.4. Circular evaluation on disjoint target pixels

Define the wrapped residual

$$
e(\mathbf r)=\operatorname{wrap}\!\left[
\widehat\theta_d(\mathbf r)-\theta_P(\mathbf r)
\right].
\tag{37}
$$

On the evaluation set $E$, circular coherence is

$$
\gamma_E=
\left|\frac{1}{|E|}\sum_{\mathbf r\in E}e^{ie(\mathbf r)}\right|,
\tag{38}
$$

and circular root-mean-square error is

$$
\operatorname{CRMSE}_E=
\sqrt{\frac{1}{|E|}\sum_{\mathbf r\in E}e(\mathbf r)^2}.
\tag{39}
$$

Coherence approaches one when the residual phase vectors point in a similar direction. Its magnitude is invariant to an additional constant piston. The circular RMSE additionally checks the piston chosen from the fitting grid. No new piston is fitted on $E$.

The two metrics are therefore complementary. Coherence alone is not a complete absolute-phase error metric.

The evaluation pixels are disjoint from the target fitting pixels, but they are spatially correlated pixels in the same crop. This is not a substitute for an independent-image or independent-acquisition test. Furthermore, the diagnostic's ordinary PNG PSNR and SSIM use either the full image or the full interior region, both of which include fitting pixels. Only the circular metrics use $E$.

### 6.5. Algorithm 2: label-assisted model diagnosis

**Input:** extracted field $C$, target $P$, fixed scale $2\pi$, and border width 16.  
**Output:** diagnostic image $\widehat P_d$, fitted sign/tilt/piston, and circular metrics on $E$.

1. Construct $S_4$ and $E$ using Equations (30) and (31).
2. For each sign $s$, initialize the correction from the wrapped and unwrapped phase difference on $S_4$ only.
3. Optimize Equation (33) separately for the two signs.
4. Select the better fitting-grid objective and estimate the piston using Equation (35).
5. Form the diagnostic phase and PNG using Equation (36).
6. Evaluate Equations (38) and (39) on $E$, without refitting any parameters.
7. Label all resulting image-comparison scores as target-assisted diagnostics.

## 7. Image metrics and the role of phase wrapping

For a normalized prediction $\widehat P$ and target $P$, define

$$
\operatorname{MSE}=\frac{1}{HW}\sum_{x,y}(\widehat P[x,y]-P[x,y])^2,
\qquad
\operatorname{PSNR}=10\log_{10}\!\left(\frac{1}{\operatorname{MSE}}\right).
\tag{40}
$$

The repository returns 99 dB when MSE is at most $10^{-12}$. Reported dataset PSNR is the arithmetic mean of per-image PSNR, not PSNR calculated from one pooled MSE.

SSIM is computed from local means, variances, and covariance:

$$
\operatorname{SSIM}(\widehat P,P)=
\frac{(2\mu_{\widehat P}\mu_P+C_1)(2\sigma_{\widehat P P}+C_2)}
{(\mu_{\widehat P}^2+\mu_P^2+C_1)
(\sigma_{\widehat P}^2+\sigma_P^2+C_2)}.
\tag{41}
$$

The implementation uses an 11 × 11 normalized Gaussian window with standard deviation 1.5, $C_1=0.01^2$, $C_2=0.03^2$, and an additional $10^{-12}$ denominator stabilizer. The SSIM map is averaged over pixels. Convolutions use zero padding. Interior metrics are obtained by cropping 16 pixels from each edge and recomputing the same metrics on that crop, including the same convolution-padding convention.

PNG metrics can differ sharply from circular phase metrics. For example, phases $\pi-0.02$ and $-\pi+0.02$ are only 0.04 radians apart on the circle, but under the encoding $(\phi+\pi)/(2\pi)$ their pixel values are approximately 0.9968 and 0.0032. An ordinary image metric regards this as a large error. A small phase offset can therefore move an entire cell region across a black/white wrap boundary while preserving much of its circular phase structure.

For this reason, we report both ordinary image fidelity and the explicitly defined circular diagnostic. Neither should silently replace the other.

## 8. Experimental protocol

### 8.1. Preliminary training audit and frozen settings

A preliminary audit used 16 evenly spaced training pairs and compared cutoff radii of 0.08, 0.11, and 0.14 cycles per pixel. It examined target-aligned phase agreement and the effect of allowing an additional phase scale. This audit supplied the evidence for retaining the 0.08 cutoff and the fixed $2\pi$ target encoding.

At cutoff 0.08, the fixed-scale training diagnostic gave mean coherence 0.99742. Allowing an additional scale yielded 0.99753 and a mean scale of $1.0022\pm0.0158$ relative to unity, where the reported spread is the standard deviation across the 16 images, computed with NumPy's default `ddof=0`. In that exploratory fit, the coefficient multiplies the recovered signed phase. These results support a one-to-one radian scale under Equation (29), up to reference conventions; they do not recover undocumented PNG export metadata.

The preliminary audit was an exploratory, target-fitted analysis. Its carrier search used $f_y<-0.08$ and radial frequency greater than 0.18, and it evaluated the candidate apertures before the final separation guard was introduced. In particular, a cutoff of 0.14 is rejected by the production guard for the illustrated carrier. The exploratory script also used a different alignment/evaluation implementation, and its image SSIM was computed with scikit-image. It should not be pooled with the production evaluation below or described as a disjoint-pixel test. The production settings were frozen for the reported validation runs; there was no exhaustive search over reconstruction priors or cell morphology models. The [original exploratory script](D:/Courses/EECS298/flow/outputs/rbc_phase_audit/audit.py) records those earlier settings.

### 8.2. Validation subsets and baselines

Two supplied-split subsets were evaluated:

- **First 32:** the first 32 sorted validation pairs, matching the earlier ADMM experiment's ordering.
- **Random 256:** 256 distinct validation indices sampled without replacement with NumPy's generator seeded by 20260905, then sorted for processing.

The exact indices and per-image parameters are saved in each run's `results.json`.

We compare input-only reconstruction against the measured hologram itself and a constant image $\widehat P=0.5$. The constant baseline matters because large, smooth backgrounds can yield apparently favorable image metrics without recovering cell structure. Historical ADMM results are included only for the matching first-32 subset; its acquisition model and implementation differ from the current method.

### 8.3. Implementation settings

| Setting | Value |
|---|---|
| Input dimensions | 256 × 256; no downsampling |
| Intensity normalization | 8-bit PNG divided by 255 |
| Detection window | Separable Hann |
| Detection region | $f_y<0$, radial frequency greater than 0.16 |
| Aperture cutoff $b$ | 0.08 cycles/pixel |
| Taper width $\tau$ | 0.015 cycles/pixel |
| Border exclusion | 16 pixels |
| Input-only fitting grid | Every second interior pixel |
| Input-only optimizer | BFGS, analytic gradient, at most 100 iterations |
| Diagnostic fitting grid | Every fourth interior pixel |
| Diagnostic optimizer | Two sign candidates; BFGS, at most 120 iterations each |
| Target phase scale | Fixed $2\pi$ radians per normalized PNG unit |
| Phase unwrapping | scikit-image, `rng=0` |
| Main numerical precision | NumPy float64 / complex128; output arrays float32 |
| Image metric precision | PyTorch float32; four CPU threads |

The local environment used NumPy 2.1.2, SciPy 1.15.3, scikit-image 0.26.0, and PyTorch 2.9.0+cu128. The reconstruction is implemented in NumPy/SciPy and ran on the CPU; the presence of a CUDA-enabled PyTorch build does not make this reconstruction a GPU algorithm.

## 9. Results

### 9.1. Input-only reconstruction

| Method | First-32 PSNR (dB) | First-32 SSIM | Random-256 PSNR (dB) | Random-256 SSIM |
|---|---:|---:|---:|---:|
| Measured hologram | 11.01 | 0.2334 | 9.57 | 0.1365 |
| Constant 0.5 | 14.10 | 0.7115 | 12.54 | 0.6440 |
| Historical ADMM, 100 µm | 13.10 | 0.1870 | Not evaluated | Not evaluated |
| Off-axis input-only | **14.44** | **0.8194** | **12.75** | **0.7406** |

On the random-256 subset, the off-axis method's interior PSNR and SSIM are 13.04 dB and 0.7514, respectively. The input-only reconstruction time recorded by that run averages approximately 35.7 ms per image, excluding file loading, image metrics, plotting, and label-assisted diagnosis. This is a local timing observation rather than a hardware-independent benchmark.

The increase in SSIM over the flat baseline is accompanied by visible recovery of cell contours and interior phase structure. The PSNR gain over the same baseline is modest. The method should therefore be described as a physically grounded baseline with remaining reference-estimation errors, rather than as a fully accurate reproduction of the supplied PNG labels.

### 9.2. Label-assisted phase compatibility

| Diagnostic quantity | First 32 | Random 256 |
|---|---:|---:|
| Coherence on $E$ | 0.99693 | 0.99534 |
| Circular RMSE on $E$ (rad) | 0.07792 | 0.09073 |
| Full PNG PSNR (dB) | 23.41 | 21.89 |
| Full PNG SSIM | 0.9298 | 0.9177 |
| Interior PNG PSNR (dB) | 24.84 | 23.43 |
| Interior PNG SSIM | 0.9431 | 0.9323 |

These results indicate that the extracted cross-term phase is highly compatible with the target after an affine reference correction. They do not mean that input-only reconstruction achieves 21.89 dB. The diagnostic obtains information about the target's phase convention by fitting to that target.

![First-32 visual comparison](D:/Courses/EECS298/flow/outputs/rbc_offaxis_first32/examples.png)

*Figure 2. The first four examples in the first-32 evaluation. Columns show the hologram, target, input-only output, and label-assisted diagnostic. The final column demonstrates phase compatibility after target-fitted reference correction; it is not an input-only result.*

![Random-256 visual comparison](D:/Courses/EECS298/flow/outputs/rbc_offaxis_random256/examples.png)

*Figure 3. The first four examples in sorted processing order from the random-256 subset. Changes in black/white cell interiors illustrate the effect of phase-reference conventions and wrap placement. These examples were selected by processing order, not ranked by reconstruction score.*

### 9.3. Synthetic physical and regression checks

Seven tests were implemented and passed. They cover fractional-carrier phase and amplitude recovery, input-only reconstruction under flips and rotation, intensity gain/offset behavior for a periodic integer-bin carrier, matched versus wrong/shifted phase labels, invalid inputs, diagnostic shape/border validation, and exclusion of evaluation labels from parameter fitting.

For the synthetic matched-label case, diagnostic coherence exceeds 0.9999. For the particular shifted and wrong labels tested, it falls below 0.48. Thus the permitted affine correction does not automatically make those unrelated phase structures match.

The target-exclusion regression is especially relevant to the diagnostic claim. It changes target pixels outside $S_4$, while keeping all values on $S_4$ identical. The fitted sign, tilt, piston, and diagnostic prediction must remain unchanged. Evaluation error is allowed to change. This checks that excluded target pixels are not used indirectly through the initializer.

These controls support implementation correctness for the tested conditions. They do not establish universal recovery for arbitrary fields or immunity to all possible target mismatches.

## 10. Discussion and limitations

### 10.1. What is supported by the results

Three observations support the revised physical explanation. First, the acquisition description specifies off-axis microscopy. Second, measured spectra contain separated conjugate orders at the frequencies suggested by the visible carrier fringes. Third, a phase extracted through an analytic sideband filter agrees closely with paired phase structure after only a low-dimensional reference correction.

The third observation is more specific than saying that every aspect of image formation has been calibrated. We have tested the phase of the measured cross term. We have not shown that a phase PNG alone can predict the raw hologram through a calibrated simulator. Object amplitude, reference amplitude, detector response, and the original reference convention are not fully supplied by the phase labels.

### 10.2. Why raw PNG reconstruction remains harder

The labels were obtained before cropping. Each crop consequently inherits a phase origin and any residual background structure from a larger reconstruction. A rule that sets each crop's circular mean to zero need not reproduce that inherited origin.

The source of a linear phase component is also ambiguous without an additional reference: it may arise from carrier error or from actual slowly varying object phase. Maximizing concentration intentionally favors a flatter background. It can remove genuine structure or choose a biased correction when the assumed background does not dominate.

Finally, selecting a conjugate sign through a third moment depends on successful unwrapping and an appropriate cell-phase distribution. This is a practical heuristic with observed limitations, rather than an independently measured sign calibration.

The gap between input-only and target-assisted metrics is therefore evidence that reference conventions matter, but it is not a proof that every residual error is exclusively a piston or tilt error. Aperture smoothing, crop leakage, noise, and unwrapping can also contribute.

### 10.3. Absolute physical quantities and unavailable corrections

The exported $|C|$ retains amplitude information but combines object and reference amplitudes and detector gain. It is not an absorption coefficient. Likewise, $\widehat\phi_{\mathrm{rel}}$ is a relative, sign-selected unwrapped field. Absolute optical path or cell thickness would require a justified phase origin and appropriate optical calibration.

The current implementation performs no flat-field division, reference-image subtraction, absorption optimization, defocus fitting, quadratic wavefront compensation, iterative phase-image retrieval, plug-and-play denoising, or neural reconstruction. These are not hidden stages behind the reported scores. Whether any is useful should be established from an appropriate acquisition model and additional evidence.

The literature-based telecentric model motivates limiting the fitted reference to an affine phase. We did not independently calibrate or rule out every residual optical aberration in each local crop.

### 10.4. Split overlap and generalization

A local filename audit found that 7,088 of 7,373 validation images, or 96.1%, have training siblings after removal of terminal `-A`/`-B` augmentation suffixes. The first ten checked sibling cases match training holograms exactly under flips. All validation files share apparent acquisition timestamp groups with training; the training filenames contain 192 such groups. The entire set of 7,088 sibling cases was not compared pixel by pixel.

The present evaluation is therefore a supplied-split reconstruction and compatibility study. It does not establish performance on independently acquired specimens. This limitation also applies when interpreting learning-based models trained and evaluated with these splits. A future generalization experiment should separate acquisition groups before augmentation.

The first-32 and random-256 results are subset means without independent-acquisition confidence intervals. They should not be presented as a complete benchmark on all 7,373 validation images or as a statistically independent comparison against every alternative reconstruction method.

## 11. Reproducibility and mapping to the source code

| Mathematical operation | Implemented location |
|---|---|
| Equations (10)–(15): carrier detection and sideband extraction | `extract_offaxis_field` |
| Equations (18)–(28): input-only reference compensation and outputs | `compensate_background` |
| Equations (29)–(39): label-assisted compatibility test | `phase_compatibility_diagnostic` |
| Dataset selection, per-image metrics, figures, and output files | `scripts/eval_rbc_offaxis.py` |
| PNG PSNR and Gaussian SSIM | `lensless_flow/metrics.py` |
| Synthetic physical and target-exclusion checks | `tests/test_holography_offaxis.py` |

Source links: [off-axis reconstruction module](D:/Courses/EECS298/flow/lensless_flow/holography_offaxis.py), [evaluation and inference script](D:/Courses/EECS298/flow/scripts/eval_rbc_offaxis.py), [metric implementation](D:/Courses/EECS298/flow/lensless_flow/metrics.py), and [tests](D:/Courses/EECS298/flow/tests/test_holography_offaxis.py).

From the repository directory, the following PowerShell commands reproduce the reported subset evaluations and run standalone inference:

```powershell
$rbcPython = 'C:\Users\12181\miniconda3\envs\py312\python.exe'

# The 32 images used for comparison with the earlier experiment.
& $rbcPython -m scripts.eval_rbc_offaxis --config configs/rbc_hologram.yaml --selection first --max_samples 32 --diagnose_labels --out_dir outputs/rbc_offaxis_first32

# A fixed random subset with the frozen reconstruction settings.
& $rbcPython -m scripts.eval_rbc_offaxis --config configs/rbc_hologram.yaml --selection random --seed 20260905 --max_samples 256 --diagnose_labels --out_dir outputs/rbc_offaxis_random256

# One hologram: this execution path does not load a paired target.
& $rbcPython -m scripts.eval_rbc_offaxis --hologram 'E:\RBCs Holograms\Holograms\Validation\Image__2021-01-28__15-49-02_1-11-B.png' --out_dir outputs/rbc_offaxis_single

# Synthetic physical and regression tests.
& $rbcPython -m unittest discover -s tests -p test_holography_offaxis.py -v
```

Omit `--diagnose_labels` to evaluate input-only predictions without running the target-assisted diagnostic. The paired evaluation path still loads targets to compute ordinary metrics. Use `--hologram` when no target is available. Add `--save_reconstructions` to export per-sample input-only arrays during dataset evaluation.

Evaluation produces `results.json`, `per_sample.csv`, `examples.png`, and `spectrum.png`. Standalone inference produces `phase.png`, `reconstruction.npz`, `metadata.json`, and `spectrum.png`. The NPZ contains the floating-point PNG representation, relative unwrapped phase, and cross-term amplitude; it does not store a target-assisted reconstruction as an input-only output.

The figures in this manuscript refer to existing local output files, which are ignored by Git and regenerated by the commands above. Their absolute paths may need updating when the document is moved to another computer.

Saved experimental records:

- [First-32 results](D:/Courses/EECS298/flow/outputs/rbc_offaxis_first32/results.json).
- [Random-256 results](D:/Courses/EECS298/flow/outputs/rbc_offaxis_random256/results.json).
- [Preliminary training encoding audit](D:/Courses/EECS298/flow/outputs/rbc_phase_audit/audit.json).
- [Filename overlap and checked-flip audit](D:/Courses/EECS298/flow/outputs/rbc_offaxis_split_audit.json).

## 12. Conclusion

The relevant reconstruction mechanism is off-axis interference demodulation. A tilted reference translates complex object information into a separated Fourier order; filtering and recentering that order recover a complex field without learning an image-valued operator. The remaining reference correction can be represented by two slopes, a constant phase, and a conjugate sign.

The current input-only implementation estimates those quantities through a dominant-background assumption and an RBC phase-sign heuristic. It recovers recognizable phase structure and improves SSIM over a constant-image baseline, while retaining substantial error relative to the supplied PNG convention. A separately evaluated label-assisted correction produces much closer phase agreement and supports the physical explanation, but it is not a deployable reconstruction score.

Further improvement should concentrate on establishing the reference convention and handling crop boundaries under an acquisition-consistent model. A blank-reference measurement or the original full-frame holograms would provide useful information for that calibration. The present results do not motivate returning to an uncalibrated in-line defocus model.

## References

1. R. Castañeda, C. Trujillo, and A. Doblas, “A human erythrocytes hologram dataset for learning-based model training,” *Data in Brief*, vol. 54, 110424, 2024. [doi:10.1016/j.dib.2024.110424](https://doi.org/10.1016/j.dib.2024.110424). Dataset: [OSF 8P7BA](https://osf.io/8p7ba/).

2. C. Trujillo, R. Castañeda, P. Piedrahita-Quintero, and J. Garcia-Sucerquia, “Automatic full compensation of quantitative phase imaging in off-axis digital holographic microscopy,” *Applied Optics*, vol. 55, no. 36, pp. 10299–10306, 2016. [doi:10.1364/AO.55.010299](https://doi.org/10.1364/AO.55.010299).

3. R. Castañeda, C. Trujillo, and A. Doblas, “pyDHM: A Python library for applications in digital holographic microscopy,” *PLOS ONE*, vol. 17, no. 10, e0275818, 2022. [doi:10.1371/journal.pone.0275818](https://doi.org/10.1371/journal.pone.0275818). [Library documentation](https://catrujilla.github.io/pyDHM/).

4. M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,” *Applied Optics*, vol. 41, no. 35, pp. 7437–7444, 2002. [doi:10.1364/AO.41.007437](https://doi.org/10.1364/AO.41.007437).

5. scikit-image contributors, “`skimage.restoration.unwrap_phase`,” official API documentation. [Documentation](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase), accessed September 5, 2026.
