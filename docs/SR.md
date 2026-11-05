# Super resolution

Real-ESRGAN, and the diffusion SR models that reuse its degradation, make
their low-resolution inputs on the GPU in the training step with BasicSR's
functions. `slangvision.sr` has drop-in replacements with the same names,
arguments and results; change the import:

```py
from slangvision.sr import (DiffJPEG, USMSharp, random_add_gaussian_noise_pt,
                            random_add_poisson_noise_pt)
```

- `USMSharp(radius=50, sigma=0)`, called `usm(img, weight=0.5,
  threshold=10)`: BasicSR's unsharp masking of the targets. `radius` is
  BasicSR's (the Gaussian's size, made odd), with OpenCV's kernel; reflect
  padding, so images must be larger than half the size. BasicSR convolves the
  51x51 Gaussian directly; here it runs as a row and a column pass, with the
  same result (within 2.4e-7 at the default size; with smaller kernels a
  residual within float rounding of the threshold can switch the mask,
  moving nearby values by up to 2e-3). `.cuda()` and `.to()` return the
  object, as BasicSR's module is used.
- `add_gaussian_noise_pt(img, sigma, gray_noise, clip, rounds)` and
  `random_add_gaussian_noise_pt(img, sigma_range, gray_prob, clip, rounds)`:
  per-image sigma (0-255 scale) and gray flag, drawn with the same torch
  calls as BasicSR, so the same seed gives the same values (a fractional
  `gray_noise` blends colour and gray noise, as in BasicSR). The noise
  itself comes from SlangVision's generator, keyed by one more draw from the
  device's torch generator, so later draws there differ from a BasicSR run.
  BasicSR draws one gray noise field and shares it across the batch; here
  each image gets its own.
- `add_poisson_noise_pt(img, scale, clip, rounds, gray_noise)` and
  `random_add_poisson_noise_pt(img, scale_range, gray_prob, clip, rounds)`:
  BasicSR's shot noise. Each value rounded to 1/255 becomes Poisson(q·n)/n,
  n the image's number of distinct values rounded up to a power of two
  (gray noise from the grayscale image, the same in all channels). BasicSR
  counts the values with `torch.unique` in a Python loop over the batch; one
  pass marks the levels here. Poisson values come from a sampler that
  matches the exact distribution (tested against the pmf up to lambda 256).
- `DiffJPEG(differentiable=True, libjpeg=False)`, called `jpeger(img,
  quality)` with a number or one quality per image (Real-ESRGAN: uniform in
  [30, 95]): BasicSR's JPEG degradation, the same values up to float
  rounding (a DCT coefficient within rounding of a quantization step's edge
  can round the other way and move its 8x8 block: about 1e-4 of values,
  0.5/255 at most at quality 95, measured). `differentiable=True`
  (BasicSR's default) gives the values of BasicSR's soft rounding; neither
  has gradients, and an input that requires them raises unless the call
  is under `torch.no_grad()`. Four kernels (the
  DCTs separable, one thread per block row): 0.30 ms instead of 1.5 ms per
  batch of 12 at 400x400 with a quality per image, 1.0 ms instead of 8.2 ms
  at 48. BasicSR overwrites the caller's quality tensor with its factors
  and truncates them for integer tensors; this one does neither. A quality
  given as a number must be above 0 and below 100, or it raises (BasicSR's
  tables are zero at 100, its output NaN); in a per-image tensor, 100 or 0
  gives NaN for that image, as in BasicSR (checking would sync the host).

Inputs are (B, 3, H, W) float16, bfloat16 or float32 CUDA tensors (BasicSR
also takes other channel counts); outputs keep the dtype and carry no
gradients (BasicSR's are differentiable; Real-ESRGAN calls them under
`no_grad`).

BasicSR's DiffJPEG is not what a JPEG encoder and decoder do: against
Pillow's quality-q 4:2:0 JPEG (libjpeg-turbo; OpenCV's output is identical)
of four photos (scikit-image's astronaut, coffee, motorcycle and chelsea,
about 400x400) it is 35.7, 37.3, 38.5 and 42.1 dB PSNR at quality 30, 50,
75 and 95, about as far from Pillow's result as Pillow's is from the
original at quality 30 (31.6 dB). Measured stage by stage:

| BasicSR's DiffJPEG | libjpeg | effect against Pillow |
|---|---|---|
| quantization tables transposed (its DCT puts the vertical frequency first, as the tables do) | ITU T.81 tables as listed | the largest: 37.3 to 43.0 dB at quality 50 |
| tables scaled by a float factor (a fractional quality works, which Real-ESRGAN's uniform draws use), not rounded | `(base * scale + 50) / 100` in integers, clamped to 1-255 (not reached at quality 30-95); quality an integer | +0.7 dB at 30, +0.2 at 75, none at 50; at quality 100 BasicSR's tables are zero (NaN) |
| black padding to a multiple of 16 | edge pixels repeated | images not a multiple of 16, in the last partial block row and column at quality 30: mean error 5.3 levels against 2.8 elsewhere; 4.7 against 1.0 with the table fixes, 0.85 with edge padding too |
| Cb and Cr repeated over 2x2 pixels | "fancy" triangle upsampling | +2.3 dB at 30, +4.1 at 95 (after the fixes above) |
| float samples and output | 8-bit integers throughout | most of the rest: 1-2.5 dB more |

The transposed tables and the black padding are mistakes; the float tables
look deliberate (continuous quality), and the rest are simplifications.
`DiffJPEG(libjpeg=True)` takes libjpeg's way for the first four (it is no longer
BasicSR's result): 45.5, 45.7, 46.0 and 47.6 dB at quality 30, 50, 75 and
95, at the same speed. Whether training on it changes a model is untested.

Where the time goes in Real-ESRGAN's training step (batch 12 of 400x400
targets, RTX PRO 4500, BasicSR's code): making the low-resolution batch takes
127 ms, against 294 ms (fp32) or 165 ms (bf16) for a RealESRNet training step
(RRDBNet, 23 blocks). 123 ms of that is USMSharp; Real-ESRGAN runs it a
second time on the 256 crop (49 ms). With these replacements the whole
degradation takes 7.5 ms (517 to 29 ms at batch 48); most of what remains is
blur kernels and JPEG, which still matters for compact models
(SRVGGNetCompact, as in realesr-general-x4v3, trains in 16 ms per batch of
12). Most of USMSharp's gain comes from splitting the Gaussian: a separable
version in plain PyTorch takes 4.3 ms. With DiffJPEG replaced too (its two
calls), the degradation takes 5.7 instead of 8.1 ms, and 25 instead of 31-33
ms at batch 48 (a later script, which times BasicSR's version at 175 ms: the
inputs and blur kernels differ). Blur kernels and resizing (BasicSR's
filter2D and F.interpolate) stay as they are.

Not here: BasicSR's filter2D (cuDNN's depthwise convolution, already fast: a
plain Slang kernel was only 1.3x faster) and resizing stay as they are.
Measurements: RTX PRO 4500 Blackwell, BasicSR commit 8d56e3a and
Real-ESRGAN a4abfb2 loaded from source, medians of 20-40 synced calls from
ad-hoc scripts (see [TESTING.md](https://github.com/tekintatar/slangvision/blob/main/docs/TESTING.md)).
