# Changelog

## 0.2.0 — 2026-11-05

### Added
- `slangvision.testing.compare_to_reference`: besides the largest
  difference from torchvision, the fraction of values, the images and each
  image's largest difference beyond the output dtype's rounding, with a
  table in the guide ("Checking your pipeline") of what to expect for each
  kind of pipeline, so step-function edge cases can be told from a wrong op.
- Compiled kernels are cached on disk (`~/.cache/slangvision/kernels`;
  `SLANGVISION_CACHE_DIR` moves or turns it off): a later process's first
  call of a pipeline compiled before takes about 1.5 s instead of 2 s
  (measured 1.0 s against 1.6 s on a warm machine).
  (SlangPy's module cache, which would save more, segfaults when one process
  loads many modules with SlangPy 0.42, so it is not used.)
- `RandomPhotometricDistort` (torchvision's SSD colour distortion, used by its
  detection and segmentation recipes), `RandomChannelPermutation` and
  `Grayscale`; with `num_output_channels=1` (torchvision's default) the
  pipeline returns one channel. `Normalize` accepts one mean and std for
  every channel.
- `slangvision.sr.DiffJPEG`: BasicSR's DiffJPEG (Real-ESRGAN's JPEG
  degradation), the same values up to float rounding, per-image quality:
  0.30 ms instead of 1.5 ms per batch of 12 at 400x400 (RTX PRO 4500), 1.0
  instead of 8.2 ms at 48. No gradients (`differentiable=True` gives the
  values of BasicSR's soft rounding). The caller's quality tensor is left
  unchanged. `libjpeg=True` follows a real encoder where BasicSR's does
  not (its tables transposed and its black padding, both mistakes; float
  tables and repeated chroma): 45.7 instead of 37.3 dB PSNR against
  Pillow's JPEG at quality 50.
- `slangvision.sr`: drop-in replacements for BasicSR's GPU degradation
  functions, which Real-ESRGAN calls in its training step: `USMSharp` (the
  same 51-tap Gaussian as a row and a column pass: 1.5 ms instead of 118 ms
  per batch of 12 at 400x400 on an RTX PRO 4500), `add_gaussian_noise_pt`,
  `random_add_gaussian_noise_pt`, `add_poisson_noise_pt` and
  `random_add_poisson_noise_pt` (per-image values from the same torch draws
  as BasicSR; Poisson noise 0.5 ms instead of 7.2 ms).
- `GaussianNoise` (torchvision's: `mean + sigma * N(0, 1)` per value, then
  clipped), new noise per pixel, channel and image from SlangVision's own
  generator; a colour op, so statistics after it (equalize, autocontrast)
  see the same noise. +0.16 ms on the ImageNet recipe (batch 256, RTX PRO
  4500). `same_on_batch=True` gives every image the same noise.
  `max_reference_error` returns NaN for pipelines with noise.
- `TrivialAugmentWide` and `AutoAugment` (torchvision's, with the ImageNet,
  CIFAR10 and SVHN policies), drawn per image on the same op pass as
  `RandAugment`: 1.9 and 3.1 ms per batch of 256 after a crop and flip (RTX
  PRO 4500), against 82 and 95 ms for torchvision per image.
- `TimmRandAugment`: timm's RandAugment from its config string (`m`, `n`,
  `p`, `mstd`, `mmax`, `inc`; default `"rand-m9-mstd0.5-inc1"`), with
  Pillow's ops as timm runs them (affine resampling with a hard edge and the
  ImageNet-mean fill, bilinear or Pillow's bicubic picked per op) on float
  values. Matches timm to within Pillow's uint8 rounding; 2.9 ms per batch
  of 256 after a crop and flip (RTX PRO 4500), against 310 ms for timm on
  one CPU core. Posterize at 0 bits gives black, as Pillow 12 does (Pillow
  13 raises there, which stops timm's own `rand-m9-mstd0.5-inc1` on about
  1 image in 600).
- `RandAugment` (torchvision's: `num_ops`, `magnitude`,
  `num_magnitude_bins`, `interpolation`, `fill`), with its ops and signs
  drawn per image; each op is a pass whose input warps, sharpens or copies
  and whose colour ops follow, with contrast, autocontrast and equalize
  statistics for the images that choose them. 2.9 ms per batch of 256 after
  a crop and flip (RTX PRO 4500), against 111 ms for torchvision per image.
  Warps gain torchvision's `fill=None` (zero padding without the fill blend).
- `RandomInvert`, `RandomPosterize`, `RandomAdjustSharpness`,
  `RandomAutocontrast` and `RandomEqualize`, following torchvision's float
  behaviour (posterize to `floor(value * 2**bits) / 2**bits`; equalize on the
  image quantized to 256 levels, with PIL's scheme as torchvision uses).
  Autocontrast and equalize read per-image statistics (min/max, 256-bin
  histograms) from a pre-pass that skips images they do not apply to.
- `RandomRotation`, `RandomAffine` and `RandomPerspective`, with
  torchvision's parameters, nearest or bilinear sampling and its fill. Each
  warp runs as its own pass over the image after the resize and crops (about
  0.2 ms per batch of 256 at 224x224); flips after a warp apply after it.
  Can be wrapped in `RandomApply` or `RandomChoice`. Not supported:
  `RandomRotation(expand=True)` and warps before a resize or crop.
- `MixUp` and `CutMix` (torchvision's, draw for draw: the same seed gives
  torchvision's result) and `TimmMixup` (timm's `Mixup`: switch
  probability, label smoothing, batch/pair/elem modes, `cutmix_minmax`,
  `mixup_enabled`),
  applied to Compose's output batch and labels by one fused Slang kernel.
- `RandomErasing` (torchvision's, after `Normalize`; `value` a number, per
  channel, or `"random"` for per-pixel normal noise, which also covers timm's
  `mode="pixel"`). `max_reference_error` skips noise-filled pixels.
- `RandomCrop` (padding, `pad_if_needed`, `fill`, constant/edge/reflect/
  symmetric padding), `Pad`, `RandomVerticalFlip` and `RandomChoice`
  (colour ops and at most one blur, also nested in RandomApply; covers DeiT
  III's ThreeAugment choice of grayscale, solarize or blur). After an
  optional resize, any sequence of `Pad`, `CenterCrop` and `RandomCrop`
  composes as torchvision applies them; reflect and symmetric padding wider
  than an image raise `ValueError`, as in torchvision. The CIFAR recipe
  `RandomCrop(32, padding=4)` works.
- `interpolation="bicubic"` for RandomResizedCrop and Resize (string or
  torchvision's `InterpolationMode`), matching torchvision's float bicubic
  with and without antialias. Used by DeiT, ConvNeXt, MAE, DINO, OpenCLIP
  and most ViT recipes. About twice the cost of bilinear (ImageNet recipe,
  batch 256: 3.4 ms against 1.75 ms on an RTX PRO 4500).
- `Compose(..., dtype=torch.bfloat16)`: bfloat16 output, rounded like
  `.to(torch.bfloat16)`, at the same speed as float16. SlangPy's torch
  bridge does not take bfloat16 tensors yet, so the kernels write bfloat16
  bits through an int16 view of the output.

### Changed
- No compiler warnings on a pipeline's first call: SlangPy's "no supported
  shader model" note at device creation is hidden, and the perspective
  solver no longer reads an uninitialized array.
- Porting hints: `A.ToDtype` (and `ToImage`, `ToTensor`, ...) explain that
  Compose does their job; MixUp or CutMix inside `RandomChoice` points to
  `random.choice`; `(H, W, 3)` images without `layout="hwc"` say so.
- The NVRTC error and the README say to pin an older NVRTC after other
  installs, since installing torch or torchvision restores PyTorch's.
- The CUDA tests stop with one message when the GPU or NVRTC is not set up.
- Between releases, `main` reports a `.dev0` version.
- `import slangvision` after torchvision or DALI decoded on the GPU warns,
  and the first pipeline call raises a RuntimeError that says to import it
  first (Linux; `SLANGVISION_ALLOW_LATE_IMPORT=1` overrides); before, the
  first call segfaulted without a word.
- `clamp01` also takes a `float`; the guide lists the stable Slang helpers
  with their signatures; a pipeline's compile error is printed once.
- The guide's "Checking your pipeline" table says float16 policies differ
  for up to a few percent of values, shows how to see whether a CPU has
  torch's 15.999999 translate table, and explains uint8 pipelines; mixers
  are picked with `v2.RandomChoice` or `torch.rand`, which follow
  `torch.manual_seed`; pipelines stay outside `torch.compile`.
- A custom op whose `apply` takes the wrong number of parameters is reported
  when the `Compose` is built; a missing reference says where it goes.
- Faster (RTX PRO 4500, batch 256): statistics pre-passes skip images that
  do not use their op and, for antialiased bicubic or photos at least 5x the
  output area, read a stored float32 resample instead of resampling again;
  GaussianBlur computes its weights once per image. RandAugment 2.90 to
  2.29 ms, TrivialAugmentWide 1.89 to 1.58, AutoAugment 3.07 to 2.49,
  TimmRandAugment 2.85 to 2.22, SimCLR 3.05 to 2.57; the ImageNet recipe
  with bicubic on 695x1024 photos 8.51 to 3.97 ms. Raw data in
  `benchmarks/results/rtx_pro_4500_review_perf_2026-11-05.json`.
- `NeighborhoodOp` is the name of the custom neighbourhood-op class, spelt
  like `ColorOp` and `Grayscale`; `NeighbourhoodOp` stays as an alias.
- Clearer errors: a resize or crop after other transforms names the
  transform it follows; a torchvision transform (`ToImage`, `ToDtype`, ...)
  or a nested `Compose` in a pipeline says what to do instead.
- `Compose` is fixed when it is built: `transforms` is a tuple and the
  generated Slang comes from the transforms as they were then. Changing a
  transform afterwards could make two ops share a parameter slot (or, for
  the last op, write into the next image's row); build a new `Compose` to
  change parameters, e.g. for a magnitude schedule.

## 0.1.0 — 2026-11-04

### Added
- `slangvision`: GPU training augmentation for PyTorch on CUDA, moved out of
  SlangImage (where it was `slangimage.augment`, never released). Pipelines
  use torchvision's transform names and match torchvision's float
  functionals image by image, draw random values per image
  (`same_on_batch=True` shares them), and compile into a few fused Slang
  kernels. Transforms: RandomResizedCrop, Resize, CenterCrop,
  RandomHorizontalFlip, ColorJitter, RandomApply, RandomGrayscale,
  GaussianBlur, RandomSolarize, Normalize; `ColorOp` and `NeighbourhoodOp`
  for custom ops; `Compose.source`; `slangvision.testing.max_reference_error`
  checks a pipeline against torchvision image by image.
- No CUDA Toolkit or environment variables needed with PyTorch from pip:
  SlangVision uses the NVRTC installed with torch, choosing a copy the
  driver supports, after a one-time `python -m slangvision setup` that
  links it into SlangPy's folder (`undo` removes it; `python -m
  slangvision` reports the setup). It explains the fix when torch's NVRTC
  is newer than the driver. A system CUDA Toolkit or `CUDA_PATH` is used
  unchanged.
- `benchmarks/compare_pipelines.py` compares pipelines against torchvision
  (batched and per image) and Kornia.

## 0.0.1 — 2026-11-04

- Placeholder release reserving the package name.
