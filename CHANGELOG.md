# Changelog

## Unreleased

### Added
- `RandomRotation`, `RandomAffine` and `RandomPerspective`, with
  torchvision's parameters, nearest or bilinear sampling and its fill. Each
  warp runs as its own pass over the image after the resize and crops (about
  0.2 ms per batch of 256 at 224x224); flips after a warp apply after it.
  Can be wrapped in `RandomApply` or `RandomChoice`. Not supported:
  `RandomRotation(expand=True)` and warps before a resize or crop.
- `MixUp` and `CutMix` (torchvision's, draw for draw: the same seed gives
  torchvision's result) and `MixUpCutMix` (timm's `Mixup`: switch
  probability, label smoothing, batch/pair/elem modes, `cutmix_minmax`),
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
