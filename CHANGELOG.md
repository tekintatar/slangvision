# Changelog

## Unreleased

### Added
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
