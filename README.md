# SlangVision

GPU training augmentation for PyTorch, written as plain
[Slang](https://shader-slang.org/) shaders: torchvision's transforms and
results, with separate random values for every image in the batch, fused
into a few CUDA kernels.

**Under development.** This release reserves the package name and contains
no functionality yet; the first usable release will follow.

Licensed under Apache-2.0. Copyright 2026 Tekin Tatar. Not affiliated with
the Slang project or Khronos.
