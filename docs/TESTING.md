# Test evidence

Every transform is checked image by image against the torchvision
`transforms.v2.functional` call it mirrors:
`slangvision.testing.max_reference_error` replays each image's sampled
values through those calls. Inputs are photo-like uint8 images of nine
sizes, including extreme aspect ratios (40x300, 300x40, 23x17) that exercise
RandomResizedCrop's centre-crop fallback. Budgets are 1e-4 for float32
output and 4e-3 for float16 output, in normalized units.

| transform | evidence (`tests/test_pipelines.py`) |
|---|---|
| RandomResizedCrop | reference: recipes, non-square size, custom scale/ratio, `antialias=False`, fallback sizes |
| Resize | reference: `(h, w)`, shorter side + CenterCrop, `antialias=False` |
| CenterCrop | reference: alone, after Resize, larger than the image (zero padding) |
| RandomHorizontalFlip | reference: `p=1`, two flips |
| ColorJitter | reference: each factor alone, all four with random order, inside RandomApply, after a blur (mean of the blurred image) |
| RandomApply | reference: colour ops, one blur |
| RandomGrayscale | reference |
| GaussianBlur | reference: square and `(kx, ky)` kernels, fixed and random sigma, two blurs in one pipeline |
| RandomSolarize | reference off the threshold; values exactly on it may differ (checked to be on it) |
| Normalize | reference (in every recipe) |
| bfloat16 output | ImageNet and SimCLR recipes within one bfloat16 step of the float32 reference; the one-pass ImageNet recipe rounds like torch for more than 95% of values |
| ColorOp, NeighbourhoodOp | reference: user-defined gamma and box blur (reflect, clamp), helper-name isolation, compile-error message |

Also on CUDA: batch tensor, CHW list, HWC list and a non-contiguous HWC view
give identical output; dtype, shape, empty input and input-error messages;
`torch.manual_seed` repeats a run; `same_on_batch` gives identical values;
sampled RandomResizedCrop areas and aspects, flip rates, ColorJitter ranges
and all 24 op orders match torchvision's sampler statistically; SlangPy's
SlangPy's Python torch bridge is allowed only for slangvision's own calls
(the process-wide flag is restored afterwards). With `slangpy-torch`
installed the same test also compares the Python and compiled bridges'
output; without it (the 2026-11-04 run below) both calls use the Python
bridge.
`tests/test_api.py` checks pipeline-structure rules, argument
validation and custom-op declarations without a GPU or torch.

## Validation records

Record the GPU, driver, Python, torch, torchvision and SlangPy versions and
the pass counts for every validation; never present an older record as
validation of newer code.

- 2026-11-03, as `slangimage.augment` (before the move to this package):
  NVIDIA H100 NVL, driver 580 (CUDA 13.0), Python 3.12, torch 2.13.0+cu130,
  torchvision 0.28.0, SlangPy 0.42.0: the 41 CUDA tests and 34 GPU-free
  tests passed. A fresh environment with SlangPy 0.43.1 and no
  `slangpy-torch` ran on SlangPy's Python torch bridge: an ImageNet recipe
  matched torchvision to 9.3e-5 (float32, normalized units).
- 2026-11-04, slangvision 0.1.0 (the wheel built from this repo, after the
  move and the split of the shader library into topic files): NVIDIA RTX
  PRO 4500 Blackwell, driver 580.126.16 (CUDA 13.0), Ubuntu 24.04, Python
  3.12.3, torch 2.13.0+cu130, torchvision 0.28.0, no `slangpy-torch` (SlangPy's
  Python torch bridge). After the review fixes, `scripts/cuda_box/run_tests.sh`:
  84 passed (45 CUDA, 39 GPU-free) with SlangPy 0.42.0, and 84 passed in a
  fresh environment where a plain install resolved SlangPy 0.43.1. The
  Python examples in README.md and docs/GUIDE.md ran. Benchmark: see the
  header of `benchmarks/compare_pipelines.py`.
- 2026-11-04, same host, two limits checked:
  - 32-bit indexing: SlangPy addresses float32 tensors beyond 4 GiB
    correctly. At the largest allowed batch (14,266 images of 224x224,
    2^31 - 51,200 values, 8 GiB of float32 output), every image matched
    the single-image result exactly, for one-pass pipelines and for a
    three-pass blur pipeline with 8 GiB intermediates. One more image is
    rejected with the 32-bit message, so the limit is in values, not bytes.
  - NVRTC: torch 2.13.0+cu130 from PyPI depends on `cuda-toolkit` 13.0.3,
    which installs NVRTC, cudart and `cuda_runtime.h` under
    `site-packages/nvidia/cu13`. With no system CUDA and no extra pip
    install, pipelines ran after the README's `libnvrtc.so` link plus both
    `CUDA_PATH` and `LD_LIBRARY_PATH`; leaving out any of those three failed
    with SlangVision's NVRTC error. (This manual setup is what the automatic
    NVRTC setup below now does.)
- 2026-11-04, NVRTC setup with consent (`python -m slangvision setup`),
  same host, fresh environments, no `CUDA_PATH` or `LD_LIBRARY_PATH`, no CUDA
  Toolkit:
  - torch 2.13.0 (NVRTC 13.0): before `setup` the first call stops and asks
    for it (with the environment-variable alternative, which also ran);
    after it, 95 passed; after `undo` the first call asks again.
  - torch 2.8.0 + torchvision 0.23 (CUDA 12.8 wheels, the older
    `nvidia/cuda_nvrtc` layout): `setup` linked NVRTC 12.8 and pipelines ran
    (the full suite passed, 90, in the earlier automatic-link version).
  - torch 2.15.0 + torchvision 0.29.2 (NVRTC 13.2, newer than the driver's
    13.0): the first call stops with the version message and the
    `cuda-toolkit` 13.0 command. After running it (pip reports a dependency
    conflict but installs), `setup` linked 13.0, the full suite passed (90,
    automatic-link version) and a bf16 training step and `torch.compile`
    still ran.
  - With the box scripts' own `CUDA_PATH`: 95 passed and no link was made.
- 2026-11-04, bfloat16 output, same host, torch 2.13.0, SlangPy 0.42.0 (the
  box scripts' own `CUDA_PATH`): 97 passed. A separate check wrote 4,194,304
  values (including rounding ties, ±inf, NaN and values near the maximum)
  through the int16 view on SlangPy 0.42.0 and 0.43.1: identical to
  `.to(torch.bfloat16)`. Batch 256 at 224x224: ImageNet recipe 1.74 ms for
  float16 and bfloat16, 1.76 ms for float32; SimCLR 3.11, 3.12 and 3.31 ms.
