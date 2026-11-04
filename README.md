# SlangVision

[![PyPI](https://img.shields.io/pypi/v/slangvision)](https://pypi.org/project/slangvision/)
[![checks](https://github.com/tekintatar/slangvision/actions/workflows/checks.yml/badge.svg)](https://github.com/tekintatar/slangvision/actions/workflows/checks.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/tekintatar/slangvision/blob/main/LICENSE)

**GPU training augmentation for PyTorch: torchvision's transforms and
results, a separate random draw for every image, fused into a few CUDA
kernels written in plain [Slang](https://shader-slang.org/).**

```python
import torch
import slangvision as A

train = A.Compose([
    A.RandomResizedCrop(224),
    A.RandomHorizontalFlip(),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
images = torch.randint(0, 256, (64, 3, 480, 640), dtype=torch.uint8, device="cuda")
x = train(images)   # (64, 3, 224, 224) float16; a list of decoded images also works
# Compose(..., dtype=torch.bfloat16) or torch.float32 for other output types
```

- **Fast.** An ImageNet augmentation batch of 256 takes 1.3 ms on an H100,
  against 7.3 ms for torchvision's batched transforms (one setting for the
  whole batch), 389 ms for torchvision image by image, and 17 ms for Kornia.
- **torchvision's results.** Each transform matches torchvision's float
  functional image by image (about 1e-5 in float32), so switching does not
  change what your model sees.
- **Per-image randomness.** Every image gets its own crop, flip, jitter
  factors and jitter order; `same_on_batch=True` on a transform shares them.
  Seeded from `torch.manual_seed`, with no host sync.
- **Reads any decoder's output in place**: a uint8 batch, or a list of
  differently sized images from `torchvision.io`, nvImageCodec, DALI or
  DataLoader workers (`layout="hwc"` for `(H, W, 3)`).
- **Your own ops in a few lines.** A custom augmentation is a Python class
  with one Slang function, fused into the same kernels as the built-ins.
- **Light.** Pure Python plus Slang source: no compiled extension to build.

**Status: pre-alpha.** APIs may still change.

## Install

Linux with an NVIDIA GPU, Python 3.10–3.13 and PyTorch with CUDA:

```bash
pip install slangvision
python -m slangvision setup     # once per environment
```

No CUDA Toolkit or environment variables are needed. SlangVision compiles
its kernels at run time with NVRTC, CUDA's run-time compiler, and uses the
copy that pip installs with PyTorch. SlangPy only finds NVRTC under a name
PyTorch's copy does not have, so `setup` adds one link, `libnvrtc.so`, to
SlangPy's package folder; `python -m slangvision undo` removes it, and
`python -m slangvision` shows what will be used. Until then, the first
pipeline call stops and asks for it. SlangVision also keeps a folder of
links in `~/.cache/slangvision`, and it never replaces a `libnvrtc.so` it
did not make.

The first call of a pipeline compiles it (about 1 s, plus about 1 s to set
up the GPU device once per process); later calls reuse the compiled
kernels.

### If NVRTC is newer than your driver

NVRTC must be no newer than the "CUDA Version" `nvidia-smi` prints: the
driver rejects code from a newer one. Current PyTorch (2.15) installs NVRTC
13.2, while NVIDIA driver 580 supports CUDA 13.0. SlangVision checks this
before it starts and says which versions it found. Either update the
driver, or install the NVRTC that matches it:

```bash
pip install "cuda-toolkit[nvrtc,cudart]==13.0.*"   # use your driver's CUDA version
```

pip then reports a dependency conflict (torch asks for 13.2), but it installs
the packages, and PyTorch keeps working: we trained a model with it, with
and without `torch.compile`.

### Your own CUDA Toolkit

A system-wide CUDA Toolkit is used instead when NVRTC is on the system
library path (on Ubuntu, `cuda-nvrtc-dev-13-0` and `cuda-cudart-dev-13-0`
from NVIDIA's apt repository, then `sudo ldconfig`), and so is a toolkit you
point to with `CUDA_PATH` (set `LD_LIBRARY_PATH` to its library folder too).
Neither needs `setup`, and SlangVision changes nothing. Its errors say which NVRTC and driver
versions it saw.

## Transforms

`RandomResizedCrop`, `Resize`, `CenterCrop`, `RandomHorizontalFlip`,
`ColorJitter`, `RandomGrayscale`, `GaussianBlur`, `RandomSolarize`,
`RandomApply` and `Normalize`, with torchvision's names and parameters,
including bilinear or bicubic resizing. Every random transform takes
`same_on_batch=False`. A SimCLR pipeline:

```python
import slangvision as A

simclr = A.Compose([
    A.RandomResizedCrop(224),
    A.RandomHorizontalFlip(),
    A.RandomApply([A.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    A.RandomGrayscale(p=0.2),
    A.RandomApply([A.GaussianBlur(23)], p=0.5),
    A.RandomSolarize(0.5, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

Pipelines are checked when you build them: geometry comes first and fixes
the output size, and `Normalize` comes last. The
[guide](https://github.com/tekintatar/slangvision/blob/main/docs/GUIDE.md)
has the full rules, decoder recipes (torchvision, nvImageCodec, DALI,
DataLoader), accuracy and limits.

## Custom ops

```python
import slangvision as A

class RandomGamma(A.ColorOp):
    """Gamma correction with a random exponent per image."""
    params = {"gamma": A.Uniform(0.8, 1.25)}
    slang = "float4 apply(float4 c, float gamma) { return clamp01(pow(c, float4(gamma))); }"

    def reference(self, img, gamma):          # optional: torch, one (3, H, W) image
        return img.clamp(0, 1) ** gamma

train = A.Compose([A.RandomResizedCrop(224), A.RandomHorizontalFlip(), RandomGamma(p=0.5)])
```

`apply` gets the pixel and one value per parameter, drawn per image. Ops that
read neighbouring pixels (blurs, filters) subclass `A.NeighbourhoodOp` the
same way. `slangvision.testing.max_reference_error(pipe, images)` checks a
pipeline against torchvision and your references, and `pipe.source` shows the
generated Slang.

## Performance

Batch 256, 512×384 uint8 in, 224×224 fp16 out, one NVIDIA H100 NVL, ms per
batch:

| Recipe | SlangVision | torchvision, batched | torchvision, per image | Kornia (per-image random) |
|---|---:|---:|---:|---:|
| ImageNet: crop, flip, ColorJitter, Normalize | **1.31** | 7.32 | 389 | 16.6 |
| Evaluation: Resize, CenterCrop | **0.68** | 2.90 | 51.9 | — |
| SimCLR: + grayscale, blur, solarize | **2.30** | 7.6–12.6 | 390 | 22.3 |

torchvision's batched mode does less work (one random setting for all 256
images); its per-image loop is the like-for-like comparison.

In real training, decoding usually decides throughput. Training from scratch
on an ImageNet subset (45,000 JPEGs, batch 256, bf16) on one H100 NVL with 18
CPU cores, images per second:

| Input pipeline | ResNet-18 | ResNet-50 |
|---|---:|---:|
| DataLoader (16 workers): decode + torchvision augmentation | 2,721 | 2,671 |
| DataLoader decode + SlangVision | 3,781 | 3,254 |
| DALI decode + DALI augmentation | 8,655 | 3,518 |
| DALI decode + SlangVision | 8,393 | 3,395 |

Moving augmentation to the GPU gives 1.4× (ResNet-18) with CPU decoding.
With DALI's GPU decoding, training runs within about 3% of DALI's own
augmentation, with torchvision's results instead of DALI's approximations.
Both tables were measured with the same kernels while they shipped as
`slangimage.augment`. `benchmarks/compare_pipelines.py` reproduces the first
(a later run on an RTX PRO 4500 is in its header); the training runs used a
script that is not in this repository. Method and caveats are in the
[guide](https://github.com/tekintatar/slangvision/blob/main/docs/GUIDE.md#performance).

## Limits

- CUDA only (Linux with an NVIDIA GPU), and not inside CUDA graph capture.
- Run pipelines from one thread (SlangPy devices are thread-affine).
- Not yet: RandAugment, TrivialAugment and the ops they use, affine and
  perspective geometry, MixUp and CutMix, masks and boxes.

## Related

[SlangImage](https://github.com/tekintatar/slangimage) is the image-processing
sibling: 44 photo-editing operators (bilateral, CLAHE, NL-means, colour,
geometry, stylization) on Metal, CUDA and Vulkan, built the same way.

## License

Copyright 2026 Tekin Tatar. Licensed under
[Apache-2.0](https://github.com/tekintatar/slangvision/blob/main/LICENSE) (see
[NOTICE](https://github.com/tekintatar/slangvision/blob/main/NOTICE)).
SlangVision is built on Slang and SlangPy but is not affiliated with the
Slang project or Khronos.
