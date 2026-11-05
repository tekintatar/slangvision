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

- **Fast.** An ImageNet augmentation batch of 256 takes 1.7 ms on an RTX PRO
  4500, against 20 ms for torchvision's batched transforms (one setting for
  the whole batch), 19 ms for Kornia, and 203 ms for torchvision one image at
  a time on the GPU (36 ms of it GPU time; the rest is launching about 86
  small kernels per image).
- **torchvision's results.** Each transform matches torchvision's float
  functional image by image (about 1e-5 in float32). Pipelines that run
  torchvision on uint8 or PIL images differ by its rounding, up to 1/255 per
  op ([Accuracy](https://github.com/tekintatar/slangvision/blob/main/docs/GUIDE.md#accuracy)).
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

This page describes `main`, which has transforms not yet released on PyPI
(see the [changelog](https://github.com/tekintatar/slangvision/blob/main/CHANGELOG.md)).
For them, install from GitHub:
`pip install git+https://github.com/tekintatar/slangvision`.

No CUDA Toolkit or environment variables are needed: SlangVision compiles
its kernels with the NVRTC that pip installs with PyTorch, and `setup` adds
the one link SlangPy needs to find it. The first call of a pipeline compiles
it (about 1 s, plus about 1 s to set up the GPU device once per process).
If NVRTC is newer than your driver, or you use your own CUDA Toolkit, see
[Requirements](https://github.com/tekintatar/slangvision/blob/main/docs/GUIDE.md#requirements)
in the guide; SlangVision's errors say which versions it found.

## Transforms

torchvision's names and parameters, per-image random values:

- **Geometry:** `RandomResizedCrop`, `Resize` (bilinear or bicubic),
  `CenterCrop`, `RandomCrop`, `Pad`, `RandomHorizontalFlip`,
  `RandomVerticalFlip`.
- **Warps:** `RandomRotation`, `RandomAffine`, `RandomPerspective`.
- **Colour:** `ColorJitter`, `RandomPhotometricDistort`,
  `RandomChannelPermutation`, `Grayscale`, `RandomGrayscale`,
  `RandomSolarize`, `RandomInvert`, `RandomPosterize`,
  `RandomAutocontrast`, `RandomEqualize`, `GaussianNoise`.
- **Filters:** `GaussianBlur`, `RandomAdjustSharpness`.
- **Policies:** `RandAugment`, `TrivialAugmentWide`, `AutoAugment`, and
  `TimmRandAugment("rand-m9-mstd0.5-inc1")`, timm's RandAugment with
  Pillow's ops (DeiT, ConvNeXt, Swin).
- **Composition and output:** `RandomApply`, `RandomChoice`, `Normalize`,
  `RandomErasing`.

Every random transform takes `same_on_batch=False`. A SimCLR pipeline:

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
has the full rules, recipes (CIFAR, DeiT/ConvNeXt, ThreeAugment), decoder
recipes (torchvision, nvImageCodec, DALI, DataLoader), a mapping from
Albumentations, accuracy and limits.

### MixUp and CutMix

```python
import slangvision as A

mix = A.TimmMixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)
# x, y = mix(train(images), labels)       # after the pipeline, on its output batch
```

`MixUp` and `CutMix` are torchvision's, draw for draw; `TimmMixup` is
timm's `Mixup`, with per-batch, per-pair or per-image mixing.

### Super resolution (Real-ESRGAN)

```python
# was: from basicsr.utils.img_process_util import USMSharp
#      from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
#      from basicsr.utils import DiffJPEG
from slangvision.sr import (DiffJPEG, USMSharp, random_add_gaussian_noise_pt,
                            random_add_poisson_noise_pt)
```

Drop-in replacements for the BasicSR functions that Real-ESRGAN's training
step calls on the GPU, with the same arguments and results: USM sharpening of
the targets takes 1.5 ms instead of 118 ms per batch of 12 at 400×400 (RTX
PRO 4500), and making Real-ESRGAN's whole low-resolution batch drops from
127 ms to under 8 ms. Details, and where BasicSR's DiffJPEG differs from a
real JPEG encoder, are in
[SR.md](https://github.com/tekintatar/slangvision/blob/main/docs/SR.md).

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
read neighbouring pixels (blurs, filters) subclass `A.NeighborhoodOp` the
same way. `slangvision.testing.max_reference_error(pipe, images)` checks a
pipeline against torchvision and your references, and `train.source` shows
the generated Slang.

## Performance

Batch 256, 512×384 uint8 in, 224×224 fp16 out, one NVIDIA RTX PRO 4500
Blackwell, ms per batch, wall time (GPU time):

| Recipe | SlangVision | torchvision, batched | torchvision, one image at a time | Kornia (per-image random) |
|---|---:|---:|---:|---:|
| ImageNet: crop, flip, ColorJitter, Normalize | **1.66** (1.47) | 20.2 (20.2) | 203 (35.7) | 19.3 (14.0) |
| Evaluation: Resize, CenterCrop | **0.87** (0.66) | 5.60 (5.80) | 27.6 (7.65) | — |
| SimCLR: + grayscale, blur, solarize | **2.54** (2.30) | 20.5 (26.5) | 221 (40.4) | 29.7 (23.8) |

torchvision's batched mode does less work (one random setting for all 256
images); its per-image loop is the like-for-like comparison, and most of its
wall time is launching small kernels. `benchmarks/compare_pipelines.py`
reproduces this table; raw data in
[`benchmarks/results/rtx_pro_4500_2026-11-05.json`](https://github.com/tekintatar/slangvision/blob/main/benchmarks/results/rtx_pro_4500_2026-11-05.json)
(an earlier H100 run is in the script's header).

In real training, decoding usually decides throughput. Training from scratch
on an ImageNet subset (45,000 JPEGs, batch 256, bf16) on one H100 NVL with 18
CPU cores, images per second:

| Input pipeline | ResNet-18 | ResNet-50 |
|---|---:|---:|
| DataLoader (16 workers): decode + torchvision augmentation | 2,721 | 2,671 |
| DataLoader decode + SlangVision | 3,781 | 3,254 |
| DALI decode + DALI augmentation | 8,655 | 3,518 |
| DALI decode + SlangVision | 8,393 | 3,395 |

Moving augmentation to the GPU gives 1.4× (ResNet-18) with CPU decoding of
ImageNet-sized photos; with larger ones decode on the GPU too (the guide
has an FGVC-Aircraft benchmark at about 0.7 MP).
With DALI's GPU decoding, training runs within about 3% of DALI's own
augmentation, with torchvision's results instead of DALI's approximations.
This table is indicative: it was measured with an earlier version of the
kernels (shipped as `slangimage.augment`), one run per row, with a training
script that is not in this repository. Method and caveats are in the
[guide](https://github.com/tekintatar/slangvision/blob/main/docs/GUIDE.md#performance).

## Limits

- CUDA only (Linux with an NVIDIA GPU), and not inside CUDA graph capture.
- Run pipelines from one thread (SlangPy devices are thread-affine).
- Not yet: AugMix, masks and boxes.

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
