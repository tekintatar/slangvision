# SlangVision guide

SlangVision runs training augmentation on CUDA for PyTorch. You
write a pipeline with torchvision's transform names; SlangVision compiles it
into a few fused Slang kernels that draw separate random values for every
image in the batch.

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
x = train(images)      # (64, 3, 224, 224) float16, on the current CUDA stream
```

- **Per-image randomness.** On a batch, torchvision's v2 transforms draw one
  random setting for the whole batch; getting a different crop and colour
  for every image takes a Python loop over the images. `slangvision`
  draws per image by default; `same_on_batch=True` on a transform shares its
  values across the batch.
- **torchvision results.** Each op matches torchvision's float functionals
  image by image, within float rounding (see [Accuracy](#accuracy)).
- **Fused.** A pipeline compiles into a handful of kernels (one pass per run
  of colour ops, one more per blur), and reads the decoded images where they
  already are.
- **Your own ops** in a few lines of Python and one Slang function
  ([Custom ops](#custom-ops)).

It does not decode images: pair it with your decoder
([Inputs and decoders](#inputs-and-decoders)).

Contents: [Requirements](#requirements) · [How a pipeline runs](#how-a-pipeline-runs)
· [Pipeline rules](#pipeline-rules) · [Randomness](#randomness) ·
[Transforms](#transforms) · [Policies](#policies) · [Recipes](#recipes) ·
[Inputs and decoders](#inputs-and-decoders) · [MixUp and CutMix](#mixup-and-cutmix)
· [Custom ops](#custom-ops) · [Coming from Albumentations](#coming-from-albumentations)
· [Accuracy](#accuracy) · [Performance](#performance) · [Limits](#limits).
Super-resolution degradations (Real-ESRGAN's) have their own page:
[SR.md](https://github.com/tekintatar/slangvision/blob/main/docs/SR.md).

## Requirements

Linux, an NVIDIA GPU and PyTorch with CUDA. SlangVision compiles its
kernels at run time with NVRTC, CUDA's run-time compiler, and uses the copy
that pip installs with PyTorch, after a one-time `setup`:

```bash
pip install slangvision torchvision
python -m slangvision setup
```

torchvision is needed only for `slangvision.testing`. SlangPy's
compiled torch bridge (`slangpy-torch`) is optional: without it, slangvision
uses SlangPy's pure-Python torch bridge for its own calls, which adds under
0.1 ms per batch.

The first call of a pipeline compiles it (about 2 s in a fresh environment,
including setting up the GPU device once per process); later calls reuse the
compiled kernels. The CUDA code NVRTC makes is also cached on disk, in
`~/.cache/slangvision/kernels` (or `$XDG_CACHE_HOME/slangvision/kernels`), so
the first call in a later process with the same pipeline takes about 1.5 s. `SLANGVISION_CACHE_DIR` moves the cache
(empty turns it off); deleting the folder is always safe.

`setup` adds one link, `libnvrtc.so`, to SlangPy's package folder, because
SlangPy looks for NVRTC under that name and PyTorch's copy has a versioned
one; `python -m slangvision undo` removes it, and `python -m slangvision`
shows what will be used. SlangVision also keeps a folder of links in
`~/.cache/slangvision`, and never replaces a `libnvrtc.so` it did not make.

### If NVRTC is newer than your driver

NVRTC must be no newer than the "CUDA Version" `nvidia-smi` prints: the
driver rejects code from a newer one. In November 2026 PyTorch 2.15 installs
NVRTC 13.2, while NVIDIA driver 580 supports CUDA 13.0. SlangVision checks
this before it starts and says which versions it found. Either update the
driver, or install the NVRTC that matches it:

```bash
pip install "cuda-toolkit[nvrtc,cudart]==13.0.*"   # use your driver's CUDA version
```

pip then reports a dependency conflict (torch asks for 13.2), but it installs
the packages, and PyTorch keeps working: we trained a model with it, with
and without `torch.compile`. Do this after your other installs: a later
`pip install` of torch or torchvision, or of anything that requires them
(`pip install -e ".[dev]"` included), puts PyTorch's NVRTC back, and
SlangVision stops with the same error until you pin it again.

### Your own CUDA Toolkit

A system-wide CUDA Toolkit is used instead when NVRTC is on the system
library path (on Ubuntu, `cuda-nvrtc-dev-13-0` and `cuda-cudart-dev-13-0`
from NVIDIA's apt repository, then `sudo ldconfig`), and so is a toolkit you
point to with `CUDA_PATH` (set `LD_LIBRARY_PATH` to its library folder too).
Neither needs `setup`, and SlangVision changes nothing. Its errors say which
NVRTC and driver versions it saw.

## How a pipeline runs

`Compose` turns the pipeline into a few GPU kernels when it is built, and
compiles them on the first call:

- A **sampler** kernel draws every image's random values (crop box, flip,
  factors, op choices) into a small table.
- A **pass** is one kernel that computes every output pixel once and writes
  the image. The first pass reads your images, crops and resizes them, and
  applies the flips and colour ops that follow.
- **Colour ops** (ColorJitter, grayscale, solarize, noise, your `ColorOp`)
  change one pixel at a time, so they run inside a pass at no extra memory
  traffic.
- **Neighbourhood ops** read the pixels around each one: blurs, sharpness,
  your `NeighborhoodOp`. **Warps** (rotation, affine, perspective, and the
  geometric ops of the policies) read the image at moved positions. Each of
  them starts a new pass over the image as it is at that point.
- A **pre-pass** runs before a pass when an op needs a statistic of each
  image: ColorJitter's contrast (the mean gray), autocontrast (each
  channel's minimum and maximum) and equalize (histograms) read the image up
  to that op and reduce it.

Between passes the image is stored in float16 (float32 for float32 output).
When a pre-pass in the first pass would make the costly resample run twice
(bicubic, or large photos), the first pass resamples once into a float32
copy instead (see [Performance](#performance)).
The number of passes, not the number of colour ops, is what mostly sets the
cost.

## Pipeline rules

Checked when you build the `Compose`:
- Geometry comes first: at most one `RandomResizedCrop` or `Resize`, then
  any sequence of `Pad`, `CenterCrop` and `RandomCrop` (each pads with its
  own `padding_mode`, so they compose as torchvision applies them one after
  the other). Like torchvision, reflect padding must be smaller than the
  image and symmetric padding at most its size, or the call raises
  `ValueError`. All images must end up the same size: `Resize(int)`
  (shorter side, aspect kept) needs a crop after it; `Resize((h, w))`,
  `RandomResizedCrop` and the crops fix the size; with only `Pad` or no
  geometric transform, all input images must be the same size. A
  `RandomCrop` larger than a (padded) image raises `ValueError`, as in
  torchvision.
- Warps (`RandomRotation`, `RandomAffine`, `RandomPerspective`) come after
  the resize, crops and pads, in any order with the other transforms; a
  warp before them raises `ValueError`.
- Flips go anywhere after the geometric transforms, but not after a
  `NeighborhoodOp` you define unless a warp comes between them. They commute
  with colour ops and Gaussian blur; a flip after a warp is applied after it.
- Each transform object appears once; create a second one to use it twice.
- `Normalize`, if present, comes after every transform except
  `RandomErasing`, which comes last (several erase several boxes, in order).
  Without `Normalize` the output is in [0, 1].

`Compose(transforms, dtype=torch.float16)` is the default; `torch.bfloat16`
(for models trained in bfloat16, at the same speed as float16) and
`torch.float32` are the other choices. Passes between blurs, warps and the
final output keep float16 (float32 for float32 output). The output is a
contiguous `(B, 3, H, W)` tensor (`(B, 1, H, W)` after
`Grayscale(num_output_channels=1)`); for `channels_last`, call `.to(...)` on
it.

A `Compose` is fixed when it is built: changing one of its transforms
afterwards is not supported. To change parameters during training (a
magnitude schedule), build a new `Compose`; each new set of values compiles
once, which takes a few seconds.

## Randomness

Random values come from a counter-based generator in the sampler kernel,
seeded on every call from PyTorch's CUDA generator: `torch.manual_seed`
makes runs repeat, and there is no host sync. Values are drawn per image;
`same_on_batch=True` on a transform gives every image the same values for
that transform (for `RandomApply`, the same apply/skip decision).

## Transforms

| transform | as in torchvision, except |
|---|---|
| `RandomResizedCrop(size, scale, ratio, interpolation="bilinear", antialias=True)` | box drawn per image |
| `Resize(size, interpolation="bilinear", antialias=True)` | an int resizes the shorter side and must be followed by `CenterCrop` or `RandomCrop`, so all images match |
| `CenterCrop(size)` | — (a crop larger than the image pads with zeros, as torchvision does) |
| `RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant")` | position drawn per image; `fill` is on the 0-1 scale |
| `Pad(padding, fill=0, padding_mode="constant")` | `fill` is on the 0-1 scale |
| `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)` | decided per image |
| `RandomRotation(degrees, interpolation="nearest", expand=False, center=None, fill=0)` | angle drawn per image; `expand=True` is not supported (it changes each image's size); `fill` is on the 0-1 scale |
| `RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=0, center=None)` | angle, translation, scale and shear drawn per image; `fill` is on the 0-1 scale |
| `RandomPerspective(distortion_scale=0.5, p=0.5, interpolation="bilinear", fill=0)` | corners drawn per image; `fill` is on the 0-1 scale |
| `ColorJitter(brightness, contrast, saturation, hue)` | factors and op order drawn per image |
| `RandomApply(transforms, p=0.5)` | decided per image; holds colour ops, or exactly one blur, warp or `NeighborhoodOp` (not flips: they have their own `p`) |
| `RandomChoice(transforms, p=None)` | chosen per image (weights `p`, equal by default); holds colour ops and at most one blur, warp or `NeighborhoodOp` |
| `RandomGrayscale(p=0.1)` | decided per image; output keeps 3 channels |
| `Grayscale(num_output_channels=1)` | with 1 (torchvision's default) the pipeline returns `(B, 1, H, W)`; it must then be at the top level (not in `RandomApply` or `RandomChoice`), and `Normalize` or a `RandomErasing` value after it takes one value, a warp's `fill` one value. Ops after it see three equal channels and the output keeps the first: contrast and grayscale use torchvision's gray weights, which sum to 0.9999, so they differ from torchvision's one-channel result by up to 1e-4 |
| `RandomPhotometricDistort(brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5)` | which ops apply, their factors, whether contrast comes before saturation or after hue, and the channel order drawn per image |
| `RandomChannelPermutation()` | one of the six channel orders drawn per image |
| `GaussianBlur(kernel_size, sigma=(0.1, 2.0))` | sigma drawn per image; the kernel radius must be smaller than the image, as torchvision's reflect padding requires |
| `RandomSolarize(threshold, p=0.5)` | `threshold` is on the 0-1 scale (0.5, not 128) |
| `RandomInvert(p=0.5)` | decided per image |
| `RandomPosterize(bits, p=0.5)` | torchvision's float behaviour: `floor(value * 2**bits) / 2**bits` (its uint8 path keeps the top bits instead: white at 4 bits is 0.9375 here, 240/255 there) |
| `RandomAdjustSharpness(sharpness_factor, p=0.5)` | decided per image; reads neighbouring pixels, so it starts a pass like a blur |
| `RandomAutocontrast(p=0.5)` | decided per image; each image's per-channel min and max come from a pre-pass |
| `RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation="nearest", fill=None)` | ops, and the signs of their magnitudes, drawn per image; comes after the resize and crops; colour ops follow torchvision's float behaviour (see below) |
| `TrivialAugmentWide(num_magnitude_bins=31, interpolation="nearest", fill=None)` | torchvision's: one op per image at a random magnitude bin; one pass after the resize and crops |
| `AutoAugment(policy="imagenet", interpolation="nearest", fill=None)` | torchvision's: one of the policy's 25 sub-policies per image (`"imagenet"`, `"cifar10"`, `"svhn"` or `AutoAugmentPolicy`); two passes |
| `TimmRandAugment(config="rand-m9-mstd0.5-inc1", interpolation="random", fill=(124, 116, 104) / 255, translate_pct=0.45)` | timm's RandAugment with Pillow's ops (see below); ops, magnitudes, signs and interpolation drawn per image; comes after the resize and crops |
| `RandomEqualize(p=0.5)` | decided per image; as in torchvision for float images, the image is quantized to 256 levels and equalized with PIL's scheme, from per-image histograms built by a pre-pass; values outside [0, 1] (bicubic overshoot) are clamped first, where torchvision's cast is undefined |
| `Normalize(mean, std)` | comes after everything except `RandomErasing`; one value applies to every channel |
| `GaussianNoise(mean=0.0, sigma=0.1, clip=True)` | adds `mean + sigma * N(0, 1)`, new for every pixel, channel and image (`same_on_batch=True`: one field for the whole batch) (SlangVision's own generator, so not torch's values), then clamps with `clip=True`; a colour op, before `Normalize` |
| `RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)` | box drawn per image; comes last (after `Normalize`); `value="random"` fills per-pixel N(0, 1) noise from SlangVision's own generator |

Every random transform takes `same_on_batch=False`. `interpolation` is
`"bilinear"` or `"bicubic"` (or torchvision's `InterpolationMode`), with
torchvision's filters: with `antialias=True` (the default) PyTorch's
antialias kernels, cubic with a = -0.5 like PIL, and without it classic
bicubic with a = -0.75. Bicubic float output can overshoot [0, 1], as
torchvision's does; it costs about twice as much as bilinear.

The warps (`RandomRotation`, `RandomAffine`, `RandomPerspective`) sample
the image as it is at their point of the pipeline with `"nearest"` or
`"bilinear"` interpolation, as torchvision does for tensors, and keep its
size. Each runs as its own pass, and costs about 0.2 ms per batch of 256 at
224x224 (see Performance). Pixels that come from outside the image get
`fill` the way torchvision's grid sampling does: with bilinear
interpolation, edge pixels blend toward `fill` (with `fill=0`, toward
black).

## Policies

`RandAugment` is torchvision's: for each image, `num_ops` times, one of 14
ops (identity, shear x/y, translate x/y, rotate, brightness, colour,
contrast, sharpness, posterize, solarize, autocontrast, equalize), each at
torchvision's magnitude for `magnitude` and a random sign where it has one.
Each op runs as its own pass, so `RandAugment()` adds two passes. Its
magnitudes are torchvision's float32 tables; torch computes those with a
vectorized CPU kernel whose last bit varies between CPUs, and SlangVision
computes them the plain way.

`TrivialAugmentWide` (the policy behind torchvision's newer classification
weights) and `AutoAugment` are torchvision's too, on the same op pass.
TrivialAugmentWide picks one of the same 14 ops per image at a magnitude bin
drawn uniformly, over its wider ranges (shear to 0.99, translate to 32
pixels, rotate to 135 degrees, enhance factors 0.01 to 1.99, posterize down
to 2 bits). AutoAugment picks one of the policy's 25 sub-policies per image
and applies each of its two ops with that op's probability, at its
magnitude bin of 10; its ops add invert. Its policies are torchvision's
tables, checked entry by entry against torchvision in the tests. The
magnitude tables are computed once, the plain float32 way; torch's own can
differ in the last bit by CPU (on an Arm Mac, TrivialAugmentWide's translate
at bin 15 of 31 is 15.999999, so torchvision shifts 15 pixels there and
SlangVision 16).

`TimmRandAugment` is timm's (`timm.data.rand_augment_transform`), the one
DeiT, ConvNeXt, Swin and most timm recipes train with. It takes timm's
config string: `"rand-m9-mstd0.5-inc1"` picks `n=2` of 15 ops per image
(auto contrast, equalize, invert, rotate, posterize, solarize, solarize-add,
colour, contrast, brightness, sharpness, shear x/y, translate x/y), with
repeats, applies each with probability `p=0.5` at magnitude `m=9` plus
Gaussian noise of standard deviation `mstd=0.5` (above 100: uniform in [0,
m]), clamped to [0, `mmax`=10]; `inc1` uses the variants whose strength
grows with the magnitude. The string is read as current timm reads it
(timm releases before August 2026 took `inc0` as `inc1`); transform sets
(`t...`) are not supported. The ops are Pillow's, as timm runs them:

- rotate, shear and translate resample like `Image.transform`: the output
  pixel centre goes through Pillow's matrix (rotation about the image
  centre, shear about the top-left corner), pixels from outside the image
  get `fill` with no blend at the edge, and `interpolation="random"` (timm's
  training default) picks bilinear or Pillow's bicubic (a = -1, clamped)
  per op; `"bilinear"` or `"bicubic"` fixes it, as timm's
  `--train-interpolation` does. `fill` defaults to the ImageNet mean that
  timm's `create_transform` passes; timm without it fills (128, 128, 128).
  `translate_pct` is timm's hparam (0.45; AugMix uses 0.3).
- colour, contrast and brightness blend like `ImageEnhance` (Pillow's luma
  weights; contrast with the mean of Pillow's uint8 luma, rounded to a
  level), sharpness with Pillow's 3x3 smoothing.
- posterize, solarize and solarize-add act on the value as uint8
  (torchvision's float-to-uint8 conversion), so they give Pillow's levels
  and thresholds; posterize keeps the image at 8 or more bits (magnitudes
  above 10 with `mmax`), as timm does, and at 0 bits or fewer, which timm
  asks for at magnitude 10 with `inc1`, gives black as Pillow 12 does.
  Pillow 13 raises a `ValueError` there, which stops timm's own
  `rand-m9-mstd0.5-inc1` pipeline: 98 of 60,000 images under Pillow 13.0
  and timm 1.0.31, as expected from 2 layers x 0.5 (applied) x 1/15
  (posterize) x 2.3% (the magnitude clamps to 10) = 0.15%.

Pillow rounds every op's result to uint8 (resampling and blends truncate);
SlangVision keeps floats between ops, as it does for torchvision's ops. Each
layer runs as its own pass, like `RandAugment`.

## Recipes

The CIFAR recipe is `[RandomCrop(32, padding=4), RandomHorizontalFlip(),
Normalize(...)]`; with rotation, add `RandomRotation(15)` after the flip.

DeiT's recipe (timm's `create_transform` with DeiT's arguments, also the
base of ConvNeXt's and MAE fine-tuning):

```python
import slangvision as A

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train = A.Compose([
    A.RandomResizedCrop(224, interpolation="bicubic"),
    A.RandomHorizontalFlip(),
    A.TimmRandAugment("rand-m9-mstd0.5-inc1", interpolation="bicubic"),
    A.Normalize(mean, std),
    A.RandomErasing(p=0.25, scale=(0.02, 1 / 3), ratio=(0.3, 1 / 0.3), value="random"),
])
mix = A.TimmMixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)
# x, y = mix(train(images), labels)
```

timm's `RandomErasing(probability=0.25, mode="pixel")` is the
`RandomErasing` line above, after `Normalize`.
`slangvision.testing.max_reference_error` skips noise-filled pixels (no
library can replay another's noise); the box itself is compared.

DeiT III's ThreeAugment is
`[Resize(224, interpolation="bicubic"), RandomCrop(224, padding=4,
padding_mode="reflect"), RandomHorizontalFlip(),
RandomChoice([RandomGrayscale(p=1), RandomSolarize(0.5, p=1),
GaussianBlur(13)]), ColorJitter(0.3, 0.3, 0.3), Normalize(...)]`; DeiT
blurs with PIL's radius-based filter, which `GaussianBlur` only
approximates. `RandomChoice(..., same_on_batch=True)` makes the same choice
for every image.

## Inputs and decoders

`pipe(images)` takes:
- a `(B, 3, H, W)` uint8 CUDA tensor, or
- a list of uint8 CUDA tensors of any sizes, `(3, H, W)` by default or
  `(H, W, 3)` with `layout="hwc"` (a `(B, H, W, 3)` tensor also works with
  `layout="hwc"`).

Images are read in place, without copying them into one buffer
(non-contiguous tensors are made contiguous first). **They must stay valid
until the GPU has run the pipeline.** Tensors PyTorch allocated on
the current stream always are, because PyTorch reuses freed memory only in
stream order. Memory another library owns needs a sync before that library
reuses it. Images decoded on another stream need
`torch.cuda.current_stream().wait_stream(that_stream)` before the call (and
`image.record_stream(torch.cuda.current_stream())` if that stream frees
them).

Decoding usually limits training throughput, so pick a fast decoder:

```py
# torchvision (0.28, 0.29) GPU decode, CHW. Wait for the previous training step
# first: torchvision's decoder can otherwise overwrite its memory (see below).
from torchvision.io import ImageReadMode, decode_jpeg
torch.cuda.current_stream().synchronize()
x = train(decode_jpeg(list_of_jpeg_byte_tensors, mode=ImageReadMode.RGB, device="cuda"))

# nvImageCodec (hardware JPEG decoder on A100/H100-class GPUs), HWC
from nvidia import nvimgcodec
decoder = nvimgcodec.Decoder()
imgs = decoder.decode(list_of_jpeg_bytes, cuda_stream=torch.cuda.current_stream().cuda_stream)
x = train([torch.as_tensor(im, device="cuda") for im in imgs], layout="hwc")
torch.cuda.current_stream().synchronize()   # before `imgs` is freed

# DALI decoding (fn.decoders.image, device="mixed"), HWC. Import
# slangvision before building the DALI pipeline (see below). DALI's
# outputs are valid only until its next run(), so wait for the augmentation
# (not the whole training step) before calling it again.
ready = None
while training:
    if ready is not None:
        ready.synchronize()
    images, labels = dali_pipeline.run()
    x = train([torch.as_tensor(images[i], device="cuda") for i in range(len(images))],
              layout="hwc")
    ready = torch.cuda.Event()
    ready.record()
    ...                                      # training step on x
```

**Import `slangvision` before DALI builds a pipeline or torchvision decodes
on the GPU.** If a GPU decoder runs before SlangPy is loaded, SlangPy
crashes with a segmentation fault while creating its CUDA device (seen with
DALI 2.4, torchvision 0.28-0.29 and SlangPy 0.42/0.43); in the other order
they work in one process. On Linux, `import slangvision` after a GPU decode
prints a `RuntimeWarning` that says so before the crash.

**torchvision's GPU decoder needs a sync while a training step is still
running** (torchvision 0.28). `decode_jpeg(..., device="cuda")` allocates its
output tensors on the current stream but fills them on a private stream that
does not wait for the current one, so it can write into memory that the
previous training step has freed but is still using. In our FGVC-Aircraft
runs this drove the loss to NaN within ten steps, with or without
SlangVision, and once stalled training. Calling
`torch.cuda.current_stream().synchronize()` before each decode fixes it, but
the decoder then no longer overlaps training: 844 images/s for ResNet-50 and
1,587 for ResNet-18, below DALI with SlangVision (1,148 and 3,164). DALI is
the faster GPU decoder.

Without a GPU decoder, keep your `DataLoader` but let the workers only
decode, and augment the batch on the GPU. This pays off for ImageNet-sized
photos (about 0.15 MP); with larger ones (FGVC-Aircraft, about 0.7 MP) the
workers ship whole decoded images instead of 224 px crops, and it was slower
than augmenting on the CPU (see Performance). Have the workers pack each batch
into one buffer: with `collate_fn=list`, PyTorch shares every image between
processes separately, which made training slower than augmenting on the CPU
(2,037 against 2,721 images/s for ResNet-18 in the README's training
benchmark).

```py
class Decoded(torch.utils.data.Dataset):     # the workers only decode
    def __init__(self, files, labels):
        self.files, self.labels = files, labels
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):                # uint8 (3, H, W), any size
        img = torchvision.io.decode_image(self.files[i], mode=torchvision.io.ImageReadMode.RGB)
        return img, self.labels[i]

def collate(batch):                  # runs in the workers
    images, labels = zip(*batch)     # decoded uint8 (3, H, W) tensors
    return (torch.cat([im.reshape(-1) for im in images]),
            torch.tensor([tuple(im.shape) for im in images]), torch.tensor(labels))

loader = torch.utils.data.DataLoader(Decoded(files, labels), batch_size=256, num_workers=16,
                                     pin_memory=True, collate_fn=collate)
for flat, shapes, labels in loader:
    parts = flat.cuda(non_blocking=True).split(shapes.prod(1).tolist())
    x = train([part.view(*s) for part, s in zip(parts, shapes.tolist())])
```

## MixUp and CutMix

They mix images with each other and need the labels, so they run after the
pipeline, on its output batch:

```py
x, y = mix(train(images), labels)       # labels: (B,) class indices or (B, classes)
```

- `MixUp(alpha=1.0, num_classes=None)` and `CutMix(alpha=1.0,
  num_classes=None)` are torchvision's `v2.MixUp` and `v2.CutMix`: one
  weight (and, for CutMix, one box) per batch, each image mixed with the
  previous one, labels one-hot and mixed by the same weight (CutMix: by the
  box's actual area). They draw from torch's CPU generator exactly as
  torchvision does, so under the same `torch.manual_seed` they give
  torchvision's result. torchvision recipes pick one per batch with
  `random.choice` or `v2.RandomChoice([cutmix, mixup])`.
- `TimmMixup(mixup_alpha=1.0, cutmix_alpha=0.0, cutmix_minmax=None,
  prob=1.0, switch_prob=0.5, mode="batch", correct_lam=True,
  label_smoothing=0.1, num_classes=1000)` is timm's `Mixup`, the DeiT,
  ConvNeXt and MAE recipe with `mixup_alpha=0.8, cutmix_alpha=1.0`. Image
  `i` mixes with `B - 1 - i`; `mode="pair"` and `"elem"` draw per mirrored
  pair and per image. timm draws with NumPy, so the random values differ
  from timm's; the rules are the same. The defaults are timm's too: label
  smoothing 0.1 and no CutMix unless `cutmix_alpha` or `cutmix_minmax` is
  set. `mix.mixup_enabled = False` turns mixing off, as timm's training
  script does for its last epochs.

One Slang kernel per call writes the mixed batch (float16, bfloat16 or
float32, as given); the parameters come from torch's CPU generator, so there
is no GPU sync.

## Custom ops

A colour op declares its random parameters and one Slang function:

```python
import slangvision as A

class RandomGamma(A.ColorOp):
    """Gamma correction with a random exponent per image."""
    params = {"gamma": A.Uniform(0.8, 1.25)}
    slang = "float4 apply(float4 c, float gamma) { return clamp01(pow(c, float4(gamma))); }"

    def reference(self, img, gamma):          # optional: torch, one (3, H, W) image
        return img.clamp(0, 1) ** gamma

pipe = A.Compose([A.RandomResizedCrop(224), RandomGamma(p=0.5)])
```

- `apply` receives the pixel (`float4`, rgb in `xyz`, values in [0, 1]) and
  one `float` per entry of `params`, in order, and returns the new pixel.
  Clamp to [0, 1] if your op can leave that range, as torchvision's ops do.
- `params` values are `A.Uniform(low, high)`, `A.Choice(values)` or a plain
  number (a constant). Pass a `params` dict in your own `__init__` to make
  the ranges arguments.
- The constructor takes `p` (probability of applying, per image) and
  `same_on_batch`.
- `gray(c)` (torchvision's luma) and `clamp01(c)` are in scope and stable;
  the other functions in `src/slangvision/shaders/` are visible too but are
  internal and may change between releases. Each op's code is placed in its
  own namespace, so helper functions you define never clash.

An op that reads neighbouring pixels subclasses `A.NeighborhoodOp`:

```python
import slangvision as A

class BoxBlur(A.NeighborhoodOp):
    params = {"r": A.Choice([1, 2])}
    border = "reflect"                        # or "clamp"
    slang = """
    float4 apply<I : IImage>(I img, int x, int y, float r)
    {
        int k = int(r);
        float4 acc = float4(0.0);
        for (int dy = -k; dy <= k; ++dy)
            for (int dx = -k; dx <= k; ++dx)
                acc += img.load(x + dx, y + dy);
        return acc / float((2 * k + 1) * (2 * k + 1));
    }"""
```

`img.load(x, y)` reads the image as it is at that point of the pipeline,
applying `border` outside it; `img.width()` and `img.height()` give its size.
Each neighbourhood op starts a new pass over the image.

Check an op against its `reference` with `slangvision.testing`:

```py
from slangvision.testing import compare_to_reference, max_reference_error

err = max_reference_error(pipe, images)       # the largest difference
diff = compare_to_reference(pipe, images)     # diff.max, diff.fraction, diff.images
```

It runs the pipeline, replays each image's random values through torch
references (torchvision for the built-in transforms, your `reference` for
your ops) and returns the largest absolute difference; `compare_to_reference`
also says how many values and which images differ by more than the output
dtype's rounding (see [Accuracy](#accuracy)). Training never
imports this module. `pipe.source` shows the generated Slang.

## Coming from Albumentations

SlangVision follows torchvision's semantics, not Albumentations': there is
no Albumentations compatibility mode, and masks, boxes and keypoints are not
supported yet. The image transforms people use most from Albumentations
mostly have a counterpart. Defaults below are those of Albumentations 2.0.8,
its last release (the repository is archived).

| Albumentations | SlangVision | notes |
|---|---|---|
| `HorizontalFlip`, `VerticalFlip` | `RandomHorizontalFlip`, `RandomVerticalFlip` | the same |
| `Rotate(limit=90)` | `RandomRotation(90, interpolation="bilinear")` | constant fill only (Albumentations' default border too; its reflect and replicate borders have no counterpart) |
| `ShiftScaleRotate()`, `Affine` | `RandomAffine(45, translate=(0.0625, 0.0625), scale=(0.9, 1.1), interpolation="bilinear")` | the line shows ShiftScaleRotate's defaults; constant fill only |
| `RandomResizedCrop`, `RandomCrop`, `CenterCrop`, `Resize` | the same names | sizes as `(height, width)`; Albumentations resizes with OpenCV, which does not antialias when downscaling: `antialias=False` is closer |
| `GaussianBlur(sigma_limit=(0.5, 3.0))` | `GaussianBlur(19, sigma=(0.5, 3.0))` | one kernel size for every image, large enough for the largest sigma, where Albumentations sizes each kernel from its sigma |
| `Blur(blur_limit=(3, 7))` | `BoxBlur` in [Custom ops](#custom-ops) (with `A.Choice([1, 2, 3])`) | |
| `GaussNoise(std_range=(0.2, 0.44))` | `GaussianNoise(sigma=0.3)` | a fixed sigma, where Albumentations draws one per image; `RandomChoice` over a few `GaussianNoise` sigmas comes close |
| `CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2))` | about one `RandomErasing(p, scale=(0.01, 0.04), ratio=(0.5, 2.0))` per hole | after `Normalize`, so `value=0` is the mean colour: for black, pass `value=[-m / s for m, s in zip(mean, std)]` |
| `ColorJitter` | `ColorJitter` | close: Albumentations works on uint8 with OpenCV's HSV |
| `ToGray` | `RandomGrayscale(p=0.5)` | |
| `Normalize` | `Normalize(mean, std)` | the same with values on the 0-1 scale |
| `RandomBrightnessContrast` | the `ColorOp` below | |
| `RandomGamma(gamma_limit=(80, 120))` | `RandomGamma` in [Custom ops](#custom-ops), with `A.Uniform(0.8, 1.2)` | |
| `CLAHE`, `MedianBlur`, `HueSaturationValue` | not available | |

Albumentations' `RandomBrightnessContrast` (default `brightness_by_max=True`)
scales by `alpha` and adds `beta`:

```python
import slangvision as A

class RandomBrightnessContrast(A.ColorOp):
    """alpha * x + beta, alpha in 1 ± contrast_limit, beta in ± brightness_limit."""
    slang = "float4 apply(float4 c, float alpha, float beta) { return clamp01(alpha * c + beta); }"

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        super().__init__(p)
        self.params = {"alpha": A.Uniform(1 - contrast_limit, 1 + contrast_limit),
                       "beta": A.Uniform(-brightness_limit, brightness_limit)}

    def reference(self, img, alpha, beta):
        return (alpha * img + beta).clamp(0, 1)
```

## Accuracy

| | difference from torchvision's float functionals |
|---|---|
| float32 output | about 1e-5 (float rounding), measured after `Normalize` (1e-5 to 2e-5 before it) |
| float16 output | fp16 rounding of the output, up to about 2e-3 at normalized values |
| bfloat16 output | torchvision's value rounded to bfloat16 (8 significant bits), as `.to(torch.bfloat16)` rounds it; at most one bfloat16 step away (about 0.016 at normalized values near 2.6) where the two land on either side of a rounding edge, which pipelines with a blur do more often |
| threshold ops (`RandomSolarize`) | a value landing exactly on the threshold may be inverted on one side and not the other; all other values match |
| `RandomPosterize`, `RandomEqualize` | step functions: a resampled value within float rounding of a level edge can land on the other side. In one pass, about 1 value in 100,000 (posterize, by one level) and 1 in 30,000 to 1 in 3,000 depending on the ops before it (equalize, by up to 2/255: one moved pixel can shift the lookup table). After a blur, warp or sharpness with float16 output, the float16 image between passes moves more values across edges: about 1 in 1,000 for posterize and 1 in 100 for equalize (by up to 3/255). On images that are not resampled they match exactly |
| `RandAugment` | float32 output: as its ops (above and below); in a recipe after a crop, under 3 in 1,000 values differ (step ops at level edges, nearest warps at rounding ties). float16 output: the float16 image between its passes makes equalize land a level (1/255) away from torchvision's float32 result for about 2% of values; `dtype=torch.float32` keeps float32 between passes (2.84 to 3.63 ms per batch of 256 in the RandAugment recipe, measured with an ad-hoc script) |
| `TrivialAugmentWide`, `AutoAugment` | as `RandAugment`: float32 output exact on images that are not resampled with bilinear warps (AutoAugment's second op can meet a level edge in the first op's output: about 1 value in 100,000); in a recipe after a crop, a few values in a million differ with nearest warps (tests allow 3 in 1,000). float16 output: 2% (TrivialAugmentWide) to 3-4% (AutoAugment's ImageNet and SVHN policies) of values a level away where equalize follows a float16 intermediate |
| `TrivialAugmentWide` translate | torch builds the magnitude table in float32 with a vectorized CPU kernel; on some CPUs (seen on an Arm Mac and an AMD EPYC) bin 15 of 31 comes out 15.999999, so torchvision shifts those images 15 pixels and SlangVision 16 (its table is computed the plain way): every image that draws translate at that bin differs as a whole, a few percent of a batch |
| `TimmRandAugment` | against timm on uint8 Pillow images, one op on images that are not resampled: invert, posterize, solarize, solarize-add and equalize match exactly; warps, brightness, contrast and autocontrast are 0 to 1 uint8 step above Pillow's (Pillow truncates); colour and sharpness up to half a step times |1 - factor| further (Pillow also rounds the luma and smoothed images: -0.45 to +1.45 steps at magnitudes up to 10). In a recipe after a crop (timm sees the crop rounded to uint8): mean difference under 0.6 steps, under 2 in 1,000 values more than 3 steps away (values near a posterize, solarize or equalize level edge), float32 and float16 output alike |
| warps with `interpolation="nearest"` | about 1 pixel in 100,000 (1 in 20,000 for `RandomPerspective`) takes the neighbouring source pixel: torchvision computes the sampling positions in float32 with a different rounding, so a position within about 1e-5 pixels of a pixel boundary can go either way. Positions exactly on a half pixel, which round magnitudes can produce on whole rows (RandAugment's ShearX and ShearY at magnitude 20 shift every fifth row or column by a whole number plus a half: those pixels are the only ones that differ), round half to even here and either way in torchvision. Bilinear warps match like the other ops |
| float16 output after a warp or blur | the image between passes is float16 too, so colour ops after it can scale that rounding (up to about 8e-3 at normalized values with `ColorJitter(0.4, 0.4, 0.4, 0.1)` after a warp), and step ops after it can cross an edge: solarize flips a value that crosses its threshold, posterize moves it a level |

torchvision's uint8 path rounds after every op; this matches its float path.

### Checking your pipeline

`slangvision.testing.compare_to_reference(pipe, images)` replays each
image's random values through torchvision and returns `max` (the largest
difference), `fraction` (the share of values off by more than the output
dtype's rounding, `tol`), `images` (the images with such values) and
`image_max` (each image's largest difference). What to expect, in normalized
units:

| your pipeline | float32 output | float16 output |
|---|---|---|
| crops, resizes, flips, colour jitter, grayscale, blur, bilinear warps, noise, erasing | `fraction` 0, `max` about 1e-5 | `fraction` 0 or a few values just over `tol` after a warp or blur (up to about 8e-3) |
| step ops (posterize, solarize, equalize) or nearest warps, alone or in a policy, after a crop | under 3 in 1,000 values, each at a level edge or a rounding tie; scattered over many images | more: equalize a level away for about 2% of values after a float16 pass; a value that crosses solarize's threshold flips (a difference up to about 4), posterize's moves a level; `fraction` under a few in 1,000 |
| `TrivialAugmentWide` or `RandAugment` on a CPU where torch's magnitude table has 15.999999 (Arm Macs, AMD EPYC) | whole images shifted one pixel: those images are almost entirely off, a few percent of a batch | the same |

A result inside these is SlangVision matching torchvision as far as float
arithmetic allows. Outside them (a pipeline without step ops or policies
with `fraction` above 0 in float32, or many images almost entirely off),
something differs: look at the worst images (`image_max`) and report it.

## Performance

The numbers are in the README's
[Performance](https://github.com/tekintatar/slangvision#performance) section;
`benchmarks/compare_pipelines.py` reproduces the batch timings (its header
has the reference runs and raw data). How to read them:

- torchvision's batched mode draws one random setting for the whole batch,
  so it does less work (and its SimCLR time swings with whether that batch
  is blurred); its per-image loop is the like-for-like comparison. Kornia
  draws per image in one batched call; its RandomResizedCrop does not
  antialias.
- `RandAugment()` after a crop and flip: 2.29 ms per batch of 256 on an RTX
  PRO 4500, against 111 ms for torchvision per image (on the uint8 crop, as
  its reference recipe runs it) and 6.3 ms for torchvision batched with one
  setting for all 256 (the `randaug` recipe in the benchmark script). Each
  op adds a pass and, for the images that choose them, contrast, autocontrast
  and equalize pre-passes.
- `TrivialAugmentWide()` and `AutoAugment()` after a crop and flip: 1.58
  and 2.49 ms per batch of 256 on an RTX PRO 4500, against 82 and 95 ms for
  torchvision per image and 4.8 and 5.2 ms batched with one setting (the
  `trivialaug` and `autoaug` recipes).
- `TimmRandAugment()` (`rand-m9-mstd0.5-inc1`) after a crop and flip: 2.22
  ms per batch of 256 on an RTX PRO 4500, against 310 ms for timm's own
  `create_transform` pipeline on Pillow images on one CPU core (AMD EPYC
  9335; a DataLoader spreads that over its workers). The `timmaug` recipe in
  the benchmark script; its baseline gives timm's posterize Pillow 12's
  0-bit result so it runs under Pillow 13.
- Ops that need a per-image statistic pay for a pre-pass over the images
  they apply to; images that skip the op cost it almost nothing. Added to
  the ImageNet recipe (RTX PRO 4500, batch 256, 1.68 ms in that run; an
  ad-hoc script, not in the repository), at `p=0.5`: `RandomAutocontrast`
  +0.45 ms, `RandomEqualize` +0.69 ms (+1.1 ms at `p=1`),
  `RandomAdjustSharpness` +0.32 ms (its own pass); `RandomInvert` and
  `RandomPosterize` cost nothing measurable.
- A statistic in the first pass would resample the images again. With
  antialiased bicubic, or bilinear from images at least five times the
  output's area when the statistic ops run on every image on average, the
  first pass resamples once into a float32 copy that the pre-pass and the
  pass read instead (it is freed after the call: about 150 MB at batch 256
  and 224 px).
- A warp adds one pass: on an RTX PRO 4500, `RandomRotation(15)` after the
  flip takes the ImageNet recipe from 1.67 to 1.84 ms per batch of 256
  (torchvision per image: 205 to 275 ms; the `rotate` recipe in the
  benchmark script).
- Each call also has a fixed host-side cost of about 0.2 ms (SlangPy
  kernel dispatches from Python), whatever the image size: the gap between
  wall and GPU time in the benchmark. It overlaps with the GPU running the
  model.
- In training, decoding usually decides throughput. On one H100 NVL with 18
  CPU cores, the CPU pipeline held both ResNet-18 and ResNet-50 near 2,700
  images/s; moving only the augmentation to the GPU gave 1.4x and 1.2x, and
  GPU decoding about 3x and 1.3x. More CPU cores narrow the gap to the CPU
  pipeline; `torch.compile` widens it.
- With DALI decoding, SlangVision runs within about 3% of DALI's own
  augmentation and gives torchvision's results; DALI's colour ops are
  linear approximations.
- The training runs (ImageNet subset, bf16, eager PyTorch, 45 s after
  warm-up, one run per row, CPU rows about ±10% between runs) used a script
  that is not part of this repository, with the same kernels while they
  shipped as `slangimage.augment`.

### Larger photos: FGVC-Aircraft

[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
(non-commercial research use) has real photos of about 0.7 MP (median
1024x695; 13% over 1 MP), about five times ImageNet's pixel count, so
crops downscale 3-4x to 224 px. NVIDIA RTX PRO 4500 Blackwell, 16 CPU
cores, torch 2.13, torchvision 0.28.

Augmentation of 256 JPEGs decoded on the GPU at their own sizes
(`benchmarks/compare_pipelines.py --images`, raw data in
`benchmarks/results/rtx_pro_4500_fgvc_aircraft_*_after_review_2026-11-05.json`),
ms per batch;
torchvision runs per image (images of different sizes cannot form one
batch):

| recipe | SlangVision bilinear | SlangVision bicubic | torchvision bilinear | torchvision bicubic |
|---|---:|---:|---:|---:|
| ImageNet: crop, flip, ColorJitter, Normalize | 2.34 | 4.36 | 205 | 204 |
| eval: Resize, CenterCrop, Normalize | 1.76 | 4.27 | 28.3 | 34.6 |
| SimCLR | 4.03 | 5.34 | 213 | 209 |

Training from scratch on its 6,667 trainval images (100 classes), batch
256, bf16, ImageNet recipe, images per second over 45 s after warm-up:

| input pipeline | ResNet-50 | ResNet-18 |
|---|---:|---:|
| model alone (no input pipeline) | 1,207 | 3,822 |
| DataLoader (16 workers): decode + torchvision augmentation | 987 | 2,051 |
| DALI decode + DALI augmentation | 1,170 | 3,064 |
| DALI decode + SlangVision | 1,148 | 3,164 |
| DataLoader decode only + SlangVision (packed batches) | 750 | 1,465 |

- With DALI decoding, SlangVision is within 2% of DALI's own augmentation
  on ResNet-50 and 3% ahead on ResNet-18, with torchvision's results.
- Bigger photos make bicubic resizing about twice bilinear's cost (the
  eval recipe, which has no statistic, more). torchvision's per-image loop
  takes 8-90x longer in wall time, much of it in launching small kernels
  (about 86 per image).
- Decoding on CPU workers and augmenting on the GPU loses here: the
  workers ship whole decoded photos (about 2 MB each) instead of crops.

## Limits

- CUDA only (Linux with an NVIDIA GPU). For image processing on Metal,
  CUDA and Vulkan, see [SlangImage](https://github.com/tekintatar/slangimage).
- Not inside CUDA graph capture: SlangPy's CUDA layer queries the stream
  while capturing. Keep augmentation outside captured regions.
- One call handles fewer than 2^31 output values (up to about batch 14,000
  at 224x224); split larger batches.
- Run pipelines from one thread: the GPU device is created by, and tied to,
  the thread that runs the first pipeline (SlangPy devices are thread-affine).
- Not yet: AugMix, masks and boxes. Warps cannot come before a resize or
  crop, and `RandomRotation` has no `expand=True`.
