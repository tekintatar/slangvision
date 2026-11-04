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

## Requirements

Linux, an NVIDIA GPU and PyTorch with CUDA. SlangVision uses the NVRTC that
pip installs with PyTorch, after a one-time `setup`; see the README's
[Install](https://github.com/tekintatar/slangvision#install) section, also
for when that NVRTC is newer than your driver:

```bash
pip install slangvision torchvision
python -m slangvision setup
```

torchvision is needed only for `slangvision.testing`. SlangPy's
compiled torch bridge (`slangpy-torch`) is optional: without it, slangvision
uses SlangPy's pure-Python torch bridge for its own calls, which adds under
0.1 ms per batch.

The first call of a pipeline compiles it (about 1 s, plus about 1 s to set up
the GPU device once per process); later calls reuse the compiled kernels.

## Transforms

| transform | as in torchvision, except |
|---|---|
| `RandomResizedCrop(size, scale, ratio, interpolation="bilinear", antialias=True)` | box drawn per image |
| `Resize(size, interpolation="bilinear", antialias=True)` | an int resizes the shorter side and must be followed by `CenterCrop` or `RandomCrop`, so all images match |
| `CenterCrop(size)` | — (a crop larger than the image pads with zeros, as torchvision does) |
| `RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant")` | position drawn per image; `fill` is on the 0-1 scale |
| `Pad(padding, fill=0, padding_mode="constant")` | `fill` is on the 0-1 scale |
| `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)` | decided per image |
| `ColorJitter(brightness, contrast, saturation, hue)` | factors and op order drawn per image |
| `RandomApply(transforms, p=0.5)` | decided per image; holds colour ops, or exactly one blur or `NeighbourhoodOp` (not flips: they have their own `p`) |
| `RandomChoice(transforms, p=None)` | chosen per image (weights `p`, equal by default); holds colour ops and at most one blur or `NeighbourhoodOp` |
| `RandomGrayscale(p=0.1)` | decided per image; output keeps 3 channels |
| `GaussianBlur(kernel_size, sigma=(0.1, 2.0))` | sigma drawn per image; the kernel radius must be smaller than the image, as torchvision's reflect padding requires |
| `RandomSolarize(threshold, p=0.5)` | `threshold` is on the 0-1 scale (0.5, not 128) |
| `Normalize(mean, std)` | must be last |

Every random transform takes `same_on_batch=False`. `interpolation` is
`"bilinear"` or `"bicubic"` (or torchvision's `InterpolationMode`), with
torchvision's filters: with `antialias=True` (the default) PyTorch's
antialias kernels, cubic with a = -0.5 like PIL, and without it classic
bicubic with a = -0.75. Bicubic float output can overshoot [0, 1], as
torchvision's does; it costs about twice as much as bilinear.

The CIFAR recipe is `[RandomCrop(32, padding=4), RandomHorizontalFlip(),
Normalize(...)]`. DeiT III's ThreeAugment is
`[Resize(224, interpolation="bicubic"), RandomCrop(224, padding=4,
padding_mode="reflect"), RandomHorizontalFlip(),
RandomChoice([RandomGrayscale(p=1), RandomSolarize(0.5, p=1),
GaussianBlur(13)]), ColorJitter(0.3, 0.3, 0.3), Normalize(...)]`; DeiT
blurs with PIL's radius-based filter, which `GaussianBlur` only
approximates. `RandomChoice(..., same_on_batch=True)` makes the same choice
for every image.

Pipeline rules, checked when you build the `Compose`:
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
- Flips go anywhere after the geometric transforms and
  before any `NeighbourhoodOp` you define (they commute with colour ops and
  with Gaussian blur).
- Each transform object appears once; create a second one to use it twice.
- `Normalize`, if present, is last. Without it the output is in [0, 1].

`Compose(transforms, dtype=torch.float16)` is the default; `torch.bfloat16`
(for models trained in bfloat16, at the same speed as float16) and
`torch.float32` are the other choices. Passes between blurs and the final
output keep float16 (float32 for float32 output). The output is a
contiguous `(B, 3, H, W)` tensor; for `channels_last`, call `.to(...)` on it.

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
reuses it.

Decoding usually limits training throughput, so pick a fast decoder:

```py
# torchvision (0.28) GPU decode, CHW
from torchvision.io import ImageReadMode, decode_jpeg
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

**With DALI, import `slangvision` before building a DALI pipeline.** If
DALI loads its native libraries first, SlangPy crashes with a segmentation
fault while creating its CUDA device (seen with DALI 2.4
and SlangPy 0.42); in the other order both work in one process.

Without a GPU decoder, keep your `DataLoader` but let the workers only
decode, and augment the batch on the GPU. Have the workers pack each batch
into one buffer: with `collate_fn=list`, PyTorch shares every image between
processes separately, which made training slower than augmenting on the CPU
(2,037 against 2,721 images/s for ResNet-18 in the benchmark below).

```py
def collate(batch):                  # runs in the workers
    images, labels = zip(*batch)     # decoded uint8 (3, H, W) tensors
    return (torch.cat([im.reshape(-1) for im in images]),
            torch.tensor([tuple(im.shape) for im in images]), torch.tensor(labels))

loader = torch.utils.data.DataLoader(decoded_dataset, batch_size=256, num_workers=16,
                                     pin_memory=True, collate_fn=collate)
for flat, shapes, labels in loader:
    parts = flat.cuda(non_blocking=True).split(shapes.prod(1).tolist())
    x = train([part.view(*s) for part, s in zip(parts, shapes.tolist())])
```

## Randomness

Random values come from a counter-based generator in the sampler kernel,
seeded on every call from PyTorch's CUDA generator: `torch.manual_seed`
makes runs repeat, and there is no host sync. Values are drawn per image;
`same_on_batch=True` on a transform gives every image the same values for
that transform (for `RandomApply`, the same apply/skip decision).

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
- Library helpers such as `gray(c)` and `clamp01(c)` are in scope (see
  `src/slangvision/shaders/`). Each op's code is placed in
  its own namespace, so helper functions you define never clash.

An op that reads neighbouring pixels subclasses `A.NeighbourhoodOp`:

```python
import slangvision as A

class BoxBlur(A.NeighbourhoodOp):
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
from slangvision.testing import max_reference_error

err = max_reference_error(pipe, images)
```

It runs the pipeline, replays each image's random values through torch
references (torchvision for the built-in transforms, your `reference` for
your ops) and returns the largest absolute difference. Training never
imports this module. `pipe.source` shows the generated Slang.

## Accuracy

| | difference from torchvision's float functionals |
|---|---|
| float32 output | about 1e-5 (float rounding), measured after `Normalize` (1e-5 to 2e-5 before it) |
| float16 output | fp16 rounding of the output, up to about 2e-3 at normalized values |
| bfloat16 output | torchvision's value rounded to bfloat16 (8 significant bits), as `.to(torch.bfloat16)` rounds it; at most one bfloat16 step away (about 0.016 at normalized values near 2.6) where the two land on either side of a rounding edge, which pipelines with a blur do more often |
| threshold ops (`RandomSolarize`) | a value landing exactly on the threshold may be inverted on one side and not the other; all other values match |

torchvision's uint8 path rounds after every op; this matches its float path.

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

## Limits

- CUDA only (Linux with an NVIDIA GPU). For image processing on Metal,
  CUDA and Vulkan, see [SlangImage](https://github.com/tekintatar/slangimage).
- Not inside CUDA graph capture: SlangPy's CUDA layer queries the stream
  while capturing. Keep augmentation outside captured regions.
- One call handles fewer than 2^31 output values (up to about batch 14,000
  at 224x224); split larger batches.
- Run pipelines from one thread: the GPU device is created by, and tied to,
  the thread that runs the first pipeline (SlangPy devices are thread-affine).
- Not yet: RandAugment, TrivialAugment and the ops they use, affine and
  perspective geometry, MixUp and CutMix, masks and boxes.
