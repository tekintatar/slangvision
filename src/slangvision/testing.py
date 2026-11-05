"""Check a pipeline against torchvision, image by image.

    from slangvision.testing import compare_to_reference, max_reference_error

    err = max_reference_error(pipe, images)    # largest absolute difference
    diff = compare_to_reference(pipe, images)  # how many values, which images

A single largest difference cannot tell one value at a rounding edge from a
wrong op: ``compare_to_reference`` also gives the fraction of values and the
images that differ by more than the output dtype's rounding, so a few
step-function edge cases (see the guide's Accuracy table) can be told apart
from a real mismatch.

The pipeline runs on the GPU as usual; then each image's sampled random
values (crop box, flip, jitter factors and order, ...) are replayed through
torchvision's functional API for the built-in transforms, and through the
``reference`` method of each ColorOp or NeighborhoodOp you define. Needs
torchvision. TimmRandAugment is replayed through timm's own Pillow ops (needs
timm) on the image rounded to uint8, as timm sees it, so it differs by
Pillow's uint8 rounding: about 1/255 per op. Training never imports this
module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

from . import transforms as T
from ._base import Binding, Transform, _UserOp
from ._compile import Compose, _walk

if TYPE_CHECKING:
    import torch

__all__ = ["ReferenceDiff", "compare_to_reference", "max_reference_error"]

#: per output dtype, the difference float rounding alone can cause (normalized
#: values up to about 2.6): what the tests allow
_TOLERANCE = {"float32": 1e-4, "float16": 4e-3, "bfloat16": 1.6e-2}


class ReferenceDiff(NamedTuple):
    """How a pipeline's output differs from the references (output units)."""
    max: float              #: largest absolute difference (NaN if nothing comparable)
    fraction: float         #: share of compared values off by more than ``tol``
    images: list[int]       #: indices of the images with any value off by more than ``tol``
    tol: float              #: the threshold used

_MODES: dict[str, Any] = {}     # filled with torchvision's InterpolationMode on first use


def max_reference_error(pipeline: Compose, images: Any, layout: str = "chw") -> float:
    """Run ``pipeline`` on ``images`` and return the largest absolute
    difference from the torch references fed the same random values, in
    output units (normalized, if the pipeline ends with Normalize). Random
    noise (GaussianNoise, RandomErasing's "random" fill) cannot be replayed,
    so those pixels are skipped; if nothing is comparable (GaussianNoise
    with sigma > 0 anywhere in the pipeline), the result is NaN."""
    return compare_to_reference(pipeline, images, layout).max


def compare_to_reference(pipeline: Compose, images: Any, layout: str = "chw",
                         tol: float | None = None) -> ReferenceDiff:
    """Run ``pipeline`` on ``images`` and compare it with the torch
    references fed the same random values: the largest difference, the
    fraction of values off by more than ``tol`` and which images they are in.
    ``tol`` defaults to what the output dtype's rounding allows (1e-4 for
    float32, 4e-3 for float16, 1.6e-2 for bfloat16). Noise pixels are skipped
    as in :func:`max_reference_error`."""
    out, params = pipeline._run(images, layout)
    ref = _replay(pipeline, images, params, layout)
    if out.shape != ref.shape:
        raise ValueError(f"output shape {tuple(out.shape)} differs from the references' "
                         f"{tuple(ref.shape)}")
    tol = _TOLERANCE[pipeline._dtype] if tol is None else float(tol)
    keep = ~ref.isnan()         # random noise (GaussianNoise, RandomErasing) is not replayed
    if not keep.any():
        return ReferenceDiff(float("nan"), float("nan"), [], tol)
    diff = (out.float() - ref).abs().nan_to_num(0.0)
    off = (diff > tol) & keep
    return ReferenceDiff(float(diff[keep].max()), float(off.sum() / keep.sum()),
                         off.flatten(1).any(1).nonzero().flatten().tolist(), tol)


def _replay(pipeline: Compose, images: Any, params: "torch.Tensor",
            layout: str = "chw") -> "torch.Tensor":
    """The references' output for the random values in ``params`` (one row
    per image, as the sampler kernel drew them)."""
    import torch

    imgs = list(images)
    if layout == "hwc":
        imgs = [im.permute(2, 0, 1) for im in imgs]
    rows = params.tolist()
    outs = []
    for i, im in enumerate(imgs):
        img = im.float() / 255.0
        for t in pipeline.transforms:
            img = _reference(t, pipeline._b, img, rows[i])
        outs.append(img)
    out = torch.stack(outs)
    # noise cannot be replayed, and ops after it (equalize) can turn its NaN
    # placeholder back into numbers: nothing after noise is comparable
    if any(isinstance(t, T.GaussianNoise) and t.sigma > 0 for t in _walk(pipeline.transforms)):
        out.fill_(float("nan"))
    return out


def _reference(t: Transform, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
    """One transform on one (3, H, W) float image, with this image's values."""
    import torchvision.transforms.v2.functional as F
    from torchvision.transforms import InterpolationMode

    _MODES.update(bilinear=InterpolationMode.BILINEAR, bicubic=InterpolationMode.BICUBIC,
                  nearest=InterpolationMode.NEAREST)

    if isinstance(t, T.RandomResizedCrop):
        left, top, w, h = (int(v) for v in row[b.geo:b.geo + 4])
        return F.resized_crop(img, top, left, h, w, list(t.size), antialias=t.antialias,
                              interpolation=_MODES[t.interpolation])
    if isinstance(t, T.Resize):
        size = [t.size[0]] if t.shorter else list(t.size)
        return F.resize(img, size, antialias=t.antialias, interpolation=_MODES[t.interpolation])
    if isinstance(t, T.CenterCrop):
        return F.center_crop(img, list(t.size))
    if isinstance(t, T.Pad):
        return _pad(F, img, list(t.padding), t)
    if isinstance(t, T.RandomCrop):
        k = b.win[id(t)]
        left, top, right, bottom = t._padded((img.shape[-2], img.shape[-1]))
        if any((left, top, right, bottom)):
            img = _pad(F, img, [left, top, right, bottom], t)
        return F.crop(img, int(row[k + 1]) + top, int(row[k]) + left, *t.size)
    if isinstance(t, T.RandomVerticalFlip):
        return F.vertical_flip(img) if row[b.k(t)] > 0.5 else img
    if isinstance(t, T.RandomHorizontalFlip):
        return F.horizontal_flip(img) if row[b.k(t)] > 0.5 else img
    if isinstance(t, T._Warp):
        k = b.k(t)
        if row[k] <= 0.5:
            return img
        raw = row[k + 9:k + 9 + t._raw]
        mode, fill = _MODES[t.interpolation], list(t.fill)[:img.shape[0]]
        center = None if t.center is None else list(t.center)
        if isinstance(t, T.RandomRotation):
            return F.rotate(img, raw[0], interpolation=mode, center=center, fill=fill)
        if isinstance(t, T.RandomAffine):
            angle, tx, ty, scale, shx, shy = raw
            return F.affine(img, angle, [int(tx), int(ty)], scale, [shx, shy],
                            interpolation=mode, fill=fill, center=center)
        h, w = img.shape[-2:]
        start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        end = [[int(raw[2 * i]), int(raw[2 * i + 1])] for i in range(4)]
        return F.perspective(img, start, end, interpolation=mode, fill=fill)
    if isinstance(t, T.ColorJitter):
        k = b.k(t)
        fns = [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue]
        order = int(row[k + 4])
        for j in range(4):
            op = (order >> (2 * j)) & 3
            if t.ranges[op]:
                img = fns[op](img, row[k + op])
        return img
    if isinstance(t, T.RandomPhotometricDistort):
        k = b.k(t)
        f = int(row[k + 4])
        if f & 1:                   # the sampler sets no bit for a disabled range
            img = F.adjust_brightness(img, row[k])
        if f & 2 and f & 16:
            img = F.adjust_contrast(img, row[k + 1])
        if f & 4:
            img = F.adjust_saturation(img, row[k + 2])
        if f & 8:
            img = F.adjust_hue(img, row[k + 3])
        if f & 2 and not f & 16:
            img = F.adjust_contrast(img, row[k + 1])
        if f & 32 and img.shape[0] == 3:    # one channel: torchvision's randperm(1)
            img = F.permute_channels(img, list(T._PERMUTATIONS[int(row[k + 5])]))
        return img
    if isinstance(t, T.RandomChannelPermutation):
        if img.shape[0] == 1:
            return img
        return F.permute_channels(img, list(T._PERMUTATIONS[int(row[b.k(t)])]))
    if isinstance(t, (T.RandAugment, T.TrivialAugmentWide, T.AutoAugment)):
        return _torchvision_policy(t, b, img, row)
    if isinstance(t, T.TimmRandAugment):
        return _timm_reference(t, b, img, row)
    if isinstance(t, (T.RandomAutocontrast, T.RandomEqualize, T.RandomAdjustSharpness)):
        if row[b.k(t)] <= 0.5:
            return img
        if isinstance(t, T.RandomAutocontrast):
            return F.autocontrast(img)
        if isinstance(t, T.RandomEqualize):
            return F.equalize(img)
        return F.adjust_sharpness(img, t.sharpness_factor)
    if isinstance(t, T.GaussianBlur):
        s = row[b.k(t)]
        return F.gaussian_blur(img, list(t.kernel_size), [s, s])
    if isinstance(t, T.RandomChoice):
        return _reference(t.transforms[int(row[b.k(t)])], b, img, row)
    if isinstance(t, T.RandomApply):
        if row[b.k(t)] > 0.5:
            for child in t.transforms:
                img = _reference(child, b, img, row)
        return img
    if isinstance(t, T.GaussianNoise):
        import torch

        if t.sigma > 0:     # noise cannot be replayed: max_reference_error skips NaN
            return torch.full_like(img, float("nan"))
        out = img + t.mean
        return out.clamp(0, 1) if t.clip else out
    if isinstance(t, T.RandomErasing):
        import torch

        k = b.k(t)
        if row[k] > 0.5:
            top, left, h, w = (int(v) for v in row[k + 1:k + 5])
            img = img.clone()
            if t.value is None:     # noise cannot be replayed: max_reference_error skips NaN
                img[:, top:top + h, left:left + w] = float("nan")
            else:
                value = torch.tensor(t.value[:img.shape[0]], device=img.device)
                img[:, top:top + h, left:left + w] = value[:, None, None]
        return img
    if isinstance(t, T.Normalize):        # one channel after Grayscale(1)
        n = img.shape[0]
        return F.normalize(img, list(t.mean[:n]), list(t.std[:n]))
    if isinstance(t, _UserOp):
        if not t._applied(b, row):
            return img
        values = t._values(b, row)
        if isinstance(t, T.RandomGrayscale):
            return F.rgb_to_grayscale(img, num_output_channels=3)
        if isinstance(t, T.Grayscale):
            return F.rgb_to_grayscale(img, num_output_channels=t.num_output_channels)
        if isinstance(t, T.RandomSolarize):
            return F.solarize(img, values["threshold"])
        if isinstance(t, T.RandomInvert):
            return F.invert(img)
        if isinstance(t, T.RandomPosterize):
            return F.posterize(img, t.bits)
        return t.reference(img, **values)
    raise TypeError(f"no torch reference for {type(t).__name__}")


def _torchvision_policy(t: Any, b: Binding, img: "torch.Tensor",
                        row: list[float]) -> "torch.Tensor":
    """torchvision's own policy: its op names, magnitude tables and op
    functions, fed this image's drawn op (AutoAugment: sub-policy), bin,
    application and sign."""
    from torchvision.transforms import v2

    kw = {"interpolation": _MODES[t.interpolation],
          "fill": None if t.fill is None else list(t.fill)}
    if isinstance(t, T.RandAugment):
        tv: Any = v2.RandAugment(t.num_ops, t.magnitude, t.num_magnitude_bins, **kw)
    elif isinstance(t, T.TrivialAugmentWide):
        tv = v2.TrivialAugmentWide(t.num_magnitude_bins, **kw)
    else:
        tv = v2.AutoAugment(v2.AutoAugmentPolicy(t.policy), **kw)
    h, w = img.shape[-2:]
    names = list(tv._AUGMENTATION_SPACE)
    for j, step in enumerate(t._children()):
        k = b.k(step)
        if row[k] < 0:
            continue
        if isinstance(t, T.AutoAugment):
            name, _, idx = tv._policies[int(row[k + 1])][j]
            bins = 10
        else:
            # RandAugment's bin is its magnitude; TrivialAugmentWide draws it
            idx = t.magnitude if isinstance(t, T.RandAugment) else int(row[k + 1])
            name, bins = names[int(row[k])], t.num_magnitude_bins
        magnitudes_fn, signed = tv._AUGMENTATION_SPACE[name]
        magnitudes = magnitudes_fn(bins, h, w)
        magnitude = 0.0 if magnitudes is None else float(magnitudes[idx])
        if signed and row[k + 2] < 0:
            magnitude = -magnitude
        img = tv._apply_image_or_video_transform(img, name, magnitude,
                                                 interpolation=tv.interpolation, fill=tv._fill)
    return img


def _timm_reference(t: Any, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
    """timm's RandAugment ops on the image as a uint8 Pillow image, with this
    image's op, magnitude, sign and interpolation."""
    import numpy as np
    import timm.data.auto_augment as aa
    import torch
    from PIL import Image

    names = aa._RAND_INCREASING_TRANSFORMS if t.increasing else aa._RAND_TRANSFORMS
    hparams = {"translate_pct": t.translate_pct}
    fill = tuple(int(round(v * 255)) for v in t.fill)
    u8 = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    pil = Image.fromarray(np.ascontiguousarray(u8))
    negate = aa._randomly_negate
    try:
        for step in t._children():
            k = b.k(step)
            if row[k] < 0:
                continue
            name = names[int(row[k])]
            sign = row[k + 2]
            aa._randomly_negate = lambda v: -v if sign < 0 else v   # the drawn sign
            level: Any = aa.LEVEL_TO_ARG[name]
            args: tuple[Any, ...] = level(row[k + 1], hparams) if level is not None else ()
            resample = Image.Resampling.BICUBIC if row[k + 3] > 0.5 else Image.Resampling.BILINEAR
            if name.startswith("Posterize") and args[0] <= 0:
                # keep no bits: black, as Pillow up to 12 gives (Pillow 13 raises)
                pil = Image.new("RGB", pil.size)
                continue
            op: Any = aa.NAME_TO_OP[name]
            pil = op(pil, *args, fillcolor=fill, resample=resample)
    finally:
        aa._randomly_negate = negate
    out = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1)
    return out.to(img.device).float() / 255.0


def _pad(F: Any, img: "torch.Tensor", padding: list[int], t: Any) -> "torch.Tensor":
    """torchvision's pad (padding as left, top, right, bottom) with t's mode and fill."""
    if t.padding_mode == "constant":
        return F.pad(img, padding, fill=list(t.fill), padding_mode="constant")
    return F.pad(img, padding, padding_mode=t.padding_mode)


# ------------------------------------------------------- super resolution
def usm_sharp_reference(x: "torch.Tensor", usm: Any, weight: float = 0.5,
                        threshold: float = 10) -> "torch.Tensor":
    """BasicSR's USMSharp.forward (basicsr/utils/img_process_util.py,
    Apache-2.0) for the slangvision.sr.USMSharp ``usm``: its 2D kernel (the
    float64 outer product rounded to float32), filter2D's reflect padding and
    conv2d, here in float32 with TF32 off (PyTorch's default lets cuDNN use
    TF32, which BasicSR's own result then carries)."""
    import torch
    import torch.nn.functional as F

    k1 = torch.tensor(usm.taps, dtype=torch.float64)
    kernel = torch.outer(k1, k1).float().to(x.device)
    k = kernel.shape[-1]

    def filter2d(img: "torch.Tensor") -> "torch.Tensor":
        b, c, h, w = img.shape
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
        return F.conv2d(img.reshape(b * c, 1, *img.shape[-2:]),
                        kernel.view(1, 1, k, k)).view(b, c, h, w)

    allow = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        img = x.float()
        residual = img - filter2d(img)
        mask = (torch.abs(residual) * 255 > threshold).float()
        soft = filter2d(mask)
        sharp = torch.clip(img + weight * residual, 0, 1)
        return soft * sharp + (1 - soft) * img
    finally:
        torch.backends.cudnn.allow_tf32 = allow


def poisson_noise_reference(img: "torch.Tensor", scale: Any = 1.0,
                            gray_noise: Any = 0) -> "torch.Tensor":
    """BasicSR's generate_poisson_noise_pt (basicsr/data/degradations.py,
    Apache-2.0) with torch.poisson: the noise, for comparing distributions
    (torch's Poisson values cannot be replayed)."""
    import torch
    from torchvision.transforms.v2.functional import rgb_to_grayscale

    b, _, h, w = img.shape

    def levels(t: "torch.Tensor") -> "torch.Tensor":
        counts = [len(torch.unique(t[i])) for i in range(b)]
        return t.new_tensor([2.0 ** math.ceil(math.log2(n)) for n in counts]).view(b, 1, 1, 1)

    g = torch.as_tensor(gray_noise, dtype=img.dtype, device=img.device).expand(b).view(b, 1, 1, 1)
    q = torch.clamp((rgb_to_grayscale(img) * 255.0).round(), 0, 255) / 255.0
    vals = levels(q)
    noise_gray = (torch.poisson(q * vals) / vals - q).expand(b, 3, h, w)
    q = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals = levels(q)
    noise = torch.poisson(q * vals) / vals - q
    noise = noise * (1 - g) + noise_gray * g
    return noise * torch.as_tensor(scale, dtype=img.dtype, device=img.device).expand(b).view(b, 1, 1, 1)


def diff_jpeg_reference(x: "torch.Tensor", quality: Any,
                        differentiable: bool = False) -> "torch.Tensor":
    """BasicSR's DiffJPEG.forward (basicsr/utils/diffjpeg.py, Apache-2.0,
    from mlomnitz/DiffJPEG, MIT) with its operations in its order, in
    float32: transposed float tables scaled by the quality factor, black
    padding to a multiple of 16, chroma averaged over 2x2 and repeated back.
    ``quality`` a number or a (B,) tensor (not modified, unlike BasicSR's)."""
    import itertools

    import numpy as np
    import torch
    import torch.nn.functional as F

    dev = x.device
    y_table = torch.tensor(np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T, device=dev)
    c = np.full((8, 8), 99, dtype=np.float32)
    c[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99],
                          [47, 66, 99, 99]]).T
    c_table = torch.tensor(c, device=dev)
    fwd = np.zeros((8, 8, 8, 8), dtype=np.float32)
    inv = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for a, b, u, v in itertools.product(range(8), repeat=4):
        fwd[a, b, u, v] = np.cos((2 * a + 1) * u * np.pi / 16) * np.cos((2 * b + 1) * v * np.pi / 16)
        inv[a, b, u, v] = np.cos((2 * u + 1) * a * np.pi / 16) * np.cos((2 * v + 1) * b * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    fwd_t, inv_t = torch.from_numpy(fwd).to(dev), torch.from_numpy(inv).to(dev)
    scale = torch.from_numpy(np.outer(alpha, alpha) * 0.25).float().to(dev)
    alpha_t = torch.from_numpy(np.outer(alpha, alpha)).float().to(dev)
    to_ycc = torch.from_numpy(np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
                                        [0.5, -0.418688, -0.081312]], dtype=np.float32).T).to(dev)
    to_rgb = torch.from_numpy(np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
                                       dtype=np.float32).T).to(dev)

    def rounding(t: "torch.Tensor") -> "torch.Tensor":
        return torch.round(t) + (t - torch.round(t)) ** 3 if differentiable else torch.round(t)

    if isinstance(quality, torch.Tensor):
        q = quality.to(dev, torch.float32).reshape(-1).expand(x.shape[0])
        factor = (torch.where(q < 50, 5000. / q, 200. - q * 2) / 100.).view(-1, 1, 1, 1)
    else:
        factor = (5000. / quality if quality < 50 else 200. - quality * 2) / 100.

    def blocks(p: "torch.Tensor") -> "torch.Tensor":
        b, h, _ = p.shape
        return p.view(b, h // 8, 8, -1, 8).permute(0, 1, 3, 2, 4).contiguous().view(b, -1, 8, 8)

    def merge(p: "torch.Tensor", h: int, w: int) -> "torch.Tensor":
        b = p.shape[0]
        return p.view(b, h // 8, w // 8, 8, 8).permute(0, 1, 3, 2, 4).contiguous().view(b, h, w)

    allow = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h, w = x.shape[-2:]
        img = F.pad(x.float(), (0, -w % 16, 0, -h % 16), mode="constant", value=0)
        hp, wp = img.shape[-2:]
        ycc = torch.tensordot((img * 255).permute(0, 2, 3, 1), to_ycc, dims=1) + \
            torch.tensor([0., 128., 128.], device=dev)
        ch = ycc.permute(0, 3, 1, 2)
        planes = {"y": (ycc[..., 0], y_table, hp, wp),
                  "cb": (F.avg_pool2d(ch[:, 1:2], 2)[:, 0], c_table, hp // 2, wp // 2),
                  "cr": (F.avg_pool2d(ch[:, 2:3], 2)[:, 0], c_table, hp // 2, wp // 2)}
        out = {}
        for k, (p, table, ph, pw) in planes.items():
            coef = scale * torch.tensordot(blocks(p) - 128, fwd_t, dims=2)
            coef = rounding(coef / (table * factor)) * (table * factor)
            out[k] = merge(0.25 * torch.tensordot(coef * alpha_t, inv_t, dims=2) + 128, ph, pw)
        up = [out[k].repeat_interleave(2, 1).repeat_interleave(2, 2) for k in ("cb", "cr")]
        ycc = torch.stack([out["y"], up[0], up[1]], 3)
        rgb = torch.tensordot(ycc + torch.tensor([0, -128., -128.], device=dev), to_rgb, dims=1)
        rgb = torch.clamp(rgb, 0, 255).permute(0, 3, 1, 2) / 255
        return rgb[:, :, :h, :w]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow
