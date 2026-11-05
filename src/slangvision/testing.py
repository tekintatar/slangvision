"""Check a pipeline against torchvision, image by image.

    from slangvision.testing import max_reference_error

    err = max_reference_error(pipe, images)    # largest absolute difference

The pipeline runs on the GPU as usual; then each image's sampled random
values (crop box, flip, jitter factors and order, ...) are replayed through
torchvision's functional API for the built-in transforms, and through the
``reference`` method of each ColorOp or NeighbourhoodOp you define. Needs
torchvision. TimmRandAugment is replayed through timm's own Pillow ops (needs
timm) on the image rounded to uint8, as timm sees it, so it differs by
Pillow's uint8 rounding: about 1/255 per op. Training never imports this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import transforms as T
from ._base import Binding, Transform, _UserOp
from ._compile import Compose, _walk

if TYPE_CHECKING:
    import torch

__all__ = ["max_reference_error"]

_MODES: dict[str, Any] = {}     # filled with torchvision's InterpolationMode on first use


def max_reference_error(pipeline: Compose, images: Any, layout: str = "chw") -> float:
    """Run ``pipeline`` on ``images`` and return the largest absolute
    difference from the torch references fed the same random values, in
    output units (normalized, if the pipeline ends with Normalize)."""
    out, params = pipeline._run(images, layout)
    ref = _replay(pipeline, images, params, layout)
    keep = ~ref.isnan()         # random noise (GaussianNoise, RandomErasing) is not replayed
    return float((out.float() - ref)[keep].abs().max()) if keep.any() else float("nan")


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
        mode, fill = _MODES[t.interpolation], list(t.fill)
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
                img[:, top:top + h, left:left + w] = torch.tensor(t.value,
                                                                 device=img.device)[:, None, None]
        return img
    if isinstance(t, T.Normalize):
        return F.normalize(img, list(t.mean), list(t.std))
    if isinstance(t, _UserOp):
        if not t._applied(b, row):
            return img
        values = t._values(b, row)
        if isinstance(t, T.RandomGrayscale):
            return F.rgb_to_grayscale(img, num_output_channels=3)
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
