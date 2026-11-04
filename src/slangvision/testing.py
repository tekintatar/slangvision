"""Check a pipeline against torchvision, image by image.

    from slangvision.testing import max_reference_error

    err = max_reference_error(pipe, images)    # largest absolute difference

The pipeline runs on the GPU as usual; then each image's sampled random
values (crop box, flip, jitter factors and order, ...) are replayed through
torchvision's functional API for the built-in transforms, and through the
``reference`` method of each ColorOp or NeighbourhoodOp you define. Needs
torchvision. Training never imports this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import transforms as T
from ._base import Binding, Transform, _UserOp
from ._compile import Compose

if TYPE_CHECKING:
    import torch

__all__ = ["max_reference_error"]

_MODES: dict[str, Any] = {}     # filled with torchvision's InterpolationMode on first use


def max_reference_error(pipeline: Compose, images: Any, layout: str = "chw") -> float:
    """Run ``pipeline`` on ``images`` and return the largest absolute
    difference from the torch references fed the same random values, in
    output units (normalized, if the pipeline ends with Normalize)."""
    out, params = pipeline._run(images, layout)
    return float((out.float() - _replay(pipeline, images, params, layout)).abs().max())


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
    return torch.stack(outs)


def _reference(t: Transform, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
    """One transform on one (3, H, W) float image, with this image's values."""
    import torchvision.transforms.v2.functional as F
    from torchvision.transforms import InterpolationMode

    _MODES.update(bilinear=InterpolationMode.BILINEAR, bicubic=InterpolationMode.BICUBIC)

    if isinstance(t, T.RandomResizedCrop):
        left, top, w, h = (int(v) for v in row[b.geo:b.geo + 4])
        return F.resized_crop(img, top, left, h, w, list(t.size), antialias=t.antialias,
                              interpolation=_MODES[t.interpolation])
    if isinstance(t, T.Resize):
        size = [t.size[0]] if t.shorter else list(t.size)
        return F.resize(img, size, antialias=t.antialias, interpolation=_MODES[t.interpolation])
    if isinstance(t, T.CenterCrop):
        return F.center_crop(img, list(t.size))
    if isinstance(t, T.RandomHorizontalFlip):
        return F.horizontal_flip(img) if row[b.k(t)] > 0.5 else img
    if isinstance(t, T.ColorJitter):
        k = b.k(t)
        fns = [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue]
        order = int(row[k + 4])
        for j in range(4):
            op = (order >> (2 * j)) & 3
            if t.ranges[op]:
                img = fns[op](img, row[k + op])
        return img
    if isinstance(t, T.GaussianBlur):
        s = row[b.k(t)]
        return F.gaussian_blur(img, list(t.kernel_size), [s, s])
    if isinstance(t, T.RandomApply):
        if row[b.k(t)] > 0.5:
            for child in t.transforms:
                img = _reference(child, b, img, row)
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
        return t.reference(img, **values)
    raise TypeError(f"no torch reference for {type(t).__name__}")
