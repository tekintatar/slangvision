# slangvision/transforms.py — the built-in transforms. Names, parameters and
# results follow torchvision.transforms.v2 on float images; random values are
# drawn per image unless same_on_batch=True. RandomGrayscale and
# RandomSolarize are written with the public ColorOp API.

from __future__ import annotations

import math
import numbers
from typing import Any, Sequence

from ._base import ST, Binding, ColorOp, P, Transform, Uniform, _probability, lit


def _size2(size: Any, name: str) -> tuple[int, int]:
    """torchvision size argument -> (height, width)."""
    if isinstance(size, numbers.Integral):
        h = w = int(size)
    elif isinstance(size, Sequence) and len(size) == 1:
        h = w = int(size[0])
    elif isinstance(size, Sequence) and len(size) == 2:
        h, w = int(size[0]), int(size[1])
    else:
        raise ValueError(f"{name}: size must be an int or (height, width), got {size!r}")
    if h <= 0 or w <= 0:
        raise ValueError(f"{name}: size must be positive, got {size!r}")
    return h, w


def _range(value: Any, name: str, center: float, bound: tuple[float, float],
           around_center: bool) -> tuple[float, float] | None:
    """torchvision ColorJitter range checking; None when the op is a no-op."""
    if isinstance(value, numbers.Real):
        value = float(value)
        if value < 0:
            raise ValueError(f"ColorJitter: {name} must be non-negative, got {value}")
        lo, hi = center - value, center + value
        if around_center:
            lo = max(lo, 0.0)
    elif isinstance(value, Sequence) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise ValueError(f"ColorJitter: {name} must be a number or (min, max), got {value!r}")
    if not (math.isfinite(hi) and bound[0] <= lo <= hi <= bound[1]):
        raise ValueError(f"ColorJitter: {name} range ({lo}, {hi}) must be finite and lie "
                         f"in {bound}")
    return None if lo == hi == center else (lo, hi)


def _interpolation(value: Any, name: str) -> str:
    """torchvision's InterpolationMode (or its string value) -> "bilinear" | "bicubic"."""
    mode = str(getattr(value, "value", value)).lower()
    if mode not in ("bilinear", "bicubic"):
        raise ValueError(f"{name}: interpolation must be bilinear or bicubic, got {value!r}")
    return mode


_PAD_MODES = ("constant", "edge", "reflect", "symmetric")


def _padding4(padding: Any, name: str) -> tuple[int, int, int, int]:
    """torchvision padding -> (left, top, right, bottom)."""
    def is_int(v: Any) -> bool:
        return isinstance(v, numbers.Integral) and not isinstance(v, bool)

    if is_int(padding):
        pads = [int(padding)] * 4
    elif (isinstance(padding, Sequence) and not isinstance(padding, str)
          and len(padding) in (1, 2, 4) and all(is_int(v) for v in padding)):
        p = [int(v) for v in padding]
        pads = p * 4 if len(p) == 1 else ([p[0], p[1], p[0], p[1]] if len(p) == 2 else p)
    else:
        raise ValueError(f"{name}: padding must be an int or a sequence of 1, 2 or 4 ints, "
                         f"got {padding!r}")
    if min(pads) < 0:
        raise ValueError(f"{name}: padding must be non-negative, got {padding!r}")
    return pads[0], pads[1], pads[2], pads[3]


def _fill3(fill: Any, name: str) -> tuple[float, float, float]:
    if fill is None:                # torchvision treats None as 0
        fill = 0
    vals = [fill] if isinstance(fill, numbers.Real) else list(fill)
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3 or not all(isinstance(v, numbers.Real) and 0 <= float(v) <= 1 for v in vals):
        raise ValueError(f"{name}: fill is one or three values on the 0-1 scale (images are "
                         f"floats in the pipeline; divide a uint8 fill by 255), got {fill!r}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _pad_mode(mode: Any, name: str) -> str:
    if mode not in _PAD_MODES:
        raise ValueError(f"{name}: padding_mode must be one of {_PAD_MODES}, got {mode!r}")
    return str(mode)


def _check_mirror(name: str, mode: str, pads: tuple[int, int, int, int],
                  hw: tuple[int, int]) -> None:
    """torchvision refuses reflect padding >= the image size and symmetric
    padding > it (a mirror needs that many pixels to copy)."""
    if mode not in ("reflect", "symmetric"):
        return
    left, top, right, bottom = pads
    h, w = hw
    limit_w, limit_h = (w - 1, h - 1) if mode == "reflect" else (w, h)
    if max(left, right) > limit_w or max(top, bottom) > limit_h:
        raise ValueError(f"{name}: {mode} padding {pads} needs an image larger than its "
                         f"padding, got {h}x{w} (as in torchvision)")


class _Window(Transform):
    """A geometric step that takes a window of the current image: an offset
    and a new size, with pixels outside the image given by ``padding_mode``
    and ``fill``. The sampler runs ``_window`` per image with the current
    size in (CW, CH); it sets the offset (wox, woy) and the new CW, CH."""

    _kind = "geometry"
    padding_mode = "constant"
    fill = (0.0, 0.0, 0.0)

    def _window(self, rng: str) -> list[str]:
        raise NotImplementedError

    def _size_after(self, hw: tuple[int, int] | None) -> tuple[int, int] | None:
        """Output (H, W) for an input of size hw (None: depends on the image)."""
        raise NotImplementedError


# ---------------------------------------------------------------- geometry
# The resample (RandomResizedCrop or Resize) sets sampler locals: the source
# box (gl, gt, gw, gh) of the W x H image, resampled to (RW, RH). Window ops
# (_Window: Pad, CenterCrop, RandomCrop) then take offsets into that grid.
class RandomResizedCrop(Transform):
    """torchvision RandomResizedCrop, the box drawn per image. ``interpolation``
    is ``"bilinear"`` (default) or ``"bicubic"``, as a string or torchvision's
    ``InterpolationMode``."""

    _kind = "geometry"
    _resamples = True

    def __init__(self, size: Any, scale: tuple[float, float] = (0.08, 1.0),
                 ratio: tuple[float, float] = (3 / 4, 4 / 3), interpolation: Any = "bilinear",
                 antialias: bool = True, same_on_batch: bool = False):
        self.size = _size2(size, "RandomResizedCrop")
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        if not 0 < self.scale[0] <= self.scale[1]:
            raise ValueError(f"RandomResizedCrop: invalid scale {scale}")
        if not 0 < self.ratio[0] <= self.ratio[1]:
            raise ValueError(f"RandomResizedCrop: invalid ratio {ratio}")
        self.interpolation = _interpolation(interpolation, "RandomResizedCrop")
        self.antialias = bool(antialias)
        self.same_on_batch = bool(same_on_batch)

    def _geometry(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"sample_rrc({rng}, W, H, {lit(self.scale[0])}, {lit(self.scale[1])}, "
                f"{lit(self.ratio[0])}, {lit(self.ratio[1])}, gl, gt, gw, gh);",
                f"RW = {w}; RH = {h};"]

    def __repr__(self) -> str:
        return (f"RandomResizedCrop(size={self.size}, scale={self.scale}, ratio={self.ratio}, "
                f"interpolation={self.interpolation!r})")


class Resize(Transform):
    """torchvision Resize: an int resizes the shorter side and keeps the aspect
    ratio; (height, width) resizes to exactly that size. ``interpolation`` is
    ``"bilinear"`` (default) or ``"bicubic"``."""

    _kind = "geometry"
    _resamples = True

    def __init__(self, size: Any, interpolation: Any = "bilinear", antialias: bool = True):
        self.shorter = isinstance(size, numbers.Integral) or (isinstance(size, Sequence)
                                                              and len(size) == 1)
        self.size = _size2(size, "Resize")
        self.interpolation = _interpolation(interpolation, "Resize")
        self.antialias = bool(antialias)

    def _geometry(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"resize_shorter(W, H, {h}, RW, RH);" if self.shorter else f"RW = {w}; RH = {h};"]

    def __repr__(self) -> str:
        return (f"Resize(size={self.size[0] if self.shorter else self.size}, "
                f"interpolation={self.interpolation!r})")


class CenterCrop(_Window):
    """torchvision CenterCrop; a crop larger than the image pads with zeros."""

    def __init__(self, size: Any):
        self.size = _size2(size, "CenterCrop")

    def _window(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"wox = center_offset(CW, {w}); woy = center_offset(CH, {h}); "
                f"CW = {w}; CH = {h};"]

    def _size_after(self, hw: tuple[int, int] | None) -> tuple[int, int] | None:
        return self.size

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class Pad(_Window):
    """torchvision Pad. ``padding`` is an int, (left/right, top/bottom) or
    (left, top, right, bottom); ``padding_mode`` is ``"constant"`` (with
    ``fill``, one or three values on the 0-1 scale), ``"edge"``,
    ``"reflect"`` or ``"symmetric"``."""

    def __init__(self, padding: Any, fill: Any = 0, padding_mode: str = "constant"):
        self.padding = _padding4(padding, "Pad")
        self.fill = _fill3(fill, "Pad")
        self.padding_mode = _pad_mode(padding_mode, "Pad")

    def _window(self, rng: str) -> list[str]:
        left, top, right, bottom = self.padding
        return [f"wox = {-left}; woy = {-top}; CW += {left + right}; CH += {top + bottom};"]

    def _size_after(self, hw: tuple[int, int] | None) -> tuple[int, int] | None:
        if hw is None:
            return None
        _check_mirror("Pad", self.padding_mode, self.padding, hw)
        left, top, right, bottom = self.padding
        return hw[0] + top + bottom, hw[1] + left + right

    def __repr__(self) -> str:
        return f"Pad(padding={self.padding}, fill={self.fill}, padding_mode={self.padding_mode!r})"


class RandomCrop(_Window):
    """torchvision RandomCrop: optional ``padding`` (as in :class:`Pad`;
    ``pad_if_needed`` pads images smaller than ``size`` on both sides), then a
    ``size`` crop at a random position, drawn per image."""

    def __init__(self, size: Any, padding: Any = None, pad_if_needed: bool = False,
                 fill: Any = 0, padding_mode: str = "constant", same_on_batch: bool = False):
        self.size = _size2(size, "RandomCrop")
        self.padding = (0, 0, 0, 0) if padding is None else _padding4(padding, "RandomCrop")
        self.pad_if_needed = bool(pad_if_needed)
        self.fill = _fill3(fill, "RandomCrop")
        self.padding_mode = _pad_mode(padding_mode, "RandomCrop")
        self.same_on_batch = bool(same_on_batch)

    def _padded(self, hw: tuple[int, int]) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) padding for an image of size hw, as
        torchvision computes it (pad_if_needed adds the shortfall to both sides)."""
        left, top, right, bottom = self.padding
        h, w = self.size
        ph, pw = hw[0] + top + bottom, hw[1] + left + right
        if self.pad_if_needed and ph < h:
            top, bottom = top + h - ph, bottom + h - ph
        if self.pad_if_needed and pw < w:
            left, right = left + w - pw, right + w - pw
        return left, top, right, bottom

    def _window(self, rng: str) -> list[str]:
        left, top, right, bottom = self.padding
        h, w = self.size
        lines = [f"int pl = {left}; int pt = {top};",
                 f"int PW = CW + {left + right}; int PH = CH + {top + bottom};"]
        if self.pad_if_needed:
            lines += [f"if (PH < {h}) {{ pt += {h} - PH; PH = 2 * {h} - PH; }}",
                      f"if (PW < {w}) {{ pl += {w} - PW; PW = 2 * {w} - PW; }}"]
        return lines + [
            # always draw, so every image advances a shared (same_on_batch) stream alike
            f"int ct = irand({rng}, PH - {h} + 1);",
            f"int cl = irand({rng}, PW - {w} + 1);",
            f"wox = cl - pl; woy = ct - pt; CW = {w}; CH = {h};"]

    def _size_after(self, hw: tuple[int, int] | None) -> tuple[int, int] | None:
        if hw is not None:
            left, top, right, bottom = self._padded(hw)
            _check_mirror("RandomCrop", self.padding_mode, (left, top, right, bottom), hw)
            if hw[0] + top + bottom < self.size[0] or hw[1] + left + right < self.size[1]:
                raise ValueError(f"RandomCrop: crop size {self.size} is larger than the "
                                 f"{'padded ' if any(self.padding) else ''}image "
                                 f"{(hw[0] + top + bottom, hw[1] + left + right)}; pass "
                                 "pad_if_needed=True or padding")
        return self.size

    def __repr__(self) -> str:
        return (f"RandomCrop(size={self.size}, padding={self.padding}, "
                f"pad_if_needed={self.pad_if_needed}, padding_mode={self.padding_mode!r})")


class _Flip(Transform):
    """A flip decided per image; _axis 0 mirrors x, 1 mirrors y."""

    _kind = "flip"
    _axis = 0

    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;"]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class RandomHorizontalFlip(_Flip):
    """torchvision RandomHorizontalFlip, decided per image."""


class RandomVerticalFlip(_Flip):
    """torchvision RandomVerticalFlip, decided per image."""

    _axis = 1



# ------------------------------------------------------------ colour ops
class ColorJitter(Transform):
    """torchvision ColorJitter: brightness, contrast, saturation and hue in a
    random order, with factors and order drawn per image. Contrast blends with
    each image's exact mean gray value, as torchvision does."""

    def __init__(self, brightness: Any = 0, contrast: Any = 0, saturation: Any = 0,
                 hue: Any = 0, same_on_batch: bool = False):
        self.ranges = [
            _range(brightness, "brightness", 1.0, (0.0, math.inf), True),
            _range(contrast, "contrast", 1.0, (0.0, math.inf), True),
            _range(saturation, "saturation", 1.0, (0.0, math.inf), True),
            _range(hue, "hue", 0.0, (-0.5, 0.5), False),
        ]
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 5        # four factors and the order

    def _stat_count(self) -> int:
        return 1 if self.ranges[1] else 0

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        lines = [f"p[uint2({k + i}, b)] = urand({rng}, {lit(r[0])}, {lit(r[1])});" if r
                 else f"p[uint2({k + i}, b)] = 0.0;" for i, r in enumerate(self.ranges)]
        return lines + [f"p[uint2({k + 4}, b)] = float(random_order({rng}));"]

    def _apply(self, b: Binding) -> list[str]:
        k = b.k(self)
        cases = []
        if self.ranges[0]:
            cases.append(f"if (op == 0) c = brightness(c, {P(k)});")
        if self.ranges[1]:
            st = b.stats(self)[0]
            cases.append(f"if (op == 1) {{ if (upto == {st}) return c; "
                         f"c = contrast(c, {P(k + 1)}, {ST(st)}); }}")
        if self.ranges[2]:
            cases.append(f"if (op == 2) c = saturation(c, {P(k + 2)});")
        if self.ranges[3]:
            cases.append(f"if (op == 3) c = hue(c, {P(k + 3)});")
        if not cases:
            return []
        return [f"{{ int order = int({P(k + 4)});",
                "  for (int j = 0; j < 4; ++j) { int op = (order >> (2 * j)) & 3;",
                *[f"    {c}" for c in cases], "  } }"]

    def __repr__(self) -> str:
        return f"ColorJitter(ranges={self.ranges})"


class RandomGrayscale(ColorOp):
    """torchvision RandomGrayscale (three equal output channels)."""

    slang = "float4 apply(float4 c) { float g = gray(c); return float4(g, g, g, 0.0); }"

    def __init__(self, p: float = 0.1, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)


class RandomSolarize(ColorOp):
    """torchvision RandomSolarize. ``threshold`` is on the 0-1 scale (the
    pipeline works on floats): use 0.5, not 128."""

    slang = """
    float4 apply(float4 c, float t)
    {
        return float4(c.x >= t ? 1.0 - c.x : c.x, c.y >= t ? 1.0 - c.y : c.y,
                      c.z >= t ? 1.0 - c.z : c.z, 0.0);
    }"""

    def __init__(self, threshold: float, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"RandomSolarize: threshold is on the 0-1 scale, got {threshold} "
                             "(divide a uint8 threshold by 255)")
        self.params = {"threshold": float(threshold)}


# ------------------------------------------------------- neighbourhood ops
class GaussianBlur(Transform):
    """torchvision GaussianBlur (reflect border), sigma drawn per image.
    Separable: a horizontal pre-pass, and the vertical half fused into the
    next pass. Wrap it in RandomApply to blur only some images."""

    _kind = "neighbourhood"

    def __init__(self, kernel_size: Any, sigma: Any = (0.1, 2.0), same_on_batch: bool = False):
        ks = [kernel_size] if isinstance(kernel_size, numbers.Integral) else list(kernel_size)
        if len(ks) == 1:
            ks = ks * 2
        if len(ks) != 2 or any(int(k) <= 0 or int(k) % 2 == 0 for k in ks):
            raise ValueError(f"GaussianBlur: kernel_size must be positive and odd, got {kernel_size}")
        kx, ky = ks
        self.kernel_size = (int(kx), int(ky))
        if isinstance(sigma, numbers.Real):
            lo = hi = float(sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            lo, hi = float(sigma[0]), float(sigma[1])
        else:
            raise ValueError(f"GaussianBlur: sigma must be a number or (min, max), got {sigma!r}")
        if not 0 < lo <= hi:
            raise ValueError(f"GaussianBlur: sigma must satisfy 0 < min <= max, got {sigma}")
        self.sigma = Uniform(lo, hi)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = {self.sigma._draw(rng)};"]

    def _taps(self, b: Binding, radius: int, coord: str, size: str, load: str) -> list[str]:
        return [f"float sig = {P(b.k(self))}; float4 acc = float4(0.0); float ws = 0.0;",
                f"for (int i = -{radius}; i <= {radius}; ++i) {{",
                "    float w = gauss(i, sig); ws += w;",
                f"    int q = reflect({coord} + i, {size}); acc += w * {load}; }}",
                "return acc / ws;"]

    def _pre(self, b: Binding) -> list[str] | None:
        return self._taps(b, self.kernel_size[0] // 2, "x", "OW", "load_mid(src, q, y, b)")

    def _input(self, b: Binding) -> list[str]:
        return self._taps(b, self.kernel_size[1] // 2, "y", "OH", "load_mid(src, x, q, b)")

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"


# --------------------------------------------------------------- others
class RandomApply(Transform):
    """torchvision RandomApply: apply ``transforms`` to an image with
    probability ``p``. Holds pointwise ops, or exactly one op that reads
    neighbouring pixels (a blur)."""

    def __init__(self, transforms: Sequence[Transform], p: float = 0.5,
                 same_on_batch: bool = False):
        self.transforms = list(transforms)
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)
        if not self.transforms:
            raise ValueError("RandomApply needs at least one transform")
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"RandomApply takes slangvision transforms, got {t!r}")
        kinds = {t._kind for t in self.transforms}
        if "neighbourhood" in kinds and len(self.transforms) != 1:
            raise ValueError("RandomApply holds either pointwise ops or exactly one op that "
                             "reads neighbouring pixels (such as GaussianBlur)")
        if kinds - {"pointwise", "neighbourhood"}:
            raise ValueError("RandomApply holds colour ops and blurs, not geometric "
                             "transforms, flips (they have their own p) or Normalize")
        self._kind = "neighbourhood" if "neighbourhood" in kinds else "pointwise"

    def _children(self) -> list[Transform]:
        return self.transforms

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;"]

    def _apply(self, b: Binding) -> list[str]:
        body = [ln for t in self.transforms for ln in t._apply(b)]
        return [f"if ({P(b.k(self))} > 0.5) {{", *[f"  {ln}" for ln in body], "}"]

    def __repr__(self) -> str:
        return f"RandomApply({self.transforms}, p={self.p})"

    def _gate(self, b: Binding, child: Transform) -> str:
        return f"{P(b.k(self))} > 0.5"

    def _post(self, b: Binding) -> list[str]:
        body = self.transforms[0]._post(b)
        return [f"if ({P(b.k(self))} > 0.5) {{", *[f"  {ln}" for ln in body], "}"] if body else []


class RandomChoice(Transform):
    """torchvision RandomChoice: apply one of ``transforms`` per image, chosen
    with weights ``p`` (equal by default). Holds colour ops and at most one op
    that reads neighbouring pixels, e.g. the choice in DeiT III's ThreeAugment:
    ``RandomChoice([RandomGrayscale(p=1), RandomSolarize(0.5, p=1), GaussianBlur(13)])``.
    With ``same_on_batch=True`` every image gets the same choice."""

    def __init__(self, transforms: Sequence[Transform], p: Sequence[float] | None = None,
                 same_on_batch: bool = False):
        self.transforms = list(transforms)
        self.same_on_batch = bool(same_on_batch)
        if not self.transforms:
            raise ValueError("RandomChoice needs at least one transform")
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"RandomChoice takes slangvision transforms, got {t!r}")
        weights = [1.0] * len(self.transforms) if p is None else [float(v) for v in p]
        if len(weights) != len(self.transforms) or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("RandomChoice: p needs one non-negative weight per transform")
        self.p = [w / sum(weights) for w in weights]
        kinds = [t._kind for t in self.transforms]
        if kinds.count("neighbourhood") > 1:
            raise ValueError("RandomChoice holds at most one op that reads neighbouring pixels")
        if set(kinds) - {"pointwise", "neighbourhood"}:
            raise ValueError("RandomChoice holds colour ops and blurs, not geometric "
                             "transforms, flips or Normalize")
        self._kind = "neighbourhood" if "neighbourhood" in kinds else "pointwise"

    def _children(self) -> list[Transform]:
        return self.transforms

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k, edges = b.k(self), []
        total = 0.0
        for w in self.p[:-1]:
            total += w
            edges.append(total)
        pick = " + ".join(f"(u < {lit(e)} ? 0 : 1)" for e in edges) or "0"
        return [f"{{ float u = rnd({rng}); p[uint2({k}, b)] = float({pick}); }}"]

    def _apply(self, b: Binding) -> list[str]:
        return self._branches(b, lambda t: t._apply(b))

    def _post(self, b: Binding) -> list[str]:
        # with a blur among the choices: the colour-op choices (and those of a
        # nested container) run in the pass after it
        return self._branches(b, lambda t: t._apply(b) if t._kind == "pointwise" else t._post(b))

    def _branches(self, b: Binding, body_of: Any) -> list[str]:
        lines = []
        for i, t in enumerate(self.transforms):
            body = body_of(t)
            if body:
                lines += [f"if (int({P(b.k(self))}) == {i}) {{", *[f"  {ln}" for ln in body], "}"]
        return lines

    def _gate(self, b: Binding, child: Transform) -> str:
        return f"int({P(b.k(self))}) == {self.transforms.index(child)}"

    def __repr__(self) -> str:
        return f"RandomChoice({self.transforms}, p={self.p})"


class Normalize(Transform):
    """torchvision Normalize; must be the last transform."""

    _kind = "normalize"

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Normalize: mean and std need three values (RGB)")
        if any(s == 0 for s in self.std):
            raise ValueError("Normalize: std must be non-zero")

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"
