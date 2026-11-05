# slangvision/transforms.py — the built-in transforms. Names, parameters and
# results follow torchvision.transforms.v2 on float images; random values are
# drawn per image unless same_on_batch=True. RandomGrayscale and
# RandomSolarize are written with the public ColorOp API.

from __future__ import annotations

import math
import numbers
import re
import struct
from typing import Any, Sequence

from ._base import ST, Binding, ColorOp, P, Transform, Uniform, _probability, lit, stop


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
           around_center: bool, owner: str = "ColorJitter") -> tuple[float, float] | None:
    """torchvision ColorJitter range checking; None when the op is a no-op."""
    if isinstance(value, numbers.Real):
        value = float(value)
        if value < 0:
            raise ValueError(f"{owner}: {name} must be non-negative, got {value}")
        lo, hi = center - value, center + value
        if around_center:
            lo = max(lo, 0.0)
    elif isinstance(value, Sequence) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise ValueError(f"{owner}: {name} must be a number or (min, max), got {value!r}")
    if not (math.isfinite(hi) and bound[0] <= lo <= hi <= bound[1]):
        raise ValueError(f"{owner}: {name} range ({lo}, {hi}) must be finite and lie "
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


# ------------------------------------------------------------------ warps
# A warp resamples the image as it is at its point of the pipeline (after
# the resize and crops) and keeps its size. Each starts a new pass that
# samples the previous one. Its slots: applied, the eight coefficients of
# shaders/warp.slang (computed by the sampler), then its raw random values
# for the torchvision reference.
def _dlit(x: float) -> str:
    """A Slang double literal."""
    return f"{float(x)!r}l"


def _warp_interpolation(value: Any, name: str) -> str:
    mode = str(getattr(value, "value", value)).lower()
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"{name}: interpolation must be nearest or bilinear (as in "
                         f"torchvision for tensors), got {value!r}")
    return mode


def _angles(value: Any, name: str, sizes: tuple[int, ...]) -> tuple[float, ...]:
    """torchvision's _setup_angle: a number d means (-d, d)."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{name}: a single number must be non-negative, got {value}")
        vals = [-float(value), float(value)]
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) in sizes:
        vals = [float(v) for v in value]
    else:
        raise ValueError(f"{name}: must be a number or a sequence of "
                         f"{' or '.join(map(str, sizes))} numbers, got {value!r}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name}: values must be finite, got {value!r}")
    return tuple(vals)


def _pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not (isinstance(value, Sequence) and len(value) == 2):
        raise ValueError(f"{name}: must be a sequence of two numbers, got {value!r}")
    a, b = float(value[0]), float(value[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"{name}: values must be finite, got {value!r}")
    return a, b


def _warp_lines(k: int, interpolation: str, fill: tuple[float, float, float] | None) -> list[str]:
    """Sample the previous pass through the coefficients in slots k .. k + 7;
    fill None keeps torch's zero padding (no fill blend)."""
    f0, f1, f2 = (lit(v) for v in (fill or (0.0, 0.0, 0.0)))
    return ["MidImage img = { src, b, OW, OH, 1 };",
            "float fx = float(x); float fy = float(y);",
            f"float den = {P(k + 6)} * fx + {P(k + 7)} * fy + 1.0;",
            f"float u = ({P(k)} * fx + {P(k + 1)} * fy + {P(k + 2)}) / den;",
            f"float v = ({P(k + 3)} * fx + {P(k + 4)} * fy + {P(k + 5)}) / den;",
            f"return warp_sample(img, u, v, {int(interpolation == 'nearest')}, "
            f"float4({f0}, {f1}, {f2}, 0.0), {int(fill is not None)});"]


class _Warp(Transform):
    _kind = "warp"
    p = 1.0
    interpolation = "nearest"
    fill = (0.0, 0.0, 0.0)
    center: tuple[float, float] | None = None
    _raw = 0            # raw random values kept for the reference

    def _slot_count(self) -> int:
        return 9 + self._raw

    def _draw(self, k: int, rng: str) -> list[str]:
        """Sampler lines that set `bool on`, fill `double m[8]` and write the
        raw values from slot k + 9 on (CW, CH: the image size here)."""
        raise NotImplementedError

    def _center(self) -> str:
        if self.center is None:
            return "double cx = 0.0l; double cy = 0.0l;"
        return (f"double cx = {_dlit(self.center[0])} - 0.5l * double(CW); "
                f"double cy = {_dlit(self.center[1])} - 0.5l * double(CH);")

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        return ["{", "    bool on = true; double m[8];",
                *[f"    {ln}" for ln in self._draw(k, rng)],
                f"    p[uint2({k}, b)] = on ? 1.0 : 0.0;",
                f"    for (int i = 0; i < 8; ++i) p[uint2({k + 1} + i, b)] = float(m[i]);",
                "}"]

    def _pre(self, b: Binding) -> list[str] | None:
        return None

    def _input(self, b: Binding) -> list[str]:
        """Body of the warp pass's input function (src, p, b, x, y, OW, OH in scope)."""
        k = b.k(self)
        gate = [f"if (!({P(k)} > 0.5)) return load_mid(src, x, y, b);"] if self.p < 1.0 else []
        return gate + _warp_lines(k + 1, self.interpolation, self.fill)


class RandomRotation(_Warp):
    """torchvision RandomRotation: rotate by an angle drawn per image from
    ``degrees`` (d means (-d, d)), counter-clockwise, about the image centre or
    ``center`` (x, y in pixels). ``interpolation`` is ``"nearest"`` (default,
    as in torchvision) or ``"bilinear"``; corners outside the image get
    ``fill`` (0-1 scale). ``expand=True`` is not supported: it changes each
    image's size."""

    _raw = 1

    def __init__(self, degrees: Any, interpolation: Any = "nearest", expand: bool = False,
                 center: Any = None, fill: Any = 0, same_on_batch: bool = False):
        self.degrees = _angles(degrees, "RandomRotation: degrees", (2,))
        if expand:
            raise ValueError("RandomRotation: expand=True is not supported (it changes each "
                             "image's size)")
        self.interpolation = _warp_interpolation(interpolation, "RandomRotation")
        self.center = _pair(center, "RandomRotation: center")
        self.fill = _fill3(fill, "RandomRotation")
        self.same_on_batch = bool(same_on_batch)

    def _draw(self, k: int, rng: str) -> list[str]:
        lo, hi = self.degrees
        return [f"float ang = urand({rng}, {lit(lo)}, {lit(hi)});",
                self._center(),
                # torchvision's rotate passes -angle to the affine matrix
                "affine_coeffs(CW, CH, -double(ang), 0.0l, 0.0l, 1.0l, 0.0l, 0.0l, cx, cy, m);",
                f"p[uint2({k + 9}, b)] = ang;"]

    def __repr__(self) -> str:
        return (f"RandomRotation(degrees={self.degrees}, interpolation={self.interpolation!r}, "
                f"center={self.center}, fill={self.fill})")


class RandomAffine(_Warp):
    """torchvision RandomAffine: rotation by ``degrees``, translation by up to
    ``translate`` (fractions of width and height, whole pixels), ``scale`` and
    ``shear`` (degrees: d, (min, max) along x, or (xmin, xmax, ymin, ymax)),
    drawn per image. ``interpolation``, ``fill`` and ``center`` as in
    :class:`RandomRotation`."""

    _raw = 6        # angle, tx, ty, scale, shear x, shear y

    def __init__(self, degrees: Any, translate: Any = None, scale: Any = None,
                 shear: Any = None, interpolation: Any = "nearest", fill: Any = 0,
                 center: Any = None, same_on_batch: bool = False):
        self.degrees = _angles(degrees, "RandomAffine: degrees", (2,))
        self.translate = _pair(translate, "RandomAffine: translate")
        if self.translate and not all(0.0 <= t <= 1.0 for t in self.translate):
            raise ValueError(f"RandomAffine: translate values must be in [0, 1], got {translate}")
        self.scale = _pair(scale, "RandomAffine: scale")
        if self.scale and not 0 < self.scale[0] <= self.scale[1]:
            raise ValueError(f"RandomAffine: scale must satisfy 0 < min <= max, got {scale}")
        self.shear = None if shear is None else _angles(shear, "RandomAffine: shear", (2, 4))
        self.interpolation = _warp_interpolation(interpolation, "RandomAffine")
        self.fill = _fill3(fill, "RandomAffine")
        self.center = _pair(center, "RandomAffine: center")
        self.same_on_batch = bool(same_on_batch)

    def _draw(self, k: int, rng: str) -> list[str]:
        lo, hi = self.degrees
        lines = [f"float ang = urand({rng}, {lit(lo)}, {lit(hi)});",
                 "float tx = 0.0; float ty = 0.0; float sc = 1.0; float shx = 0.0; float shy = 0.0;"]
        if self.translate:
            fx, fy = self.translate
            lines += [f"float mdx = float({_dlit(fx)} * double(CW)); "
                      f"float mdy = float({_dlit(fy)} * double(CH));",
                      # torchvision rounds half to even (Python round)
                      f"tx = round_even(urand({rng}, -mdx, mdx)); "
                      f"ty = round_even(urand({rng}, -mdy, mdy));"]
        if self.scale:
            lines.append(f"sc = urand({rng}, {lit(self.scale[0])}, {lit(self.scale[1])});")
        if self.shear:
            lines.append(f"shx = urand({rng}, {lit(self.shear[0])}, {lit(self.shear[1])});")
            if len(self.shear) == 4:
                lines.append(f"shy = urand({rng}, {lit(self.shear[2])}, {lit(self.shear[3])});")
        return lines + [
            self._center(),
            "affine_coeffs(CW, CH, double(ang), double(tx), double(ty), double(sc), "
            "double(shx), double(shy), cx, cy, m);",
            f"p[uint2({k + 9}, b)] = ang; p[uint2({k + 10}, b)] = tx; "
            f"p[uint2({k + 11}, b)] = ty;",
            f"p[uint2({k + 12}, b)] = sc; p[uint2({k + 13}, b)] = shx; "
            f"p[uint2({k + 14}, b)] = shy;"]

    def __repr__(self) -> str:
        return (f"RandomAffine(degrees={self.degrees}, translate={self.translate}, "
                f"scale={self.scale}, shear={self.shear}, "
                f"interpolation={self.interpolation!r}, fill={self.fill}, center={self.center})")


class RandomPerspective(_Warp):
    """torchvision RandomPerspective: with probability ``p``, move each corner
    inwards by up to ``distortion_scale`` of half the width and height (drawn
    per image) and warp the image to match. ``interpolation`` is
    ``"bilinear"`` (default) or ``"nearest"``; ``fill`` as in
    :class:`RandomRotation`."""

    _raw = 8        # the moved corners' x, y

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5,
                 interpolation: Any = "bilinear", fill: Any = 0, same_on_batch: bool = False):
        self.distortion_scale = float(distortion_scale)
        if not 0.0 <= self.distortion_scale <= 1.0:
            raise ValueError("RandomPerspective: distortion_scale must be in [0, 1], got "
                             f"{distortion_scale}")
        self.p = _probability(p)
        self.interpolation = _warp_interpolation(interpolation, "RandomPerspective")
        self.fill = _fill3(fill, "RandomPerspective")
        self.same_on_batch = bool(same_on_batch)

    def _draw(self, k: int, rng: str) -> list[str]:
        ds = _dlit(self.distortion_scale)
        return [
            f"on = rnd({rng}) < {lit(self.p)};",
            f"int bw = int({ds} * double(CW / 2)) + 1; int bh = int({ds} * double(CH / 2)) + 1;",
            "int4 ex; int4 ey;",
            f"ex.x = irand({rng}, bw); ey.x = irand({rng}, bh);",
            f"ex.y = CW - bw + irand({rng}, bw); ey.y = irand({rng}, bh);",
            f"ex.z = CW - bw + irand({rng}, bw); ey.z = CH - bh + irand({rng}, bh);",
            f"ex.w = irand({rng}, bw); ey.w = CH - bh + irand({rng}, bh);",
            "perspective_coeffs(CW, CH, ex, ey, m);",
            "for (int i = 0; i < 4; ++i) {",
            f"    p[uint2({k + 9} + 2 * i, b)] = float(ex[i]); "
            f"p[uint2({k + 10} + 2 * i, b)] = float(ey[i]); }}"]

    def __repr__(self) -> str:
        return (f"RandomPerspective(distortion_scale={self.distortion_scale}, p={self.p}, "
                f"interpolation={self.interpolation!r}, fill={self.fill})")



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
            cases.append(f"if (op == 1) {{ {stop(st)} c = contrast(c, {P(k + 1)}, {ST(st)}); }}")
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


class RandomPhotometricDistort(Transform):
    """torchvision RandomPhotometricDistort (SSD's colour distortion):
    brightness, contrast, saturation and hue, each applied with probability
    ``p`` and a factor from its (min, max) range; contrast comes before
    saturation or after hue (even odds); then, with probability ``p``, one of
    the six channel orders. All drawn per image."""

    def __init__(self, brightness: tuple[float, float] = (0.875, 1.125),
                 contrast: tuple[float, float] = (0.5, 1.5),
                 saturation: tuple[float, float] = (0.5, 1.5),
                 hue: tuple[float, float] = (-0.05, 0.05), p: float = 0.5,
                 same_on_batch: bool = False):
        name = "RandomPhotometricDistort"
        self.ranges = []
        for value, label, center, bound in ((brightness, "brightness", 1.0, (0.0, math.inf)),
                                            (contrast, "contrast", 1.0, (0.0, math.inf)),
                                            (saturation, "saturation", 1.0, (0.0, math.inf)),
                                            (hue, "hue", 0.0, (-0.5, 0.5))):
            if not (isinstance(value, Sequence) and len(value) == 2):
                raise ValueError(f"{name}: {label} must be (min, max), got {value!r}")
            self.ranges.append(_range(value, label, center, bound, False, name))
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 6        # four factors, which ops apply (bits), the channel order

    def _stat_count(self) -> int:
        return 1 if self.ranges[1] else 0

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        lines = [f"p[uint2({k + i}, b)] = urand({rng}, {lit(r[0])}, {lit(r[1])});" if r
                 else f"p[uint2({k + i}, b)] = 0.0;" for i, r in enumerate(self.ranges)]
        # bits 0-3: op i applies; 4: contrast before saturation; 5: permute
        bits = [f"if (rnd({rng}) < {lit(self.p)}) f |= {1 << i};"
                for i, r in enumerate(self.ranges) if r]
        return lines + ["{ int f = 0;", *bits,
                        f"if (rnd({rng}) < 0.5) f |= 16;",
                        f"if (rnd({rng}) < {lit(self.p)}) f |= 32;",
                        f"p[uint2({k + 4}, b)] = float(f);",
                        f"p[uint2({k + 5}, b)] = float(irand({rng}, 6)); }}"]

    def _apply(self, b: Binding) -> list[str]:
        k = b.k(self)
        contrast = ""
        if self.ranges[1]:
            st = b.stats(self)[0]
            contrast = f"{{ {stop(st)} c = contrast(c, {P(k + 1)}, {ST(st)}); }}"
        lines = [f"{{ int f = int({P(k + 4)});",
                 f"  if ((f & 1) != 0) c = brightness(c, {P(k)});" if self.ranges[0] else "",
                 f"  if ((f & 18) == 18) {contrast}" if contrast else "",
                 f"  if ((f & 4) != 0) c = saturation(c, {P(k + 2)});" if self.ranges[2] else "",
                 f"  if ((f & 8) != 0) c = hue(c, {P(k + 3)});" if self.ranges[3] else "",
                 f"  if ((f & 18) == 2) {contrast}" if contrast else "",
                 f"  if ((f & 32) != 0) c = permute_channels(c, int({P(k + 5)})); }}"]
        return [ln for ln in lines if ln]

    def __repr__(self) -> str:
        return f"RandomPhotometricDistort(ranges={self.ranges}, p={self.p})"


class RandomChannelPermutation(Transform):
    """torchvision RandomChannelPermutation: one of the six channel orders
    (the identity included), drawn per image."""

    def __init__(self, same_on_batch: bool = False):
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = float(irand({rng}, 6));"]

    def _apply(self, b: Binding) -> list[str]:
        return [f"c = permute_channels(c, int({P(b.k(self))}));"]


#: permute_channels' channel orders, by index
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


class RandomGrayscale(ColorOp):
    """torchvision RandomGrayscale (three equal output channels)."""

    slang = "float4 apply(float4 c) { float g = gray(c); return float4(g, g, g, 0.0); }"

    def __init__(self, p: float = 0.1, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)


class Grayscale(ColorOp):
    """torchvision Grayscale. With ``num_output_channels=1`` (torchvision's
    default) the pipeline returns ``(B, 1, H, W)``: it must sit at the top
    level of the pipeline, and a Normalize or RandomErasing value after it
    takes one value, a fill one value or three equal ones. Colour ops after
    it see three equal channels, and the output keeps the first."""

    slang = RandomGrayscale.slang

    def __init__(self, num_output_channels: int = 1):
        super().__init__(1.0)
        if num_output_channels not in (1, 3) or isinstance(num_output_channels, bool):
            raise ValueError(f"Grayscale: num_output_channels must be 1 or 3, got "
                             f"{num_output_channels!r}")
        self.num_output_channels = int(num_output_channels)

    def __repr__(self) -> str:
        return f"Grayscale(num_output_channels={self.num_output_channels})"


class RandomSolarize(ColorOp):
    """torchvision RandomSolarize. ``threshold`` is on the 0-1 scale (the
    pipeline works on floats): use 0.5, not 128."""

    slang = "float4 apply(float4 c, float t) { return solarize(c, t); }"

    def __init__(self, threshold: float, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"RandomSolarize: threshold is on the 0-1 scale, got {threshold} "
                             "(divide a uint8 threshold by 255)")
        self.params = {"threshold": float(threshold)}


class RandomInvert(ColorOp):
    """torchvision RandomInvert: 1 - value."""

    slang = "float4 apply(float4 c) { return invert(c); }"

    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)


class RandomPosterize(ColorOp):
    """torchvision RandomPosterize on float images: ``2**bits`` levels,
    ``floor(value * levels) / levels``. (torchvision's uint8 path keeps the top
    ``bits`` bits instead, which gives slightly different values, e.g. 240/255
    rather than 15/16 for white at 4 bits.)"""

    slang = "float4 apply(float4 c, float n) { return posterize(c, n); }"

    bits: int

    def __init__(self, bits: int, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)
        if not (isinstance(bits, numbers.Integral) and not isinstance(bits, bool)
                and 0 <= bits <= 8):
            raise ValueError(f"RandomPosterize: bits must be an int in [0, 8], got {bits!r}")
        self.bits = int(bits)
        self.params = {"levels": float(1 << self.bits)}

    def __repr__(self) -> str:
        return f"RandomPosterize(bits={self.bits}, p={self.p})"


class _Gated(Transform):
    """A built-in op applied with probability p, decided per image."""

    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;"]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class RandomAutocontrast(_Gated):
    """torchvision RandomAutocontrast: stretch each channel to [0, 1] using the
    image's own min and max (a pre-pass computes them per image)."""

    _stat_kind = "minmax"

    def _stat_count(self) -> int:
        return 6

    def _apply(self, b: Binding) -> list[str]:
        st = b.stats(self)[0]
        args = ", ".join(ST(st + i) for i in range(6))
        return [f"if ({P(b.k(self))} > 0.5) {{ {stop(st)} c = autocontrast(c, {args}); }}"]


class RandomEqualize(_Gated):
    """torchvision RandomEqualize: histogram equalization per channel, on the
    image quantized to 256 levels as torchvision does for float images (so the
    output has 256 levels). A pre-pass builds each image's histograms."""

    _stat_kind = "lut"

    def _stat_count(self) -> int:
        return 768

    def _apply(self, b: Binding) -> list[str]:
        st = b.stats(self)[0]
        return [f"if ({P(b.k(self))} > 0.5) {{ {stop(st)} c = equalize(c, st, b, {st}); }}"]


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

    # Slots: sigma, then the normalized weights at offsets 0..radius of the
    # horizontal kernel, then of the vertical one, computed once per image by
    # the sampler so the passes do not evaluate exp per tap and pixel.
    def _slot_count(self) -> int:
        return 3 + self.kernel_size[0] // 2 + self.kernel_size[1] // 2

    def _weights(self, b: Binding) -> tuple[int, int]:
        """The first weight slot of the horizontal and the vertical kernel."""
        k = b.k(self) + 1
        return k, k + 1 + self.kernel_size[0] // 2

    def _sample(self, b: Binding, rng: str) -> list[str]:
        lines = [f"{{ float sig = {self.sigma._draw(rng)}; p[uint2({b.k(self)}, b)] = sig;"]
        for first, ksize in zip(self._weights(b), self.kernel_size):
            r = ksize // 2      # torchvision's taps exp(-x^2 / 2 sigma^2), normalized
            lines += [f"  {{ float ws = 0.0; for (int i = -{r}; i <= {r}; ++i) ws += gauss(abs(i), sig);",
                      f"    for (int i = 0; i <= {r}; ++i) p[uint2({first} + i, b)] = "
                      "gauss(i, sig) / ws; }"]
        return lines + ["}"]

    def _taps(self, first: int, radius: int, coord: str, size: str, load: str) -> list[str]:
        return ["float4 acc = float4(0.0);",
                f"for (int i = -{radius}; i <= {radius}; ++i) {{",
                f"    int q = reflect({coord} + i, {size}); acc += prm(p, b, {first} + abs(i)) * {load}; }}",
                "return acc;"]

    def _pre(self, b: Binding) -> list[str] | None:
        return self._taps(self._weights(b)[0], self.kernel_size[0] // 2, "x", "OW",
                          "load_mid(src, q, y, b)")

    def _input(self, b: Binding) -> list[str]:
        return self._taps(self._weights(b)[1], self.kernel_size[1] // 2, "y", "OH",
                          "load_mid(src, x, q, b)")

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"


class RandomAdjustSharpness(_Gated):
    """torchvision RandomAdjustSharpness: blend with torchvision's 3x3
    smoothing kernel (border pixels only clamped) by ``sharpness_factor`` (0
    smooths, 1 keeps, 2 sharpens), with probability ``p``."""

    _kind = "neighbourhood"

    def __init__(self, sharpness_factor: float, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)
        self.sharpness_factor = float(sharpness_factor)
        if not (math.isfinite(self.sharpness_factor) and self.sharpness_factor >= 0):
            raise ValueError("RandomAdjustSharpness: sharpness_factor must be non-negative, "
                             f"got {sharpness_factor}")

    def _pre(self, b: Binding) -> list[str] | None:
        return None

    def _input(self, b: Binding) -> list[str]:
        return [f"if (!({P(b.k(self))} > 0.5)) return load_mid(src, x, y, b);",
                "MidImage img = { src, b, OW, OH, 1 };",
                f"return sharpen(img, x, y, {lit(1.0 - self.sharpness_factor)});"]

    def __repr__(self) -> str:
        return f"RandomAdjustSharpness(sharpness_factor={self.sharpness_factor}, p={self.p})"


# ---------------------------------------------------------------- policies
def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _lin_f32(start: float, end: float, steps: int, i: int) -> float:
    """torch.linspace(start, end, steps)[i] in float32, as lin_f32 in
    shaders/warp.slang (torch's vectorized CPU kernel can differ in the last bit)."""
    s, e = _f32(start), _f32(end)
    if steps == 1:
        return s
    step = _f32(_f32(e - s) / (steps - 1))
    if i < steps // 2:
        return _f32(s + _f32(step * i))
    return _f32(e - _f32(step * (steps - i - 1)))


class _Policy(Transform):
    """An augmentation policy: Compose runs each of its steps as a pass."""

    _kind = "policy"
    _steps: list[Transform]

    def _children(self) -> list[Transform]:
        return list(self._steps)


class _PolicyOp(Transform):
    """One step of a policy: a pass whose input warps or sharpens (or copies)
    the previous one, followed by the step's colour ops."""

    _kind = "warp"      # it may warp: flips after it apply after it

    def __init__(self, parent: Any):
        self.parent = parent

    @property
    def same_on_batch(self) -> bool:     # type: ignore[override]
        return bool(self.parent.same_on_batch)

    def _pre(self, b: Binding) -> list[str] | None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(of {self.parent!r})"


# torchvision's policy ops (index = the op's code in the passes below):
# RandAugment and TrivialAugmentWide choose among the first 14, AutoAugment's
# sub-policies name theirs
_TV_OPS = ("Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Brightness",
           "Color", "Contrast", "Sharpness", "Posterize", "Solarize", "AutoContrast", "Equalize",
           "Invert")

# torchvision's AutoAugment sub-policies: two (op, probability, magnitude bin) each
_AA_POLICIES: dict[str, list[tuple[tuple[str, float, int | None], ...]]] = {
    "imagenet": [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)), (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)), (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)), (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, None)), (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)), (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)), (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)), (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)), (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)), (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, None)), (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)), (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)), (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    ],
    "cifar10": [
        (("Invert", 0.1, None), ("Contrast", 0.2, 6)), (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
        (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)), (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)), (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
        (("Color", 0.4, 3), ("Brightness", 0.6, 7)), (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
        (("Equalize", 0.6, None), ("Equalize", 0.5, None)), (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
        (("Color", 0.7, 7), ("TranslateX", 0.5, 8)), (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
        (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)), (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
        (("Solarize", 0.5, 2), ("Invert", 0.0, None)), (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.2, None), ("Equalize", 0.6, None)), (("Color", 0.9, 9), ("Equalize", 0.6, None)),
        (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)), (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
        (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)), (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)), (("Equalize", 0.8, None), ("Invert", 0.1, None)),
        (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
    ],
    "svhn": [
        (("ShearX", 0.9, 4), ("Invert", 0.2, None)), (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.6, None), ("Solarize", 0.6, 6)), (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Equalize", 0.6, None), ("Rotate", 0.9, 3)), (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
        (("ShearY", 0.9, 8), ("Invert", 0.4, None)), (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
        (("Invert", 0.9, None), ("AutoContrast", 0.8, None)), (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)), (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)), (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)), (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
        (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)), (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
        (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)), (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
        (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)), (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
        (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)), (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
        (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
    ],
}


def _is_int(v: Any) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _magnitude_bins(name: str, value: Any) -> int:
    if not (_is_int(value) and value >= 2):
        # torchvision's posterize table divides by num_magnitude_bins - 1
        raise ValueError(f"{name}: num_magnitude_bins must be an int >= 2, got {value!r}")
    return int(value)


class _TorchvisionPolicy(_Policy):
    """torchvision's policies (RandAugment, TrivialAugmentWide, AutoAugment):
    each step draws an op code, a magnitude bin and a sign per image and runs
    torchvision's op at that bin's magnitude. ``interpolation`` is
    ``"nearest"`` or ``"bilinear"``; ``fill=None`` keeps torch's zero padding,
    otherwise one or three values on the 0-1 scale."""

    num_magnitude_bins: int
    _wide = False           # TrivialAugmentWide's larger magnitude ranges

    def _setup(self, interpolation: Any, fill: Any, same_on_batch: bool, steps: int) -> None:
        name = type(self).__name__
        self.interpolation = _warp_interpolation(interpolation, name)
        self.fill = None if fill is None else _fill3(fill, name)
        self.same_on_batch = bool(same_on_batch)
        self._steps = [_TorchvisionPolicyOp(self, j) for j in range(steps)]

    def _choose(self, j: int, k: int, b: Binding, rng: str) -> list[str]:
        """Sampler lines that set `int op` (a _TV_OPS code), `int idx` (the
        magnitude bin), `bool on` and `int pick` (the value kept for the
        reference in slot k + 1) for step j, whose slots start at k."""
        raise NotImplementedError


class _TorchvisionPolicyOp(_PolicyOp):
    """One step of a torchvision policy. Slots: the op code if applied (else
    -1), the policy's pick (the magnitude bin; AutoAugment: the
    sub-policy), the sign (+-1), the eight warp coefficients of
    shaders/warp.slang, torchvision's signed magnitude."""

    parent: _TorchvisionPolicy

    def __init__(self, parent: _TorchvisionPolicy, j: int):
        super().__init__(parent)
        self.j = j

    def _slot_count(self) -> int:
        return 12

    def _stat_layout(self) -> list[tuple[str, int]]:
        return [("mean", 1), ("minmax", 6), ("lut", 768)]      # contrast, autocontrast, equalize

    def _sample(self, b: Binding, rng: str) -> list[str]:
        pol, k = self.parent, b.k(self)
        n, wide = pol.num_magnitude_bins, pol._wide
        shear, rotate, enhance = (0.99, 135.0, 0.99) if wide else (0.3, 30.0, 0.9)

        def table(kind: str, values: Sequence[float]) -> str:
            fmt = _dlit if kind == "double" else lit
            return f"{kind} {{}}[{n}] = {{{{ {', '.join(fmt(v) for v in values)} }}}};"

        def lin(start: float, end: float) -> list[float]:
            return [_lin_f32(start, end, n, i) for i in range(n)]

        # torch.round(8 - arange(n) / ((n - 1) / 4 or 6)) in float32
        div = _f32((n - 1) / (6 if wide else 4))
        bits = [float(round(_f32(8 - _f32(_f32(i) / div)))) for i in range(n)]
        shears = lin(0.0, shear)
        if wide:        # a fixed 32 pixels
            translate = "tr[idx]"
        else:           # 150 / 331 of the image's size, as a float32 end point
            translate = (f"float(lin_f32(0.0l, double(float({_dlit(150.0 / 331.0)} * "
                         f"double(op == 3 ? CW : CH))), {n}, idx))")
        corner = "-0.5l * double(CW), -0.5l * double(CH)"      # center=[0, 0]
        return [
            "{",
            *[f"    {ln}" for ln in pol._choose(self.j, k, b, rng)],
            f"    float sg = rnd({rng}) <= 0.5 ? -1.0 : 1.0;",
            "    double sd = double(sg);",
            # torchvision's magnitude per bin (its float32 tables)
            "    " + table("float", shears).format("sh"),
            "    " + table("double", [math.degrees(math.atan(v)) for v in shears]).format("shd"),
            "    " + table("float", lin(0.0, rotate)).format("rot"),
            "    " + table("float", lin(0.0, enhance)).format("enh"),
            "    " + table("float", bits).format("bits"),
            "    " + table("float", lin(1.0, 0.0)).format("sol"),
            *(["    " + table("float", lin(0.0, 32.0)).format("tr")] if wide else []),
            "    float m = 0.0;",
            "    if (op == 1 || op == 2) m = sh[idx];",
            f"    if (op == 3 || op == 4) m = {translate};",
            "    if (op == 5) m = rot[idx];",
            "    if (op >= 6 && op <= 9) m = enh[idx];",
            "    if (op == 10) m = bits[idx];",
            "    if (op == 11) m = sol[idx];",
            "    if (op >= 1 && op <= 9) m *= sg;",
            "    double c[8]; for (int i = 0; i < 8; ++i) c[i] = 0.0l;",
            f"    if (op == 1) affine_coeffs(CW, CH, 0.0l, 0.0l, 0.0l, 1.0l, sd * shd[idx], 0.0l, "
            f"{corner}, c);",
            f"    if (op == 2) affine_coeffs(CW, CH, 0.0l, 0.0l, 0.0l, 1.0l, 0.0l, sd * shd[idx], "
            f"{corner}, c);",
            # torchvision truncates the translation to whole pixels: int(magnitude)
            "    if (op == 3 || op == 4) { double t = double(int(m));",
            "        affine_coeffs(CW, CH, 0.0l, op == 3 ? t : 0.0l, op == 4 ? t : 0.0l, 1.0l, "
            "0.0l, 0.0l, 0.0l, 0.0l, c); }",
            # rotate passes -angle to the affine matrix
            "    if (op == 5) affine_coeffs(CW, CH, -double(m), 0.0l, 0.0l, 1.0l, 0.0l, 0.0l, "
            "0.0l, 0.0l, c);",
            f"    p[uint2({k}, b)] = on ? float(op) : -1.0; p[uint2({k + 1}, b)] = float(pick);",
            f"    p[uint2({k + 2}, b)] = sg; p[uint2({k + 11}, b)] = m;",
            f"    for (int i = 0; i < 8; ++i) p[uint2({k + 3} + i, b)] = float(c[i]);",
            "}"]

    def _input(self, b: Binding) -> list[str]:
        pol, k = self.parent, b.k(self)
        return [f"int op = int({P(k)});",
                "if (op == 9) {", "    MidImage img = { src, b, OW, OH, 1 };",
                # adjust_sharpness(1 + m) blends with alpha = 1 - factor = -m
                f"    return sharpen(img, x, y, -{P(k + 11)});", "}",
                "if (op < 1 || op > 5) return load_mid(src, x, y, b);",
                *_warp_lines(k + 3, pol.interpolation, pol.fill)]

    def _post(self, b: Binding) -> list[str]:
        k = b.k(self)
        mean, minmax, lut = b.stats(self)
        f = f"(1.0 + {P(k + 11)})"
        mm = ", ".join(ST(minmax + i) for i in range(6))
        return [f"{{ int op = int({P(k)});",
                f"  if (op == 6) c = brightness(c, {f});",
                f"  if (op == 7) c = saturation(c, {f});",
                f"  if (op == 8) {{ {stop(mean)} c = contrast(c, {f}, {ST(mean)}); }}",
                f"  if (op == 10) c = posterize(c, float(1 << clamp(int({P(k + 11)}), 0, 8)));",
                f"  if (op == 11) c = solarize(c, {P(k + 11)});",
                f"  if (op == 12) {{ {stop(minmax)} c = autocontrast(c, {mm}); }}",
                f"  if (op == 13) {{ {stop(lut)} c = equalize(c, st, b, {lut}); }}",
                "  if (op == 14) c = invert(c);",
                "}"]


class RandAugment(_TorchvisionPolicy):
    """torchvision RandAugment: ``num_ops`` times per image, one of 14 ops
    (identity, shear, translate, rotate, brightness, colour, contrast,
    sharpness, posterize, solarize, autocontrast, equalize) chosen uniformly,
    at ``magnitude`` (an index into ``num_magnitude_bins`` levels) with a
    random sign where the op has one. Each op runs as its own pass after the
    resize and crops. ``interpolation`` is ``"nearest"`` (default) or
    ``"bilinear"``; ``fill=None`` (default) keeps torch's zero padding,
    otherwise one or three values on the 0-1 scale. Colour ops follow
    torchvision's float behaviour (posterize to ``floor(value * 2**bits) /
    2**bits``)."""

    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                 interpolation: Any = "nearest", fill: Any = None, same_on_batch: bool = False):
        if not (_is_int(num_ops) and num_ops >= 0):
            raise ValueError(f"RandAugment: num_ops must be a non-negative int, got {num_ops!r}")
        self.num_magnitude_bins = _magnitude_bins("RandAugment", num_magnitude_bins)
        if not (_is_int(magnitude) and 0 <= magnitude < num_magnitude_bins):
            raise ValueError(f"RandAugment: magnitude must be an int in [0, num_magnitude_bins), "
                             f"got {magnitude!r}")
        self.num_ops, self.magnitude = int(num_ops), int(magnitude)
        self._setup(interpolation, fill, same_on_batch, self.num_ops)

    def _choose(self, j: int, k: int, b: Binding, rng: str) -> list[str]:
        return [f"int op = irand({rng}, 14); int idx = {self.magnitude}; bool on = true; "
                "int pick = idx;"]

    def __repr__(self) -> str:
        return (f"RandAugment(num_ops={self.num_ops}, magnitude={self.magnitude}, "
                f"num_magnitude_bins={self.num_magnitude_bins}, "
                f"interpolation={self.interpolation!r}, fill={self.fill})")


class TrivialAugmentWide(_TorchvisionPolicy):
    """torchvision TrivialAugmentWide: one of RandAugment's 14 ops per image,
    chosen uniformly, at a magnitude bin drawn uniformly from
    ``num_magnitude_bins`` over its wider ranges (shear to 0.99, translate to
    32 pixels, rotate to 135 degrees, enhance factors 0.01 to 1.99, posterize
    to 2 bits), with a random sign where the op has one. One pass after the
    resize and crops; ``interpolation`` and ``fill`` as in RandAugment."""

    _wide = True

    def __init__(self, num_magnitude_bins: int = 31, interpolation: Any = "nearest",
                 fill: Any = None, same_on_batch: bool = False):
        self.num_magnitude_bins = _magnitude_bins("TrivialAugmentWide", num_magnitude_bins)
        self._setup(interpolation, fill, same_on_batch, 1)

    def _choose(self, j: int, k: int, b: Binding, rng: str) -> list[str]:
        return [f"int op = irand({rng}, 14); int idx = irand({rng}, {self.num_magnitude_bins}); "
                "bool on = true; int pick = idx;"]

    def __repr__(self) -> str:
        return (f"TrivialAugmentWide(num_magnitude_bins={self.num_magnitude_bins}, "
                f"interpolation={self.interpolation!r}, fill={self.fill})")


class AutoAugment(_TorchvisionPolicy):
    """torchvision AutoAugment: per image, one of the ``policy``'s 25
    sub-policies (``"imagenet"``, ``"cifar10"`` or ``"svhn"``, or
    torchvision's ``AutoAugmentPolicy``), chosen uniformly; each of its two
    ops applies with its probability, at its magnitude bin of 10 and a random
    sign where the op has one. Two passes after the resize and crops;
    ``interpolation`` and ``fill`` as in RandAugment."""

    num_magnitude_bins = 10

    def __init__(self, policy: Any = "imagenet", interpolation: Any = "nearest",
                 fill: Any = None, same_on_batch: bool = False):
        name = str(getattr(policy, "value", policy)).lower()
        if name not in _AA_POLICIES:
            raise ValueError("AutoAugment: policy must be imagenet, cifar10 or svhn, "
                             f"got {policy!r}")
        self.policy = name
        self._setup(interpolation, fill, same_on_batch, 2)

    def _choose(self, j: int, k: int, b: Binding, rng: str) -> list[str]:
        subs = _AA_POLICIES[self.policy]
        ops = ", ".join(str(_TV_OPS.index(s[j][0])) for s in subs)
        probs = ", ".join(lit(s[j][1]) for s in subs)
        idxs = ", ".join(str(s[j][2] or 0) for s in subs)
        # the second step reads the sub-policy the first one drew
        pick = (f"irand({rng}, {len(subs)})" if j == 0
                else f"int(p[uint2({b.k(self._steps[0]) + 1}, b)])")
        return [f"int pick = {pick};",
                f"int ops[{len(subs)}] = {{ {ops} }};",
                f"float probs[{len(subs)}] = {{ {probs} }};",
                f"int idxs[{len(subs)}] = {{ {idxs} }};",
                f"int op = ops[pick]; int idx = idxs[pick]; bool on = rnd({rng}) <= probs[pick];"]

    def __repr__(self) -> str:
        return (f"AutoAugment(policy={self.policy!r}, interpolation={self.interpolation!r}, "
                f"fill={self.fill})")


# timm's RandAugment ops, in its order (index = the op's code below); with
# inc1, Posterize, Solarize, Color, Contrast, Brightness and Sharpness are
# its *Increasing variants
_TIMM_RA_OPS = ("AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize",
                "SolarizeAdd", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY",
                "TranslateXRel", "TranslateYRel")
# the fill timm's create_transform gives its ops: the ImageNet mean as uint8
_TIMM_FILL = (124 / 255, 116 / 255, 104 / 255)


class TimmRandAugment(_Policy):
    """timm's RandAugment (``timm.data.rand_augment_transform``), the one
    DeiT, ConvNeXt, Swin and most timm training recipes use, from timm's
    config string: ``"rand-m9-mstd0.5-inc1"`` (the default) picks 2 ops per
    image (``n``) from 15, uniformly with repeats, applies each with
    probability ``p`` (0.5) at magnitude ``m`` plus Gaussian noise of
    standard deviation ``mstd`` (above 100: uniform in [0, m]), clamped to
    [0, ``mmax``] (10); ``inc1`` uses the ops whose strength grows with the
    magnitude. Transform sets (``t...``) are not supported.

    The ops are Pillow's, as timm runs them: rotate, shear and translate
    resample like ``Image.transform`` (bilinear or Pillow's bicubic,
    ``interpolation="random"`` picking one per op as timm's default does)
    and fill with ``fill`` (default: the ImageNet mean that
    ``create_transform`` passes, (124, 116, 104) / 255). Values stay floats
    between ops: the uint8 rounding after each Pillow op is dropped, while
    posterize, solarize and equalize act on the value as uint8."""

    def __init__(self, config: str = "rand-m9-mstd0.5-inc1", interpolation: Any = "random",
                 fill: Any = _TIMM_FILL, translate_pct: float = 0.45,
                 same_on_batch: bool = False):
        self.config = str(config)
        self.magnitude, self.num_layers, self.prob = 10, 2, 0.5
        self.magnitude_std, self.magnitude_max, self.increasing = 0.0, 10, False
        self._parse(self.config)
        mode = str(getattr(interpolation, "value", interpolation)).lower()
        if mode not in ("random", "bilinear", "bicubic"):
            raise ValueError("TimmRandAugment: interpolation must be random, bilinear or "
                             f"bicubic (Pillow's), got {interpolation!r}")
        self.interpolation = mode
        self.fill = _fill3(fill, "TimmRandAugment")
        self.translate_pct = float(translate_pct)
        if not math.isfinite(self.translate_pct):
            raise ValueError(f"TimmRandAugment: translate_pct must be finite, got {translate_pct}")
        self.same_on_batch = bool(same_on_batch)
        self._steps = [_TimmRandAugmentOp(self) for _ in range(self.num_layers)]

    def _parse(self, config: str) -> None:
        """Current timm's rand_augment_transform parsing: sections without a
        number are skipped, the first mstd and mmax win, mmax0 means 10 and
        inc0 means off (timm releases before its August 2026 fix read inc0 as
        on)."""
        sections = config.split("-")
        if sections[0] != "rand":
            raise ValueError(f"TimmRandAugment: config must start with 'rand', got {config!r}")
        convert = {"mstd": float, "mmax": int, "inc": int, "m": int, "n": int, "p": float}
        values: dict[str, Any] = {}
        for c in sections[1:]:
            if c.startswith("t"):
                raise ValueError(f"TimmRandAugment: transform sets ({c!r}) are not supported")
            cs = re.split(r"(\d.*)", c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key not in convert:
                raise ValueError(f"TimmRandAugment: unknown config section {c!r}")
            try:
                v = convert[key](val)
            except ValueError:
                raise ValueError(f"TimmRandAugment: bad value in config section {c!r}") from None
            if key not in ("mstd", "mmax") or key not in values:
                values[key] = v
        self.magnitude = values.get("m", self.magnitude)
        self.num_layers = values.get("n", self.num_layers)
        self.prob = values.get("p", self.prob)          # above 1 always applies, as in timm
        self.increasing = bool(values.get("inc", 0))
        std = values.get("mstd", 0.0)
        self.magnitude_std = math.inf if std > 100 else std
        self.magnitude_max = values.get("mmax", 0) or 10
        if self.num_layers < 0:
            raise ValueError(f"TimmRandAugment: n must be non-negative, got {self.num_layers}")

    def __repr__(self) -> str:
        return (f"TimmRandAugment({self.config!r}, interpolation={self.interpolation!r}, "
                f"fill={self.fill}, translate_pct={self.translate_pct})")


class _TimmRandAugmentOp(_PolicyOp):
    """One of timm's RandAugment layers. Slots: the op code if applied (else
    -1), the magnitude, the sign (+-1), bicubic (1) or bilinear (0), the op's
    level (degrees, bits, threshold, the amount to add on the 0-255 scale,
    the factor or the fraction), the six warp coefficients of timm_affine in
    shaders/pil.slang."""

    parent: TimmRandAugment

    def _slot_count(self) -> int:
        return 11

    def _stat_layout(self) -> list[tuple[str, int]]:
        return [("mean_pil", 1), ("minmax", 6), ("lut", 768)]   # contrast, autocontrast, equalize

    def _sample(self, b: Binding, rng: str) -> list[str]:
        ra, k = self.parent, b.k(self)
        m = _dlit(ra.magnitude)
        if ra.magnitude_std == math.inf:
            draw = f"double(urand({rng}, 0.0, {lit(ra.magnitude)}))"
        elif ra.magnitude_std > 0:
            draw = f"double({lit(ra.magnitude)} + {lit(ra.magnitude_std)} * nrand({rng}))"
        else:
            draw = m
        upper = _dlit(ra.magnitude_max)
        bicubic = {"random": f"rnd({rng}) < 0.5 ? 1 : 0", "bicubic": "1", "bilinear": "0"}
        inc = int(ra.increasing)
        return [
            "{",
            f"    int op = irand({rng}, {len(_TIMM_RA_OPS)});",
            f"    bool on = rnd({rng}) < {lit(ra.prob)};",
            # timm clamps max(0, min(m, upper)) in double; the level uses the stored float32
            f"    float mf = float(max(0.0l, min({draw}, {upper})));",
            "    double lv = double(mf) / 10.0l;",
            f"    double sd = rnd({rng}) > 0.5 ? -1.0l : 1.0l;",
            f"    int bicubic = {bicubic[ra.interpolation]};",
            "    double a = 0.0l;",
            "    if (op == 3) a = sd * (lv * 30.0l);",
            f"    if (op == 4) a = {inc} != 0 ? 4.0l - double(int(lv * 4.0l)) : double(int(lv * 4.0l));",
            "    if (op == 5) { int t = min(256, int(lv * 256.0l));",
            f"                   a = double({inc} != 0 ? 256 - t : t); }}",
            "    if (op == 6) a = double(min(128, int(lv * 110.0l)));",
            "    if (op >= 7 && op <= 10)",
            f"        a = {inc} != 0 ? max(0.1l, 1.0l + sd * (lv * 0.9l)) : lv * 1.8l + 0.1l;",
            "    if (op == 11 || op == 12) a = sd * (lv * 0.3l);",
            f"    if (op == 13 || op == 14) a = sd * (lv * {_dlit(ra.translate_pct)});",
            "    double m[6]; timm_affine(op, a, CW, CH, m);",
            f"    p[uint2({k}, b)] = on ? float(op) : -1.0; p[uint2({k + 1}, b)] = mf;",
            f"    p[uint2({k + 2}, b)] = float(sd); p[uint2({k + 3}, b)] = float(bicubic);",
            f"    p[uint2({k + 4}, b)] = float(a);",
            f"    for (int i = 0; i < 6; ++i) p[uint2({k + 5} + i, b)] = float(m[i]);",
            "}"]

    def _input(self, b: Binding) -> list[str]:
        ra, k = self.parent, b.k(self)
        f0, f1, f2 = (lit(v) for v in ra.fill)
        return [f"int op = int({P(k)});",
                "MidImage img = { src, b, OW, OH, 1 };",
                # ImageEnhance.Sharpness blends toward the 3x3 smoothing by the factor
                f"if (op == 10) return sharpen(img, x, y, 1.0 - {P(k + 4)});",
                "if (op != 3 && op < 11) return load_mid(src, x, y, b);",
                "float fx = float(x); float fy = float(y);",
                f"float u = {P(k + 5)} * fx + {P(k + 6)} * fy + {P(k + 7)};",
                f"float v = {P(k + 8)} * fx + {P(k + 9)} * fy + {P(k + 10)};",
                f"return pil_sample(img, u, v, int({P(k + 3)}), float4({f0}, {f1}, {f2}, 0.0));"]

    def _post(self, b: Binding) -> list[str]:
        k = b.k(self)
        mean, minmax, lut = b.stats(self)
        a = P(k + 4)
        mm = ", ".join(ST(minmax + i) for i in range(6))
        return [f"{{ int op = int({P(k)});",
                f"  if (op == 0) {{ {stop(minmax)} c = autocontrast(c, {mm}); }}",
                f"  if (op == 1) {{ {stop(lut)} c = equalize(c, st, b, {lut}); }}",
                "  if (op == 2) c = invert(c);",
                f"  if (op == 4) c = posterize_pil(c, int({a}));",
                f"  if (op == 5) c = solarize_pil(c, int({a}));",
                f"  if (op == 6) c = solarize_add(c, {a} * (1.0 / 255.0));",
                f"  if (op == 7) c = color_pil(c, {a});",
                f"  if (op == 8) {{ {stop(mean)} c = contrast(c, {a}, {ST(mean)}); }}",
                f"  if (op == 9) c = brightness(c, {a});",
                "}"]


# --------------------------------------------------------------- others
class RandomApply(Transform):
    """torchvision RandomApply: apply ``transforms`` to an image with
    probability ``p``. Holds pointwise ops, or exactly one op that reads
    neighbouring pixels (a blur) or warps (RandomRotation, RandomAffine,
    RandomPerspective)."""

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
        if kinds - {"pointwise", "neighbourhood", "warp"}:
            raise ValueError("RandomApply holds colour ops, blurs and warps, not resizes, "
                             "crops, flips (they have their own p), policies or Normalize")
        if kinds - {"pointwise"} and len(self.transforms) != 1:
            raise ValueError("RandomApply holds either pointwise ops or exactly one op that "
                             "reads neighbouring pixels (such as GaussianBlur or "
                             "RandomRotation)")
        self._kind = next(iter(kinds - {"pointwise"}), "pointwise")

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
    that reads neighbouring pixels (a blur or a warp), e.g. the choice in DeiT
    III's ThreeAugment:
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
        if set(kinds) - {"pointwise", "neighbourhood", "warp"}:
            raise ValueError("RandomChoice holds colour ops, blurs and warps, not resizes, "
                             "crops, flips, policies or Normalize")
        if len(kinds) - kinds.count("pointwise") > 1:
            raise ValueError("RandomChoice holds at most one op that reads neighbouring pixels "
                             "(a blur or a warp)")
        self._kind = next((k for k in kinds if k != "pointwise"), "pointwise")

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
        # with a blur or warp among the choices: the colour-op choices (and those of a
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
    """torchvision Normalize; comes after every transform except RandomErasing.
    One mean or std value applies to every channel."""

    _kind = "normalize"

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)
        if len(self.mean) not in (1, 3) or len(self.std) not in (1, 3):
            raise ValueError("Normalize: mean and std need one value or three (RGB)")
        self._channels = max(len(self.mean), len(self.std))     # as given
        self.mean, self.std = self.mean * (3 // len(self.mean)), self.std * (3 // len(self.std))
        if any(s == 0 for s in self.std):
            raise ValueError("Normalize: std must be non-zero")

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


# ----------------------------------------------------------------- noise
class GaussianNoise(Transform):
    """torchvision GaussianNoise: adds ``mean + sigma * N(0, 1)`` to every
    value, new noise for every pixel, channel and image, then clamps to [0, 1]
    with ``clip=True``. A colour op (before Normalize); the noise comes from
    SlangVision's own generator, so it differs from torch's.
    ``same_on_batch=True`` adds the same noise field to every image."""

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True,
                 same_on_batch: bool = False):
        self.mean, self.sigma, self.clip = float(mean), float(sigma), bool(clip)
        self.same_on_batch = bool(same_on_batch)
        if not math.isfinite(self.mean):
            raise ValueError(f"GaussianNoise: mean must be finite, got {mean}")
        if not (math.isfinite(self.sigma) and self.sigma >= 0):
            raise ValueError(f"GaussianNoise: sigma must be non-negative, got {sigma}")

    def _slot_count(self) -> int:
        return 1            # the image's noise key

    def _sample(self, b: Binding, rng: str) -> list[str]:
        # advance the stream: the next draw must not reuse the key's bits
        return [f"{rng} = pcg({rng}); p[uint2({b.k(self)}, b)] = float({rng} >> 8);   "
                "// 24 bits: exact in a float"]

    def _apply(self, b: Binding) -> list[str]:
        image = "0" if self.same_on_batch else "b"
        noise = f"normal4(uint({P(b.k(self))}), {image}, x, y)"
        c = f"c + ({lit(self.mean)} + {lit(self.sigma)} * {noise})"
        return [f"c = clamp01({c});" if self.clip else f"c = {c};"]

    def __repr__(self) -> str:
        return (f"GaussianNoise(mean={self.mean}, sigma={self.sigma}, clip={self.clip}, "
                f"same_on_batch={self.same_on_batch})")


# --------------------------------------------------------------- erasing
class RandomErasing(Transform):
    """torchvision RandomErasing: with probability ``p``, fill one box per image
    with ``value``: a number, one value per channel, or ``"random"`` for
    per-pixel standard normal noise. The box takes ``scale`` of the image's
    area at a log-uniform aspect within ``ratio``; after 10 tries that do not
    fit, the image is left alone. Runs at the end of the pipeline, after
    Normalize, so ``value`` is in output units, as in torchvision recipes.
    timm's ``RandomErasing(probability=p, mode="pixel")`` is
    ``RandomErasing(p, scale=(0.02, 1 / 3), ratio=(0.3, 1 / 0.3), value="random")``.
    Use several to erase several boxes."""

    _kind = "erase"

    def __init__(self, p: float = 0.5, scale: Sequence[float] = (0.02, 0.33),
                 ratio: Sequence[float] = (0.3, 3.3), value: Any = 0.0,
                 same_on_batch: bool = False):
        self.p = _probability(p)
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        if not 0 <= self.scale[0] <= self.scale[1] <= 1:
            raise ValueError(f"RandomErasing: scale must satisfy 0 <= min <= max <= 1, got {scale}")
        if not 0 < self.ratio[0] <= self.ratio[1]:
            raise ValueError(f"RandomErasing: ratio must satisfy 0 < min <= max, got {ratio}")
        if isinstance(value, str):
            if value != "random":
                raise ValueError(f"RandomErasing: a string value must be 'random', got {value!r}")
            self.value: tuple[float, float, float] | None = None
        else:
            vals = [value] if isinstance(value, numbers.Real) else list(value)
            self._channels = len(vals)      # as given
            if len(vals) == 1:
                vals = vals * 3
            if len(vals) != 3 or not all(isinstance(v, numbers.Real) and math.isfinite(v)
                                         for v in vals):
                raise ValueError("RandomErasing: value is a number, three numbers (RGB) or "
                                 f"'random', got {value!r}")
            self.value = (float(vals[0]), float(vals[1]), float(vals[2]))
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 6            # applied, top, left, height, width, noise key

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        return [
            "{",
            f"    bool on = rnd({rng}) < {lit(self.p)};",
            "    int et = 0; int el = 0; int eh = 0; int ew = 0;",
            f"    bool found = sample_box({rng}, CW, CH, {lit(self.scale[0])}, {lit(self.scale[1])}, "
            f"{lit(self.ratio[0])}, {lit(self.ratio[1])}, et, el, eh, ew);",
            f"    p[uint2({k}, b)] = on && found ? 1.0 : 0.0;",
            f"    p[uint2({k + 1}, b)] = float(et); p[uint2({k + 2}, b)] = float(el);",
            f"    p[uint2({k + 3}, b)] = float(eh); p[uint2({k + 4}, b)] = float(ew);",
            f"    {rng} = pcg({rng}); p[uint2({k + 5}, b)] = float({rng} >> 8);   "
            "// 24 bits: exact in a float",
            "}"]

    def _erase(self, b: Binding) -> list[str]:
        """Lines for the final pass (x, y, b and the normalized pixel c in scope)."""
        k = b.k(self)
        inside = (f"{P(k)} > 0.5 && y >= int({P(k + 1)}) && x >= int({P(k + 2)}) && "
                  f"y < int({P(k + 1)}) + int({P(k + 3)}) && x < int({P(k + 2)}) + int({P(k + 4)})")
        if self.value is None:
            fill = f"normal4(uint({P(k + 5)}), b, x, y)"
        else:
            fill = f"float4({lit(self.value[0])}, {lit(self.value[1])}, {lit(self.value[2])}, 0.0)"
        return [f"if ({inside}) c = {fill};"]

    def __repr__(self) -> str:
        value = "'random'" if self.value is None else self.value
        return f"RandomErasing(p={self.p}, scale={self.scale}, ratio={self.ratio}, value={value})"
