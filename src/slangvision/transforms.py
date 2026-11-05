# slangvision/transforms.py — the built-in transforms. Names, parameters and
# results follow torchvision.transforms.v2 on float images; random values are
# drawn per image unless same_on_batch=True. RandomGrayscale and
# RandomSolarize are written with the public ColorOp API.

from __future__ import annotations

import math
import numbers
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
        f0, f1, f2 = (lit(v) for v in self.fill)
        lines = ["MidImage img = { src, b, OW, OH, 1 };"]
        if self.p < 1.0:
            lines.append(f"if (!({P(k)} > 0.5)) return load_mid(src, x, y, b);")
        return lines + [
            "float fx = float(x); float fy = float(y);",
            f"float den = {P(k + 7)} * fx + {P(k + 8)} * fy + 1.0;",
            f"float u = ({P(k + 1)} * fx + {P(k + 2)} * fy + {P(k + 3)}) / den;",
            f"float v = ({P(k + 4)} * fx + {P(k + 5)} * fy + {P(k + 6)}) / den;",
            f"return warp_sample(img, u, v, {int(self.interpolation == 'nearest')}, "
            f"float4({f0}, {f1}, {f2}, 0.0));"]


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


class RandomInvert(ColorOp):
    """torchvision RandomInvert: 1 - value."""

    slang = "float4 apply(float4 c) { return float4(1.0 - c.x, 1.0 - c.y, 1.0 - c.z, 0.0); }"

    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)


class RandomPosterize(ColorOp):
    """torchvision RandomPosterize on float images: ``2**bits`` levels,
    ``floor(value * levels) / levels``. (torchvision's uint8 path keeps the top
    ``bits`` bits instead, which gives slightly different values, e.g. 240/255
    rather than 15/16 for white at 4 bits.)"""

    slang = """
    float4 apply(float4 c, float n)
    {
        float top = n - 1.0;
        float k = 1.0 / n;
        return float4(clamp(floor(c.x * n), 0.0, top) * k, clamp(floor(c.y * n), 0.0, top) * k,
                      clamp(floor(c.z * n), 0.0, top) * k, 0.0);
    }"""

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
        return [
            "float4 c = load_mid(src, x, y, b);",
            f"if (!({P(b.k(self))} > 0.5)) return c;",
            # torchvision clamps the whole output, the unblended border included
            "if (x == 0 || y == 0 || x == OW - 1 || y == OH - 1) return clamp01(c);",
            "float4 ring = float4(0.0);",
            "for (int dy = -1; dy <= 1; ++dy)",
            "    for (int dx = -1; dx <= 1; ++dx)",
            "        if (dx != 0 || dy != 0) ring += load_mid(src, x + dx, y + dy, b);",
            "float4 blur = ring * (1.0 / 13.0) + c * (5.0 / 13.0);",
            f"return clamp01(c + {lit(1.0 - self.sharpness_factor)} * (blur - c));"]

    def __repr__(self) -> str:
        return f"RandomAdjustSharpness(sharpness_factor={self.sharpness_factor}, p={self.p})"


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
                             "crops, flips (they have their own p) or Normalize")
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
                             "crops, flips or Normalize")
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
    """torchvision Normalize; comes after every transform except RandomErasing."""

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
            f"    p[uint2({k + 5}, b)] = float(pcg({rng}) >> 8);     // 24 bits: exact in a float",
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
