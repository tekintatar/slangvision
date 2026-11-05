# slangvision/_base.py — building blocks shared by every transform: random
# distributions, the parameter-slot binding, and the two public op base
# classes (ColorOp, NeighbourhoodOp).
#
# How a transform takes part in a pipeline. Compose gives every transform a
# few slots in a per-image parameter table (one row of floats per image) and,
# for ops that need a per-image statistic, statistic slots. A transform then
# contributes Slang text: lines that draw its random values into its slots
# (run by the sampler kernel, one thread per image) and lines that apply it
# to the pixel `c`. slangvision.testing checks the result against torch
# references, which live there, not on the transforms.

from __future__ import annotations

import hashlib
import math
import numbers
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


def lit(x: float) -> str:
    """A Slang float literal."""
    return repr(float(x))


def _probability(p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return p


# ---------------------------------------------------------- distributions
class Uniform:
    """A float drawn uniformly from ``[low, high]``, separately per image."""

    def __init__(self, low: float, high: float):
        self.low, self.high = float(low), float(high)
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.high >= self.low):
            raise ValueError(f"Uniform needs finite low <= high, got ({self.low}, {self.high})")

    def _draw(self, rng: str) -> str:
        return f"urand({rng}, {lit(self.low)}, {lit(self.high)})"

    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


class Choice:
    """One of ``values``, picked uniformly, separately per image."""

    def __init__(self, values: Any):
        self.values = [float(v) for v in values]
        if not self.values:
            raise ValueError("Choice needs at least one value")
        if not all(math.isfinite(v) for v in self.values):
            raise ValueError(f"Choice values must be finite, got {self.values}")

    def _pick(self) -> str:
        """Expression selecting the value for an index `i` already drawn."""
        expr = lit(self.values[-1])
        for i in range(len(self.values) - 2, -1, -1):
            expr = f"(i == {i} ? {lit(self.values[i])} : {expr})"
        return expr

    def __repr__(self) -> str:
        return f"Choice({self.values})"


def _check_params(owner: str, params: Any) -> Mapping[str, Any]:
    if not isinstance(params, Mapping):
        raise TypeError(f"{owner}.params must be a dict of name -> Uniform/Choice/number")
    for name, value in params.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{owner}.params: {name!r} is not a valid name")
        if isinstance(value, bool) or not isinstance(value, (Uniform, Choice, numbers.Real)):
            raise TypeError(f"{owner}.params[{name!r}] must be Uniform, Choice or a number, "
                            f"got {value!r}")
        if isinstance(value, numbers.Real) and not math.isfinite(value):
            raise ValueError(f"{owner}.params[{name!r}] must be finite, got {value!r}")
    return params


def _draw_lines(dist: Any, rng: str, slot: int) -> list[str]:
    """Sampler lines that store one draw of ``dist`` in ``slot``."""
    target = f"p[uint2({slot}, b)]"
    if isinstance(dist, Uniform):
        return [f"{target} = {dist._draw(rng)};"]
    if len(dist.values) == 1:
        return [f"{target} = {lit(dist.values[0])};"]
    return [f"{{ int i = irand({rng}, {len(dist.values)}); {target} = {dist._pick()}; }}"]


# ------------------------------------------------------------ slot binding
def P(k: int) -> str:
    """Slang expression reading this image's parameter slot k."""
    return f"prm(p, b, {k})"


def ST(k: int) -> str:
    """Slang expression reading this image's statistic slot k."""
    return f"prm(st, b, {k})"


def stop(k: int) -> str:
    """Chain line where the statistic pre-pass of slot group k reads the image:
    the op that needs the statistic puts it just before itself."""
    return f"if (upto == {k}) {{ hit = true; return c; }}"


class Binding:
    """Where each transform's slots are in one pipeline's parameter table.
    Kept outside the transforms so one transform object can be used in
    several pipelines."""

    def __init__(self) -> None:
        self.nparams = 0
        self.nstats = 0
        self.stage = 0
        self.geo = 0            # Compose reserves the first slots for geometry
        self.win: dict[int, int] = {}   # window op -> its first geometry slot
        # one per op that needs per-image statistics: (kind, first slot, pass)
        self.stat_groups: list[tuple[str, int, int]] = []
        self._k: dict[int, int] = {}
        self._st: dict[int, list[int]] = {}

    def bind(self, t: "Transform") -> None:
        if id(t) in self._k:
            raise ValueError(f"{t!r} appears twice in the pipeline; create a separate "
                             "transform for each use")
        self._k[id(t)] = self.nparams
        self.nparams += t._slot_count()
        firsts = []
        for kind, n in t._stat_layout():
            self.stat_groups.append((kind, self.nstats, self.stage))
            firsts.append(self.nstats)
            self.nstats += n
        self._st[id(t)] = firsts
        for child in t._children():
            self.bind(child)

    def k(self, t: "Transform") -> int:
        return self._k[id(t)]

    def stats(self, t: "Transform") -> list[int]:
        """The first slot of each of t's statistic groups."""
        return self._st[id(t)]


# -------------------------------------------------------------- transforms
class Transform:
    """Base of every augmentation transform."""

    #: geometry | flip | pointwise | neighbourhood | warp | normalize | erase
    _kind = "pointwise"
    #: what the statistic pre-pass computes: "mean" (gray mean, 1 slot),
    #: "mean_pil" (Pillow's uint8 luma mean, rounded to a level: 1 slot), "minmax"
    #: (per-channel min, then max - min: 6 slots) or "lut" (per-channel
    #: equalization tables from 256-bin histograms: 768 slots)
    _stat_kind = "mean"
    same_on_batch = False

    def _slot_count(self) -> int:
        return 0

    def _stat_count(self) -> int:
        return 0

    def _stat_layout(self) -> list[tuple[str, int]]:
        """(kind, slot count) of each statistic group this op needs."""
        n = self._stat_count()
        return [(self._stat_kind, n)] if n else []

    def _children(self) -> list["Transform"]:
        return []

    def _functions(self) -> str:
        """Slang functions this transform needs, placed before the generated code."""
        return ""

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return []

    def _apply(self, b: Binding) -> list[str]:
        """For a pointwise op: lines that update `float4 c` (with p, st, b and
        the output pixel x, y in scope)."""
        return []

    def _post(self, b: Binding) -> list[str]:
        """For an op that reads neighbouring pixels: lines to run at the start
        of the next pass's colour chain (a container's colour-op parts)."""
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _UserOp(Transform):
    """Shared parts of ColorOp and NeighbourhoodOp."""

    # read-only default, so ops that forget to define their own cannot share
    # one mutable dict; p and same_on_batch default even without super().__init__
    params: Mapping[str, Any] = MappingProxyType({})
    slang: str = ""
    p = 1.0
    same_on_batch = False
    _signature = ""

    def __init__(self, p: float = 1.0, same_on_batch: bool = False):
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)

    def _validate(self) -> None:
        name = type(self).__name__
        _check_params(name, self.params)
        if not isinstance(self.slang, str) or not re.search(r"\bapply\s*[<(]", self.slang):
            raise ValueError(f"{name}.slang must define `{self._signature}`")

    def _namespace(self) -> str:
        digest = hashlib.sha1(self.slang.encode()).hexdigest()[:10]
        return f"op_{type(self).__name__}_{digest}"

    def _random(self) -> list[str]:
        return [n for n, v in self.params.items() if isinstance(v, (Uniform, Choice))]

    def _slot_count(self) -> int:
        return (self.p < 1.0) + len(self._random())

    def _functions(self) -> str:
        return f"namespace {self._namespace()} {{\n{self.slang}\n}}"

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        lines = []
        if self.p < 1.0:
            lines.append(f"p[uint2({k}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;")
            k += 1
        for i, name in enumerate(self._random()):
            lines += _draw_lines(self.params[name], rng, k + i)
        return lines

    def _flag(self, b: Binding) -> str | None:
        return f"{P(b.k(self))} > 0.5" if self.p < 1.0 else None

    def _args(self, b: Binding) -> list[str]:
        k = b.k(self) + (self.p < 1.0)
        args, i = [], 0
        for value in self.params.values():
            if isinstance(value, (Uniform, Choice)):
                args.append(P(k + i))
                i += 1
            else:
                args.append(lit(value))
        return args

    def _values(self, b: Binding, row: list[float]) -> dict[str, float]:
        k = b.k(self) + (self.p < 1.0)
        out, i = {}, 0
        for name, value in self.params.items():
            if isinstance(value, (Uniform, Choice)):
                out[name] = row[k + i]
                i += 1
            else:
                out[name] = float(value)
        return out

    def _applied(self, b: Binding, row: list[float]) -> bool:
        return self.p >= 1.0 or row[b.k(self)] > 0.5

    def reference(self, img: "torch.Tensor", **params: float) -> "torch.Tensor":
        """The op in torch on one (3, H, W) float image in [0, 1]; optional,
        used only by :func:`slangvision.testing.max_reference_error`."""
        raise NotImplementedError(f"{type(self).__name__} defines no reference")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, params={self.params})"


class ColorOp(_UserOp):
    """A pointwise op you define: random parameters plus one Slang function.

    Set ``slang`` to Slang source that defines::

        float4 apply(float4 c, float a, float b, ...)

    It receives one pixel (rgb in ``xyz``, values in [0, 1], ``w`` unused)
    and one float per entry of ``params``, in order, and returns the new
    pixel. ``params`` maps names to :class:`Uniform`, :class:`Choice` or a
    plain number (a constant). The library helpers in
    ``slangvision/shaders/`` (``gray``, ``clamp01``, ...) are in
    scope; each op's code sits in its own namespace, so helper names never
    clash. Torchvision-style ops clamp their result to [0, 1] themselves.

    ``p`` is the probability of applying the op to an image;
    ``same_on_batch=True`` makes every image share the op's random values.
    Define :meth:`reference` (torch ops on one image) to check the op with
    :func:`slangvision.testing.max_reference_error`.
    """

    _signature = "float4 apply(float4 c, ...)"

    def _apply(self, b: Binding) -> list[str]:
        call = f"c = {self._namespace()}::apply({', '.join(['c'] + self._args(b))});"
        flag = self._flag(b)
        return [f"if ({flag}) {call}" if flag else call]


class NeighbourhoodOp(_UserOp):
    """An op that reads neighbouring pixels, which you define.

    Set ``slang`` to Slang source that defines::

        float4 apply<I : IImage>(I img, int x, int y, float a, ...)

    ``img.load(x, y)`` returns the pixel at (x, y) of the image as it is at
    this point of the pipeline, with ``border`` applied outside it
    (``"reflect"`` like torch's reflect padding, or ``"clamp"``);
    ``img.width()`` and ``img.height()`` give its size. ``params``, ``p``,
    ``same_on_batch`` and :meth:`reference` work as in :class:`ColorOp`.
    Each neighbourhood op starts a new pass over the image.
    """

    _kind = "neighbourhood"
    border = "reflect"
    _signature = "float4 apply<I : IImage>(I img, int x, int y, ...)"

    def _validate(self) -> None:
        super()._validate()
        if self.border not in ("reflect", "clamp"):
            raise ValueError(f"{type(self).__name__}.border must be 'reflect' or 'clamp'")

    def _input(self, b: Binding) -> list[str]:
        """Body of the next pass's input function (src, p, b, x, y, OW, OH in scope)."""
        border = 0 if self.border == "reflect" else 1
        lines = [f"MidImage img = {{ src, b, OW, OH, {border} }};"]
        flag = self._flag(b)
        if flag:
            lines.append(f"if (!({flag})) return img.load(x, y);")
        call = ", ".join(["img", "x", "y"] + self._args(b))
        return lines + [f"return {self._namespace()}::apply({call});"]

    def _pre(self, b: Binding) -> list[str] | None:
        """Body of an optional pre-pass (separable ops); None for single-pass ops."""
        return None
