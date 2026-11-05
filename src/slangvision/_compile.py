# slangvision/_compile.py — Compose: checks a pipeline's structure, generates
# its Slang and runs it.
#
# Generated kernels, in launch order:
#   sample_params      one thread per image draws every random value
#   stats_<k>          per-image statistic k (contrast's mean): the same
#                      pointwise chain, stopped at the op that needs it
#   pre_<s>            optional pre-pass of a separable neighbourhood op
#   stage_<s>          pass s: its input (pass 0 resamples the source image;
#                      later passes read the previous pass through their
#                      neighbourhood op or warp), then its pointwise chain;
#                      the last pass normalizes and writes the output
# Containers become per-image branches (RandomApply flags, ColorJitter's
# order).

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ._base import P, Binding, NeighbourhoodOp, Transform, _UserOp, lit
from .transforms import (CenterCrop, ColorJitter, GaussianBlur, Normalize, Pad, RandAugment,
                         RandomAdjustSharpness, RandomAffine, RandomApply, RandomAutocontrast,
                         RandomChoice, RandomCrop, RandomEqualize, RandomErasing,
                         RandomHorizontalFlip, RandomPerspective, RandomResizedCrop,
                         RandomRotation, RandomVerticalFlip, Resize, _RandAugmentOp, _Window)

if TYPE_CHECKING:
    import torch

# Parameter slots for geometry: the resample's source box and output grid
# (gl, gt, gw, gh, RW, RH), then four per window op (Pad, CenterCrop,
# RandomCrop): its offset into the previous grid and that grid's size.
_RESAMPLE_SLOTS = 6
_WINDOW_SLOTS = 4
_CHUNK = 8              # columns per thread in statistic passes

# Intermediates between passes are (B, 3, H, W) tensors of type Mid (float16,
# or float32 for float32 output); the output is (B, 3, H, W) of type Out. For
# bfloat16 output, Out is int16_t: SlangPy's torch bridge does not take
# bfloat16 tensors yet, so the kernel writes rounded bfloat16 bits into an
# int16 view of the output tensor.
_DTYPES = {             # dtype -> (Mid, Out, how store_out converts a float)
    "float16": ("half", "half", "Out({})"),
    "float32": ("float", "float", "{}"),
    "bfloat16": ("half", "int16_t", "bf16_bits({})"),
}

_MID_HELPERS = """
float4 load_mid(Tensor<Mid, 4> t, int x, int y, int b)
{
    return float4(float(t[uint4(x, y, 0, b)]), float(t[uint4(x, y, 1, b)]),
                  float(t[uint4(x, y, 2, b)]), 0.0);
}
void store_mid(RWTensor<Mid, 4> t, int x, int y, int b, float4 c)
{
    t[uint4(x, y, 0, b)] = Mid(c.x); t[uint4(x, y, 1, b)] = Mid(c.y); t[uint4(x, y, 2, b)] = Mid(c.z);
}
// the previous pass's output, as neighbourhood ops see it
struct MidImage : IImage
{
    Tensor<Mid, 4> t;
    int b;
    int w;
    int h;
    int border;     // 0 reflect, 1 clamp
    float4 load(int x, int y)
    {
        if (border == 0) { x = reflect(x, w); y = reflect(y, h); }
        else             { x = clamp(x, 0, w - 1); y = clamp(y, 0, h - 1); }
        return load_mid(t, x, y, b);
    }
    int width() { return w; }
    int height() { return h; }
}
"""


def _stats_kernel(kind: str, k: int, s: int, args: str, inp: str) -> list[str]:
    """The statistic pre-pass of group k in pass s: one thread per _CHUNK
    pixels of a row, reading the image just before the op that needs it.
    Images that do not apply the op stop at their first pixel (`hit` is the
    same for the whole image); their statistic slots are never read."""
    def loop(skip: str) -> list[str]:
        return [f"    for (int j = 0; j < {_CHUNK}; ++j) {{",
                f"        int x = pix.x * {_CHUNK} + j;",
                "        if (x >= OW) break;",
                f"        float4 c = chain_{s}({inp}, p, st, b, {k}, hit);",
                f"        if (!hit) return{skip};"]
    head = f"(int3 pix, {args}, Tensor<float, 2> p, Tensor<float, 2> st"
    if kind == "mean":         # gray sum; Compose divides by the pixel count
        return [f"float stats_{k}{head}, int OW, int OH)", "{",
                "    int b = pix.z; int y = pix.y; bool hit; float sum = 0.0;",
                *loop(" 0.0"), "        sum += gray(c);", "    }", "    return sum;", "}", ""]
    if kind == "minmax":       # per-channel min and max of these pixels
        return [f"void stats_{k}{head}, RWTensor<float, 4> part, int OW, int OH)", "{",
                "    int b = pix.z; int y = pix.y; bool hit;",
                "    float4 lo = float4(3.0e38); float4 hi = float4(-3.0e38);",
                *loop(""), "        lo = min(lo, c); hi = max(hi, c);", "    }",
                "    part[uint4(0, pix.x, y, b)] = lo.x; part[uint4(1, pix.x, y, b)] = lo.y;",
                "    part[uint4(2, pix.x, y, b)] = lo.z; part[uint4(3, pix.x, y, b)] = hi.x;",
                "    part[uint4(4, pix.x, y, b)] = hi.y; part[uint4(5, pix.x, y, b)] = hi.z;",
                "}", ""]
    # "lut": 256-bin histograms of the quantized channels, for images that apply the op
    return [f"void stats_{k}{head}, uint64_t hist, int OW, int OH)", "{",
            "    int b = pix.z; int y = pix.y; bool hit;",
            "    uint* h = (uint*)(hist) + b * 768;",
            "    int qx = -1; int qy = -1; int qz = -1; uint nx = 0u; uint ny = 0u; uint nz = 0u;",
            *loop(""),
            "        hist_add(h, q255(c.x), qx, nx); hist_add(h + 256, q255(c.y), qy, ny);",
            "        hist_add(h + 512, q255(c.z), qz, nz);",
            "    }",
            "    hist_flush(h, qx, nx); hist_flush(h + 256, qy, ny); hist_flush(h + 512, qz, nz);",
            "}", ""]


def _autocontrast_stats(part: "torch.Tensor", st: "torch.Tensor", k: int) -> None:
    """torchvision's autocontrast_image: per-channel min and (max - min), with
    (0, 1) for a constant channel."""
    lo = part[..., :3].amin((1, 2))
    hi = part[..., 3:].amax((1, 2))
    flat = hi == lo
    st[:, k:k + 3] = lo.masked_fill(flat, 0.0)
    st[:, k + 3:k + 6] = (hi - lo).masked_fill(flat, 1.0)


def _equalize_stats(hist: "torch.Tensor", pixels: int, st: "torch.Tensor", k: int) -> None:
    """torchvision's equalize_image lookup tables (PIL's scheme), scaled to
    0-1: an image whose step is 0 keeps its quantized values."""
    import torch

    hist = hist.long()
    cum = hist.cumsum(-1)
    last = cum.argmax(-1, keepdim=True)          # the last non-empty bin
    step = (pixels - hist.gather(-1, last)) // 255
    lut = ((cum[..., :-1] + step // 2) // step.clamp(min=1)).clamp(0, 255)
    lut = torch.cat([torch.zeros_like(lut[..., :1]), lut], -1)
    lut = torch.where(step != 0, lut, torch.arange(256, device=lut.device))
    st[:, k:k + 768] = lut.float().mul_(1.0 / 255.0).reshape(-1, 768)


def _dtype_name(dtype: Any) -> str:
    name = "float16" if dtype is None else str(dtype).replace("torch.", "")
    if name not in _DTYPES:
        raise ValueError("dtype must be torch.float16, torch.bfloat16 or torch.float32, "
                         f"got {dtype}")
    return name


_BUILT_IN = (RandomResizedCrop, Resize, CenterCrop, Pad, RandomCrop, RandomHorizontalFlip,
             RandomVerticalFlip, RandomRotation, RandomAffine, RandomPerspective, ColorJitter,
             GaussianBlur, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize,
             RandomApply, RandomChoice, Normalize, RandomErasing, RandAugment)


def _expand(transforms: Sequence[Transform]) -> list[Transform]:
    """The pipeline as Compose plans it: a policy becomes its op passes."""
    return [s for t in transforms for s in (t._children() if isinstance(t, RandAugment) else [t])]


def _walk(transforms: Sequence[Transform]):
    for t in _expand(transforms):
        if not isinstance(t, (_BUILT_IN, _UserOp, _RandAugmentOp)):
            raise TypeError(f"Compose takes slangvision transforms; define your own "
                            f"with ColorOp or NeighbourhoodOp, got {t!r}")
        yield t
        yield from _walk(t._children())


def _unwrap(t: Transform) -> tuple[list[tuple[Transform, Transform]], Transform]:
    """A neighbourhood op or warp and the (RandomApply | RandomChoice, child)
    wrappers that gate it."""
    gates: list[tuple[Transform, Transform]] = []
    while isinstance(t, (RandomApply, RandomChoice)):
        child = next(c for c in t.transforms if c._kind != "pointwise")
        gates.append((t, child))
        t = child
    return gates, t


def _resize_shorter(hw: tuple[int, int], size: int) -> tuple[int, int]:
    """torchvision Resize(int): shorter side to size (as resize_shorter in geometry.slang)."""
    h, w = hw
    return (size * h // w, size) if w <= h else (size, size * w // h)


class Compose:
    """A GPU augmentation pipeline: torchvision-style transforms compiled into
    a few Slang kernels. Random values are drawn per image.

    ``Compose(transforms, dtype=torch.float16)``; ``torch.bfloat16`` and
    ``torch.float32`` output are also available. Geometric transforms come
    first: an optional RandomResizedCrop or Resize, then any sequence of Pad,
    CenterCrop and RandomCrop; they fix the output size. Flips, warps
    (RandomRotation, RandomAffine, RandomPerspective) and colour ops follow in
    any order; Normalize, if present, comes next, and RandomErasing last.

    Calling the pipeline returns a contiguous ``(B, 3, H, W)`` tensor of
    ``dtype`` on the current CUDA stream. ``images`` is a ``(B, 3, H, W)``
    uint8 CUDA tensor, or a list of ``(3, H, W)`` uint8 CUDA tensors of any
    sizes; pass ``layout="hwc"`` for ``(H, W, 3)`` images (DALI, nvImageCodec).
    Images are read in place on the current stream: they must be written
    before it and stay valid until the GPU has run the pipeline. Tensors
    produced on the same stream always are. Images decoded on another stream
    need ``torch.cuda.current_stream().wait_stream(that_stream)`` before the
    call (and ``image.record_stream(torch.cuda.current_stream())`` if that
    stream frees them); memory owned by another library (a DALI output,
    valid until its next ``run()``) needs
    ``torch.cuda.current_stream().synchronize()`` before it is reused.
    ``torch.manual_seed`` makes the random values reproducible.
    """

    def __init__(self, transforms: Sequence[Transform], dtype: Any = None):
        self.transforms = list(transforms)
        self._dtype = _dtype_name(dtype)
        for t in _walk(self.transforms):
            if isinstance(t, _UserOp):
                t._validate()
        self._plan()
        self._sources: dict[str, str] = {}

    # ------------------------------------------------------------ structure
    def _plan(self) -> None:
        geometry: list[Transform] = []
        self._flips: list[Transform] = []
        self._flip_stage: dict[int, int] = {}   # flip -> the pass whose input applies it
        warp_stage = 0
        self._stages: list[list[Transform]] = [[]]
        self._neighbourhood: list[Transform] = []
        self._normalize: Normalize | None = None
        self._erasing: list[RandomErasing] = []
        user_neighbourhood = False
        for t in _expand(self.transforms):
            kind = t._kind
            if self._erasing and kind != "erase":
                raise ValueError("RandomErasing must come last (after Normalize)")
            if self._normalize is not None and kind != "erase":
                raise ValueError("Normalize must come after every transform except "
                                 "RandomErasing")
            if kind == "geometry":
                if self._flips or self._neighbourhood or any(self._stages):
                    raise ValueError(f"{t!r}: resizes, crops and pads must come first (warps "
                                     "work on the image after them)")
                if getattr(t, "_resamples", False) and geometry:
                    raise ValueError(f"{t!r}: one Resize or RandomResizedCrop per pipeline, "
                                     "and it must be the first transform")
                geometry.append(t)
            elif kind == "flip":
                if user_neighbourhood:
                    raise ValueError(f"{t!r} must come before your NeighbourhoodOps")
                # a flip after a warp mirrors the warp's output coordinates;
                # it commutes with everything else
                self._flips.append(t)
                self._flip_stage[id(t)] = warp_stage
            elif kind == "pointwise":
                self._stages[-1].append(t)
            elif kind == "neighbourhood":
                self._neighbourhood.append(t)
                # its _post lines (a container's colour ops) start the next pass's chain
                self._stages.append([t])
                user_neighbourhood |= isinstance(_unwrap(t)[1], NeighbourhoodOp)
            elif kind == "warp":
                self._neighbourhood.append(t)
                self._stages.append([t])
                warp_stage = len(self._stages) - 1
                user_neighbourhood = False
            elif kind == "normalize":
                self._normalize = t     # type: ignore[assignment]
            elif kind == "erase":
                self._erasing.append(t)     # type: ignore[arg-type]
            else:
                raise TypeError(f"{t!r}: unknown transform kind {kind!r}")
        self._resample = geometry[0] if geometry and getattr(geometry[0], "_resamples",
                                                             False) else None
        self._windows: list[_Window] = [g for g in geometry if isinstance(g, _Window)]
        self._size = self._chain_size(None)
        self._check_per_image = any(isinstance(w, RandomCrop) or w.padding_mode != "constant"
                                    for w in self._windows)
        if isinstance(self._resample, Resize) and self._resample.shorter and self._size is None:
            raise ValueError("Resize(int) keeps each image's aspect ratio, so the images end up "
                             "different sizes: follow it with CenterCrop or RandomCrop, or give "
                             "Resize a (height, width)")
        self._antialias = getattr(self._resample, "antialias", True)
        self._bicubic = getattr(self._resample, "interpolation", "") == "bicubic"
        if self._size:
            self._check_blur(self._size)

        b = Binding()
        b.nparams = _RESAMPLE_SLOTS
        for w in self._windows:
            b.win[id(w)] = b.nparams
            b.nparams += _WINDOW_SLOTS
        for t in _expand(self.transforms):
            if t._kind in ("neighbourhood", "warp"):
                b.stage += 1
            b.bind(t)
        self._b = b

    def _chain_size(self, hw: tuple[int, int] | None) -> tuple[int, int] | None:
        """Output (H, W) for an input of size hw; None when it depends on the image."""
        r = self._resample
        if isinstance(r, RandomResizedCrop):
            hw = r.size
        elif isinstance(r, Resize):
            hw = (None if hw is None else _resize_shorter(hw, r.size[0])) if r.shorter else r.size
        for w in self._windows:
            hw = w._size_after(hw)
        return hw

    def _check_blur(self, size: tuple[int, int]) -> None:
        """torchvision's reflect padding needs each blur radius below the image size."""
        for t in self._neighbourhood:
            op = _unwrap(t)[1]
            if isinstance(op, GaussianBlur):
                kx, ky = op.kernel_size
                if kx // 2 >= size[1] or ky // 2 >= size[0]:
                    raise ValueError(f"{op!r} needs an image larger than its radius; the images "
                                     f"are {size[1]}x{size[0]} at that point")

    @property
    def source(self) -> str:
        """The generated Slang for (3, H, W) inputs, for debugging."""
        return self._source("chw")

    # -------------------------------------------------------------- codegen
    def _source(self, layout: str) -> str:
        if layout in self._sources:
            return self._sources[layout]
        from ._runtime import library_source

        b = self._b
        mid, out, conv = _DTYPES[self._dtype]
        put = [conv.format(f"c.{ch}") for ch in "xyz"]
        src = [library_source(),
               "// ---------------- generated by slangvision.Compose ----------------",
               f"typealias Mid = {mid};", f"typealias Out = {out};", _MID_HELPERS,
               "void store_out(RWTensor<Out, 4> t, int x, int y, int b, float4 c)",
               "{",
               f"    t[uint4(x, y, 0, b)] = {put[0]}; t[uint4(x, y, 1, b)] = {put[1]}; "
               f"t[uint4(x, y, 2, b)] = {put[2]};",
               "}", ""]
        src += list(dict.fromkeys(t._functions() for t in _walk(self.transforms)
                                  if isinstance(t, _UserOp)))

        # sampler
        src += ["void sample_params(int1 pix, Tensor<int, 1> seed, Tensor<int, 2> meta,",
                "                   RWTensor<float, 2> p)",
                "{",
                "    int b = pix.x;",
                "    uint s0 = uint(seed[uint1(0)]);",
                "    uint s = pcg(s0 ^ pcg(uint(b) * 2654435769u + 1u));   // this image",
                "    uint sb = pcg(s0 + 1758394347u);                       // shared by the batch",
                "    int H = meta[uint2(0, b)];",
                "    int W = meta[uint2(1, b)];",
                "    int gl = 0; int gt = 0; int gw = W; int gh = H;",
                "    int RW = W; int RH = H;"]
        r = self._resample
        if r is not None:
            src += [f"    {ln}" for ln in r._geometry("sb" if r.same_on_batch else "s")]  # type: ignore[attr-defined]
        for i, name in enumerate(("gl", "gt", "gw", "gh", "RW", "RH")):
            src.append(f"    p[uint2({b.geo + i}, b)] = float({name});")
        src.append("    int CW = RW; int CH = RH; int wox = 0; int woy = 0;")
        for w in self._windows:
            k = b.win[id(w)]
            rng = "sb" if w.same_on_batch else "s"
            src += [f"    p[uint2({k + 2}, b)] = float(CW); p[uint2({k + 3}, b)] = float(CH);",
                    "    {", *[f"        {ln}" for ln in w._window(rng)], "    }",
                    f"    p[uint2({k}, b)] = float(wox); p[uint2({k + 1}, b)] = float(woy);"]
        for t in _walk(self.transforms):
            src += [f"    {ln}" for ln in t._sample(b, "sb" if t.same_on_batch else "s")]
        src += ["}", ""]

        # pass 0 input: flips, window ops (last first), resample
        g = b.geo
        src += ["float4 input_0<S : IImageSource>(S src, Tensor<float, 2> p, int b, int x, int y,",
                "                                 int OW, int OH)",
                "{", *self._flip_lines(0), "    int rx = x; int ry = y;"]
        for w in reversed(self._windows):
            k = b.win[id(w)]
            src += [f"    rx += int({P(k)}); ry += int({P(k + 1)});",
                    f"    {{ int Wp = int({P(k + 2)}); int Hp = int({P(k + 3)});"]
            if w.padding_mode == "constant":
                f0, f1, f2 = (lit(v) for v in w.fill)
                src.append(f"      if (rx < 0 || ry < 0 || rx >= Wp || ry >= Hp) "
                           f"return float4({f0}, {f1}, {f2}, 0.0); }}")
            else:
                fn = {"edge": "clamp_index", "reflect": "reflect",
                      "symmetric": "symmetric"}[w.padding_mode]
                src.append(f"      rx = {fn}(rx, Wp); ry = {fn}(ry, Hp); }}")
        src += [f"    return resample(src, int({P(g)}), int({P(g + 1)}), int({P(g + 2)}), "
                f"int({P(g + 3)}), rx, ry, int({P(g + 4)}), int({P(g + 5)}), "
                f"{int(self._antialias)}, {int(self._bicubic)});",
                "}", ""]

        # later pass inputs and pre-passes
        for s, t in enumerate(self._neighbourhood, start=1):
            gates, op = _unwrap(t)
            gate = [f"    if (!({r._gate(b, c)})) return load_mid(src, x, y, b);"  # type: ignore[attr-defined]
                    for r, c in gates]
            head = "(Tensor<Mid, 4> src, Tensor<float, 2> p, int b, int x, int y, int OW, int OH)"
            pre = op._pre(b)    # type: ignore[attr-defined]
            if pre is not None:
                src += [f"float4 pre_{s}_pixel{head}", "{", *gate,
                        *[f"    {ln}" for ln in pre], "}",
                        f"void pre_{s}(int3 pix, Tensor<Mid, 4> src, Tensor<float, 2> p,",
                        "             RWTensor<Mid, 4> dst, int OW, int OH)",
                        "{",
                        "    int x = pix.x; int y = pix.y; int b = pix.z;",
                        f"    store_mid(dst, x, y, b, pre_{s}_pixel(src, p, b, x, y, OW, OH));",
                        "}", ""]
            src += [f"float4 input_{s}{head}", "{", *self._flip_lines(s), *gate,
                    *[f"    {ln}" for ln in op._input(b)], "}", ""]  # type: ignore[attr-defined]

        # pointwise chains; `upto` stops at statistic group k for its pre-pass
        # and sets `hit` (left false when this image does not apply that op)
        for s, stage in enumerate(self._stages):
            src += [f"float4 chain_{s}(float4 c, Tensor<float, 2> p, Tensor<float, 2> st, int b,"
                    " int upto, out bool hit)", "{", "    hit = false;"]
            src += [f"    {ln}" for t in stage
                    for ln in (t._post(b) if t._kind in ("neighbourhood", "warp")
                               else t._apply(b))]
            src += ["    return c;", "}", ""]

        # pass kernels and statistic pre-passes
        last = len(self._stages) - 1
        for s in range(len(self._stages)):
            if s == 0:
                args = "Tensor<int64_t, 1> ptrs, Tensor<int, 2> meta"
                inp = f"input_0(source_{layout}(ptrs, meta, b), p, b, x, y, OW, OH)"
            else:
                args = "Tensor<Mid, 4> src"
                inp = f"input_{s}(src, p, b, x, y, OW, OH)"
            dst = "RWTensor<Out, 4>" if s == last else "RWTensor<Mid, 4>"
            if s == last:
                erase = [ln for e in self._erasing for ln in e._erase(b)]
                store = " ".join(["c = (c - mean) * inv_std;", *erase,
                                  "store_out(dst, x, y, b, c);"])
            else:
                store = "store_mid(dst, x, y, b, c);"
            src += [f"void stage_{s}(int3 pix, {args}, Tensor<float, 2> p, Tensor<float, 2> st,",
                    f"             {dst} dst, int OW, int OH, float4 mean, float4 inv_std)",
                    "{",
                    "    int b = pix.z; int x = pix.x; int y = pix.y; bool hit;",
                    f"    float4 c = chain_{s}({inp}, p, st, b, -1, hit);",
                    f"    {store}",
                    "}", ""]
            for kind, k, stat_stage in b.stat_groups:
                if stat_stage == s:
                    src += _stats_kernel(kind, k, s, args, inp)
        self._sources[layout] = "\n".join(src)
        return self._sources[layout]

    def _flip_lines(self, stage: int) -> list[str]:
        """The flips that pass `stage`'s input applies to its coordinates."""
        lines = []
        for axis, coord, size in ((0, "x", "OW"), (1, "y", "OH")):
            flips = [f"({P(self._b.k(f))} > 0.5)" for f in self._flips
                     if f._axis == axis and self._flip_stage[id(f)] == stage]  # type: ignore[attr-defined]
            if flips:
                lines.append(f"    if ({' != '.join(flips)}) {coord} = {size} - 1 - {coord};")
        return lines

    # ------------------------------------------------------------------ run
    def __call__(self, images: Any, layout: str = "chw") -> "torch.Tensor":
        return self._run(images, layout)[0]

    def _run(self, images: Any, layout: str, params: "torch.Tensor | None" = None):
        import torch

        from . import _runtime

        table = _runtime.image_table(images, layout)
        size = self._size
        if size is None or self._check_per_image:
            # RandomCrop sizes and mirror paddings are checked per image size
            sizes = {self._chain_size(hw) for hw in set(table.dims)}
            if size is None and not table.dims:
                raise ValueError("no images: the output size comes from the images (no "
                                 "fixed-size geometric transform), so an empty list has none")
            if size is None and len(sizes) > 1:
                raise ValueError("the output size comes from the images (no fixed-size "
                                 "geometric transform), so all images must end up the same "
                                 "size")
            size = size or next(iter(sizes))
        assert size is not None     # an image size always gives a concrete output size
        if not self._size:
            self._check_blur(size)
        OH, OW = size
        dtype = getattr(torch, self._dtype)
        dev = table.device
        values = table.count * 3 * OH * OW
        if values >= 2**31:             # intermediates between passes have the same size
            raise ValueError(f"output of {values} values is too large for one call "
                             "(32-bit indexing); split the batch")
        out = torch.empty(table.count, 3, OH, OW, dtype=dtype, device=dev)
        if table.count == 0:
            return out, torch.empty(0, self._b.nparams, dtype=torch.float32, device=dev)
        with _runtime.torch_bridge():
            return self._launch(table, layout, OH, OW, out, params)

    def _launch(self, table: Any, layout: str, OH: int, OW: int, out: "torch.Tensor",
                params: "torch.Tensor | None"):
        import slangpy as spy
        import torch

        from . import _runtime

        dev = out.device
        dtype = torch.float32 if self._dtype == "float32" else torch.float16   # Mid
        final = out.view(torch.int16) if out.dtype == torch.bfloat16 else out
        m = _runtime.load_module(self._source(layout))
        B = table.count
        if params is None:
            params = torch.empty(B, self._b.nparams, dtype=torch.float32, device=dev)
            seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=dev)
            m.sample_params(spy.grid((B,)), seed, table.meta, params)
        st = torch.zeros(B, max(1, self._b.nstats), dtype=torch.float32, device=dev)
        norm = self._normalize
        m3 = norm.mean if norm else (0.0, 0.0, 0.0)
        s3 = norm.std if norm else (1.0, 1.0, 1.0)
        mean = spy.float4(m3[0], m3[1], m3[2], 0.0)
        inv_std = spy.float4(1.0 / s3[0], 1.0 / s3[1], 1.0 / s3[2], 1.0)
        chunks = (OW + _CHUNK - 1) // _CHUNK
        cur = None
        for s in range(len(self._stages)):
            if s > 0 and _unwrap(self._neighbourhood[s - 1])[1]._pre(self._b) is not None:  # type: ignore[attr-defined]
                tmp = torch.empty(B, 3, OH, OW, dtype=dtype, device=dev)
                getattr(m, f"pre_{s}")(spy.grid((B, OH, OW)), cur, params, tmp, OW, OH)
                cur = tmp
            inputs = (table.ptrs, table.meta) if s == 0 else (cur,)
            for kind, k, stat_stage in self._b.stat_groups:
                if stat_stage != s:
                    continue
                fn, grid = getattr(m, f"stats_{k}"), spy.grid((B, OH, chunks))
                if kind == "mean":
                    part = fn(grid, *inputs, params, st, OW, OH)
                    st[:, k] = part.sum((1, 2)) / float(OH * OW)
                elif kind == "minmax":
                    part = torch.empty(B, OH, chunks, 6, dtype=torch.float32, device=dev)
                    fn(grid, *inputs, params, st, part, OW, OH)
                    _autocontrast_stats(part, st, k)
                else:
                    hist = torch.zeros(B, 3, 256, dtype=torch.int32, device=dev)
                    # a raw address SlangPy does not track: the call is ordered
                    # on the current stream by its tensor arguments (params, st)
                    fn(grid, *inputs, params, st, hist.data_ptr(), OW, OH)
                    _equalize_stats(hist, OH * OW, st, k)
            dst = final if s == len(self._stages) - 1 else torch.empty(B, 3, OH, OW, dtype=dtype,
                                                                       device=dev)
            getattr(m, f"stage_{s}")(spy.grid((B, OH, OW)), *inputs, params, st, dst, OW, OH,
                                     mean, inv_std)
            cur = dst
        return out, params

    def __repr__(self) -> str:
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n], dtype=torch.{self._dtype})"
