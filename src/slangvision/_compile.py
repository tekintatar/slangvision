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
#                      neighbourhood op), then its pointwise chain; the last
#                      pass normalizes and writes the output
# Containers become per-image branches (RandomApply flags, ColorJitter's
# order).

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ._base import P, Binding, NeighbourhoodOp, Transform, _UserOp
from .transforms import (CenterCrop, ColorJitter, GaussianBlur, Normalize, RandomApply,
                         RandomHorizontalFlip, RandomResizedCrop, Resize)

if TYPE_CHECKING:
    import torch

_GEO_SLOTS = 8          # gl, gt, gw, gh, RW, RH, gcx, gcy
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


def _dtype_name(dtype: Any) -> str:
    name = "float16" if dtype is None else str(dtype).replace("torch.", "")
    if name not in _DTYPES:
        raise ValueError("dtype must be torch.float16, torch.bfloat16 or torch.float32, "
                         f"got {dtype}")
    return name


_BUILT_IN = (RandomResizedCrop, Resize, CenterCrop, RandomHorizontalFlip, ColorJitter,
             GaussianBlur, RandomApply, Normalize)


def _walk(transforms: Sequence[Transform]):
    for t in transforms:
        if not isinstance(t, (_BUILT_IN, _UserOp)):
            raise TypeError(f"Compose takes slangvision transforms; define your own "
                            f"with ColorOp or NeighbourhoodOp, got {t!r}")
        yield t
        yield from _walk(t._children())


def _unwrap(t: Transform) -> tuple[list[RandomApply], Transform]:
    """A neighbourhood op and the RandomApply wrappers that gate it."""
    gates = []
    while isinstance(t, RandomApply):
        gates.append(t)
        t = t.transforms[0]
    return gates, t


class Compose:
    """A GPU augmentation pipeline: torchvision-style transforms compiled into
    a few Slang kernels. Random values are drawn per image.

    ``Compose(transforms, dtype=torch.float16)``; ``torch.bfloat16`` and
    ``torch.float32`` output are also available. Geometric transforms
    (RandomResizedCrop, or Resize then CenterCrop, or CenterCrop) come first
    and fix the output size; RandomHorizontalFlip and pointwise ops follow in
    any order; Normalize, if present, is last.

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
        self._stages: list[list[Transform]] = [[]]
        self._neighbourhood: list[Transform] = []
        self._normalize: Normalize | None = None
        user_neighbourhood = False
        for i, t in enumerate(self.transforms):
            kind = t._kind
            if kind == "geometry":
                if self._flips or self._neighbourhood or any(self._stages):
                    raise ValueError(f"{t!r}: geometric transforms must come first")
                if getattr(t, "_resamples", False) and geometry:
                    raise ValueError(f"{t!r}: one Resize or RandomResizedCrop per pipeline, "
                                     "and it must be the first transform")
                if isinstance(t, CenterCrop) and any(isinstance(g, CenterCrop) for g in geometry):
                    raise ValueError("one CenterCrop per pipeline")
                geometry.append(t)
            elif kind == "flip":
                if user_neighbourhood:
                    raise ValueError(f"{t!r} must come before your NeighbourhoodOps")
                self._flips.append(t)
            elif kind == "pointwise":
                self._stages[-1].append(t)
            elif kind == "neighbourhood":
                self._neighbourhood.append(t)
                self._stages.append([])
                user_neighbourhood |= isinstance(_unwrap(t)[1], NeighbourhoodOp)
            elif kind == "normalize":
                if i != len(self.transforms) - 1:
                    raise ValueError("Normalize must be the last transform")
                self._normalize = t     # type: ignore[assignment]
            else:
                raise TypeError(f"{t!r}: unknown transform kind {kind!r}")
        self._geometry = geometry
        self._size: tuple[int, int] | None = None
        for g in geometry:
            if isinstance(g, (RandomResizedCrop, CenterCrop)):
                self._size = g.size
            elif isinstance(g, Resize):
                self._size = None if g.shorter else g.size
        if geometry and self._size is None:
            raise ValueError("Resize(int) keeps each image's aspect ratio, so the images end up "
                             "different sizes: follow it with CenterCrop, or give Resize a "
                             "(height, width)")
        self._antialias = all(getattr(g, "antialias", True) for g in geometry)
        if self._size:
            self._check_blur(self._size)

        b = Binding()
        b.nparams = _GEO_SLOTS
        for t in self.transforms:
            if t._kind == "neighbourhood":
                b.stage += 1
            b.bind(t)
        self._b = b

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
                "    int RW = W; int RH = H; int gcx = 0; int gcy = 0;"]
        for geo in self._geometry:
            src += [f"    {ln}" for ln in geo._geometry("sb" if geo.same_on_batch else "s")]  # type: ignore[attr-defined]
        for i, name in enumerate(("gl", "gt", "gw", "gh", "RW", "RH", "gcx", "gcy")):
            src.append(f"    p[uint2({b.geo + i}, b)] = float({name});")
        for t in _walk(self.transforms):
            src += [f"    {ln}" for ln in t._sample(b, "sb" if t.same_on_batch else "s")]
        src += ["}", ""]

        # pass 0 input: flips, crop offset, resample
        g = b.geo
        flips = [f"({P(b.k(f))} > 0.5)" for f in self._flips]
        src += ["float4 input_0<S : IImageSource>(S src, Tensor<float, 2> p, int b, int x, int y,",
                "                                 int OW, int OH)",
                "{"]
        if flips:
            src.append(f"    if ({' != '.join(flips)}) x = OW - 1 - x;")
        src += [f"    int RW = int({P(g + 4)}); int RH = int({P(g + 5)});",
                f"    int rx = x + int({P(g + 6)}); int ry = y + int({P(g + 7)});",
                "    if (rx < 0 || ry < 0 || rx >= RW || ry >= RH) return float4(0.0);",
                f"    return resample(src, int({P(g)}), int({P(g + 1)}), int({P(g + 2)}), "
                f"int({P(g + 3)}), rx, ry, RW, RH, {int(self._antialias)});",
                "}", ""]

        # later pass inputs and pre-passes
        for s, t in enumerate(self._neighbourhood, start=1):
            gates, op = _unwrap(t)
            gate = [f"    if (!({P(b.k(r))} > 0.5)) return load_mid(src, x, y, b);" for r in gates]
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
            src += [f"float4 input_{s}{head}", "{", *gate,
                    *[f"    {ln}" for ln in op._input(b)], "}", ""]  # type: ignore[attr-defined]

        # pointwise chains; `upto` stops at statistic k for its pre-pass
        for s, stage in enumerate(self._stages):
            src += [f"float4 chain_{s}(float4 c, Tensor<float, 2> p, Tensor<float, 2> st, int b,"
                    " int upto)", "{"]
            src += [f"    {ln}" for t in stage for ln in t._apply(b)]
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
            store = ("store_out(dst, x, y, b, (c - mean) * inv_std);" if s == last
                     else "store_mid(dst, x, y, b, c);")
            src += [f"void stage_{s}(int3 pix, {args}, Tensor<float, 2> p, Tensor<float, 2> st,",
                    f"             {dst} dst, int OW, int OH, float4 mean, float4 inv_std)",
                    "{",
                    "    int b = pix.z; int x = pix.x; int y = pix.y;",
                    f"    float4 c = chain_{s}({inp}, p, st, b, -1);",
                    f"    {store}",
                    "}", ""]
            for k, stat_stage in enumerate(b.stats_stage):
                if stat_stage != s:
                    continue
                src += [f"float stats_{k}(int3 pix, {args}, Tensor<float, 2> p, "
                        "Tensor<float, 2> st, int OW, int OH)",
                        "{",
                        "    int b = pix.z; int y = pix.y; float sum = 0.0;",
                        f"    for (int j = 0; j < {_CHUNK}; ++j) {{",
                        f"        int x = pix.x * {_CHUNK} + j;",
                        "        if (x >= OW) break;",
                        f"        sum += gray(chain_{s}({inp}, p, st, b, {k}));",
                        "    }",
                        "    return sum;",
                        "}", ""]
        self._sources[layout] = "\n".join(src)
        return self._sources[layout]

    # ------------------------------------------------------------------ run
    def __call__(self, images: Any, layout: str = "chw") -> "torch.Tensor":
        return self._run(images, layout)[0]

    def _run(self, images: Any, layout: str, params: "torch.Tensor | None" = None):
        import torch

        from . import _runtime

        table = _runtime.image_table(images, layout)
        size = self._size or table.size
        if size is None and table.count == 0:
            raise ValueError("no images: with no geometric transform the output size comes "
                             "from the images, so an empty list has none")
        if size is None:
            raise ValueError("with no geometric transform, all images must have the same size")
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
            for k, stat_stage in enumerate(self._b.stats_stage):
                if stat_stage == s:
                    part = getattr(m, f"stats_{k}")(spy.grid((B, OH, chunks)), *inputs, params,
                                                    st, OW, OH)
                    st[:, k] = part.sum((1, 2)) / float(OH * OW)
            dst = final if s == len(self._stages) - 1 else torch.empty(B, 3, OH, OW, dtype=dtype,
                                                                       device=dev)
            getattr(m, f"stage_{s}")(spy.grid((B, OH, OW)), *inputs, params, st, dst, OW, OH,
                                     mean, inv_std)
            cur = dst
        return out, params

    def __repr__(self) -> str:
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n], dtype=torch.{self._dtype})"
