"""Drop-in replacements for BasicSR's GPU degradation functions
(basicsr.utils.img_process_util, basicsr.data.degradations,
basicsr.utils.diffjpeg), which Real-ESRGAN and the diffusion SR models built
on it call in their training step on float (B, 3, H, W) batches. Same names
and arguments, and the same results apart from the noise values themselves;
swap the import::

    from slangvision.sr import USMSharp       # was basicsr.utils.img_process_util
    from slangvision.sr import DiffJPEG       # was basicsr.utils.diffjpeg

See docs/SR.md. Each has its own Slang kernels (shaders/kernels/);
slangvision.testing has the references.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .mixing import _CONVERT

if TYPE_CHECKING:
    import torch

__all__ = ["DiffJPEG", "USMSharp", "add_gaussian_noise_pt", "add_poisson_noise_pt",
           "random_add_gaussian_noise_pt", "random_add_poisson_noise_pt"]

_KERNELS = Path(__file__).resolve().parent / "shaders" / "kernels"

# OpenCV's fixed kernels for small sizes when sigma <= 0 (getGaussianKernel,
# checked against OpenCV 5.0)
_SMALL = {1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
          7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
          9: [0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875,
              0.05078125, 0.015625]}


def gaussian_kernel(ksize: int, sigma: float) -> list[float]:
    """cv2.getGaussianKernel(ksize, sigma) (float64 values): sigma <= 0 means
    0.3 * ((ksize - 1) / 2 - 1) + 0.8."""
    if sigma <= 0 and ksize in _SMALL:
        return list(_SMALL[ksize])
    s = sigma if sigma > 0 else 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    scale = -0.5 / (s * s)
    t = [math.exp(scale * (i - (ksize - 1) * 0.5) ** 2) for i in range(ksize)]
    total = sum(t)
    return [v * (1.0 / total) for v in t]


def _view(dtype: str) -> Any:
    """How a tensor of this dtype travels to Slang (bfloat16 as an int16 view)."""
    import torch

    return (lambda t: t.view(torch.int16)) if dtype == "bfloat16" else (lambda t: t)


def _module(dtype: str, kernel: str) -> Any:
    from . import _runtime

    return _runtime.load_module(f"import slangpy;\n{_CONVERT[dtype]}\n"
                                f"{(_KERNELS / kernel).read_text()}")


def _check_batch(x: Any, name: str) -> str:
    import torch

    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"{name}: images must be a (B, 3, H, W) tensor, got "
                         f"{getattr(x, 'shape', type(x).__name__)}")
    dtype = str(x.dtype).replace("torch.", "")
    if dtype not in _CONVERT or not x.is_cuda:
        raise ValueError(f"{name}: images must be a float16, bfloat16 or float32 CUDA tensor, "
                         f"got {x.dtype} on {x.device}")
    if x.requires_grad and torch.is_grad_enabled():
        raise NotImplementedError(f"{name}: no gradients (the kernels have no backward pass); "
                                  "run it under torch.no_grad() or on a detached tensor")
    return dtype


class USMSharp:
    """BasicSR's USMSharp, which Real-ESRGAN applies to its high-resolution
    targets: ``usm = USMSharp().cuda(); gt_usm = usm(gt, weight=0.5,
    threshold=10)``. ``sharp = clamp(x + weight * (x - blur), 0, 1)`` is
    blended into the image where ``|x - blur| * 255 > threshold`` (the mask
    itself blurred). ``radius`` is BasicSR's: the Gaussian's size, made odd
    (50 gives 51 taps), with OpenCV's sigma for ``sigma=0``; reflect padding,
    so H and W must exceed half the size. The 2D Gaussian runs as a row and a
    column pass, which is what makes it fast: BasicSR convolves 51x51
    directly.

    Takes a (B, 3, H, W) float16, bfloat16 or float32 CUDA tensor in [0, 1]
    and returns the sharpened batch in the same dtype, without gradients
    (Real-ESRGAN calls it under no_grad). ``.cuda()`` and ``.to()`` return
    the object itself, as BasicSR's module is used."""

    def __init__(self, radius: int = 50, sigma: float = 0.0):
        if not isinstance(radius, numbers.Integral) or isinstance(radius, bool) or radius < 0:
            raise ValueError(f"USMSharp: radius must be a non-negative int, got {radius!r}")
        self.radius: int = int(radius)
        self.sigma: float = float(sigma)
        if not math.isfinite(self.sigma):
            raise ValueError(f"USMSharp: sigma must be finite, got {sigma}")
        self.size: int = self.radius + 1 if self.radius % 2 == 0 else self.radius
        self.taps: list[float] = gaussian_kernel(self.size, self.sigma)
        self._weights: dict[Any, "torch.Tensor"] = {}

    def cuda(self, *args: Any, **kwargs: Any) -> "USMSharp":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "USMSharp":
        return self

    def __call__(self, x: "torch.Tensor", weight: float = 0.5,
                 threshold: float = 10) -> "torch.Tensor":
        import slangpy as spy
        import torch

        from . import _runtime

        weight, threshold = float(weight), float(threshold)
        dtype = _check_batch(x, "USMSharp")
        view = _view(dtype)
        B, _, H, W = x.shape
        r = self.size // 2
        if B and min(H, W) <= r:
            raise ValueError(f"USMSharp: images must be larger than {r} pixels on each side "
                             f"for reflect padding (radius {self.radius}), got {H}x{W}")
        x = x.detach().contiguous()     # SlangPy refuses tensors that require grad
        out = torch.empty_like(x)
        if B == 0:
            return out
        w = self._weights.get(x.device)
        if w is None:
            w = self._weights[x.device] = torch.tensor(self.taps, dtype=torch.float32,
                                                       device=x.device)
        tmp = torch.empty(B, 3, H, W, dtype=torch.float32, device=x.device)
        res = torch.empty_like(tmp)
        grid = spy.grid((B, H, W))
        with _runtime.torch_bridge():
            m = _module(dtype, "usm.slang")
            m.usm_rows(grid, view(x), w, tmp, r, W)
            m.usm_residual(grid, view(x), tmp, w, res, r, H)
            m.usm_mask_rows(grid, res, w, tmp, r, threshold, W)
            m.usm_blend(grid, view(x), tmp, res, w, view(out), r, weight, H)
        return out

    def __repr__(self) -> str:
        return f"USMSharp(radius={self.radius}, sigma={self.sigma})"


# ----------------------------------------------------------------- noise
def _per_image(value: Any, x: "torch.Tensor") -> "torch.Tensor":
    """A number or a (B,) tensor as (B,) float32 on x's device."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.reshape(-1).to(device=x.device, dtype=torch.float32).expand(x.shape[0])
    return torch.full((x.shape[0],), float(value), dtype=torch.float32, device=x.device)


def _noise(kernel: str, x: "torch.Tensor", columns: list["torch.Tensor"], clip: bool,
           rounds: bool, name: str) -> "torch.Tensor":
    import slangpy as spy
    import torch

    from . import _runtime

    dtype = _check_batch(x, name)
    x = x.detach().contiguous()     # SlangPy refuses tensors that require grad
    out = torch.empty_like(x)
    if x.shape[0] == 0:
        return out
    # the noise key: one more draw from the device's generator (24 bits, exact
    # in a float), so the CPU generator is untouched and nothing syncs
    key = torch.randint(0, 2**24, (1,), device=x.device).float().expand(x.shape[0])
    prm = torch.stack(columns + [key], 1).contiguous()
    view = _view(dtype)
    with _runtime.torch_bridge():
        getattr(_module(dtype, "noise.slang"), kernel)(
            spy.grid((x.shape[0],) + tuple(x.shape[2:])), view(x), view(out), prm,
            int(bool(clip)), int(bool(rounds)))
    return out


def add_gaussian_noise_pt(img: "torch.Tensor", sigma: Any = 10, gray_noise: Any = 0,
                          clip: bool = True, rounds: bool = False) -> "torch.Tensor":
    """BasicSR's add_gaussian_noise_pt: adds N(0, sigma / 255) noise per
    value (``sigma`` a number or one per image); ``gray_noise`` (a number or
    one per image) blends in noise that is the same for all three channels,
    ``noise * (1 - g) + gray * g``; then clamps to [0, 1] (``clip``) and/or
    rounds to 1/255 (``rounds``). BasicSR draws one gray noise field and
    shares it across the batch; here every image gets its own. No
    gradients (BasicSR's is differentiable; Real-ESRGAN runs it under
    no_grad)."""
    x = img
    return _noise("gaussian_noise", x, [_per_image(sigma, x), _per_image(gray_noise, x)],
                  clip, rounds, "add_gaussian_noise_pt")


def random_add_gaussian_noise_pt(img: "torch.Tensor", sigma_range: Any = (0, 1.0),
                                 gray_prob: float = 0, clip: bool = True,
                                 rounds: bool = False) -> "torch.Tensor":
    """BasicSR's random_add_gaussian_noise_pt (Real-ESRGAN's noise): sigma
    uniform in ``sigma_range`` (0-255 scale) and gray noise with probability
    ``gray_prob``, per image, drawn with the same torch calls as BasicSR, so
    the same seed gives the same sigmas and choices. The noise itself comes
    from SlangVision's generator, keyed by one draw from the device's torch
    generator: later draws on that generator differ from a BasicSR run."""
    import torch

    b = img.shape[0]
    sigma = torch.rand(b, dtype=img.dtype, device=img.device) * (sigma_range[1] -
                                                                 sigma_range[0]) + sigma_range[0]
    gray = (torch.rand(b, dtype=img.dtype, device=img.device) < gray_prob).float()
    return add_gaussian_noise_pt(img, sigma, gray, clip, rounds)


def add_poisson_noise_pt(img: "torch.Tensor", scale: Any = 1.0, clip: bool = True,
                         rounds: bool = False, gray_noise: Any = 0) -> "torch.Tensor":
    """BasicSR's add_poisson_noise_pt: each value rounded to 1/255 (q) is
    replaced by Poisson(q * n) / n, with n = 2**ceil(log2(the image's number
    of distinct rounded values)); the noise (that minus q) times ``scale`` is
    added to the image. ``gray_noise`` blends in noise from the grayscale
    image (torchvision's weights), the same for all channels, as in
    add_gaussian_noise_pt. Then ``clip`` and ``rounds`` as there. BasicSR
    counts the distinct values with torch.unique per image; here one pass
    marks which of the 256 levels occur. No gradients."""
    import slangpy as spy
    import torch

    from . import _runtime

    x = img
    dtype = _check_batch(x, "add_poisson_noise_pt")
    B = x.shape[0]
    if B == 0:
        return x.clone()
    x = x.detach().contiguous()     # SlangPy refuses tensors that require grad
    color = torch.zeros(B, 256, dtype=torch.int32, device=x.device)
    gray = torch.zeros_like(color)
    view = _view(dtype)
    with _runtime.torch_bridge():
        _module(dtype, "noise.slang").levels(spy.grid((B,) + tuple(x.shape[2:])), view(x),
                                             color, gray)

    def vals(present: "torch.Tensor") -> "torch.Tensor":
        return torch.exp2(torch.ceil(torch.log2(present.sum(1).float())))

    return _noise("poisson_noise", x, [_per_image(scale, x), _per_image(gray_noise, x),
                                      vals(color), vals(gray)], clip, rounds,
                  "add_poisson_noise_pt")


def random_add_poisson_noise_pt(img: "torch.Tensor", scale_range: Any = (0, 1.0),
                                gray_prob: float = 0, clip: bool = True,
                                rounds: bool = False) -> "torch.Tensor":
    """BasicSR's random_add_poisson_noise_pt (Real-ESRGAN's noise): scale
    uniform in ``scale_range`` and gray noise with probability ``gray_prob``,
    per image, drawn with the same torch calls as BasicSR (the noise as in
    random_add_gaussian_noise_pt)."""
    import torch

    b = img.shape[0]
    scale = torch.rand(b, dtype=img.dtype, device=img.device) * (scale_range[1] -
                                                                 scale_range[0]) + scale_range[0]
    gray = (torch.rand(b, dtype=img.dtype, device=img.device) < gray_prob).float()
    return add_poisson_noise_pt(img, scale, clip, rounds, gray)


# ------------------------------------------------------------------ JPEG
# ITU T.81 Annex K tables K.1 (luminance) and K.2 (chrominance), rows the
# vertical frequency, as libjpeg stores them. BasicSR's DiffJPEG uses them
# transposed (its DCT puts the vertical frequency first too).
_Y_TABLE = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_C_TABLE = [(([17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99])[r][c]
             if r < 4 and c < 4 else 99) for r in range(8) for c in range(8)]
_JPEG_CONSTS: dict[Any, tuple["torch.Tensor", ...]] = {}


def _jpeg_consts(device: Any, libjpeg: bool) -> tuple["torch.Tensor", ...]:
    """The cosines and BasicSR's alpha(u) alpha(v) (float64 products rounded
    to float32, as it builds them), and the base Y then C tables, transposed
    as BasicSR has them unless ``libjpeg``; on ``device``."""
    key = (str(device), bool(libjpeg))
    consts = _JPEG_CONSTS.get(key)
    if consts is None:
        import numpy as np
        import torch

        n = np.arange(8)
        cos = np.cos((2 * n[:, None] + 1) * n[None, :] * np.pi / 16)          # [x, u]
        alpha = np.array([1.0 / np.sqrt(2)] + [1.0] * 7)
        y, c = np.array(_Y_TABLE).reshape(8, 8), np.array(_C_TABLE).reshape(8, 8)
        if not libjpeg:
            y, c = y.T, c.T
        consts = _JPEG_CONSTS[key] = tuple(
            torch.tensor(np.asarray(a, dtype=np.float32).reshape(-1), device=device)
            for a in (cos, np.outer(alpha, alpha), np.concatenate([y.reshape(-1), c.reshape(-1)])))
    return consts


class DiffJPEG:
    """BasicSR's DiffJPEG, which Real-ESRGAN uses as its JPEG degradation:
    ``jpeger = DiffJPEG(differentiable=False).cuda(); out = jpeger(out,
    quality=q)`` with ``q`` a number or one per image (BasicSR's quality,
    above 0 and below 100, may be fractional). The same steps as BasicSR, in
    four passes (the DCTs separable, one thread per block row): RGB to YCbCr
    (JFIF), Cb and Cr averaged over 2x2 pixels, 8x8 DCTs quantized with the
    standard tables scaled by quality, then back, with chroma repeated over
    2x2 pixels; sizes not a multiple of 16 are padded with black and cropped
    back.

    ``differentiable=True`` (BasicSR's default) gives the values of BasicSR's
    soft rounding, ``round(t) + (t - round(t))**3``, but no gradients: it
    raises NotImplementedError when ``x`` requires them. ``libjpeg=True`` is
    closer to a real encoder and decoder and no longer BasicSR's result (the
    differences: docs/SR.md); it always rounds plainly.

    Takes a (B, 3, H, W) float16, bfloat16 or float32 CUDA tensor in [0, 1]
    (BasicSR takes float32 only) and returns the batch in the same dtype.
    The caller's quality tensor is left as it is (BasicSR overwrites it with
    its quality factors). ``.cuda()`` and ``.to()`` return the object
    itself."""

    def __init__(self, differentiable: bool = True, libjpeg: bool = False):
        self.differentiable: bool = bool(differentiable)
        self.libjpeg: bool = bool(libjpeg)

    def cuda(self, *args: Any, **kwargs: Any) -> "DiffJPEG":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "DiffJPEG":
        return self

    def _quality(self, quality: Any, x: "torch.Tensor") -> tuple["torch.Tensor | None", float]:
        """A (B,) float32 quality tensor for the kernel to scale per image
        (left unchanged, no host sync), or the table scale for all: BasicSR's
        quality factor (in double, as BasicSR computes it for a number) or
        libjpeg's scale for the quality rounded to 1-100. The kernel turns
        per-image qualities into tables: in torch that takes about seven
        launches (+0.085 ms per call at batch 12)."""
        import torch

        b = x.shape[0]
        if isinstance(quality, torch.Tensor):
            q = quality.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
            if q.numel() not in (1, b):
                raise ValueError(f"DiffJPEG: quality must be a number or one per image ({b}), "
                                 f"got shape {tuple(quality.shape)}")
            return q.expand(b).contiguous(), 0.0
        value = float(quality)
        if self.libjpeg and math.isfinite(value):
            qi = min(max(round(value), 1), 100)
            return None, float(5000 // qi if qi < 50 else 200 - 2 * qi)
        if 0 < value < 100:
            return None, (5000.0 / value if value < 50 else 200.0 - value * 2) / 100.0
        raise ValueError(f"DiffJPEG: quality must be above 0 and below 100, got {quality} "
                         "(BasicSR's tables are zero at 100; libjpeg=True takes 1-100)")

    def __call__(self, x: "torch.Tensor", quality: Any) -> "torch.Tensor":
        import slangpy as spy
        import torch

        from . import _runtime

        dtype = _check_batch(x, "DiffJPEG")
        B, _, H, W = x.shape
        x = x.detach().contiguous()     # SlangPy refuses tensors that require grad
        out = torch.empty_like(x)
        if B == 0:
            return out
        qual, value = self._quality(quality, x)
        cos, alpha, tab = _jpeg_consts(x.device, self.libjpeg)
        hp, wp = -(-H // 16) * 16, -(-W // 16) * 16
        # luma, then Cb and Cr side by side below it: every block on the grid of 8
        p = torch.empty(B, hp * 3 // 2, wp, dtype=torch.float32, device=x.device)
        q = torch.empty_like(p)
        blocks = spy.grid((B, hp * 3 // 2, wp // 8))      # one thread per block row
        view = _view(dtype)
        with _runtime.torch_bridge():
            m = _module(dtype, "jpeg.slang")
            m.jpeg_ycc(spy.grid((B, hp // 2, wp // 2)), view(x), p, H, W, hp, wp,
                       int(self.libjpeg))
            m.jpeg_quant(blocks, p, cos, alpha, tab, tab if qual is None else qual, q, hp,
                         int(qual is not None), value, int(self.libjpeg),
                         int(self.differentiable and not self.libjpeg))
            m.jpeg_idct(blocks, q, cos, alpha, p)
            m.jpeg_out(spy.grid((B, H, W)), p, view(out), hp, wp, (H + 1) // 2, (W + 1) // 2,
                       int(self.libjpeg))
        return out

    def __repr__(self) -> str:
        return f"DiffJPEG(differentiable={self.differentiable}, libjpeg={self.libjpeg})"
