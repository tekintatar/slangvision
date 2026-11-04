# slangvision/_runtime.py — the CUDA device, the table of input images and the
# compiled-module cache. Kernels run on PyTorch's current CUDA stream.

from __future__ import annotations

import contextlib
import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import slangpy as spy

if TYPE_CHECKING:
    import torch

_SHADERS = Path(__file__).resolve().parent / "shaders"

_DEVICES: dict[int, spy.Device] = {}
_MODULES: dict[tuple[int, str], spy.Module] = {}


@functools.lru_cache(maxsize=1)
def library_source() -> str:
    """The op library: every .slang file in shaders/, core.slang first and the
    rest in name order, joined into one source. Each pipeline appends its
    generated entry points and compiles the whole as one module, so the file
    split costs nothing at run time (Slang does not need declarations before
    use)."""
    files = sorted(_SHADERS.glob("*.slang"), key=lambda f: (f.name != "core.slang", f.name))
    return "\n".join(f.read_text() for f in files)


# The first device creation compiles and runs a one-pixel probe kernel, so a
# missing or unusable NVRTC (_cuda.py sets up the one installed with PyTorch)
# fails here with a clear message instead of in the first pipeline.
_NVRTC_HINT = (
    "SlangPy compiles CUDA kernels at run time with NVRTC, which must be no newer "
    "than the CUDA version your NVIDIA driver supports (nvidia-smi). SlangVision "
    "uses the NVRTC installed with PyTorch, or yours if CUDA_PATH is set or a CUDA "
    "Toolkit is installed system-wide (run `sudo ldconfig` after an apt install). "
    "See the README section 'Install'"
)

_PROBE_SOURCE = """
void slangvision_probe(uint2 coord, [format("rgba32f")] RWTexture2D<float4> dst)
{
    dst[coord] = float4(0.25, 0.5, 0.75, 1.0);
}
"""
_PROBE_VALUE = (0.25, 0.5, 0.75, 1.0)

# SlangPy appends the native source location ("…/device.cpp:264 in function
# Device") to its errors; it says nothing actionable to a user.
_NATIVE_LOCATION = re.compile(r"^\S+\.(?:c|cc|cpp|h|hpp):\d+ in function \S+$")


def _summarize(err: BaseException, limit: int = 300) -> str:
    """One-line, length-bounded description of a possibly multi-line error."""
    lines = (line.strip() for line in str(err).splitlines())
    kept = [line for line in lines if line and not _NATIVE_LOCATION.match(line)]
    text = " ".join(" ".join(kept).split()) or type(err).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _verify_runtime_compiler(dev: spy.Device) -> None:
    """Compile and run a one-pixel kernel on ``dev``; raise if that fails."""
    module = spy.Module.load_from_source(dev, "slangvision_probe", _PROBE_SOURCE)
    texture = dev.create_texture(
        width=1,
        height=1,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    module.slangvision_probe(spy.grid((1, 1)), texture)
    value = np.asarray(texture.to_numpy(), dtype=np.float32).reshape(-1)[:4]
    if not np.allclose(value, _PROBE_VALUE):
        raise RuntimeError(f"test kernel wrote {value.tolist()} instead of {list(_PROBE_VALUE)}")


def device() -> spy.Device:
    """SlangPy's CUDA device sharing PyTorch's context on the current GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("slangvision needs a CUDA GPU and a CUDA build of PyTorch")
    index = torch.cuda.current_device()
    if index in _DEVICES:
        return _DEVICES[index]
    from . import _cuda

    try:
        _cuda.configure()
    except _cuda.SetupError as e:
        raise RuntimeError(f"slangvision could not set up NVRTC: {e}.") from None
    try:
        dev = spy.create_torch_device(type=spy.DeviceType.cuda)
    except RuntimeError as e:
        raise RuntimeError(f"slangvision could not create SlangPy's CUDA device "
                           f"({_summarize(e)}). {_NVRTC_HINT}. {_cuda.describe()}.") from e
    try:
        _verify_runtime_compiler(dev)
    except Exception as e:
        try:
            dev.close()
        except Exception:
            pass
        raise RuntimeError("slangvision created SlangPy's CUDA device but could not "
                           f"compile a test kernel on it ({_summarize(e)}). "
                           f"{_NVRTC_HINT}. {_cuda.describe()}.") from e
    _DEVICES[index] = dev
    return dev


@contextlib.contextmanager
def torch_bridge():
    """Allow SlangPy's Python torch bridge for augmentation calls only.

    It costs well under 0.1 ms per batch here, so the compiled slangpy-torch
    extension is optional. The setting is process-wide in SlangPy, so it is
    restored afterwards: other SlangPy code in the process keeps its own
    choice."""
    if not spy.is_torch_bridge_using_fallback():
        yield
        return
    from slangpy.core import calldata

    previous = bool(getattr(calldata, "_ALLOW_TORCH_FALLBACK", False))
    spy.set_allow_torch_fallback(True)
    try:
        yield
    finally:
        spy.set_allow_torch_fallback(previous)


def load_module(source: str) -> spy.Module:
    """Compile a generated pipeline once per GPU (cached by source)."""
    import torch

    dev = device()
    digest = hashlib.sha1(source.encode()).hexdigest()[:16]
    key = (torch.cuda.current_device(), digest)
    if key not in _MODULES:
        try:
            _MODULES[key] = spy.Module.load_from_source(dev, f"slangvision_{digest}", source)
        except Exception as e:
            raise RuntimeError("slangvision could not compile this pipeline; "
                               "Compose.source shows the generated Slang (a ColorOp or "
                               f"NeighbourhoodOp you defined is the usual cause): {e}") from e
    return _MODULES[key]


@dataclass
class ImageTable:
    """Device addresses and sizes of the input images, read in place."""

    ptrs: "torch.Tensor | None"     # (B,) int64 device addresses
    meta: "torch.Tensor | None"     # (B, 2) int32 rows: (H, W)
    count: int
    dims: list                      # (H, W) of each image (one entry for a batch tensor)
    device: "torch.device"
    keep: list                      # the (contiguous) inputs, alive during the call


def _checked(t: Any, dev: "torch.device", layout: str, batched: bool) -> "torch.Tensor":
    import torch

    want = 4 if batched else 3
    shape = "(B, 3, H, W)" if batched else "(3, H, W)"
    if layout == "hwc":
        shape = "(B, H, W, 3)" if batched else "(H, W, 3)"
    if not isinstance(t, torch.Tensor) or t.ndim != want:
        raise ValueError(f"expected uint8 CUDA images of shape {shape}, got "
                         f"{getattr(t, 'shape', type(t).__name__)}")
    if t.dtype != torch.uint8:
        raise ValueError(f"images must be uint8 (decoder output), got {t.dtype}")
    if t.device != dev:
        raise ValueError(f"images must be on the current CUDA device {dev}, got {t.device}")
    channels = t.shape[-1] if layout == "hwc" else t.shape[-3]
    if channels != 3:
        raise ValueError(f"images must have 3 (RGB) channels in layout {layout!r}, "
                         f"got shape {tuple(t.shape)}")
    hw = t.shape[-2:] if layout == "chw" else t.shape[-3:-1]
    if 0 in hw:
        raise ValueError(f"images must not be empty, got shape {tuple(t.shape)}")
    if 3 * hw[0] * hw[1] >= 2**31:
        raise ValueError(f"an image of {hw[0]}x{hw[1]} pixels is too large (the kernels "
                         "index each image with 32-bit integers); downscale it first")
    return t.contiguous()


def image_table(images: Any, layout: str) -> ImageTable:
    import torch

    if layout not in ("chw", "hwc"):
        raise ValueError(f"layout must be 'chw' or 'hwc', got {layout!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("slangvision needs a CUDA GPU and a CUDA build of PyTorch")
    dev = torch.device("cuda", torch.cuda.current_device())
    if isinstance(images, torch.Tensor):
        x = _checked(images, dev, layout, batched=True)
        B = x.shape[0]
        H, W = (x.shape[2], x.shape[3]) if layout == "chw" else (x.shape[1], x.shape[2])
        if B == 0:
            return ImageTable(None, None, 0, [(H, W)], dev, [])
        ptrs = x.data_ptr() + torch.arange(B, dtype=torch.int64, device=dev) * (3 * H * W)
        meta = torch.tensor([H, W], dtype=torch.int32, device=dev).repeat(B, 1)
        return ImageTable(ptrs, meta, B, [(H, W)], dev, [x])
    imgs = [_checked(t, dev, layout, batched=False) for t in images]
    if not imgs:
        return ImageTable(None, None, 0, [], dev, [])
    dims = [(t.shape[1], t.shape[2]) if layout == "chw" else (t.shape[0], t.shape[1])
            for t in imgs]
    ptrs = torch.tensor([t.data_ptr() for t in imgs], dtype=torch.int64).pin_memory()
    meta = torch.tensor(dims, dtype=torch.int32).pin_memory()
    return ImageTable(ptrs.to(dev, non_blocking=True), meta.to(dev, non_blocking=True),
                      len(imgs), dims, dev, imgs)
