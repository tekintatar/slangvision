"""Checks that run when slangvision is imported, before SlangPy loads."""

from __future__ import annotations

import sys
import warnings


def _gpu_decoder_loaded() -> bool:
    """Whether a GPU decoder already ran in this process (Linux): torchvision's
    GPU decode and DALI's load NVIDIA's video codec library, libnvcuvid."""
    try:
        with open("/proc/self/maps") as maps:
            return any("libnvcuvid" in line for line in maps)
    except OSError:
        return False


# If torchvision's GPU decoder or DALI ran before SlangPy is loaded, SlangPy
# segfaults creating its CUDA device (seen with torchvision 0.28-0.29, DALI
# 2.4, SlangPy 0.42-0.43); in the other order they work in one process. A
# warning, not an error: other GPU video libraries load libnvcuvid too, and
# with them SlangPy may well work.
if "slangpy" not in sys.modules and _gpu_decoder_loaded():
    warnings.warn(
        "slangvision was imported after a GPU decoder ran in this process (torchvision."
        "io.decode_jpeg(..., device=\"cuda\"), DALI, ...). With those, SlangPy crashes with a "
        "segmentation fault when the first pipeline runs: put `import slangvision` at the "
        "top, before the first decode", RuntimeWarning, stacklevel=2)
