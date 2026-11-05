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
# 2.4, SlangPy 0.42-0.43); in the other order they work in one process. So
# the import warns, and the first pipeline call raises instead of crashing
# (_runtime.device). SLANGVISION_ALLOW_LATE_IMPORT=1 skips that: other GPU
# video libraries load libnvcuvid too, and with them SlangPy may well work.
DECODER_FIRST = "slangpy" not in sys.modules and _gpu_decoder_loaded()

MESSAGE = (
    "slangvision was imported after a GPU decoder ran in this process (torchvision.io."
    "decode_jpeg(..., device=\"cuda\"), DALI, ...). In that order SlangPy crashes with a "
    "segmentation fault creating its CUDA device: put `import slangvision` before the "
    "first decode, at the top of your training script (set SLANGVISION_ALLOW_LATE_IMPORT=1 "
    "if the decoder was some other library and you want to try anyway)")

if DECODER_FIRST:
    warnings.warn(MESSAGE, RuntimeWarning, stacklevel=2)
