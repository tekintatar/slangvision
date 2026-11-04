# slangvision/_cuda.py — help SlangPy find NVRTC with PyTorch from pip.
#
# SlangPy compiles CUDA kernels at run time with NVRTC, and looks for it in
# two ways: its device layer opens $CUDA_PATH/lib/libnvrtc.so, and Slang's
# compiler opens the bare name "libnvrtc.so" through the dynamic loader
# (LD_LIBRARY_PATH, then SlangPy's own folder, which is in its RUNPATH, then
# the system library cache). A system CUDA Toolkit satisfies both. PyTorch
# from pip ships NVRTC in its nvidia wheels instead (nvidia/cu13/lib for CUDA
# 13, nvidia/cuda_nvrtc/lib for CUDA 12) under a versioned name only, which
# neither lookup finds.
#
# Unless the user set CUDA_PATH or has a system NVRTC, configure() picks the
# newest wheel NVRTC the driver supports (a newer one makes kernels the
# driver rejects), makes a folder of links to it in the user cache and sets
# it as CUDA_PATH in-process (SlangPy reads it when it creates the device).
# The compiler's bare-name lookup cannot be redirected in-process
# (LD_LIBRARY_PATH is read at start-up), so it needs a "libnvrtc.so" link in
# SlangPy's package folder. That changes another package, so it is made only
# with the user's consent: `python -m slangvision setup` once per
# environment. After that, configure() keeps its own link current; it never
# replaces a libnvrtc.so it did not make.

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class Nvrtc:
    lib: str                    # the real libnvrtc.so.<major> file
    include: str                # folder with cuda_runtime.h
    libdir: str                 # folder with the NVRTC and cudart libraries
    version: tuple[int, int]


def _version(v: tuple[int, int] | None) -> str:
    return "unknown" if v is None else f"{v[0]}.{v[1]}"


def driver_cuda_version() -> tuple[int, int] | None:
    """The newest CUDA version the installed NVIDIA driver supports."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    v = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(v)) != 0:
        return None
    return v.value // 1000, (v.value % 1000) // 10


def nvrtc_version(path: str) -> tuple[int, int] | None:
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    major, minor = ctypes.c_int(), ctypes.c_int()
    if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
        return None
    return major.value, minor.value


def _wheel_layouts(root: str) -> list[tuple[str, str]]:
    """(libdir, include) pairs for NVIDIA's pip wheels under one nvidia/ folder."""
    pairs = [(os.path.join(d, "lib"), os.path.join(d, "include"))
             for d in sorted(glob.glob(os.path.join(root, "cu[0-9]*")))]        # CUDA 13+
    pairs.append((os.path.join(root, "cuda_nvrtc", "lib"),                       # CUDA 12
                  os.path.join(root, "cuda_runtime", "include")))
    return pairs


def wheel_nvrtcs() -> list[Nvrtc]:
    """NVRTC copies installed by pip (normally as PyTorch dependencies)."""
    try:
        import nvidia
    except ImportError:
        return []
    found = []
    for root in getattr(nvidia, "__path__", []):
        for libdir, include in _wheel_layouts(root):
            libs = sorted(glob.glob(os.path.join(libdir, "libnvrtc.so.*")))
            if not libs or not os.path.isfile(os.path.join(include, "cuda_runtime.h")):
                continue
            version = nvrtc_version(libs[0])
            if version is not None:
                found.append(Nvrtc(libs[0], include, libdir, version))
    return found


def choose(found: list[Nvrtc], driver: tuple[int, int] | None) -> Nvrtc | None:
    """The newest copy the driver can run (any copy if the driver is unknown)."""
    usable = [n for n in found if driver is None or n.version <= driver]
    return max(usable, key=lambda n: n.version) if usable else None


def _system_nvrtc() -> bool:
    try:
        ctypes.CDLL("libnvrtc.so")
    except OSError:
        return False
    return True


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "slangvision")


def _shim(n: Nvrtc) -> str:
    """A CUDA_PATH-style folder of links to one wheel NVRTC (made once, atomically,
    so concurrent processes such as DDP ranks agree)."""
    key = hashlib.sha1(n.lib.encode()).hexdigest()[:10]
    shim = os.path.join(_cache_dir(), f"cuda-{_version(n.version)}-{key}")
    if os.path.isfile(os.path.join(shim, "lib", "libnvrtc.so")):
        return shim
    os.makedirs(os.path.dirname(shim), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".tmp-", dir=os.path.dirname(shim))
    os.symlink(n.include, os.path.join(tmp, "include"))
    os.mkdir(os.path.join(tmp, "lib"))
    for f in glob.glob(os.path.join(n.libdir, "libnvrtc*")) + \
            glob.glob(os.path.join(n.libdir, "libcudart*")):
        os.symlink(f, os.path.join(tmp, "lib", os.path.basename(f)))
    os.symlink(n.lib, os.path.join(tmp, "lib", "libnvrtc.so"))
    try:
        os.rename(tmp, shim)
    except OSError:             # another process made it first
        shutil.rmtree(tmp, ignore_errors=True)
    return shim


_SETUP = "python -m slangvision setup"


def slangpy_link() -> str:
    """Where Slang's compiler finds NVRTC by its bare name."""
    import slangpy

    return os.path.join(os.path.dirname(slangpy.__file__), "libnvrtc.so")


def _is_ours(link: str) -> bool:
    """A link SlangVision made: one pointing at an NVRTC from NVIDIA's pip wheels."""
    if not os.path.islink(link):
        return False
    target = os.readlink(link)
    return (os.sep + "nvidia" + os.sep) in target and \
        os.path.basename(target).startswith("libnvrtc.so.")


def _ld_path_has_nvrtc() -> bool:
    dirs = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    return any(d and os.path.exists(os.path.join(d, "libnvrtc.so")) for d in dirs)


def _write_link(link: str, n: Nvrtc) -> None:
    tmp = f"{link}.{os.getpid()}.tmp"
    os.symlink(n.lib, tmp)
    os.replace(tmp, link)


class SetupError(RuntimeError):
    pass


def _plan() -> Nvrtc | None:
    """The wheel NVRTC to use, or None when the user's own setup applies."""
    if os.environ.get("CUDA_PATH") or _system_nvrtc():
        return None
    found = wheel_nvrtcs()
    if not found:
        return None             # nothing to set up; device creation reports it
    driver = driver_cuda_version()
    n = choose(found, driver)
    if n is None:
        newest = max(found, key=lambda f: f.version)
        raise SetupError(
            f"the NVRTC installed with PyTorch is CUDA {_version(newest.version)}, newer "
            f"than the CUDA {_version(driver)} your NVIDIA driver supports, so the "
            "driver would reject the kernels it compiles. Install an NVRTC that "
            f'matches the driver: pip install "cuda-toolkit[nvrtc,cudart]=='
            f'{_version(driver)}.*" (pip reports a conflict because torch asks for '
            f"{_version(newest.version)}, but installs it, and torch keeps working), or "
            "update the driver")
    return n


def configure() -> str | None:
    """Point SlangPy at a usable NVRTC before the device is created; returns
    what is used, or None when the user's own setup (CUDA_PATH or a system
    CUDA Toolkit) applies."""
    n = _plan()
    if n is None:
        return None
    shim = _shim(n)
    if not _ld_path_has_nvrtc():
        link = slangpy_link()
        if _is_ours(link):
            if os.path.realpath(link) != os.path.realpath(n.lib):
                try:            # consent was given; keep our own link current
                    _write_link(link, n)
                except OSError as e:
                    raise SetupError(f"could not update {link} ({e}); run {_SETUP}") from e
        else:
            raise SetupError(
                "SlangVision needs a one-time setup in this environment to use the "
                f"NVRTC installed with PyTorch: run `{_SETUP}`. It links NVRTC "
                f"{_version(n.version)} into SlangPy's package folder ({link}); "
                "`python -m slangvision undo` removes it. Without it, set "
                f"CUDA_PATH={shim} and LD_LIBRARY_PATH={shim}/lib before starting "
                "Python")
    os.environ["CUDA_PATH"] = shim
    return f"NVRTC {_version(n.version)} from {n.libdir}"


def setup() -> list[str]:
    """`python -m slangvision setup`: link the chosen wheel NVRTC into SlangPy's
    folder, with the user's consent (they ran the command)."""
    n = _plan()
    if n is None:
        why = ("CUDA_PATH is set" if os.environ.get("CUDA_PATH") else
               "a system NVRTC is installed" if _system_nvrtc() else
               "no NVRTC from pip was found (is PyTorch with CUDA installed?)")
        return [f"Nothing to do: {why}."]
    link = slangpy_link()
    if os.path.lexists(link) and not _is_ours(link):
        raise SetupError(f"{link} exists and was not made by SlangVision; leaving it alone")
    _shim(n)
    _write_link(link, n)
    return [f"Linked NVRTC {_version(n.version)} ({n.lib})",
            f"  into SlangPy's package folder: {link}",
            "Undo with: python -m slangvision undo"]


def undo() -> list[str]:
    link = slangpy_link()
    if not os.path.lexists(link):
        return ["Nothing to undo."]
    if not _is_ours(link):
        raise SetupError(f"{link} was not made by SlangVision; leaving it alone")
    os.remove(link)
    return [f"Removed {link}"]


def describe() -> str:
    """One line on the NVRTC and driver in use, for error messages."""
    driver = driver_cuda_version()
    path = os.environ.get("CUDA_PATH")
    where = f"CUDA_PATH={path}" if path else "no CUDA_PATH"
    wheels = ", ".join(f"{_version(n.version)} ({n.libdir})" for n in wheel_nvrtcs())
    return (f"Driver supports CUDA {_version(driver)}; {where}; NVRTC from pip: "
            f"{wheels or 'none'}; system NVRTC: {'yes' if _system_nvrtc() else 'no'}")
