# tests/test_cuda_setup.py — the automatic NVRTC setup (slangvision/_cuda.py)
# on a fake tree of NVIDIA pip wheels: no GPU, NVRTC or torch needed.

from __future__ import annotations

import os
import sys
import types

import pytest

from slangvision import _cuda


def _wheel(root, layout, major):
    """A fake NVIDIA wheel layout with an NVRTC library and cuda_runtime.h."""
    if layout == "cu13":
        lib, inc = root / f"cu{major}" / "lib", root / f"cu{major}" / "include"
    else:
        lib, inc = root / "cuda_nvrtc" / "lib", root / "cuda_runtime" / "include"
    lib.mkdir(parents=True)
    inc.mkdir(parents=True, exist_ok=True)
    (lib / f"libnvrtc.so.{major}").write_text("")
    (lib / f"libnvrtc-builtins.so.{major}.0").write_text("")
    (inc / "cuda_runtime.h").write_text("")
    return str(lib / f"libnvrtc.so.{major}")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Fake nvidia/ and slangpy packages, cache folder, driver and NVRTC versions."""
    nvidia = tmp_path / "site" / "nvidia"
    nvidia.mkdir(parents=True)
    slangpy_dir = tmp_path / "site" / "slangpy"
    slangpy_dir.mkdir()
    monkeypatch.setitem(sys.modules, "nvidia", types.SimpleNamespace(__path__=[str(nvidia)]))
    monkeypatch.setitem(sys.modules, "slangpy",
                        types.SimpleNamespace(__file__=str(slangpy_dir / "__init__.py")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    # setenv first so the undo also removes the CUDA_PATH configure() sets
    monkeypatch.setenv("CUDA_PATH", "")
    monkeypatch.delenv("CUDA_PATH")
    monkeypatch.setattr(_cuda, "_system_nvrtc", lambda: False)
    versions = {}
    monkeypatch.setattr(_cuda, "nvrtc_version", lambda path: versions.get(path))
    state = types.SimpleNamespace(root=nvidia, slangpy=slangpy_dir, versions=versions,
                                  driver=(13, 0))
    monkeypatch.setattr(_cuda, "driver_cuda_version", lambda: state.driver)
    return state


def test_choose_takes_the_newest_copy_the_driver_supports():
    a = _cuda.Nvrtc("a", "i", "d", (12, 8))
    b = _cuda.Nvrtc("b", "i", "d", (13, 0))
    c = _cuda.Nvrtc("c", "i", "d", (13, 2))
    assert _cuda.choose([a, b, c], (13, 0)) is b
    assert _cuda.choose([c], (13, 0)) is None
    assert _cuda.choose([a, c], None) is c


def test_finds_cuda_13_and_cuda_12_wheel_layouts(fake):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    fake.versions[_wheel(fake.root, "cu12", 12)] = (12, 8)
    assert sorted(n.version for n in _cuda.wheel_nvrtcs()) == [(12, 8), (13, 0)]


def test_configure_asks_for_consent_before_touching_slangpy(fake):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    with pytest.raises(_cuda.SetupError, match="python -m slangvision setup"):
        _cuda.configure()
    assert not os.path.lexists(fake.slangpy / "libnvrtc.so")
    assert "CUDA_PATH" not in os.environ


def test_setup_then_configure(fake):
    lib = _wheel(fake.root, "cu13", 13)
    fake.versions[lib] = (13, 0)
    assert "Linked NVRTC 13.0" in _cuda.setup()[0]
    assert os.path.realpath(fake.slangpy / "libnvrtc.so") == os.path.realpath(lib)
    assert "NVRTC 13.0" in _cuda.configure()
    shim = os.environ["CUDA_PATH"]
    assert os.path.realpath(os.path.join(shim, "lib", "libnvrtc.so")) == os.path.realpath(lib)
    assert os.path.isfile(os.path.join(shim, "include", "cuda_runtime.h"))
    del os.environ["CUDA_PATH"]
    assert "NVRTC 13.0" in _cuda.configure()        # second run reuses both


def test_configure_keeps_its_own_link_current(fake):
    old = _wheel(fake.root / "old", "cu13", 13)
    new = _wheel(fake.root, "cu13", 13)
    fake.versions[new] = (13, 0)
    os.symlink(old, fake.slangpy / "libnvrtc.so")    # ours, from an earlier torch
    _cuda.configure()
    assert os.path.realpath(fake.slangpy / "libnvrtc.so") == os.path.realpath(new)


def test_never_replaces_a_libnvrtc_it_did_not_make(fake):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    (fake.slangpy / "libnvrtc.so").write_text("someone else's")
    with pytest.raises(_cuda.SetupError, match="not made by SlangVision"):
        _cuda.setup()
    with pytest.raises(_cuda.SetupError, match="one-time setup"):
        _cuda.configure()
    with pytest.raises(_cuda.SetupError, match="not made by SlangVision"):
        _cuda.undo()
    assert (fake.slangpy / "libnvrtc.so").read_text() == "someone else's"


def test_undo_removes_only_our_link(fake):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    _cuda.setup()
    assert _cuda.undo()[0].startswith("Removed")
    assert not os.path.lexists(fake.slangpy / "libnvrtc.so")
    assert _cuda.undo() == ["Nothing to undo."]


def test_ld_library_path_with_nvrtc_needs_no_link(fake, tmp_path, monkeypatch):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    own = tmp_path / "own"
    own.mkdir()
    (own / "libnvrtc.so").write_text("")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(own))
    assert "NVRTC 13.0" in _cuda.configure()
    assert not os.path.lexists(fake.slangpy / "libnvrtc.so")


def test_configure_rejects_nvrtc_newer_than_the_driver(fake):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 2)
    with pytest.raises(_cuda.SetupError, match=r'cuda-toolkit\[nvrtc,cudart\]==13\.0\.\*'):
        _cuda.configure()
    assert "CUDA_PATH" not in os.environ


def test_configure_leaves_user_setups_alone(fake, monkeypatch):
    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
    assert _cuda.configure() is None
    assert _cuda.setup()[0].startswith("Nothing to do: CUDA_PATH")
    monkeypatch.delenv("CUDA_PATH")
    monkeypatch.setattr(_cuda, "_system_nvrtc", lambda: True)
    assert _cuda.configure() is None
    assert not os.path.lexists(fake.slangpy / "libnvrtc.so")


def test_command_line(fake, capsys):
    from slangvision.__main__ import main

    fake.versions[_wheel(fake.root, "cu13", 13)] = (13, 0)
    assert main([]) == 0 and "Not ready" in capsys.readouterr().out
    assert main(["setup"]) == 0 and "Linked NVRTC 13.0" in capsys.readouterr().out
    assert main([]) == 0 and "will use: NVRTC 13.0" in capsys.readouterr().out
    assert main(["undo"]) == 0
    assert main(["bogus"]) == 2
