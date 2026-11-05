# Contributing to SlangVision

Thanks for your interest! SlangVision is pre-alpha and moving quickly, so
please open an issue to discuss anything non-trivial before sending a large
PR.

## Development setup

The CUDA tests need Linux, an NVIDIA GPU and PyTorch with CUDA (NVRTC comes
with it; run `python -m slangvision setup` once, see the README's
*Install*). The GPU-free tests run anywhere SlangPy
installs, including macOS. Python 3.10–3.13; name the interpreter
explicitly when creating the environment.

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # or 3.10, 3.11, 3.13
pip install -e ".[dev]"
pytest                        # CUDA tests skip themselves without a GPU
```

## Checking a change

```bash
scripts/check.sh              # whitespace, ruff, mypy and the full test suite
scripts/check.sh --no-gpu     # only the tests that need no GPU
scripts/check.sh -k blur      # extra arguments go to pytest
```

`tests/test_pipelines.py` needs torch, torchvision and a CUDA GPU and is
skipped otherwise; a skipped module means the kernels were not validated.
`tests/test_api.py` checks pipeline rules, argument validation and custom-op
declarations without a GPU or torch.

GitHub Actions runs `scripts/check.sh --no-gpu` on Python 3.10 and 3.13,
shellcheck and the wheel/sdist build. Hosted runners have no GPU, so run the
CUDA tests before opening a PR and state the GPU, driver and torch,
torchvision and SlangPy versions you tested. Maintainers validate on a
Linux NVIDIA host with the scripts in `scripts/cuda_box/`.

## Adding a transform

1. **Match torchvision's float functional image by image.** Write it in
   `src/slangvision/transforms.py`, as a `ColorOp` or `NeighborhoodOp` (the
   public custom-op API) where it fits, the way `RandomInvert` or
   `RandomSolarize` are; its Slang functions go in the matching file in
   `src/slangvision/shaders/` (a new topic gets a new file; every `.slang`
   file there is loaded). Follow the CUDA hot-loop rules in AGENTS.md (no
   float3/float2).
2. **Export it** from `src/slangvision/__init__.py`: the import and
   `__all__` (both alphabetical).
3. **Give it a reference** in `src/slangvision/testing.py` (`_reference`):
   the torchvision call that replays the transform's per-image random values.
   Built-in transforms use torchvision here, not a `reference` method; that
   method is for users' own ops, which have no torchvision call to replay.
4. **Test it on CUDA** in `tests/test_pipelines.py`: add it to the
   `test_op_matches_torchvision` list (or a recipe list), which compares it
   with its reference image by image, plus anything specific to it. Argument and pipeline-rule checks that need no GPU go in
   `tests/test_api.py`.
5. **Document it** in four places: the transform table of
   [docs/GUIDE.md](docs/GUIDE.md), the transform table in
   [README.md](README.md), [docs/TESTING.md](docs/TESTING.md) (its
   evidence row) and `CHANGELOG.md` under `Unreleased`.

## Releasing

Maintainers release by setting the version in `pyproject.toml`,
`src/slangvision/__init__.py` and `CITATION.cff`, moving the CHANGELOG
entries under a dated version heading, and pushing a `vX.Y.Z` tag on `main`.
The `publish` workflow checks that the tag matches the package version,
builds the archives and uploads them to PyPI through trusted publishing (no
stored token). CI cannot run the CUDA tests, so before tagging run the
full suite on real hardware against the commit you will tag and add its
validation record (commit, GPU, driver, versions, result) to
[docs/TESTING.md](docs/TESTING.md); do not tag a commit without one. `twine check` does not render Markdown, so keep README links
absolute: relative links break on PyPI.

## Pull requests

- Keep PRs focused; one transform or one fix per PR where possible.
- Run `scripts/check.sh` before pushing.
- Update `CHANGELOG.md` under the `Unreleased` heading.
- By contributing you agree your work is licensed under the project's
  [Apache-2.0](LICENSE) license.
