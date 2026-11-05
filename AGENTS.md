# SlangVision — notes for coding agents

GPU training augmentation for PyTorch on CUDA, written in Slang through
SlangPy. Human contributors: start with [CONTRIBUTING.md](CONTRIBUTING.md);
this file adds the conventions that are expensive to rediscover. `CLAUDE.md`
only imports this file, so edit this one. Put machine- or person-specific
notes (interpreter paths, GPU hosts, identities) in your own untracked
`CLAUDE.local.md`, never here.

## Commands

- `pip install -e ".[dev]"` (Linux + NVIDIA for the CUDA tests; the GPU-free
  tests run anywhere SlangPy installs, including macOS).
- **Before finishing a change, run `scripts/check.sh`** (whitespace, ruff,
  mypy, tests; `--no-gpu` without an NVIDIA GPU; extra arguments go to
  pytest). If the CUDA tests did not run, say so instead of implying they
  passed.
- CUDA validation on a remote Linux NVIDIA host: `scripts/cuda_box/deploy.sh`
  rebuilds the host environment from the built wheel, then
  `scripts/cuda_box/run_tests.sh` and `run_bench.sh`. `deploy.sh` syncs with
  `rsync --delete`, so point it only at a disposable host. Never present
  older results as validation of new code.

## Architecture

- `src/slangvision/_compile.py` — `Compose`: checks a pipeline's structure
  (`_plan`), generates one Slang entry point per pass from the transforms'
  snippets (`_source`), appends them to the op library and runs
  them. Inspect generated code with `Compose.source`.
- `src/slangvision/_base.py` — distributions (`Uniform`, `Choice`), the
  parameter-slot binding and the public custom-op bases `ColorOp` and
  `NeighbourhoodOp`.
- `src/slangvision/transforms.py` — the built-in transforms.
- `src/slangvision/testing.py` — torch/torchvision references and
  `max_reference_error`; never imported by training.
- `src/slangvision/_runtime.py` — the CUDA device (one per torch device),
  the input-address table, the module cache and the scoped SlangPy torch
  bridge.
- `src/slangvision/shaders/` — the op library, one file per topic:
  `core.slang` (parameter access, pixel helpers, RNG), `color.slang`,
  `geometry.slang`, `neighbourhood.slang`, `sources.slang`, `warp.slang`. `_runtime.py`
  joins them (core first) into one source, so a new file needs no
  registration and the split has no run-time cost.

## Conventions & gotchas (expensive to rediscover)

- Every transform must match torchvision's float functional image by image.
  Its torch reference lives in `testing.py` (not on the class, which keeps
  only kernel code and random draws) and replays the sampled per-image
  values; `tests/test_pipelines.py` (CUDA) checks each one through
  `testing.max_reference_error`. Pipeline and argument checks that need no
  GPU go in `tests/test_api.py`. Built-in colour ops use the public
  `ColorOp` API where they can.
- Inputs are read in place through device addresses (`uint8_t*` cast from
  int64), so the caller keeps them alive until the GPU has run.
- SlangPy `Tensor` vector subscripts are reversed (x = last dimension) and
  use 32-bit index math: one call handles fewer than 2^31 output values.
- CUDA hot-loop rules: no float3/float2 in kernels (NVRTC spills the structs
  Slang emits; even one `float4(x.rgb, y)` per pixel costs about 2.6x), no
  dynamically indexed local arrays; hoist invariants.
- Kernels run on PyTorch's current CUDA stream through
  `spy.create_torch_device`. Not inside CUDA graph capture (SlangPy queries
  the stream while capturing).
- With DALI, `slangpy` must be imported before DALI builds a pipeline, or
  SlangPy segfaults creating its CUDA device (DALI 2.4, SlangPy 0.42).
- SlangPy devices are thread-affine: pipelines run from one thread.
- bfloat16: Slang compiles `BFloat16`, but SlangPy's torch bridge rejects
  torch.bfloat16 tensors (no bfloat16 in its ScalarType/DataType tables, 0.42
  and 0.43.1). The final pass therefore writes `bf16_bits()` (core.slang,
  round to nearest even like torch) into `out.view(torch.int16)`;
  intermediates stay float16. Switch to `RWTensor<BFloat16, 4>` once SlangPy
  maps torch.bfloat16.
- NVRTC setup (`_cuda.py`): SlangPy's device layer opens
  `$CUDA_PATH/lib/libnvrtc.so`, Slang's compiler `dlopen("libnvrtc.so")`
  (searched in LD_LIBRARY_PATH, the system cache and SlangPy's RUNPATH,
  which includes its package folder). Pip's torch ships only
  `libnvrtc.so.<major>`, so unless CUDA_PATH is set or a system NVRTC loads,
  `configure()` links the newest wheel NVRTC the driver supports into a
  cache folder (set as CUDA_PATH in-process, which works because SlangPy
  reads it at device creation). LD_LIBRARY_PATH cannot be changed
  in-process, so the bare name needs a link in SlangPy's folder: made only
  with the user's consent (`python -m slangvision setup`), kept current by
  `configure()` afterwards, never replacing a file SlangVision did not make
  (`_is_ours`), removed by `undo`. NVRTC newer than the driver's CUDA version
  makes kernels the driver rejects, so it is never chosen. The GPU box
  scripts set CUDA_PATH themselves, so they exercise the user-CUDA_PATH path;
  test the automatic path in an environment without it.
- Stay with plain Slang through SlangPy: no CUDA C++, Triton or
  ahead-of-time kernel builds.

## Where things are documented

- docs/GUIDE.md — the user guide (transforms, rules, decoders, custom ops,
  accuracy, performance, limits).
- docs/TESTING.md — per-transform evidence and validation records; update
  it when adding transforms or tests.
- docs/ROADMAP.md — what is planned next.
- benchmarks/compare_pipelines.py — reference numbers and honest notes in
  its header.
