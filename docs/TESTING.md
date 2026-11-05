# Test evidence

Every transform is checked image by image against the torchvision
`transforms.v2.functional` call it mirrors:
`slangvision.testing.max_reference_error` replays each image's sampled
values through those calls. Inputs are photo-like uint8 images of nine
sizes, including extreme aspect ratios (40x300, 300x40, 23x17) that exercise
RandomResizedCrop's centre-crop fallback. Budgets are 1e-4 for float32
output and 4e-3 for float16 output, in normalized units.

| transform | evidence (`tests/test_pipelines.py`) |
|---|---|
| RandomResizedCrop | reference: recipes, non-square size, custom scale/ratio, `antialias=False`, fallback sizes; bicubic with and without antialias, small scale, in a full recipe |
| Resize | reference: `(h, w)`, shorter side + CenterCrop, `antialias=False`; bicubic upscale and downscale with and without antialias |
| CenterCrop | reference: alone, after Resize, larger than the image (zero padding), after Pad |
| RandomCrop | reference on 64 CIFAR-sized images (24-40 px), so padding shows in most crops: CIFAR (padding 4), reflect, symmetric, edge and reflect with `pad_if_needed`, per-side padding with RGB fill, three padding modes chained, two RandomCrops, after Resize + Pad and RandomResizedCrop + Pad; offsets reach the padding on all four sides; per-image positions and `same_on_batch`; too-large crops and too-wide mirror padding raise; mirror padding at torchvision's limits |
| Pad | reference: constant fill on a batch, reflect before CenterCrop, edge before RandomCrop, uneven padding before flips |
| RandomVerticalFlip | reference: `p=1` alone and after uneven padding, random with RandomCrop and RandomHorizontalFlip |
| RandomChoice | reference: ThreeAugment choice (grayscale, solarize, blur), contrast or blur (contrast mean after the choice), weighted, inside RandomApply, nested RandomChoice; sampled frequencies follow `p`; `same_on_batch` gives one choice |
| RandomHorizontalFlip | reference: `p=1`, two flips, before, after and between warps, after your NeighbourhoodOp with a warp between |
| RandomRotation | reference (bilinear): on resized crops, about an off-centre `center`, between flips with ColorJitter after it, CIFAR-style after RandomCrop, inside RandomChoice after a perspective; nearest: fewer than 1e-3 of values differ (the test's bound; measured separately: about 1e-5 of pixels, where torchvision's float32 grid rounds positions near pixel boundaries differently); angles stay in range; `same_on_batch` shares the angle |
| RandomAffine | reference (bilinear): translate, scale, x/y shear and RGB fill; full-turn rotation with scale, shear and `center` on an odd-sized image; inside RandomApply before a blur and a flip; nearest as for RandomRotation; whole-pixel translations rounded (not truncated: the share of 0) within torchvision's bounds, scale and shear in range |
| RandomPerspective | reference (bilinear, so the corners' coefficients are torchvision's least squares): `p=0.7`, full distortion with fill 0.5, after a blur and before RandomErasing; nearest as for RandomRotation; corner positions span exactly torchvision's bounds; `p` rate |
| ColorJitter | reference: each factor alone, all four with random order, inside RandomApply, after a blur (mean of the blurred image) |
| RandomApply | reference: colour ops, one blur |
| RandomGrayscale | reference |
| GaussianBlur | reference: square and `(kx, ky)` kernels, fixed and random sigma, two blurs in one pipeline |
| RandomSolarize | reference off the threshold; values exactly on it may differ (checked to be on it) |
| Normalize | reference (in every recipe) |
| MixUp, CutMix | draws identical to torchvision's `make_params` under the same seeds (CPU); images (1e-6) and labels (exact) equal to torchvision's `v2.MixUp`/`v2.CutMix` on CUDA; float16 and bfloat16 outputs; input errors and empty batches |
| MixUpCutMix | kernel output equals the mixing its parameters describe in batch, pair and elem modes; partners mirrored; CutMix label weight from the clipped box; timm's smoothed labels; switch/prob rates and Beta mean; `cutmix_minmax` box sizes |
| RandomErasing | reference: constant and per-channel fills after Normalize, without Normalize, two boxes in order, random fill (box compared, noise skipped); noise mean 0 and std 1 with a normal tail, different per image; box area, aspect and rate follow torchvision's sampler; `same_on_batch` gives one box |
| bfloat16 output | ImageNet and SimCLR recipes within one bfloat16 step of the float32 reference; the one-pass ImageNet recipe rounds like torch for more than 95% of values |
| RandomInvert, RandomPosterize | reference: exact on images that are not resampled (posterize at 0, 3 and 8 bits); after RandomResizedCrop and ColorJitter, posterize off its level edges (under 2e-3 of values; measured about 5e-6 of pixels) |
| RandomAdjustSharpness | reference: factors 0 and 2.5 exact on unresampled images; bicubic overshoot clamped at the border; after a crop and ColorJitter, inside RandomApply, inside RandomChoice after a blur |
| RandomAutocontrast | reference: exact on unresampled images, flat/two-valued/ramp images; after ColorJitter (the min/max read after a contrast change), in RandomChoice after a blur; `p=0` leaves images untouched |
| RandomEqualize | reference: exact on unresampled images and on flat (PIL's step 0), two-valued, dominant-value and ramp images; after a crop, ColorJitter and a blur, inside RandomChoice (only some images choose it), off quantization edges (under 2e-3 of values; measured about 3e-4 of pixels) |
| RandAugment | reference (torchvision's own op functions and magnitude tables, fed the drawn op and sign): one op on 120 unresampled images, every op drawn: exact with bilinear warps (magnitude 9, and 30 with RGB fill); nearest (magnitude 9, 0 with 5 bins) within 1e-3 of values, at magnitude 20 with fill 0.5 within 5e-3 (its shear puts whole rows on half pixels, exact ties); equalize and autocontrast after it; two and three ops in a recipe between flips (under 3e-3 of values, edge cases of step ops and nearest ties); float16 output (under 5% of values, 90% within float16 rounding); op frequencies uniform and signs even per image; `same_on_batch` shares the ops |
| ColorOp, NeighbourhoodOp | reference: user-defined gamma and box blur (reflect, clamp), helper-name isolation, compile-error message |

Also on CUDA: batch tensor, CHW list, HWC list and a non-contiguous HWC view
give identical output; dtype, shape, empty input and input-error messages;
`torch.manual_seed` repeats a run; `same_on_batch` gives identical values;
sampled RandomResizedCrop areas and aspects, flip rates, ColorJitter ranges
and all 24 op orders match torchvision's sampler statistically; SlangPy's
SlangPy's Python torch bridge is allowed only for slangvision's own calls
(the process-wide flag is restored afterwards). With `slangpy-torch`
installed the same test also compares the Python and compiled bridges'
output; without it (the 2026-11-04 run below) both calls use the Python
bridge.
`tests/test_api.py` checks pipeline-structure rules, argument
validation and custom-op declarations without a GPU or torch.

## Validation records

Record the GPU, driver, Python, torch, torchvision and SlangPy versions and
the pass counts for every validation; never present an older record as
validation of newer code.

- 2026-11-03, as `slangimage.augment` (before the move to this package):
  NVIDIA H100 NVL, driver 580 (CUDA 13.0), Python 3.12, torch 2.13.0+cu130,
  torchvision 0.28.0, SlangPy 0.42.0: the 41 CUDA tests and 34 GPU-free
  tests passed. A fresh environment with SlangPy 0.43.1 and no
  `slangpy-torch` ran on SlangPy's Python torch bridge: an ImageNet recipe
  matched torchvision to 9.3e-5 (float32, normalized units).
- 2026-11-04, slangvision 0.1.0 (the wheel built from this repo, after the
  move and the split of the shader library into topic files): NVIDIA RTX
  PRO 4500 Blackwell, driver 580.126.16 (CUDA 13.0), Ubuntu 24.04, Python
  3.12.3, torch 2.13.0+cu130, torchvision 0.28.0, no `slangpy-torch` (SlangPy's
  Python torch bridge). After the review fixes, `scripts/cuda_box/run_tests.sh`:
  84 passed (45 CUDA, 39 GPU-free) with SlangPy 0.42.0, and 84 passed in a
  fresh environment where a plain install resolved SlangPy 0.43.1. The
  Python examples in README.md and docs/GUIDE.md ran. Benchmark: see the
  header of `benchmarks/compare_pipelines.py`.
- 2026-11-04, same host, two limits checked:
  - 32-bit indexing: SlangPy addresses float32 tensors beyond 4 GiB
    correctly. At the largest allowed batch (14,266 images of 224x224,
    2^31 - 51,200 values, 8 GiB of float32 output), every image matched
    the single-image result exactly, for one-pass pipelines and for a
    three-pass blur pipeline with 8 GiB intermediates. One more image is
    rejected with the 32-bit message, so the limit is in values, not bytes.
  - NVRTC: torch 2.13.0+cu130 from PyPI depends on `cuda-toolkit` 13.0.3,
    which installs NVRTC, cudart and `cuda_runtime.h` under
    `site-packages/nvidia/cu13`. With no system CUDA and no extra pip
    install, pipelines ran after the README's `libnvrtc.so` link plus both
    `CUDA_PATH` and `LD_LIBRARY_PATH`; leaving out any of those three failed
    with SlangVision's NVRTC error. (This manual setup is what the automatic
    NVRTC setup below now does.)
- 2026-11-04, NVRTC setup with consent (`python -m slangvision setup`),
  same host, fresh environments, no `CUDA_PATH` or `LD_LIBRARY_PATH`, no CUDA
  Toolkit:
  - torch 2.13.0 (NVRTC 13.0): before `setup` the first call stops and asks
    for it (with the environment-variable alternative, which also ran);
    after it, 95 passed; after `undo` the first call asks again.
  - torch 2.8.0 + torchvision 0.23 (CUDA 12.8 wheels, the older
    `nvidia/cuda_nvrtc` layout): `setup` linked NVRTC 12.8 and pipelines ran
    (the full suite passed, 90, in the earlier automatic-link version).
  - torch 2.15.0 + torchvision 0.29.2 (NVRTC 13.2, newer than the driver's
    13.0): the first call stops with the version message and the
    `cuda-toolkit` 13.0 command. After running it (pip reports a dependency
    conflict but installs), `setup` linked 13.0, the full suite passed (90,
    automatic-link version) and a bf16 training step and `torch.compile`
    still ran.
  - With the box scripts' own `CUDA_PATH`: 95 passed and no link was made.
- 2026-11-04, bfloat16 output, same host, torch 2.13.0, SlangPy 0.42.0 (the
  box scripts' own `CUDA_PATH`): 97 passed. A separate check wrote 4,194,304
  values (including rounding ties, ±inf, NaN and values near the maximum)
  through the int16 view on SlangPy 0.42.0 and 0.43.1: identical to
  `.to(torch.bfloat16)`. Batch 256 at 224x224: ImageNet recipe 1.74 ms for
  float16 and bfloat16, 1.76 ms for float32; SimCLR 3.11, 3.12 and 3.31 ms.
- 2026-11-04, bicubic resizing, RandomCrop/Pad/RandomVerticalFlip/
  RandomChoice and the review fixes, same host and stack (box scripts'
  `CUDA_PATH`): 145 passed. Benchmark: header of
  `benchmarks/compare_pipelines.py`, raw data in
  `benchmarks/results/rtx_pro_4500_2026-11-04.json`.
- 2026-11-05, RandomErasing and MixUp/CutMix/MixUpCutMix added, same host
  and stack (box scripts' `CUDA_PATH`): 182 passed (the GPU-free mixing
  tests also run on CPU torch in CI).
- 2026-11-05, RandomRotation/RandomAffine/RandomPerspective (the warp
  pass) and the review fixes, same host and stack (box scripts' `CUDA_PATH`): 216 passed.
  Benchmark: `rotate` recipe, raw data in
  `benchmarks/results/rtx_pro_4500_rotate_2026-11-05.json`.
- 2026-11-05, RandomInvert/RandomPosterize/RandomAdjustSharpness/
  RandomAutocontrast/RandomEqualize and statistic pre-passes by kind, same
  host and stack (box scripts' `CUDA_PATH`): 238 passed. Costs in the
  ImageNet recipe measured with an ad-hoc script (median of 30 synced calls,
  batch 256 of 512x384 photo-like images). Review fixes (sharpness clamps
  its border; float16 step-op rates measured after a blur): 239 passed.
- 2026-11-05, RandAugment, same host and stack (box scripts' `CUDA_PATH`):
  256 passed; with the review's added tests 260 passed (the nearest
  mismatches at magnitude 20 checked to be ShearX rows and ShearY columns
  on exact half pixels only). Benchmark: `randaug` recipe, raw data in
  `benchmarks/results/rtx_pro_4500_randaug_2026-11-05.json`.
