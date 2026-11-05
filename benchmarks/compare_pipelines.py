# benchmarks/compare_pipelines.py
# Training-augmentation pipelines on a batch: slangvision vs
# torchvision v2 (batched, and one image at a time) vs Kornia. CUDA only.
# Each library gets the same recipe and uint8 input; output is fp16 NCHW.
#
#   pip install torch torchvision kornia
#   python benchmarks/compare_pipelines.py [--batch 256] [--iters 20]
#       [--json out.json]
#
# Recipes (torchvision.transforms.v2 names):
#   imagenet  RandomResizedCrop(224) + flip + ColorJitter(0.4, 0.4, 0.4, 0.1)
#             + Normalize
#   eval      Resize(256) + CenterCrop(224) + Normalize
#   simclr    RandomResizedCrop(224) + flip + RandomApply(ColorJitter, 0.8)
#             + RandomGrayscale(0.2) + RandomApply(GaussianBlur(23), 0.5)
#             + RandomSolarize(0.5, 0.2) + Normalize
#   rotate    imagenet with RandomRotation(15) (nearest, as torchvision's
#             default) after the flip
#   randaug   RandomResizedCrop(224) + flip + RandAugment(2, 9) + Normalize
#             (torchvision runs RandAugment on the uint8 crop, as its
#             reference recipe does)
#   trivialaug, autoaug  RandomResizedCrop(224) + flip + TrivialAugmentWide()
#             or AutoAugment() (ImageNet policy) + Normalize (torchvision on
#             the uint8 crop, as for randaug)
#   timmaug   RandomResizedCrop(224) + flip + timm's RandAugment
#             rand-m9-mstd0.5-inc1 + Normalize, against timm's own
#             create_transform on Pillow images, one CPU core
#   pip install timm (for timmaug)
#   --recipes imagenet,randaug runs a subset
# Input: a batch of photo-like 512x384 uint8 images already on the GPU
# (--input-size HxW changes their size, e.g. 695x1024 for FGVC-sized
# photos), or with --images DIR the first --batch JPEGs under DIR decoded on the GPU as a
# list of their own sizes (then torchvision runs per image only: different
# sizes cannot form one batch). Decoding is not timed. --interpolation
# bicubic switches RandomResizedCrop/Resize in both libraries. Times are the median wall time of synced calls
# and the summed GPU kernel time per call (torch profiler).
#
# Reference run (2026-11-03, NVIDIA H100 NVL, torch 2.13, torchvision 0.28,
# Kornia 0.9.0, SlangPy 0.42; batch 256), wall ms (GPU ms). Measured with
# the same kernels while they shipped as slangimage.augment:
#
#   recipe     slangvision   torchvision batch   torchvision per image   Kornia fp16
#   imagenet   1.31 (1.05)   7.32 (7.05)         389 (50.9)              16.6 (6.45)
#   eval       0.68 (0.42)   2.90 (2.87)         51.9 (8.55)             -
#   simclr     2.30 (2.00)   7.6-12.6 (13.6-14.2) 390 (53.1)              22.3 (10.8)
#
# Second run (2026-11-04, slangvision 0.1.0, NVIDIA RTX PRO 4500 Blackwell,
# driver 580, torch 2.13, torchvision 0.28, Kornia 0.9, SlangPy 0.42;
# raw data in benchmarks/results/rtx_pro_4500_2026-11-04.json):
#
#   recipe     slangvision   torchvision batch   torchvision per image   Kornia fp16
#   imagenet   1.71 (1.47)   19.9 (20.1)         202 (35.7)              19.2 (14.0)
#   eval       0.89 (0.67)   5.61 (5.85)         28.6 (7.64)             -
#   simclr     3.16 (2.89)   20.5 (27.1)         230 (41.7)              30.3 (23.8)
#
# rotate, added 2026-11-05 on the same box (rtx_pro_4500_rotate_2026-11-05.json;
# slangvision's other recipes matched the run above within 3%, torchvision's
# per-image loop within 8%, its batched SimCLR GPU time varied more):
#
#   rotate     1.84 (1.62)   23.5 (23.2)         275 (43.2)              -
#
# randaug, added 2026-11-05 (rtx_pro_4500_randaug_2026-11-05.json):
#
#   randaug    2.88 (2.58)   6.32 (6.48)         111 (15.2)              -
#
# trivialaug and autoaug, added 2026-11-05 (rtx_pro_4500_policies_2026-11-05.json;
# slangvision's imagenet and randaug in the same run within 2% of the rows
# above, torchvision's per-image loop within 11%, its batched randaug 11%):
#
#   trivialaug 1.89 (1.60)   4.80 (4.54)         81.9 (10.3)             -
#   autoaug    3.09 (2.76)   5.18 (4.40)         94.9 (14.1)             -
#
# timmaug, added 2026-11-05 (rtx_pro_4500_timmaug_2026-11-05.json; imagenet
# and randaug in the same run within 2% of the rows above, torchvision's
# per-image loop within 12%). timm 1.0.31, Pillow 13.0.0, one core of an AMD
# EPYC 9335 (the host has 16; 334 ms in an earlier run):
#
#   recipe     slangvision   timm per image (Pillow, 1 CPU core)
#   timmaug    2.87 (2.52)   310
#
# Review fixes, 2026-11-05 (rtx_pro_4500_review_perf_2026-11-05.json; before
# and after interleaved, two runs each, within 2% of each other): statistic
# pre-passes skip images without the op before reading pixels, GaussianBlur
# weights come from the sampler, and pass-0 statistics read a stored float32
# resample with bicubic or inputs of 5x the output's area or more. slangvision,
# wall ms (GPU ms), before -> after:
#
#   recipe      384x512 bilinear          695x1024 bilinear       695x1024 bicubic
#   imagenet    1.68 (1.47) -> 1.69 (1.48) 2.89 (2.65) -> 2.07 (1.88) 8.51 (8.2) -> 3.97 (3.93)
#   eval        -                         1.45 (1.25) -> 1.45 (1.25) 3.72 (3.61) -> 3.72 (3.61)
#   simclr      3.05 (2.78) -> 2.57 (2.35) 4.10 (4.06) -> 3.03 (2.83) 9.18 (9.18) -> 4.96 (5.09)
#   randaug     2.90 (2.58) -> 2.29 (1.96)
#   trivialaug  1.89 (1.60) -> 1.58 (1.30)
#   autoaug     3.07 (2.74) -> 2.49 (2.17)
#   timmaug     2.85 (2.51) -> 2.22 (1.87)
#
# Final code after the review fixes, 2026-11-05 (rtx_pro_4500_2026-11-05.json;
# the README's table), same box:
#
#   recipe     slangvision   torchvision batch   torchvision per image   Kornia fp16
#   imagenet   1.66 (1.47)   20.2 (20.2)         203 (35.7)              19.3 (14.0)
#   eval       0.87 (0.66)   5.60 (5.80)         27.6 (7.65)             -
#   simclr     2.54 (2.30)   20.5 (26.5)         221 (40.4)              29.7 (23.8)
#
# and on 256 FGVC-Aircraft JPEGs (--images; rtx_pro_4500_fgvc_aircraft_
# {bilinear,bicubic}_after_review_2026-11-05.json), slangvision bilinear /
# bicubic: imagenet 2.34 / 4.36, eval 1.76 / 4.27, simclr 4.03 / 5.34
# (before: 3.24 / 9.40, 1.79 / 4.30, 4.70 / 11.1).
#
# Honest read:
# - torchvision's batch mode draws ONE random setting for the whole batch, so
#   it does less work than the per-image pipelines; its per-image loop is the
#   like-for-like comparison. slangvision and Kornia draw per image.
# - With one setting per batch, torchvision's cost swings from call to call
#   (RandomApply blurs all 256 images or none): its simclr median was 7.6 ms
#   in one run and 12.6 ms in another.
# - Kornia's RandomResizedCrop does not antialias and its colour ops convert
#   fp16 to fp32 internally; ours antialias like torchvision.
# - In training, decoding usually limits throughput; these numbers are the GPU
#   time augmentation takes from the model. See docs/GUIDE.md.

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import slangvision as A

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def photo_batch(n: int, h: int = 384, w: int = 512) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    low = torch.rand(n, 3, h // 16, w // 16, device="cuda", generator=g)
    x = torch.nn.functional.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)
    x = x + 0.04 * torch.randn(n, 3, h, w, device="cuda", generator=g)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()


def ours(interp: str = "bilinear") -> dict:
    return {
        "imagenet": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                               A.RandomHorizontalFlip(),
                               A.ColorJitter(0.4, 0.4, 0.4, 0.1), A.Normalize(MEAN, STD)]),
        "eval": A.Compose([A.Resize(256, interpolation=interp), A.CenterCrop(224),
                           A.Normalize(MEAN, STD)]),
        "simclr": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                             A.RandomHorizontalFlip(),
                             A.RandomApply([A.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                             A.RandomGrayscale(p=0.2),
                             A.RandomApply([A.GaussianBlur(23)], p=0.5),
                             A.RandomSolarize(0.5, p=0.2), A.Normalize(MEAN, STD)]),
        "rotate": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                             A.RandomHorizontalFlip(), A.RandomRotation(15),
                             A.ColorJitter(0.4, 0.4, 0.4, 0.1), A.Normalize(MEAN, STD)]),
        "randaug": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                              A.RandomHorizontalFlip(), A.RandAugment(),
                              A.Normalize(MEAN, STD)]),
        "trivialaug": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                                 A.RandomHorizontalFlip(), A.TrivialAugmentWide(),
                                 A.Normalize(MEAN, STD)]),
        "autoaug": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                              A.RandomHorizontalFlip(), A.AutoAugment(),
                              A.Normalize(MEAN, STD)]),
        "timmaug": A.Compose([A.RandomResizedCrop(224, interpolation=interp),
                              A.RandomHorizontalFlip(),
                              A.TimmRandAugment("rand-m9-mstd0.5-inc1", interpolation=interp),
                              A.Normalize(MEAN, STD)]),
    }


def timm_per_image(x: torch.Tensor, interp: str):
    """timm's training transform (crop, flip, RandAugment, ToTensor,
    Normalize) on Pillow copies of the batch, one image at a time on one CPU
    core, as one DataLoader worker runs it. Converting to Pillow is not timed."""
    import timm.data.auto_augment as aa
    from PIL import Image
    from timm.data import create_transform

    def posterize(img, bits, **_):
        # timm asks for 0 bits at magnitude 10, which Pillow 13 rejects
        # (ValueError) and Pillow 12 turned black: keep Pillow 12's result
        return Image.new(img.mode, img.size) if bits == 0 else aa.posterize(img, bits)

    for name in ("Posterize", "PosterizeIncreasing", "PosterizeOriginal"):
        aa.NAME_TO_OP[name] = posterize
    t = create_transform(224, is_training=True, auto_augment="rand-m9-mstd0.5-inc1",
                         interpolation=interp, mean=MEAN, std=STD)
    pils = [Image.fromarray(im.permute(1, 2, 0).cpu().numpy()) for im in x]

    def run(_: torch.Tensor) -> torch.Tensor:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return torch.stack([t(im) for im in pils])
        finally:
            torch.set_num_threads(threads)
    return run


def torchvision_recipes(interp: str = "bilinear") -> dict:
    from torchvision.transforms import InterpolationMode, v2

    mode = InterpolationMode(interp)
    tail = [v2.ToDtype(torch.float16, scale=True), v2.Normalize(list(MEAN), list(STD))]
    crop = [v2.RandomResizedCrop(224, interpolation=mode, antialias=True),
            v2.RandomHorizontalFlip()]
    return {
        "imagenet": v2.Compose(crop + [v2.ColorJitter(0.4, 0.4, 0.4, 0.1)] + tail),
        "eval": v2.Compose([v2.Resize(256, interpolation=mode, antialias=True),
                            v2.CenterCrop(224)] + tail),
        "simclr": v2.Compose(crop + [v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                                     v2.RandomGrayscale(p=0.2),
                                     v2.RandomApply([v2.GaussianBlur(23)], p=0.5),
                                     v2.RandomSolarize(128, p=0.2)] + tail),
        "rotate": v2.Compose(crop + [v2.RandomRotation(15), v2.ColorJitter(0.4, 0.4, 0.4, 0.1)]
                             + tail),
        "randaug": v2.Compose(crop + [v2.RandAugment()] + tail),
        "trivialaug": v2.Compose(crop + [v2.TrivialAugmentWide()] + tail),
        "autoaug": v2.Compose(crop + [v2.AutoAugment()] + tail),
    }


def kornia_recipes() -> dict:
    try:
        import kornia.augmentation as K
    except ImportError:
        return {}
    norm = K.Normalize(mean=torch.tensor(MEAN), std=torch.tensor(STD))
    crop = [K.RandomResizedCrop((224, 224)), K.RandomHorizontalFlip()]
    return {
        "imagenet": K.AugmentationSequential(*crop, K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0),
                                             norm).cuda(),
        "simclr": K.AugmentationSequential(
            *crop, K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8), K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((23, 23), (0.1, 2.0), p=0.5),
            K.RandomSolarize(thresholds=0.0, additions=0.0, p=0.2), norm).cuda(),
    }


def jpeg_list(folder: str, n: int) -> list:
    """The first n JPEGs under folder (sorted), decoded on the GPU (not timed)."""
    from torchvision.io import decode_jpeg, read_file

    files = sorted(Path(folder).rglob("*.jpg"))[:n]
    if len(files) < n:
        raise SystemExit(f"found {len(files)} JPEGs under {folder}, need {n}")
    return decode_jpeg([read_file(str(f)) for f in files], device="cuda")


def wall_ms(fn, x, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def gpu_ms(fn, x, iters: int) -> tuple[float, int]:
    from torch.profiler import ProfilerActivity, profile

    fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
    events = [e for e in prof.events()
              if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time > 0]
    return sum(e.device_time for e in events) / iters / 1e3, round(len(events) / iters)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare augmentation pipelines on one GPU batch.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--interpolation", default="bilinear", choices=["bilinear", "bicubic"])
    ap.add_argument("--input-size", default="384x512",
                    help="HxW of the generated photo-like images (ignored with --images)")
    ap.add_argument("--images", help="folder of JPEGs: time on the first --batch of them, "
                                     "decoded on the GPU as a list of their own sizes")
    ap.add_argument("--recipes", help="comma-separated subset of the recipes")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    h, w = (int(v) for v in args.input_size.split("x"))
    x = jpeg_list(args.images, args.batch) if args.images else photo_batch(args.batch, h, w)
    tv = torchvision_recipes(args.interpolation)
    ko = {} if args.images or args.interpolation != "bilinear" else kornia_recipes()
    rows = []
    wanted = args.recipes.split(",") if args.recipes else None
    for recipe, pipe in ours(args.interpolation).items():
        if wanted and recipe not in wanted:
            continue
        methods = {"slangvision": pipe}
        if recipe == "timmaug":
            methods["timm per image (Pillow, 1 CPU core)"] = timm_per_image(x, args.interpolation)
        if recipe in tv and not args.images:    # images of different sizes cannot form one batch
            methods["torchvision batch (one setting)"] = tv[recipe]
        if recipe in tv:
            methods["torchvision per image"] = lambda b, t=tv[recipe]: torch.stack([t(im) for im in b])
        if recipe in ko:
            methods["Kornia fp16 per image"] = lambda b, k=ko[recipe]: k(b.half() / 255.0)
        for name, fn in methods.items():
            loop = "per image" in name and "Kornia" not in name
            ms = wall_ms(fn, x, iters=3 if loop else args.iters, warmup=1 if loop else 5)
            g, kernels = gpu_ms(fn, x, iters=1 if loop else 5)
            row = {"recipe": recipe, "method": name, "wall_ms": round(ms, 2),
                   "gpu_ms": round(g, 2), "kernels": kernels,
                   "images_per_s": round(len(x) / ms * 1e3)}
            rows.append(row)
            print(f"{recipe:9} {name:32} {ms:9.2f} ms  gpu {g:8.2f} ms  {kernels:6} kernels")
    if args.json:
        meta = {"date": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__, "slangvision": A.__version__,
                "python": platform.python_version(), "batch": args.batch,
                "interpolation": args.interpolation,
                "input_size": None if args.images else args.input_size,
                "images": args.images and Path(args.images).name}
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
