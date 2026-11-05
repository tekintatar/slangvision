# Roadmap

- **Phase 1** ✅ (2026-11-03, first developed as `slangimage.augment`):
  torchvision-style pipelines compiled into fused Slang kernels, random
  values per image (`same_on_batch` optional), in-place reads of batches or
  lists of decoded images (CHW/HWC), public `ColorOp`/`NeighbourhoodOp` for
  custom ops. Transforms: RandomResizedCrop, Resize, CenterCrop,
  RandomHorizontalFlip, ColorJitter, RandomApply, RandomGrayscale,
  GaussianBlur, RandomSolarize, Normalize. CUDA only.
- **Since 0.1.0** (unreleased): bfloat16 output, bicubic resizing,
  RandomCrop (padding modes), Pad, RandomVerticalFlip, RandomChoice,
  RandomErasing, MixUp/CutMix, RandomRotation/RandomAffine/RandomPerspective
  (a warp pass), RandomInvert/RandomPosterize/RandomAdjustSharpness/
  RandomAutocontrast/RandomEqualize (torchvision's float behaviour; per-image
  statistics from pre-passes), torchvision's and timm's RandAugment. Next:
  the other policies below.
- **Phase 2 — policies**: torchvision's RandAugment ✅ (2026-11-05), timm's
  RandAugment ✅ (2026-11-05, Pillow's ops on float values), then TrivialAugmentWide and
  AutoAugment, as per-image choices among the same ops (geometric ones
  through the warp pass). Float behaviour, decided 2026-11-05: torchvision's uint8 path cannot
  be matched exactly anyway (its uint8 resize differs); most ops differ by
  at most 1/255, posterize by its rounding scheme, and equalize can move a
  value by a level where the two quantize an input differently. timm's SolarizeAdd comes with
  timm's policies. A rounding-to-uint8 mode only if someone needs exact uint8
  parity.
- **Phase 3 — targets**, only if users ask: masks, boxes and keypoints with
  the same per-image values.
