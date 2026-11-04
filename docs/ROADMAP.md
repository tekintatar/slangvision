# Roadmap

- **Phase 1** ✅ (2026-11-03, first developed as `slangimage.augment`):
  torchvision-style pipelines compiled into fused Slang kernels, random
  values per image (`same_on_batch` optional), in-place reads of batches or
  lists of decoded images (CHW/HWC), public `ColorOp`/`NeighbourhoodOp` for
  custom ops. Transforms: RandomResizedCrop, Resize, CenterCrop,
  RandomHorizontalFlip, ColorJitter, RandomApply, RandomGrayscale,
  GaussianBlur, RandomSolarize, Normalize. CUDA only.
- **Phase 2 — policies**, if phase 1 is used: complete RandAugment and
  TrivialAugmentWide (their geometric ops through a per-image 3x3 matrix and
  one resample, which also brings RandomAffine/RandomRotation/
  RandomPerspective) and the ops they need (sharpness, posterize, invert,
  autocontrast, equalize), RandomChoice. Decide first whether posterize and
  equalize match torchvision's float or uint8 paths (RandAugment usually
  runs on uint8).
- **Phase 3 — targets**, only if users ask: masks, boxes and keypoints with
  the same per-image values; MixUp/CutMix.
