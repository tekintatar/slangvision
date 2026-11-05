# Roadmap

What is done is in the [changelog](../CHANGELOG.md); this page lists what may
come next and what was decided against.

- **Next, small:** RandomBrightnessContrast and RandomGamma as built-ins if
  people ask (today they are a few lines as a `ColorOp`; see the guide's
  "Coming from Albumentations"). Albumentations' other popular ops map onto
  existing transforms, except CLAHE and MedianBlur.
- **Performance, measured first:** separable resampling for large
  downscales (bicubic from large photos), and fewer statistics passes in
  the policies when no image picks contrast, autocontrast or equalize.
- **Targets**, if users ask: masks, boxes and keypoints transformed with the
  same per-image values, variable output sizes (scale jitter, letterbox).
  This is what detection, segmentation and most Albumentations users need.
- **Maybe:** AugMix (a different pass structure), ElasticTransform, timm's
  AutoAugment policies (per-entry level functions on the TimmRandAugment
  pass).
- **Decided against for now:** an Albumentations compatibility mode
  (SlangVision keeps torchvision's semantics); a uint8 rounding mode between
  ops (torchvision's uint8 path cannot be matched exactly anyway: most ops
  differ by at most 1/255); BasicSR's filter2D on the GPU (cuDNN is already
  fast). Batched augmentation stays in this package; image-processing
  operators are SlangImage's.
