"""SlangVision: GPU training augmentation for PyTorch, written in Slang.

    import slangvision as A

    train = A.Compose([
        A.RandomResizedCrop(224),
        A.RandomHorizontalFlip(),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = train(images)   # (B, 3, 224, 224) float16 on the current CUDA stream

``images`` is a ``(B, 3, H, W)`` uint8 CUDA tensor or a list of decoded
images of any sizes. Names, parameters and results follow
``torchvision.transforms.v2`` on float images, but random values are drawn
for every image (``same_on_batch=True`` shares them across the batch).
Each pipeline compiles into a few generated Slang kernels. Add your own ops
with :class:`ColorOp` and :class:`NeighbourhoodOp`. Needs PyTorch with CUDA.
See docs/GUIDE.md.
"""

__version__ = "0.1.0"

# Load SlangPy (and, through it, torch) at import: if DALI loads its native
# libraries before SlangPy, SlangPy crashes creating its CUDA device, so
# importing slangvision first must be enough.
from . import _runtime  # noqa: F401
from ._base import Choice, ColorOp, NeighbourhoodOp, Uniform
from ._compile import Compose
from .mixing import CutMix, MixUp, MixUpCutMix
from .transforms import (
    CenterCrop,
    ColorJitter,
    GaussianBlur,
    Normalize,
    Pad,
    RandAugment,
    RandomAdjustSharpness,
    RandomAffine,
    RandomApply,
    RandomAutocontrast,
    RandomChoice,
    RandomCrop,
    RandomEqualize,
    RandomErasing,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomInvert,
    RandomPerspective,
    RandomPosterize,
    RandomResizedCrop,
    RandomRotation,
    RandomSolarize,
    RandomVerticalFlip,
    Resize,
)

__all__ = [
    "__version__",
    "CenterCrop",
    "Choice",
    "ColorJitter",
    "ColorOp",
    "CutMix",
    "Compose",
    "GaussianBlur",
    "MixUp",
    "MixUpCutMix",
    "NeighbourhoodOp",
    "Normalize",
    "Pad",
    "RandAugment",
    "RandomAdjustSharpness",
    "RandomAffine",
    "RandomApply",
    "RandomAutocontrast",
    "RandomChoice",
    "RandomCrop",
    "RandomEqualize",
    "RandomErasing",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomInvert",
    "RandomPerspective",
    "RandomPosterize",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomSolarize",
    "RandomVerticalFlip",
    "Resize",
    "Uniform",
]
