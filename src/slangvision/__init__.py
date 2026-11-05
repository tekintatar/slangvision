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
with :class:`ColorOp` and :class:`NeighborhoodOp`. Needs PyTorch with CUDA.
See docs/GUIDE.md.
"""

__version__ = "0.2.0.dev0"

# Load SlangPy (and, through it, torch) at import: if DALI loads its native
# libraries before SlangPy, SlangPy crashes creating its CUDA device, so
# importing slangvision first must be enough.
from . import _runtime  # noqa: F401
from ._base import Choice, ColorOp, NeighborhoodOp, NeighbourhoodOp, Uniform
from ._compile import Compose
from .mixing import CutMix, MixUp, TimmMixup
from .transforms import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    GaussianBlur,
    GaussianNoise,
    Grayscale,
    Normalize,
    Pad,
    RandAugment,
    RandomAdjustSharpness,
    RandomAffine,
    RandomApply,
    RandomAutocontrast,
    RandomChannelPermutation,
    RandomChoice,
    RandomCrop,
    RandomEqualize,
    RandomErasing,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomInvert,
    RandomPerspective,
    RandomPhotometricDistort,
    RandomPosterize,
    RandomResizedCrop,
    RandomRotation,
    RandomSolarize,
    RandomVerticalFlip,
    Resize,
    TimmRandAugment,
    TrivialAugmentWide,
)

__all__ = [
    "__version__",
    "AutoAugment",
    "CenterCrop",
    "Choice",
    "ColorJitter",
    "ColorOp",
    "CutMix",
    "Compose",
    "GaussianBlur",
    "GaussianNoise",
    "Grayscale",
    "MixUp",
    "NeighborhoodOp",
    "NeighbourhoodOp",
    "Normalize",
    "Pad",
    "RandAugment",
    "RandomAdjustSharpness",
    "RandomAffine",
    "RandomApply",
    "RandomAutocontrast",
    "RandomChannelPermutation",
    "RandomChoice",
    "RandomCrop",
    "RandomEqualize",
    "RandomErasing",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomInvert",
    "RandomPerspective",
    "RandomPhotometricDistort",
    "RandomPosterize",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomSolarize",
    "RandomVerticalFlip",
    "Resize",
    "TimmMixup",
    "TimmRandAugment",
    "TrivialAugmentWide",
    "Uniform",
]

# torchvision transforms with no counterpart here, because Compose already
# does their job: a hint instead of a bare AttributeError when porting
_PORTING = {
    "ToImage": "images go in as uint8 CUDA tensors (or lists of them) already",
    "PILToTensor": "decode to uint8 CUDA tensors (see the guide's Inputs and decoders)",
    "ToTensor": "images go in as uint8 tensors and come out as floats in [0, 1]",
    "ToDtype": "the output is already float: choose it with Compose(dtype=...)",
    "ConvertImageDtype": "the output is already float: choose it with Compose(dtype=...)",
    "ToPureTensor": "the output is already a plain tensor",
}


def __getattr__(name: str):
    if name in _PORTING:
        raise AttributeError(f"slangvision has no {name}: drop it; {_PORTING[name]}")
    raise AttributeError(f"module 'slangvision' has no attribute {name!r}")
