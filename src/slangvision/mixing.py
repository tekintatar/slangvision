"""MixUp and CutMix on an augmented batch, after Compose::

    x, y = mix(train(images), labels)

They need labels and other images' augmented pixels, so they are a separate
step: one fused Slang kernel per call (shaders/kernels/mix.slang) writes the
mixed batch, with parameters drawn by torch's CPU generator (no GPU sync;
torch.manual_seed repeats them). MixUp and CutMix follow
torchvision.transforms.v2 draw for draw, so with the same seed they give
torchvision's results; TimmMixup follows timm's Mixup.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["CutMix", "MixUp", "TimmMixup"]

_KERNEL = Path(__file__).resolve().parent / "shaders" / "kernels" / "mix.slang"
_CONVERT = {
    "float16": "typealias T = half;\nfloat ld(T v) { return float(v); }\nT st(float f) { return T(f); }",
    "float32": "typealias T = float;\nfloat ld(T v) { return v; }\nT st(float f) { return f; }",
    # torch.bfloat16 travels as an int16 view (SlangPy's bridge has no bfloat16)
    "bfloat16": ("typealias T = int16_t;\n"
                 "float ld(T v) { return asfloat(uint(uint16_t(v)) << 16); }\n"
                 "T st(float f) { return bf16_bits(f); }\n"
                 "int16_t bf16_bits(float f)\n{\n    uint u = asuint(f);\n"
                 "    if ((u & 0x7fffffffu) > 0x7f800000u) return int16_t(0x7fc0);\n"
                 "    u += 0x7fffu + ((u >> 16) & 1u);\n    return int16_t(u >> 16);\n}"),
}


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


def _unit(value: Any, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


class _Mixer:
    """Shared call: checks, the kernel, and label mixing."""

    num_classes: int | None = None
    label_smoothing = 0.0

    def _params(self, B: int, H: int, W: int) -> tuple[list[int], list[float], list[tuple],
                                                       list[float]]:
        """Per image: partner, own weight outside the box (1 for CutMix), box
        (x1, y1, x2, y2; empty for MixUp), and the label weight."""
        raise NotImplementedError

    def __call__(self, images: "torch.Tensor", labels: "torch.Tensor"
                 ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be a (B, 3, H, W) tensor (Compose's output), got "
                             f"{getattr(images, 'shape', type(images).__name__)}")
        name = str(images.dtype).replace("torch.", "")
        if name not in _CONVERT or not images.is_cuda:
            raise ValueError("images must be a float16, bfloat16 or float32 CUDA tensor, got "
                             f"{images.dtype} on {images.device}")
        B, _, H, W = images.shape
        if images.requires_grad and torch.is_grad_enabled():
            raise NotImplementedError(f"{type(self).__name__}: no gradients (the kernel has no "
                                      "backward pass); run it under torch.no_grad() or on a "
                                      "detached tensor")
        if not isinstance(labels, torch.Tensor) or labels.ndim not in (1, 2) or len(labels) != B:
            raise ValueError(f"labels must be a (B,) tensor of class indices or (B, classes) "
                             f"probabilities for B={B}, got "
                             f"{getattr(labels, 'shape', type(labels).__name__)}")
        if labels.ndim == 1 and self.num_classes is None:
            raise ValueError("num_classes is needed for class-index labels")
        if labels.ndim == 2 and self.num_classes is not None and labels.shape[1] != self.num_classes:
            raise ValueError(f"labels have {labels.shape[1]} classes, num_classes is "
                             f"{self.num_classes}")
        if B == 0:
            return images.clone(), self._mix_labels(labels, [], [])
        partner, own, boxes, lam = self._params(B, H, W)
        return self._mix_images(images, partner, own, boxes), \
            self._mix_labels(labels, partner, lam)

    def _mix_images(self, x: "torch.Tensor", partner: list[int], own: list[float],
                    boxes: list[tuple]) -> "torch.Tensor":
        import slangpy as spy
        import torch

        from . import _runtime

        rows = [(float(j), a, 1.0 - a, *(float(v) for v in box))
                for j, a, box in zip(partner, own, boxes)]
        # the partner weight is 1 - a in double precision, as torch computes it
        prm = torch.tensor(rows, dtype=torch.float32).pin_memory().to(x.device, non_blocking=True)
        x = x.detach().contiguous()     # SlangPy refuses tensors that require grad
        out = torch.empty_like(x)
        name = str(x.dtype).replace("torch.", "")
        src = f"import slangpy;\n{_CONVERT[name]}\n{_KERNEL.read_text()}"
        with _runtime.torch_bridge():
            m = _runtime.load_module(src)
            view = (lambda t: t.view(torch.int16)) if name == "bfloat16" else (lambda t: t)
            m.mix(spy.grid(tuple(x.shape[:1]) + tuple(x.shape[2:])), view(x), view(out), prm)
        return out

    def _mix_labels(self, labels: "torch.Tensor", partner: list[int], lam: list[float]
                    ) -> "torch.Tensor":
        import torch

        if labels.ndim == 1:
            n = int(self.num_classes)       # type: ignore[arg-type]
            off = self.label_smoothing / n
            y = torch.full((len(labels), n), off, device=labels.device)
            y.scatter_(1, labels.long().view(-1, 1), 1.0 - self.label_smoothing + off)
        else:
            y = labels.float()
        if not len(partner):
            return y
        idx = torch.tensor(partner, device=y.device)
        w = torch.tensor(lam, device=y.device, dtype=y.dtype).view(-1, 1)
        return y[idx].mul(1.0 - w).add_(y.mul(w))


# ---------------------------------------------------------- torchvision
class _TorchvisionMix(_Mixer):
    def __init__(self, alpha: float = 1.0, num_classes: int | None = None):
        import torch

        self.alpha = float(alpha)
        if not self.alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.num_classes = None if num_classes is None else _positive_int(num_classes,
                                                                          "num_classes")
        self._dist = torch.distributions.Beta(torch.tensor([self.alpha]),
                                              torch.tensor([self.alpha]))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha}, num_classes={self.num_classes})"


class MixUp(_TorchvisionMix):
    """torchvision.transforms.v2.MixUp on an augmented batch: one weight
    ``lam ~ Beta(alpha, alpha)`` per batch, each image mixed with the previous
    one (``roll(1)``), labels one-hot and mixed the same way. Draws from
    torch's CPU generator exactly as torchvision does, so under the same seed
    the result is torchvision's. ``x, y = MixUp(num_classes=100)(x, labels)``.
    """

    def _params(self, B: int, H: int, W: int):
        lam = float(self._dist.sample(()))
        return [(i - 1) % B for i in range(B)], [lam] * B, [(0, 0, 0, 0)] * B, [lam] * B


class CutMix(_TorchvisionMix):
    """torchvision.transforms.v2.CutMix on an augmented batch: one box per
    batch, centred uniformly with sides ``sqrt(1 - lam)`` of the image, pasted
    from the previous image (``roll(1)``); labels mixed by the box's actual
    area. Same draws as torchvision, so the same seed gives its result."""

    def _params(self, B: int, H: int, W: int):
        import torch

        lam = float(self._dist.sample(()))
        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))
        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half, r_h_half = int(r * W), int(r * H)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        return ([(i - 1) % B for i in range(B)], [1.0] * B, [(x1, y1, x2, y2)] * B,
                [lam_adjusted] * B)


# ----------------------------------------------------------------- timm
class TimmMixup(_Mixer):
    """timm's Mixup: MixUp and/or CutMix with label smoothing, the recipe of
    DeiT, ConvNeXt and MAE (``mixup_alpha=0.8, cutmix_alpha=1.0``). Each image
    mixes with its mirror in the batch (image ``i`` with ``B - 1 - i``).

    ``mode`` ``"batch"`` draws one choice and weight per batch, ``"pair"``
    one per mirrored pair, ``"elem"`` one per image. ``prob`` is the chance of
    mixing at all, ``switch_prob`` the chance of CutMix when both alphas are
    set; ``cutmix_minmax`` draws box sides uniformly within that fraction of
    the image instead of from ``cutmix_alpha``; ``correct_lam`` sets CutMix's
    label weight from the box's clipped area. Labels are smoothed like timm's
    ``mixup_target``. Draws use torch's CPU generator, not timm's NumPy one,
    so the random values differ from timm's while following the same rules.
    As in timm, ``mixup_enabled = False`` turns mixing off (timm's train.py
    does so for the last epochs) and the defaults are timm's: label smoothing
    0.1 and no CutMix unless ``cutmix_alpha`` or ``cutmix_minmax`` is set.
    """

    def __init__(self, mixup_alpha: float = 1.0, cutmix_alpha: float = 0.0,
                 cutmix_minmax: tuple[float, float] | None = None, prob: float = 1.0,
                 switch_prob: float = 0.5, mode: str = "batch", correct_lam: bool = True,
                 label_smoothing: float = 0.1, num_classes: int = 1000):
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        if self.mixup_alpha < 0 or self.cutmix_alpha < 0:
            raise ValueError("mixup_alpha and cutmix_alpha must be non-negative")
        self.cutmix_minmax = None
        if cutmix_minmax is not None:
            lo, hi = (float(v) for v in cutmix_minmax)
            if not 0 < lo < hi <= 1:
                raise ValueError(f"cutmix_minmax must satisfy 0 < min < max <= 1, "
                                 f"got {cutmix_minmax}")
            self.cutmix_minmax = (lo, hi)
            self.cutmix_alpha = 1.0         # as timm: minmax turns CutMix on
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            raise ValueError("set mixup_alpha, cutmix_alpha or cutmix_minmax")
        self.prob = _unit(prob, "prob")
        self.switch_prob = _unit(switch_prob, "switch_prob")
        if mode not in ("batch", "pair", "elem"):
            raise ValueError(f"mode must be 'batch', 'pair' or 'elem', got {mode!r}")
        self.mode = mode
        self.correct_lam = bool(correct_lam)
        self.label_smoothing = _unit(label_smoothing, "label_smoothing")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.mixup_enabled = True

    def _draw(self, n: int) -> tuple[list[float], list[bool]]:
        """Per draw: the weight (1 = no mixing) and whether it is CutMix."""
        import torch

        if not self.mixup_enabled:
            return [1.0] * n, [False] * n

        def beta(alpha: float) -> "torch.Tensor":
            return torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample((n,))

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            cut = torch.rand(n) < self.switch_prob
            lam = torch.where(cut, beta(self.cutmix_alpha), beta(self.mixup_alpha))
        elif self.mixup_alpha > 0:
            cut, lam = torch.zeros(n, dtype=torch.bool), beta(self.mixup_alpha)
        else:
            cut, lam = torch.ones(n, dtype=torch.bool), beta(self.cutmix_alpha)
        lam = torch.where(torch.rand(n) < self.prob, lam, torch.ones(n))
        return lam.tolist(), cut.tolist()

    def _box(self, lam: float, H: int, W: int) -> tuple[tuple[int, int, int, int], float]:
        """timm's cutmix_bbox_and_lam: (x1, y1, x2, y2) and the label weight."""
        import torch

        if self.cutmix_minmax is not None:
            lo, hi = self.cutmix_minmax
            ch = int(torch.randint(int(H * lo), int(H * hi), (1,)))
            cw = int(torch.randint(int(W * lo), int(W * hi), (1,)))
            y1 = int(torch.randint(0, H - ch + 1, (1,)))
            x1 = int(torch.randint(0, W - cw + 1, (1,)))
            y2, x2 = y1 + ch, x1 + cw
        else:
            ratio = math.sqrt(1.0 - lam)
            ch, cw = int(H * ratio), int(W * ratio)
            cy, cx = int(torch.randint(0, H, (1,))), int(torch.randint(0, W, (1,)))
            y1, y2 = min(max(cy - ch // 2, 0), H), min(max(cy + (ch + 1) // 2, 0), H)
            x1, x2 = min(max(cx - cw // 2, 0), W), min(max(cx + (cw + 1) // 2, 0), W)
        if self.correct_lam or self.cutmix_minmax is not None:
            lam = 1.0 - (y2 - y1) * (x2 - x1) / float(H * W)
        return (x1, y1, x2, y2), lam

    def _params(self, B: int, H: int, W: int):
        partner = [B - 1 - i for i in range(B)]
        own, boxes, label = [1.0] * B, [(0, 0, 0, 0)] * B, [1.0] * B
        groups = {"batch": [list(range(B))],
                  "pair": [[i, B - 1 - i] for i in range(B // 2)],
                  "elem": [[i] for i in range(B)]}[self.mode]
        lams, cuts = self._draw(len(groups))
        for members, lam, cut in zip(groups, lams, cuts):
            if lam == 1.0:
                continue
            if cut:
                box, lam = self._box(lam, H, W)
                for i in members:
                    boxes[i], label[i] = box, lam
            else:
                for i in members:
                    own[i], label[i] = lam, lam
        return partner, own, boxes, label

    def __repr__(self) -> str:
        return (f"TimmMixup(mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha}, "
                f"cutmix_minmax={self.cutmix_minmax}, prob={self.prob}, "
                f"switch_prob={self.switch_prob}, mode={self.mode!r}, "
                f"label_smoothing={self.label_smoothing}, num_classes={self.num_classes})")
