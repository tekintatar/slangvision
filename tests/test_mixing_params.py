# tests/test_mixing_params.py — MixUp/CutMix parameter draws and label mixing,
# on the CPU (torch and torchvision, no GPU). The image kernel is covered by
# tests/test_pipelines.py.

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("torchvision.transforms.v2")

import slangvision as A  # noqa: E402


def test_mixup_draws_match_torchvision():
    for seed in range(20):
        torch.manual_seed(seed)
        partner, own, boxes, lam = A.MixUp(alpha=0.8, num_classes=10)._params(8, 32, 48)
        torch.manual_seed(seed)
        tv = v2.MixUp(alpha=0.8, num_classes=10).make_params([])
        assert own == [tv["lam"]] * 8 and lam == own
        assert partner == [7, 0, 1, 2, 3, 4, 5, 6]          # roll(1): image i mixes with i - 1
        assert boxes == [(0, 0, 0, 0)] * 8


def test_cutmix_draws_match_torchvision():
    for seed in range(50):
        torch.manual_seed(seed)
        _, own, boxes, lam = A.CutMix(alpha=1.0, num_classes=10)._params(4, 32, 48)
        torch.manual_seed(seed)
        tv = v2.CutMix(alpha=1.0, num_classes=10).make_params([torch.zeros(4, 3, 32, 48)])
        assert boxes == [tv["box"]] * 4 and lam == [tv["lam_adjusted"]] * 4 and own == [1.0] * 4


def test_labels_match_torchvision_and_timm():
    labels = torch.tensor([3, 1, 4, 1])
    lam = 0.3
    mixed = A.MixUp(num_classes=5)._mix_labels(labels, [3, 0, 1, 2], [lam] * 4)
    one_hot = torch.nn.functional.one_hot(labels, 5).float()
    assert torch.equal(mixed, one_hot.roll(1, 0).mul_(1 - lam).add_(one_hot.mul(lam)))
    # timm's mixup_target: smoothed one-hot, mixed with the mirrored batch
    m = A.TimmMixup(num_classes=5, label_smoothing=0.1)
    got = m._mix_labels(labels, [3, 2, 1, 0], [lam] * 4)
    y1 = torch.full((4, 5), 0.1 / 5)
    y1.scatter_(1, labels.view(-1, 1), 1 - 0.1 + 0.1 / 5)
    assert torch.allclose(got, y1 * lam + y1.flip(0) * (1 - lam))
    probs = torch.rand(4, 5)
    assert torch.allclose(A.MixUp()._mix_labels(probs, [3, 0, 1, 2], [lam] * 4),
                          probs.roll(1, 0) * (1 - lam) + probs * lam)


@pytest.mark.parametrize("mode", ["batch", "pair", "elem"])
def test_timm_style_parameters_follow_its_rules(mode):
    m = A.TimmMixup(mixup_alpha=0.8, cutmix_alpha=1.0, mode=mode, num_classes=10)
    torch.manual_seed(0)
    B, H, W = 64, 40, 60
    partner, own, boxes, lam = m._params(B, H, W)
    assert partner == [B - 1 - i for i in range(B)]
    for i in range(B):
        cut = boxes[i] != (0, 0, 0, 0)
        if cut:
            x1, y1, x2, y2 = boxes[i]
            assert own[i] == 1.0 and 0 <= x1 <= x2 <= W and 0 <= y1 <= y2 <= H
            assert math.isclose(lam[i], 1 - (x2 - x1) * (y2 - y1) / (H * W))
        else:
            assert own[i] == lam[i]
    if mode == "batch":
        assert len(set(boxes)) == 1 and len(set(own)) == 1
    if mode == "pair":
        assert all(boxes[i] == boxes[B - 1 - i] and own[i] == own[B - 1 - i] for i in range(B))
    if mode == "elem":
        assert len(set(own)) > B // 4


def test_timm_mixup_can_be_turned_off():
    m = A.TimmMixup(mixup_alpha=0.8, cutmix_alpha=1.0, mode="elem", num_classes=10)
    m.mixup_enabled = False
    partner, own, boxes, lam = m._params(8, 32, 32)
    assert own == lam == [1.0] * 8 and boxes == [(0, 0, 0, 0)] * 8


def test_timm_style_draw_statistics():
    m = A.TimmMixup(mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.3, prob=0.7,
                    mode="elem", num_classes=10)
    torch.manual_seed(0)
    lam, cut = m._draw(20000)
    lam_t, cut_t = torch.tensor(lam), torch.tensor(cut)
    mixed = lam_t < 1.0
    assert abs(mixed.float().mean().item() - 0.7) < 0.02
    assert abs(cut_t.float().mean().item() - 0.3) < 0.02
    assert abs(lam_t[mixed].mean().item() - 0.5) < 0.02            # Beta(a, a) has mean 1/2
    # ... and variance 1 / (4 (2a + 1)): tells mixup_alpha 0.8 from cutmix_alpha 1.0
    assert abs(lam_t[mixed & ~cut_t].var().item() - 1 / 10.4) < 0.005
    assert abs(lam_t[mixed & cut_t].var().item() - 1 / 12) < 0.005
    minmax = A.TimmMixup(cutmix_minmax=(0.2, 0.6), mixup_alpha=0.0, num_classes=10)
    box, w = minmax._box(0.5, 100, 200)
    x1, y1, x2, y2 = box
    assert 20 <= y2 - y1 < 60 and 40 <= x2 - x1 < 120 and math.isclose(w, 1 - (y2 - y1) * (x2 - x1) / 20000)


@pytest.mark.parametrize("make, match", [
    (lambda: A.MixUp(alpha=0), "alpha must be positive"),
    (lambda: A.CutMix(num_classes=0), "num_classes"),
    (lambda: A.TimmMixup(mixup_alpha=0, cutmix_alpha=0), "set mixup_alpha"),
    (lambda: A.TimmMixup(mode="row"), "mode"),
    (lambda: A.TimmMixup(prob=1.5), "prob"),
    (lambda: A.TimmMixup(cutmix_minmax=(0.5, 0.2)), "cutmix_minmax"),
    (lambda: A.TimmMixup(label_smoothing=-0.1), "label_smoothing"),
])
def test_argument_errors(make, match):
    with pytest.raises(ValueError, match=match):
        make()
