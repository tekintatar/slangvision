# tests/test_parameters.py — argument parsing checked against torchvision's own
# classes and OpenCV's kernels, on the CPU (torch and torchvision, no GPU).
# The references in slangvision.testing reuse the parsed values, so a parsing
# bug would cancel out there: these tests compare them with independent ones.

from __future__ import annotations

import math

import pytest

v2 = pytest.importorskip("torchvision.transforms.v2")

import slangvision as A  # noqa: E402
from slangvision import sr  # noqa: E402

JITTER = [
    dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(brightness=(0.2, 1.8), contrast=1.5, saturation=0, hue=(-0.3, 0.2)),
    dict(brightness=0.0, contrast=(1.0, 1.0), saturation=(0.5, 0.5), hue=0.5),
    dict(contrast=(0.0, 3.0)),
]


@pytest.mark.parametrize("kw", JITTER, ids=repr)
def test_color_jitter_ranges_match_torchvision(kw):
    tv = v2.ColorJitter(**kw)
    expected = [tv.brightness, tv.contrast, tv.saturation, tv.hue]
    got = A.ColorJitter(**kw).ranges
    assert [None if r is None else tuple(r) for r in got] == \
        [None if r is None else tuple(float(v) for v in r) for r in expected]


@pytest.mark.parametrize("kw", [
    {}, dict(brightness=(0.5, 1.5), contrast=(0.2, 1.8), saturation=(1, 1), hue=(-0.3, 0.3)),
], ids=repr)
def test_photometric_distort_ranges_match_torchvision(kw):
    tv = v2.RandomPhotometricDistort(**kw)
    centers = (1.0, 1.0, 1.0, 0.0)
    got = A.RandomPhotometricDistort(**kw).ranges
    for r, c, want in zip(got, centers, (tv.brightness, tv.contrast, tv.saturation, tv.hue)):
        assert (r or (c, c)) == tuple(float(v) for v in want)


@pytest.mark.parametrize("args", [(5,), ((3, 7), 1.5), (9, (0.2, 3.0)), (23, (0.1, 2.0))],
                         ids=repr)
def test_gaussian_blur_arguments_match_torchvision(args):
    tv, ours = v2.GaussianBlur(*args), A.GaussianBlur(*args)
    assert ours.kernel_size == tuple(tv.kernel_size)      # both (kx, ky)
    assert (ours.sigma.low, ours.sigma.high) == tuple(float(v) for v in tv.sigma)


# cv2.getGaussianKernel(k, s, cv2.CV_64F) from OpenCV 5.0.0: first tap, the tap
# before the centre, the centre (k <= 9 with s = 0 are OpenCV's fixed tables)
OPENCV = {
    (3, 0): [0.25, 0.25, 0.5],
    (5, 0): [0.0625, 0.25, 0.375],
    (7, 0): [0.03125, 0.21875, 0.28125],
    (9, 0): [0.015625, 0.19921875, 0.234375],
    (11, 0): [0.008812229292562288, 0.1769983568313557, 0.20056541423882085],
    (21, 0): [0.0019290645132252324, 0.10971120494447857, 0.11428185740027867],
    (51, 0): [0.00037832129423682147, 0.04955031115428864, 0.04993893906128147],
    (21, 3.0): [0.0005143181739659887, 0.12584950778634324, 0.13303900634536284],
    (7, 1.5): [0.03663284536919403, 0.2167453214037078, 0.27068214949642655],
}


def test_usm_sharp_uses_basicsrs_kernel():
    """BasicSR makes the radius odd and asks OpenCV for that many taps."""
    for usm, key in ((sr.USMSharp(), (51, 0)), (sr.USMSharp(radius=20, sigma=3.0), (21, 3.0)),
                     (sr.USMSharp(radius=7, sigma=1.5), (7, 1.5))):
        n = len(usm.taps)
        got = [usm.taps[0], usm.taps[n // 2 - 1], usm.taps[n // 2]]
        assert n == key[0] and all(math.isclose(a, b, rel_tol=1e-12)
                                   for a, b in zip(got, OPENCV[key]))


@pytest.mark.parametrize("size, sigma", OPENCV, ids=str)
def test_usm_kernel_matches_opencv(size, sigma):
    taps = sr.gaussian_kernel(size, sigma)
    got = [taps[0], taps[size // 2 - 1], taps[size // 2]]
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(got, OPENCV[size, sigma]))
    assert math.isclose(sum(taps), 1.0, rel_tol=1e-12) and taps == taps[::-1]


def test_small_kernels_are_opencvs_tables():
    """OpenCV's fixed tables for sizes up to 9 with sigma 0, in full."""
    assert sr.gaussian_kernel(3, 0) == [0.25, 0.5, 0.25]
    assert sr.gaussian_kernel(5, 0) == [0.0625, 0.25, 0.375, 0.25, 0.0625]
    assert sr.gaussian_kernel(7, 0) == [0.03125, 0.109375, 0.21875, 0.28125, 0.21875,
                                        0.109375, 0.03125]
    assert sr.gaussian_kernel(9, 0) == [0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375,
                                        0.19921875, 0.1171875, 0.05078125, 0.015625]
