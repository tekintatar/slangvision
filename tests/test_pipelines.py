# tests/test_pipelines.py
# slangvision on CUDA: every pipeline is checked image by image
# against torchvision.transforms.v2 functionals fed the same random values
# (slangvision.testing.max_reference_error), plus inputs, randomness and both SlangPy
# torch bridges. Requires torch + torchvision + CUDA — skipped elsewhere;
# runs on a CUDA host, e.g. via scripts/cuda_box/run_tests.sh.

from __future__ import annotations

import math
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
if not torch.cuda.is_available():
    pytest.skip("slangvision requires CUDA", allow_module_level=True)

import slangpy as spy  # noqa: E402

import slangvision as A  # noqa: E402
from slangvision.testing import _replay, max_reference_error  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TOL32 = 1e-4        # fp32 output vs torchvision (normalized units)
TOL16 = 4e-3        # fp16 output rounding at |values| up to ~2.6


def _image(h: int, w: int, g: "torch.Generator") -> "torch.Tensor":
    """Photo-like uint8 (3, h, w): smooth colour fields plus grain."""
    low = torch.rand(1, 3, max(2, h // 16), max(2, w // 16), device="cuda", generator=g)
    x = torch.nn.functional.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)[0]
    x = x + 0.04 * torch.randn(3, h, w, device="cuda", generator=g)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


@pytest.fixture(scope="module")
def images():
    """Varied sizes, including extreme aspect ratios that make
    RandomResizedCrop fall back to its centre crop."""
    g = torch.Generator(device="cuda").manual_seed(0)
    sizes = [(300, 400), (480, 360), (250, 250), (600, 900), (200, 520), (375, 500),
             (40, 300), (300, 40), (23, 17)]
    return [_image(h, w, g) for h, w in sizes]


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator(device="cuda").manual_seed(1)
    return torch.stack([_image(96, 128, g) for _ in range(6)])


def _check(transforms, images, layout="chw", tol16=TOL16):
    torch.manual_seed(0)
    assert max_reference_error(A.Compose(transforms, dtype=torch.float32), images, layout) < TOL32
    torch.manual_seed(0)
    assert max_reference_error(A.Compose(transforms), images, layout) < tol16


# ---------------------------------------------------------------- recipes
RECIPES = {
    "imagenet": [A.RandomResizedCrop(224), A.RandomHorizontalFlip(),
                 A.ColorJitter(0.4, 0.4, 0.4, 0.1), A.Normalize(MEAN, STD)],
    "eval": [A.Resize(256), A.CenterCrop(224), A.Normalize(MEAN, STD)],
    "simclr": [A.RandomResizedCrop(224), A.RandomHorizontalFlip(),
               A.RandomApply([A.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
               A.RandomGrayscale(p=0.2), A.RandomApply([A.GaussianBlur(23)], p=0.5),
               A.RandomSolarize(0.5, p=0.2), A.Normalize(MEAN, STD)],
}


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_matches_torchvision(name, images):
    _check(RECIPES[name], images)


@pytest.mark.parametrize("transforms", [
    [A.RandomResizedCrop((96, 160), scale=(0.3, 0.9), ratio=(0.5, 2.0))],
    [A.RandomResizedCrop(64, antialias=False)],
    [A.Resize((120, 90))],
    [A.Resize(100, antialias=False), A.CenterCrop(80)],
    [A.Resize(64), A.CenterCrop((90, 50))],           # larger than the image: zero padding
    [A.CenterCrop((70, 333))],
    [A.RandomResizedCrop(64), A.RandomHorizontalFlip(p=1.0)],
    [A.RandomResizedCrop(64), A.RandomHorizontalFlip(), A.RandomHorizontalFlip()],
], ids=lambda ts: "+".join(type(t).__name__ for t in ts))
def test_geometry_matches_torchvision(transforms, images):
    _check(transforms, images)


@pytest.fixture(scope="module")
def small():
    """64 CIFAR-sized images of 24-40 px, so padding shows up in most crops."""
    g = torch.Generator(device="cuda").manual_seed(2)
    sizes = torch.randint(24, 41, (64, 2), generator=torch.Generator().manual_seed(2))
    return [_image(int(h), int(w), g) for h, w in sizes]


@pytest.mark.parametrize("transforms", [
    [A.RandomCrop(32, padding=4)],                                      # the CIFAR recipe
    [A.RandomCrop(32, padding=4, padding_mode="reflect"), A.RandomHorizontalFlip()],
    [A.RandomCrop((30, 34), padding=(5, 3), padding_mode="symmetric")],
    [A.RandomCrop(48, pad_if_needed=True, padding_mode="edge")],         # larger than every image
    [A.RandomCrop(48, padding=2, pad_if_needed=True, padding_mode="reflect")],
    [A.RandomCrop(28, padding=(1, 2, 3, 4), fill=(0.2, 0.5, 0.9))],
    [A.Pad((3, 7), padding_mode="reflect"), A.CenterCrop(30)],
    [A.Pad(3, padding_mode="edge"), A.RandomCrop(36, padding=6, padding_mode="reflect"),
     A.CenterCrop(40)],                                                  # three padding modes
    [A.RandomCrop(36, padding=6), A.RandomCrop(28, padding=3, padding_mode="symmetric")],
    [A.Pad((1, 2, 3, 4), fill=0.7), A.CenterCrop(30), A.RandomVerticalFlip(p=1.0),
     A.RandomHorizontalFlip(p=1.0)],                                     # flips after uneven padding
    [A.Resize(40), A.Pad(5, padding_mode="edge"), A.RandomCrop(36)],
    [A.RandomResizedCrop(32), A.Pad(4), A.RandomCrop(32)],
    [A.RandomCrop(24), A.RandomVerticalFlip(), A.RandomHorizontalFlip()],
], ids=lambda ts: "+".join(type(t).__name__ for t in ts))
def test_crops_pads_and_flips_match_torchvision(transforms, small):
    _check(transforms, small)


def test_crop_positions_reach_the_padding_on_every_side(small):
    """The CIFAR recipe draws offsets into the padding on all four sides."""
    crop = A.RandomCrop(32, padding=4)
    pipe = A.Compose([crop])
    torch.manual_seed(0)
    _, params = pipe._run(small, "chw")
    k = pipe._b.win[id(crop)]
    wox, woy, cw, ch = (params[:, k + i] for i in range(4))
    assert (wox < 0).any() and (woy < 0).any()
    assert (wox + 32 > cw).any() and (woy + 32 > ch).any()


def test_pad_alone_keeps_a_same_size_batch(batch):
    _check([A.Pad(6, fill=0.5)], batch)
    _check([A.RandomVerticalFlip(p=1.0)], batch)


@pytest.mark.parametrize("transforms", [
    # DeiT III's ThreeAugment choice
    [A.RandomCrop(32, padding=4), A.RandomChoice([A.RandomGrayscale(p=1.0),
                                                  A.RandomSolarize(0.5, p=1.0),
                                                  A.GaussianBlur(9)])],
    [A.RandomCrop(32), A.RandomChoice([A.ColorJitter(contrast=(0.2, 1.8)), A.GaussianBlur(5)])],
    [A.RandomCrop(32), A.RandomChoice([A.ColorJitter(contrast=(0.2, 1.8)),
                                       A.RandomGrayscale(p=1.0)], p=[3, 1])],
    [A.RandomCrop(32), A.RandomApply([A.RandomChoice([A.RandomGrayscale(p=1.0),
                                                      A.GaussianBlur(5)])], p=0.7)],
    [A.RandomCrop(32), A.RandomChoice([A.RandomChoice([A.RandomGrayscale(p=1.0),
                                                       A.GaussianBlur(5)]),
                                       A.RandomSolarize(0.5, p=1.0)])],
], ids=lambda ts: "+".join(type(t).__name__ for t in ts))
def test_random_choice_matches_torchvision(transforms, small):
    _check(transforms, [im for im in small if min(im.shape[-2:]) >= 32])


def test_random_crop_and_choice_draw_per_image_and_respect_same_on_batch(batch):
    # one image size: a shared draw then gives every image the same offset
    def first_slots(t, **kw):
        pipe = A.Compose([A.RandomCrop(24, **kw), t] if t else [A.RandomCrop(24, **kw)])
        torch.manual_seed(0)
        _, params = pipe._run(torch.cat([batch] * 8), "chw")
        return params

    assert first_slots(None, padding=8)[:, 6].unique().numel() > 1
    shared = first_slots(None, padding=8, same_on_batch=True)
    assert shared[:, 6].unique().numel() == 1 and shared[:, 7].unique().numel() == 1
    choice = A.RandomChoice([A.RandomGrayscale(p=1.0), A.GaussianBlur(5)], same_on_batch=True)
    assert first_slots(choice, padding=8)[:, 10].unique().numel() == 1


def test_random_choice_follows_its_weights():
    pipe = A.Compose([A.RandomChoice([A.RandomGrayscale(p=1.0), A.RandomSolarize(0.5, p=1.0),
                                      A.RandomGrayscale(p=1.0)], p=[1, 2, 1])])
    torch.manual_seed(0)
    _, params = pipe._run(torch.zeros(4096, 3, 16, 16, dtype=torch.uint8, device="cuda"), "chw")
    counts = torch.bincount(params[:, -1].long(), minlength=3).float() / 4096
    assert torch.allclose(counts, torch.tensor([0.25, 0.5, 0.25], device="cuda"), atol=0.03)


@pytest.mark.parametrize("transforms, match", [
    ([A.RandomCrop(1000)], "larger than the image"),
    ([A.Pad(30, padding_mode="reflect")], "reflect padding"),             # wider than 23x17
    ([A.RandomCrop(200, pad_if_needed=True, padding_mode="symmetric")], "symmetric padding"),
])
def test_per_image_size_errors(transforms, match, images):
    with pytest.raises(ValueError, match=match):
        A.Compose(transforms)(images)


def test_mirror_padding_equal_to_the_limit_matches_torchvision(batch):
    # torchvision's limits: reflect padding up to size - 1, symmetric up to size
    _check([A.Pad((127, 95), padding_mode="reflect"), A.CenterCrop((96, 128))], batch)
    _check([A.Pad((128, 96), padding_mode="symmetric"), A.CenterCrop((96, 128))], batch)


@pytest.mark.parametrize("transforms", [
    [A.RandomResizedCrop(64, interpolation="bicubic")],
    [A.RandomResizedCrop((48, 80), scale=(0.02, 0.3), interpolation="bicubic")],
    [A.RandomResizedCrop(64, interpolation="bicubic", antialias=False)],
    [A.Resize(100, interpolation="bicubic"), A.CenterCrop(80)],
    [A.Resize((300, 410), interpolation="bicubic")],                     # upscale
    [A.Resize((300, 410), interpolation="bicubic", antialias=False)],
    [A.Resize((40, 30), interpolation="bicubic", antialias=False)],      # downscale, no antialias
    [A.RandomResizedCrop(64, interpolation="bicubic"), A.ColorJitter(0.4, 0.4, 0.4, 0.1),
     A.Normalize(MEAN, STD)],
], ids=str)
def test_bicubic_matches_torchvision(transforms, images):
    # float bicubic overshoots [0, 1] like torchvision's; nothing is clamped
    _check(transforms, images)


@pytest.mark.parametrize("op", [
    A.ColorJitter(brightness=0.5),
    A.ColorJitter(contrast=(0.2, 1.8)),
    A.ColorJitter(saturation=0.8),
    A.ColorJitter(hue=0.5),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
    A.RandomGrayscale(p=1.0),
    A.GaussianBlur(5, sigma=(0.5, 3.0)),
    A.GaussianBlur((3, 9), sigma=1.5),
    A.RandomApply([A.ColorJitter(0.4, 0.4), A.RandomGrayscale(p=0.5)], p=0.5),
    A.RandomApply([A.GaussianBlur(7)], p=0.5),
], ids=repr)
def test_op_matches_torchvision(op, images):
    _check([A.RandomResizedCrop(96), op, A.Normalize(MEAN, STD)], images)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.71])
def test_solarize_matches_torchvision_off_the_threshold(threshold, images):
    """A value landing exactly on the threshold (0.3 is 76.5/255, which
    bilinear sampling of uint8 pixels can produce) may be inverted on one side
    and not the other: float rounding differs. Every other value matches."""
    pipe = A.Compose([A.RandomResizedCrop(96), A.RandomSolarize(threshold, p=1.0)],
                     dtype=torch.float32)
    torch.manual_seed(0)
    out, params = pipe._run(images, "chw")
    ref = _replay(pipe, images, params, "chw")
    bad = (out - ref).abs() > TOL32
    assert bad.float().mean().item() < 1e-4
    on_threshold = ((ref - threshold).abs() < 1e-5) | ((1.0 - ref - threshold).abs() < 1e-5)
    assert on_threshold[bad].all()


def test_ops_after_a_blur_use_the_blurred_image(images):
    # contrast's mean is taken in the pass after the blur
    _check([A.RandomResizedCrop(96), A.GaussianBlur(9), A.ColorJitter(0.4, 0.6, 0.4, 0.1),
            A.GaussianBlur(3), A.RandomSolarize(0.5, p=0.5)], images)


def test_no_geometry_keeps_the_batch_size(batch):
    pipe = A.Compose([A.ColorJitter(0.4, 0.4, 0.4, 0.1), A.Normalize(MEAN, STD)],
                     dtype=torch.float32)
    assert pipe(batch).shape == (6, 3, 96, 128)
    assert max_reference_error(pipe, batch) < TOL32
    with pytest.raises(ValueError, match="same size"):
        pipe([batch[0], batch[1][:, :50]])


# ------------------------------------------------------------ custom ops
class Gamma(A.ColorOp):
    params = {"gamma": A.Uniform(0.7, 1.4), "gain": 1.0}
    slang = "float4 apply(float4 c, float gamma, float gain) { return clamp01(gain * pow(c, float4(gamma))); }"

    def reference(self, img, gamma, gain):
        return (gain * img.pow(gamma)).clamp(0, 1)


class BoxBlur(A.NeighbourhoodOp):
    params = {"r": A.Choice([1, 2, 3])}
    slang = """
    float4 apply<I : IImage>(I img, int x, int y, float r)
    {
        int k = int(r);
        float4 acc = float4(0.0);
        for (int dy = -k; dy <= k; ++dy)
            for (int dx = -k; dx <= k; ++dx)
                acc += img.load(x + dx, y + dy);
        return acc / float((2 * k + 1) * (2 * k + 1));
    }"""

    def __init__(self, border="reflect", p=1.0):
        super().__init__(p)
        self.border = border

    def reference(self, img, r):
        k = int(r)
        mode = "reflect" if self.border == "reflect" else "replicate"
        x = torch.nn.functional.pad(img[None], (k, k, k, k), mode=mode)
        return torch.nn.functional.avg_pool2d(x, 2 * k + 1, stride=1)[0]


@pytest.mark.parametrize("border", ["reflect", "clamp"])
def test_custom_ops_match_their_references(border, images):
    _check([A.RandomResizedCrop((80, 112)), Gamma(p=0.6), BoxBlur(border, p=0.7),
            A.ColorJitter(0.3, 0.3), A.Normalize(MEAN, STD)], images)


def test_custom_op_helper_names_do_not_clash(images):
    class Darken(A.ColorOp):
        slang = "float k() { return 0.5; } float4 apply(float4 c) { return c * k(); }"

        def reference(self, img):
            return img * 0.5

    class Brighten(A.ColorOp):
        slang = "float k() { return 1.5; } float4 apply(float4 c) { return clamp01(c * k()); }"

        def reference(self, img):
            return (img * 1.5).clamp(0, 1)

    _check([A.RandomResizedCrop(48), Darken(), Brighten(p=0.5)], images)


def test_custom_op_may_reuse_a_library_helper_name(batch):
    # the library defines posterize(float4, float); an op's own one wins inside its namespace
    class Halve(A.ColorOp):
        slang = ("float4 posterize(float4 c, float n) { return c * (1.0 / n); } "
                 "float4 apply(float4 c) { return posterize(c, 2.0); }")

        def reference(self, img):
            return img * 0.5

    # unresampled images: no value lands on one of posterize's level edges
    _check([Halve(), A.RandomPosterize(3, p=0.5)], batch)


def test_one_op_class_used_twice(images):
    _check([A.RandomResizedCrop(48), A.RandomApply([A.ColorJitter(0.4), A.RandomGrayscale(p=1.0)]),
            A.RandomGrayscale(p=0.2)], images)


def test_bad_slang_reports_the_generated_source(images):
    class Broken(A.ColorOp):
        slang = "float4 apply(float4 c) { return undefined_name(c); }"

    with pytest.raises(RuntimeError, match="Compose.source"):
        A.Compose([A.RandomResizedCrop(32), Broken()])(images)


# ----------------------------------------------------------------- inputs
def test_batch_and_lists_of_either_layout_agree(batch):
    pipe = A.Compose(RECIPES["simclr"], dtype=torch.float32)
    out, params = pipe._run(batch, "chw")
    hwc = batch.permute(0, 2, 3, 1).contiguous()
    for images, layout in ((list(batch), "chw"), (hwc, "hwc"), (list(hwc), "hwc"),
                           (batch.permute(0, 2, 3, 1), "hwc")):     # non-contiguous view
        assert torch.equal(pipe._run(images, layout, params=params)[0], out)


@pytest.mark.parametrize("name", ["imagenet", "simclr"])
def test_bfloat16_output_is_torchvision_rounded(name, images):
    """bfloat16 output is within one bfloat16 step of the float32 reference:
    rounded like torch, except where the two land on either side of a rounding
    edge. Pipelines with several passes keep float16 between them, which
    moves values by up to about 1e-3 after Normalize, so they cross edges
    more often."""
    pipe = A.Compose(RECIPES[name], dtype=torch.bfloat16)
    torch.manual_seed(0)
    out, params = pipe._run(images, "chw")
    assert out.dtype == torch.bfloat16
    ref = _replay(pipe, images, params, "chw")
    step = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2**-14))) - 7)
    assert ((out.float() - ref).abs() <= step + 2e-3).all()
    if len(pipe._stages) == 1:      # no float16 intermediates: torch's rounding nearly always
        assert ((out.float() - ref).abs() <= step / 2 + 1e-6).float().mean() > 0.95


def test_output_dtype_shape_and_empty_input(images):
    out = A.Compose([A.RandomResizedCrop((40, 56))])(images)
    assert out.dtype == torch.float16 and out.shape == (len(images), 3, 40, 56)
    assert out.is_contiguous() and 0.0 <= out.min() and out.max() <= 1.0
    assert A.Compose([A.RandomResizedCrop(8)], dtype=torch.float32)([]).shape == (0, 3, 8, 8)
    bf = A.Compose([A.RandomResizedCrop(8)], dtype=torch.bfloat16)
    assert bf(images).dtype == torch.bfloat16 and bf([]).dtype == torch.bfloat16
    empty = torch.empty(0, 3, 32, 32, dtype=torch.uint8, device="cuda")
    assert A.Compose([A.RandomResizedCrop(8)])(empty).shape == (0, 3, 8, 8)
    assert A.Compose([A.ColorJitter(0.4)])(empty).shape == (0, 3, 32, 32)
    assert A.Compose([A.Pad(4)])(empty).shape == (0, 3, 40, 40)


@pytest.mark.parametrize("bad, match", [
    (lambda b: b.float(), "uint8"),
    (lambda b: b[:, :1], "3 \\(RGB\\) channels"),
    (lambda b: b.cpu(), "current CUDA device"),
    (lambda b: b[0], "shape"),
    # a stride-0 view: the size check runs before any copy
    (lambda b: b[:1, :, :1, :1].expand(1, 3, 30000, 30000), "too large"),
])
def test_input_errors(bad, match, batch):
    with pytest.raises(ValueError, match=match):
        A.Compose([A.RandomResizedCrop(16)])(bad(batch))


def test_empty_list_without_geometry_says_why():
    with pytest.raises(ValueError, match="no images"):
        A.Compose([A.ColorJitter(0.4)])([])


def test_parameter_tables_ignore_the_default_dtype(batch):
    pipe = A.Compose([A.RandomResizedCrop(32), A.ColorJitter(0.4, 0.4)], dtype=torch.float32)
    torch.manual_seed(0)
    expected = pipe(batch)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        assert torch.equal(pipe(batch), expected)
    finally:
        torch.set_default_dtype(previous)


def test_disabled_jitter_factors_are_written(batch):
    _, params = A.Compose([A.ColorJitter(brightness=0.4)])._run(batch, "chw")
    assert torch.isfinite(params).all()


# -------------------------------------------------------------- randomness
def test_seeded_runs_repeat_and_differ(images):
    pipe = A.Compose(RECIPES["simclr"])
    torch.manual_seed(7)
    a = pipe(images)
    torch.manual_seed(7)
    assert torch.equal(pipe(images), a)
    assert not torch.equal(pipe(images), a)


def test_same_on_batch_shares_values(batch):
    jitter = A.ColorJitter(0.4, 0.4, 0.4, 0.1, same_on_batch=True)
    crop = A.RandomResizedCrop(32, same_on_batch=True)
    flip = A.RandomHorizontalFlip(same_on_batch=True)
    free = A.ColorJitter(0.4, 0.4, 0.4, 0.1)
    pipe = A.Compose([crop, flip, jitter, free])
    _, params = pipe._run(batch, "chw")
    b = pipe._b

    def rows(k, n):
        return params[:, k:k + n]

    for k, n in ((b.geo, 8), (b.k(flip), 1), (b.k(jitter), 5)):
        assert (rows(k, n) == rows(k, n)[0]).all()
    assert not (rows(b.k(free), 5) == rows(b.k(free), 5)[0]).all()


def test_random_values_follow_torchvision(images):
    from torchvision.transforms import RandomResizedCrop

    crop, flip = A.RandomResizedCrop(8), A.RandomHorizontalFlip(p=0.3)
    jitter = A.ColorJitter(0.4, 0.4, 0.4, 0.1)
    pipe = A.Compose([crop, flip, jitter])
    img = images[0]                             # 300 x 400
    _, params = pipe._run([img] * 4096, "chw")
    b = pipe._b
    area = (params[:, b.geo + 2] * params[:, b.geo + 3] / (300 * 400)).mean().item()
    aspect = torch.log(params[:, b.geo + 2] / params[:, b.geo + 3]).mean().item()
    torch.manual_seed(0)
    boxes = [RandomResizedCrop.get_params(img, [0.08, 1.0], [3 / 4, 4 / 3]) for _ in range(4096)]
    tv_area = sum(h * w for _, _, h, w in boxes) / (4096 * 300 * 400)
    tv_aspect = sum(math.log(w / h) for _, _, h, w in boxes) / 4096
    assert abs(area - tv_area) < 0.02 and abs(aspect - tv_aspect) < 0.03
    assert abs(params[:, b.k(flip)].mean().item() - 0.3) < 0.03
    k = b.k(jitter)
    assert abs(params[:, k].mean().item() - 1.0) < 0.02                 # brightness in [0.6, 1.4]
    assert params[:, k + 3].abs().max().item() <= 0.1                  # hue in [-0.1, 0.1]
    orders = Counter(params[:, k + 4].int().tolist())
    assert len(orders) == 24 and min(orders.values()) > 4096 / 24 * 0.7


# ------------------------------------------------------------------ erasing
@pytest.mark.parametrize("transforms", [
    [A.RandomResizedCrop(64), A.Normalize(MEAN, STD), A.RandomErasing(p=1.0)],
    [A.RandomResizedCrop(64), A.Normalize(MEAN, STD),
     A.RandomErasing(p=0.7, scale=(0.1, 0.5), ratio=(0.5, 2.0), value=(0.5, -1.0, 2.0))],
    [A.RandomResizedCrop(64), A.ColorJitter(0.4), A.RandomErasing(p=1.0, value=0.3)],
    [A.RandomCrop(32, padding=4), A.Normalize(MEAN, STD), A.RandomErasing(p=1.0),
     A.RandomErasing(p=1.0, value=1.5)],                                # two boxes, in order
    [A.RandomResizedCrop(64), A.Normalize(MEAN, STD), A.RandomErasing(p=1.0, value="random")],
], ids=lambda ts: "+".join(type(t).__name__ for t in ts))
def test_random_erasing_matches_torchvision(transforms, small):
    # value="random": the box matches; its noise is checked in the next test
    _check(transforms, [im for im in small if min(im.shape[-2:]) >= 32])


def test_random_erasing_noise_is_standard_normal(batch):
    erase = A.RandomErasing(p=1.0, scale=(0.3, 0.3), ratio=(1.0, 1.0), value="random")
    pipe = A.Compose([A.Normalize(MEAN, STD), erase], dtype=torch.float32)
    torch.manual_seed(0)
    x = torch.cat([batch] * 16)
    out, params = pipe._run(x, "chw")
    k = pipe._b.k(erase)
    vals, firsts = [], []
    for i in range(len(x)):
        top, left, h, w = (int(v) for v in params[i, k + 1:k + 5])
        box = out[i, :, top:top + h, left:left + w]
        vals.append(box.reshape(-1))
        firsts.append(box[:, :4, :4].reshape(-1))
    v = torch.cat(vals)
    assert abs(v.mean().item()) < 0.01 and abs(v.std().item() - 1.0) < 0.01
    assert abs(((v > 1.96).float().mean() - 0.025).item()) < 0.003      # normal tail
    assert torch.stack(firsts).std(0).min() > 0.5                       # differs per image


def test_random_erasing_boxes_follow_torchvision(images):
    from torchvision.transforms import v2

    erase = A.RandomErasing(p=0.6)
    pipe = A.Compose([A.CenterCrop((120, 160)), erase])
    img = images[0]
    torch.manual_seed(0)
    _, params = pipe._run([img] * 4096, "chw")
    k = pipe._b.k(erase)
    on = params[:, k] > 0.5
    area = (params[on, k + 3] * params[on, k + 4] / (120 * 160)).mean().item()
    aspect = torch.log(params[on, k + 3] / params[on, k + 4]).mean().item()
    tv = v2.RandomErasing(p=1.0)
    torch.manual_seed(0)
    boxes = [tv.make_params([torch.zeros(3, 120, 160)]) for _ in range(4096)]
    boxes = [bx for bx in boxes if bx["v"] is not None]
    tv_area = sum(bx["h"] * bx["w"] for bx in boxes) / (len(boxes) * 120 * 160)
    tv_aspect = sum(math.log(bx["h"] / bx["w"]) for bx in boxes) / len(boxes)
    assert abs(on.float().mean().item() - 0.6) < 0.03
    assert abs(area - tv_area) < 0.01 and abs(aspect - tv_aspect) < 0.05
    shared = A.RandomErasing(p=1.0, same_on_batch=True)
    pipe = A.Compose([A.CenterCrop((120, 160)), shared])
    _, params = pipe._run([img] * 64, "chw")
    k = pipe._b.k(shared)
    assert params[:, k + 1:k + 5].unique(dim=0).shape[0] == 1


# ------------------------------------------------------------- mixing
@pytest.mark.parametrize("cls", ["MixUp", "CutMix"])
def test_mixup_and_cutmix_match_torchvision(cls, batch):
    from torchvision.transforms import v2

    x = A.Compose([A.RandomResizedCrop(64), A.Normalize(MEAN, STD)],
                  dtype=torch.float32)(torch.cat([batch] * 2))
    labels = torch.arange(12, device="cuda") % 5
    for seed in range(5):
        torch.manual_seed(seed)
        ours = getattr(A, cls)(alpha=1.0, num_classes=5)(x, labels)
        torch.manual_seed(seed)
        theirs = getattr(v2, cls)(alpha=1.0, num_classes=5)(x, labels)
        assert (ours[0] - theirs[0]).abs().max() < 1e-6       # one rounding differs at most
        assert torch.equal(ours[1], theirs[1])


@pytest.mark.parametrize("mode", ["batch", "pair", "elem"])
def test_timm_style_mixing_follows_its_parameters(mode, batch):
    x = A.Compose([A.RandomResizedCrop(48), A.Normalize(MEAN, STD)],
                  dtype=torch.float32)(torch.cat([batch] * 4))
    labels = torch.arange(24, device="cuda") % 10
    m = A.MixUpCutMix(mixup_alpha=0.8, cutmix_alpha=1.0, mode=mode, num_classes=10)
    torch.manual_seed(0)
    out, y = m(x, labels)
    torch.manual_seed(0)
    partner, own, boxes, lam = m._params(*x.shape[:1], *x.shape[2:])
    ref = x.clone()
    for i, (j, a, (x1, y1, x2, y2)) in enumerate(zip(partner, own, boxes)):
        ref[i] = x[i] * a + x[j] * (1 - a)
        ref[i, :, y1:y2, x1:x2] = x[j, :, y1:y2, x1:x2]
    assert (out - ref).abs().max() < 1e-5
    assert torch.allclose(y, m._mix_labels(labels, partner, lam))
    assert torch.allclose(y.sum(1), torch.ones(24, device="cuda"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixing_keeps_the_output_dtype(dtype, batch):
    x32 = A.Compose([A.RandomResizedCrop(32), A.Normalize(MEAN, STD)],
                    dtype=torch.float32)(batch)
    x = x32.to(dtype)
    torch.manual_seed(0)
    out, _ = A.MixUp(alpha=1.0, num_classes=3)(x, torch.zeros(6, dtype=torch.long, device="cuda"))
    torch.manual_seed(0)
    ref, _ = A.MixUp(alpha=1.0, num_classes=3)(x.float(), torch.zeros(6, dtype=torch.long,
                                                                      device="cuda"))
    assert out.dtype == dtype
    assert (out.float() - ref.to(dtype).float()).abs().max() <= 2 * (ref.abs().max() * 2**-7)


def test_mixing_input_errors(batch):
    x = A.Compose([A.RandomResizedCrop(32)])(batch)
    with pytest.raises(ValueError, match="num_classes"):
        A.MixUp()(x, torch.zeros(6, dtype=torch.long, device="cuda"))
    with pytest.raises(ValueError, match="labels"):
        A.MixUp(num_classes=3)(x, torch.zeros(5, dtype=torch.long, device="cuda"))
    with pytest.raises(ValueError, match="CUDA"):
        A.MixUp(num_classes=3)(x.cpu(), torch.zeros(6, dtype=torch.long))
    empty_x, empty_y = A.CutMix(num_classes=3)(x[:0], torch.zeros(0, dtype=torch.long,
                                                                device="cuda"))
    assert empty_x.shape == (0, 3, 32, 32) and empty_y.shape == (0, 3)


# ------------------------------------------------------------------ bridges
def test_python_torch_bridge_is_scoped_to_slangvision(images):
    # with slangpy-torch installed this also compares the two bridges
    from slangpy.core import calldata

    pipe = A.Compose(RECIPES["imagenet"])
    out, params = pipe._run(images, "chw")
    was = spy.is_torch_bridge_using_fallback()
    allowed = calldata._ALLOW_TORCH_FALLBACK
    try:
        spy.set_torch_bridge_python_fallback(True)
        spy.set_allow_torch_fallback(False)
        # slangvision allows the Python bridge for its own calls only
        assert torch.equal(pipe._run(images, "chw", params=params)[0], out)
        assert calldata._ALLOW_TORCH_FALLBACK is False
    finally:
        spy.set_torch_bridge_python_fallback(was)
        spy.set_allow_torch_fallback(allowed)


# ------------------------------------------------------------------ warps
@pytest.mark.parametrize("transforms", [
    [A.RandomResizedCrop((96, 128)), A.RandomRotation(30, interpolation="bilinear")],
    [A.Resize((80, 100)), A.RandomAffine(20, translate=(0.2, 0.1), scale=(0.7, 1.3),
                                         shear=(-10, 10, -5, 5), interpolation="bilinear",
                                         fill=(0.2, 0.5, 0.8))],
    [A.Resize((63, 47)), A.RandomAffine((-180, 180), scale=(0.5, 1.5), shear=15,
                                        interpolation="bilinear", center=(5.5, 40.0))],
    [A.RandomResizedCrop(64), A.RandomPerspective(0.6, p=0.7)],
    [A.RandomResizedCrop((61, 77)), A.RandomPerspective(1.0, p=1.0, fill=0.5)],
    [A.RandomResizedCrop(64), A.RandomApply([A.RandomAffine(15, interpolation="bilinear")]),
     A.GaussianBlur(5), A.RandomHorizontalFlip()],
    [A.RandomResizedCrop(64), A.GaussianBlur(5), A.RandomPerspective(p=1.0),
     A.RandomChoice([A.RandomRotation(40, interpolation="bilinear"), A.RandomGrayscale(p=1.0)]),
     A.RandomErasing(p=1.0)],
])
def test_warps_match_torchvision(transforms, images):
    _check(transforms, images)


def test_flips_and_jitter_around_a_warp_match_torchvision(images):
    # float16: the resampled image is stored as float16 between the passes,
    # and the jitter after the warp scales that rounding by up to 1.4 * 1.4
    _check([A.RandomResizedCrop(64), A.RandomHorizontalFlip(),
            A.RandomRotation(25, interpolation="bilinear", center=(10, 40)),
            A.RandomVerticalFlip(), A.ColorJitter(0.4, 0.4, 0.4, 0.1), A.Normalize(MEAN, STD)],
           images, tol16=8e-3)


def test_flips_between_warps_and_after_your_neighbourhood_op_match_torchvision(images):
    # a flip after your NeighbourhoodOp is allowed once a warp comes between them
    _check([A.RandomResizedCrop(64), BoxBlur(), A.RandomRotation(20, interpolation="bilinear"),
            A.RandomHorizontalFlip(), A.RandomPerspective(p=0.8), A.RandomVerticalFlip(),
            A.RandomHorizontalFlip()], images)


def test_cifar_style_rotation_matches_torchvision(small):
    _check([A.RandomCrop(32, padding=4, pad_if_needed=True), A.RandomHorizontalFlip(),
            A.RandomRotation(15, interpolation="bilinear"), A.Normalize(MEAN, STD)], small)


@pytest.mark.parametrize("transforms", [
    [A.Resize((72, 88)), A.RandomRotation(30)],
    [A.Resize((72, 88)), A.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10,
                                        fill=0.3)],
    [A.RandomResizedCrop(64), A.RandomPerspective(p=1.0, interpolation="nearest")],
])
def test_nearest_warps_match_torchvision_but_for_rounding_ties(transforms, images):
    # torchvision computes the sampling grid in float32 with its own rounding,
    # so a position within ~1e-5 px of a pixel boundary can pick the other pixel
    torch.manual_seed(0)
    pipe = A.Compose(transforms, dtype=torch.float32)
    out, params = pipe._run(images, "chw")
    bad = ((out - _replay(pipe, images, params)).abs() > TOL32).float().mean().item()
    assert bad < 1e-3


def test_warp_random_values_follow_torchvision(batch):
    rot = A.RandomRotation((-10, 30))
    aff = A.RandomAffine(0, translate=(0.25, 0.1), scale=(0.8, 1.2), shear=(-5, 5, 2, 4))
    per = A.RandomPerspective(0.4, p=0.3)
    same = A.RandomRotation(90, same_on_batch=True)
    pipe = A.Compose([A.Resize((61, 90)), rot, aff, per, same])
    torch.manual_seed(0)
    _, params = pipe._run(batch.repeat(200, 1, 1, 1), "chw")
    k = pipe._b.k
    angle = params[:, k(rot) + 9]
    assert angle.min() >= -10 and angle.max() <= 30 and angle.std() > 9
    tx, ty, sc, shx, shy = params[:, k(aff) + 10:k(aff) + 15].T
    assert torch.equal(tx, tx.round()) and torch.equal(ty, ty.round())
    assert tx.abs().max() == 22 and ty.abs().max() == 6
    # rounded, not truncated: 0 takes 1/45 of (-22.5, 22.5), truncation 2/45
    assert abs((tx == 0).float().mean().item() - 1 / 45) < 0.012
    assert sc.min() >= 0.8 and sc.max() <= 1.2
    assert shx.abs().max() <= 5 and shy.min() >= 2 and shy.max() <= 4
    applied = params[:, k(per)] > 0.5
    assert abs(applied.float().mean().item() - 0.3) < 0.05
    # corners: bound int(0.4 * 90 // 2) + 1 = 19 across, int(0.4 * 61 // 2) + 1 = 13 down
    ex = params[:, k(per) + 9:k(per) + 17:2]
    ey = params[:, k(per) + 10:k(per) + 17:2]
    assert ex[:, [0, 3]].min() == 0 and ex[:, [0, 3]].max() == 18
    assert ex[:, [1, 2]].min() == 90 - 19 and ex[:, [1, 2]].max() == 89
    assert ey[:, [0, 1]].min() == 0 and ey[:, [0, 1]].max() == 12
    assert ey[:, [2, 3]].min() == 61 - 13 and ey[:, [2, 3]].max() == 60
    shared = params[:, k(same) + 9]
    assert torch.all(shared == shared[0])


# ------------------------------------------- policy ops and per-image statistics
def _mismatch(transforms, images, seed=0):
    """Share of output values off torchvision's by more than TOL32 (float32)."""
    pipe = A.Compose(transforms, dtype=torch.float32)
    torch.manual_seed(seed)
    out, params = pipe._run(images, "chw")
    return ((out - _replay(pipe, images, params)).abs() > TOL32).float().mean().item()


@pytest.mark.parametrize("op", [
    A.RandomInvert(p=1.0),
    A.RandomPosterize(3, p=1.0),
    A.RandomPosterize(0, p=1.0),
    A.RandomPosterize(8, p=1.0),
    A.RandomAutocontrast(p=1.0),
    A.RandomEqualize(p=1.0),
    A.RandomAdjustSharpness(2.5, p=1.0),
    A.RandomAdjustSharpness(0.0, p=1.0),
], ids=repr)
def test_policy_ops_on_unresampled_images_match_torchvision_exactly(op, batch):
    # without resampling every value is k/255, never on a posterize or
    # quantization edge, so even the step functions must match
    _check([op], batch)


@pytest.mark.parametrize("op", [
    A.RandomInvert(p=0.5),
    A.RandomAutocontrast(p=0.5),
    A.RandomAdjustSharpness(2.0, p=0.5),
    A.RandomApply([A.RandomAdjustSharpness(0.2, p=1.0)], p=0.5),
], ids=repr)
def test_continuous_policy_ops_match_torchvision(op, images):
    _check([A.RandomResizedCrop(96), A.ColorJitter(0.4, 0.4, 0.4, 0.1), op,
            A.Normalize(MEAN, STD)], images)


def test_sharpness_clamps_bicubic_overshoot_at_the_border(images):
    _check([A.RandomResizedCrop(96, interpolation="bicubic"), A.RandomAdjustSharpness(2.0, p=1.0)],
           images)


@pytest.mark.parametrize("op", [A.RandomPosterize(4, p=0.7), A.RandomEqualize(p=0.7)], ids=repr)
def test_step_policy_ops_match_torchvision_off_their_edges(op, images):
    # resampled values can land within float rounding of a quantization edge,
    # where the two sides may round differently (as RandomSolarize's threshold)
    assert _mismatch([A.RandomResizedCrop(96), A.ColorJitter(0.4, 0.4), op], images) < 2e-3


def test_statistics_in_later_passes_and_containers_match_torchvision(images):
    # the min/max and histograms are taken in the pass after a blur, after a
    # contrast change (its own statistic), and only for images that choose them
    assert _mismatch([A.RandomResizedCrop(96), A.GaussianBlur(5), A.ColorJitter(0.4, 0.6),
                      A.RandomChoice([A.RandomEqualize(p=1.0), A.RandomAutocontrast(p=1.0),
                                      A.RandomInvert(p=1.0), A.RandomAdjustSharpness(0.3, p=1.0)]),
                      A.RandomAutocontrast(p=0.5), A.RandomEqualize(p=0.5)], images) < 2e-3


def test_equalize_edge_cases_match_torchvision():
    # flat (PIL's step 0: the image is left quantized), two values, one
    # dominant value plus a ramp, and a channel without its maximum bin
    flat = torch.full((3, 40, 50), 100, dtype=torch.uint8, device="cuda")
    two = flat.clone()
    two[:, :, :25] = 30
    dominant = flat.clone()
    dominant[0, 0, :] = torch.arange(50, dtype=torch.uint8, device="cuda") * 5
    ramp = torch.arange(2000, device="cuda").remainder(200).to(torch.uint8).view(1, 40, 50)
    ramp = torch.cat([ramp, ramp.flip(-1), 255 - ramp])
    _check([A.RandomEqualize(p=1.0)], torch.stack([flat, two, dominant, ramp]))
    _check([A.RandomAutocontrast(p=1.0)], torch.stack([flat, two, dominant, ramp]))


def test_statistic_ops_with_p_0_leave_images_unchanged(batch):
    pipe = A.Compose([A.RandomEqualize(p=0.0), A.RandomAutocontrast(p=0.0)],
                     dtype=torch.float32)
    assert torch.equal(pipe(batch), batch.float() * (1.0 / 255.0))


# --------------------------------------------------------------- RandAugment
@pytest.fixture(scope="module")
def many(batch):
    """120 unresampled images: enough for every RandAugment op to come up."""
    return batch.repeat(20, 1, 1, 1)


@pytest.mark.parametrize("ra", [
    A.RandAugment(num_ops=1, interpolation="bilinear"),
    A.RandAugment(num_ops=1, magnitude=30, interpolation="bilinear", fill=(0.2, 0.4, 0.6)),
    A.RandAugment(num_ops=1),
    A.RandAugment(num_ops=1, magnitude=20, fill=0.5),
    A.RandAugment(num_ops=1, magnitude=0, num_magnitude_bins=5),
], ids=repr)
def test_one_rand_augment_op_on_unresampled_images_matches_torchvision(ra, many):
    # one op on k/255 values: the step ops (posterize, solarize, equalize) see
    # no edge cases, so only the nearest warps may differ, at rounding ties.
    # ShearX at magnitude 20 (0.2) about the corner puts every fifth row of
    # these sizes exactly on half pixels, where torch's float32 grid rounds
    # either way: those images differ on whole rows
    pipe = A.Compose([ra], dtype=torch.float32)
    torch.manual_seed(0)
    out, params = pipe._run(many, "chw")
    ops = params[:, pipe._b.k(ra._children()[0])].int()
    assert set(ops.tolist()) == set(range(14))
    bad = ((out - _replay(pipe, many, params)).abs() > TOL32).float().mean().item()
    limit = 1e-9 if ra.interpolation == "bilinear" else 5e-3 if ra.magnitude == 20 else 1e-3
    assert bad < limit


@pytest.mark.parametrize("ra", [
    A.RandAugment(),
    A.RandAugment(num_ops=3, magnitude=15, interpolation="bilinear"),
    A.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=11, fill=0.5),
], ids=repr)
def test_rand_augment_in_a_recipe_matches_torchvision_off_level_edges(ra, images):
    # after resampling, a value on a posterize/solarize/equalize edge can land
    # on either side, and nearest warps can pick the neighbouring pixel
    assert _mismatch([A.RandomResizedCrop(96), A.RandomHorizontalFlip(), ra,
                      A.RandomVerticalFlip(), A.Normalize(MEAN, STD)], images) < 3e-3


def test_statistic_ops_after_rand_augment_match_torchvision(many):
    # equalize and autocontrast after the op pass read their own statistics;
    # after a bilinear warp, equalize can meet a level edge
    assert _mismatch([A.RandAugment(num_ops=1, interpolation="bilinear"),
                      A.RandomAutocontrast(p=0.5), A.RandomEqualize(p=0.5)], many[:48]) < 2e-3


def test_rand_augment_float16_output(images):
    # the float16 image between passes moves values across equalize's 256
    # quantization levels far more often than float32 does, and each moved
    # value can shift the image's table: about 2% of values end a level
    # (1/255, 0.017 after Normalize) from torchvision's float32 result
    torch.manual_seed(0)
    pipe = A.Compose([A.RandomResizedCrop(96), A.RandAugment(interpolation="bilinear"),
                      A.Normalize(MEAN, STD)])
    out, params = pipe._run(images, "chw")
    d = (out.float() - _replay(pipe, images, params)).abs()
    assert (d > TOL16).float().mean().item() < 5e-2
    assert torch.quantile(d.flatten()[::7], 0.9).item() < TOL16


def test_rand_augment_draws_ops_and_signs_per_image(batch):
    ra = A.RandAugment(num_ops=2)
    shared = A.RandAugment(num_ops=1, same_on_batch=True)
    pipe = A.Compose([A.Resize((40, 50)), ra, shared])
    torch.manual_seed(0)
    _, params = pipe._run(batch.repeat(500, 1, 1, 1), "chw")
    for step in ra._children():
        k = pipe._b.k(step)
        counts = torch.bincount(params[:, k].long(), minlength=14).float() / len(params)
        assert (counts - 1 / 14).abs().max() < 0.015
        assert abs((params[:, k + 1] > 0).float().mean().item() - 0.5) < 0.03
    k = pipe._b.k(shared._children()[0])
    assert torch.all(params[:, k] == params[0, k]) and torch.all(params[:, k + 1] == params[0, k + 1])
