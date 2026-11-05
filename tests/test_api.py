# tests/test_api.py
# slangvision pipeline and op checks that need no GPU and no torch:
# argument validation, pipeline structure rules and custom-op declarations.
# The GPU behaviour is covered by tests/test_pipelines.py.

from __future__ import annotations

import pytest

import slangvision as A

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Box(A.NeighbourhoodOp):
    slang = "float4 apply<I : IImage>(I img, int x, int y) { return img.load(x + 1, y); }"


@pytest.mark.parametrize("transforms, match", [
    ([A.ColorJitter(0.4), A.RandomResizedCrop(224)], "must come first"),
    ([A.RandomHorizontalFlip(), A.CenterCrop(224)], "must come first"),
    ([A.CenterCrop(256), A.Resize(224)], "one Resize or RandomResizedCrop"),
    ([A.RandomResizedCrop(224), A.Resize(200)], "one Resize or RandomResizedCrop"),
    ([A.RandomResizedCrop(16), A.RandomCrop(32)], "larger than the image"),
    ([A.Resize(256), A.Pad(4)], "follow it with CenterCrop or RandomCrop"),
    ([A.Resize(256)], "follow it with CenterCrop or RandomCrop"),
    ([A.Normalize(MEAN, STD), A.ColorJitter(0.4)], "Normalize must come after every transform"),
    ([A.RandomErasing(), A.Normalize(MEAN, STD)], "RandomErasing must come last"),
    ([A.RandomErasing(), A.ColorJitter(0.4)], "RandomErasing must come last"),
    ([A.RandomResizedCrop(64), Box(), A.RandomHorizontalFlip()], "before your NeighbourhoodOps"),
    ([A.RandomResizedCrop(8), A.GaussianBlur(23)], "larger than its radius"),
    ([A.Resize((64, 6)), A.RandomApply([A.GaussianBlur((13, 3))])], "larger than its radius"),
    ([A.RandomRotation(10), A.RandomResizedCrop(224)], "warps work on the image after them"),
    ([A.Resize((64, 64)), A.RandomAffine(5), A.CenterCrop(32)], "must come first"),
    ([A.RandomResizedCrop(64), A.RandomRotation(10), Box(), A.RandomHorizontalFlip()],
     "before your NeighbourhoodOps"),
])
def test_pipeline_structure_errors(transforms, match):
    with pytest.raises(ValueError, match=match):
        A.Compose(transforms)


def test_valid_pipelines_build_without_a_gpu():
    A.Compose([A.RandomResizedCrop(224), A.RandomHorizontalFlip(),
               A.RandomApply([A.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
               A.RandomGrayscale(p=0.2), A.RandomApply([A.GaussianBlur(23)], p=0.5),
               A.RandomSolarize(0.5, p=0.2), A.Normalize(MEAN, STD)])
    A.Compose([A.Resize(256), A.CenterCrop(224), A.Normalize(MEAN, STD)])
    A.Compose([A.Resize((200, 300))], dtype="float32")
    A.Compose([A.Resize((200, 300))], dtype="bfloat16")
    A.Compose([A.ColorJitter(0.4)])
    cifar = A.Compose([A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(), A.Normalize(MEAN, STD)])
    three = A.Compose([A.Resize(224, interpolation="bicubic"),
                       A.RandomCrop(224, padding=4, padding_mode="reflect"),
                       A.RandomHorizontalFlip(),
                       A.RandomChoice([A.RandomGrayscale(p=1.0), A.RandomSolarize(0.5, p=1.0),
                                       A.GaussianBlur(13)]),
                       A.ColorJitter(0.3, 0.3, 0.3), A.Normalize(MEAN, STD)])
    assert "sample_params" in cifar.source and "stage_1" in three.source
    assert A.Pad(4, fill=None).fill == (0.0, 0.0, 0.0)
    deit = A.Compose([A.RandomResizedCrop(224, interpolation="bicubic"),
                      A.RandomHorizontalFlip(), A.Normalize(MEAN, STD),
                      A.RandomErasing(p=0.25, scale=(0.02, 1 / 3), ratio=(0.3, 1 / 0.3),
                                      value="random"),
                      A.RandomErasing(p=0.25, value=(0.0, 0.5, -1.0))])
    assert deit.source.count("normal4(") >= 2      # the library function and its one use
    A.Compose([A.ColorJitter(0.4), A.RandomErasing()])      # without Normalize


def test_warps_start_a_pass_and_take_the_flips_after_them():
    pipe = A.Compose([A.RandomResizedCrop(64), A.RandomHorizontalFlip(), A.RandomRotation(15),
                      A.GaussianBlur(5), A.RandomVerticalFlip(), A.Normalize(MEAN, STD)])
    src = pipe.source
    assert "stage_2" in src and "stage_3" not in src         # resample, warp, blur
    k_h = pipe._b.k(pipe.transforms[1])
    k_v = pipe._b.k(pipe.transforms[4])
    input_0 = src[src.index("float4 input_0"):src.index("float4 input_1")]
    input_1 = src[src.index("float4 input_1"):src.index("float4 input_2")]
    assert f"prm(p, b, {k_h})" in input_0 and f"prm(p, b, {k_v})" not in input_0
    assert f"prm(p, b, {k_v})" in input_1      # after the warp, before the blur: commutes
    # a flip after your NeighbourhoodOp is fine when a warp follows the op
    A.Compose([A.RandomResizedCrop(64), Box(), A.RandomRotation(10), A.RandomHorizontalFlip()])
    A.Compose([A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(),
               A.RandomApply([A.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                             shear=(-5, 5, -2, 2), center=(3, 4))], p=0.5),
               A.RandomChoice([A.RandomPerspective(p=1.0), A.RandomGrayscale(p=1.0)]),
               A.Normalize(MEAN, STD)])
    A.Compose([A.RandomRotation((-90, 90), interpolation="bilinear", fill=(0.5, 0.5, 0.5))])


def test_rand_augment_builds_one_pass_per_op():
    ra = A.RandAugment(num_ops=3)
    pipe = A.Compose([A.RandomResizedCrop(224), A.RandomHorizontalFlip(), ra,
                      A.Normalize(MEAN, STD), A.RandomErasing(p=0.25)])
    src = pipe.source
    assert "stage_3" in src and "stage_4" not in src
    # contrast, autocontrast and equalize statistics for each op pass
    assert [kind for kind, _, _ in pipe._b.stat_groups] == ["mean", "minmax", "lut"] * 3
    assert "stage_0" in A.Compose([A.RandAugment(num_ops=0)]).source
    assert "RandAugment(num_ops=3" in repr(pipe)


def test_trivial_augment_and_auto_augment_build_their_passes():
    class Policy:                   # stands in for torchvision's AutoAugmentPolicy
        value = "svhn"

    assert A.AutoAugment(Policy()).policy == "svhn"
    for policy, passes in ((A.TrivialAugmentWide(), 1), (A.AutoAugment("cifar10"), 2)):
        pipe = A.Compose([A.RandomResizedCrop(224), policy, A.Normalize(MEAN, STD)])
        assert f"stage_{passes}" in pipe.source and f"stage_{passes + 1}" not in pipe.source
        assert [kind for kind, _, _ in pipe._b.stat_groups] == ["mean", "minmax", "lut"] * passes
    assert "AutoAugment(policy='cifar10'" in repr(A.AutoAugment("cifar10"))
    assert "TrivialAugmentWide(num_magnitude_bins=31" in repr(A.TrivialAugmentWide())


def test_timm_rand_augment_parses_timm_configs_and_builds_one_pass_per_layer():
    ra = A.TimmRandAugment()
    assert (ra.magnitude, ra.num_layers, ra.prob, ra.magnitude_std, ra.increasing) == \
        (9, 2, 0.5, 0.5, True)
    ra = A.TimmRandAugment("rand-m7-n3-p0.3-mmax20-mstd101-inc0-mstd1")
    assert (ra.magnitude, ra.num_layers, ra.prob, ra.magnitude_max, ra.increasing) == \
        (7, 3, 0.3, 20, False)
    assert ra.magnitude_std == float("inf")         # above 100: uniform; the first mstd wins
    assert A.TimmRandAugment("rand-inc-m5").magnitude == 5      # timm skips sections without a number
    other = A.TimmRandAugment("rand-m9-inc1-inc0-mmax0-p1.5")
    assert (other.increasing, other.magnitude_max, other.prob) == (False, 10, 1.5)  # current timm
    pipe = A.Compose([A.RandomResizedCrop(224), A.RandomHorizontalFlip(), ra,
                      A.Normalize(MEAN, STD)])
    assert "stage_3" in pipe.source and "stage_4" not in pipe.source
    assert [kind for kind, _, _ in pipe._b.stat_groups] == ["mean_pil", "minmax", "lut"] * 3
    assert "TimmRandAugment('rand-m7-n3" in repr(pipe)
    assert "stage_0" in A.Compose([A.TimmRandAugment("rand-n0")]).source


@pytest.mark.parametrize("make, match", [
    (lambda: A.RandomSolarize(128), "0-1 scale"),
    (lambda: A.ColorJitter(brightness=-0.1), "non-negative"),
    (lambda: A.ColorJitter(hue=0.6), "lie in"),
    (lambda: A.ColorJitter(contrast=(1.5, 0.5)), "lie in"),
    (lambda: A.ColorJitter(brightness=float("inf")), "finite"),
    (lambda: A.Uniform(0.0, float("inf")), "finite"),
    (lambda: A.Choice([1.0, float("nan")]), "finite"),
    (lambda: A.GaussianBlur(4), "positive and odd"),
    (lambda: A.GaussianBlur(5, sigma=(2.0, 1.0)), "sigma"),
    (lambda: A.RandomHorizontalFlip(p=1.5), "p must be in"),
    (lambda: A.RandomResizedCrop(0), "positive"),
    (lambda: A.RandomResizedCrop(224, scale=(0.5, 0.1)), "scale"),
    (lambda: A.CenterCrop((1, 2, 3)), "size"),
    (lambda: A.Normalize((0.5, 0.5), (0.5, 0.5)), "one value or three"),
    (lambda: A.RandomPhotometricDistort(brightness=0.4), r"\(min, max\)"),
    (lambda: A.RandomPhotometricDistort(hue=(-0.6, 0.0)), "RandomPhotometricDistort: hue"),
    (lambda: A.RandomPhotometricDistort(p=1.5), "p must be in"),
    (lambda: A.Grayscale(2), "1 or 3"),
    (lambda: A.Compose([A.RandomApply([A.Grayscale()], p=0.5)]), "channel count"),
    (lambda: A.Compose([A.Grayscale(), A.Normalize(MEAN, STD)]), "one value"),
    (lambda: A.Compose([A.Grayscale(), A.RandomErasing(value=(0.1, 0.2, 0.3))]), "one value"),
    (lambda: A.Compose([A.Grayscale(), A.Normalize((0.5,) * 3, (0.5,) * 3)]), "one value"),
    (lambda: A.Compose([A.RandomChoice([A.Grayscale(), A.RandomInvert(p=1.0)])]), "channel count"),
    (lambda: A.Compose([A.Grayscale(), A.Grayscale(3)]), "channel count again"),
    (lambda: A.Compose([A.Grayscale(), A.RandomApply([A.RandomRotation(9, fill=(1, 0, 0))])]),
     "one value"),
    (lambda: A.Compose([A.Grayscale(), A.TimmRandAugment()]), "one value"),
    (lambda: A.Normalize(MEAN, (0.2, 0.0, 0.2)), "non-zero"),
    (lambda: A.RandomApply([]), "at least one"),
    (lambda: A.RandomApply([A.GaussianBlur(3), A.ColorJitter(0.1)]), "exactly one"),
    (lambda: A.RandomApply([A.CenterCrop(8)]), "crops"),
    (lambda: A.Compose([A.ColorJitter(0.4)], dtype="float64"), "dtype"),
    (lambda: A.Uniform(1.0, 0.0), "low <= high"),
    (lambda: A.Compose([A.RandomResizedCrop(8)] + [A.RandomHorizontalFlip()] * 2), "appears twice"),
    (lambda: A.Choice([]), "at least one"),
    (lambda: A.GaussianBlur(3, sigma=[1.0]), "sigma"),
    (lambda: A.RandomResizedCrop(8, interpolation="nearest"), "interpolation"),
    (lambda: A.RandomCrop(32, padding=4, fill=255), "0-1 scale"),
    (lambda: A.Pad(4, padding_mode="wrap"), "padding_mode"),
    (lambda: A.Pad((1, 2, 3)), "padding"),
    (lambda: A.Pad(-1), "non-negative"),
    (lambda: A.Pad("4"), "padding"),
    (lambda: A.Pad([2.7]), "padding"),
    (lambda: A.RandomChoice([]), "at least one"),
    (lambda: A.RandomErasing(value="noise"), "'random'"),
    (lambda: A.RandomErasing(value=(0.1, 0.2)), "three numbers"),
    (lambda: A.RandomErasing(scale=(0.5, 0.2)), "scale"),
    (lambda: A.RandomErasing(scale=(0.1, 1.5)), "scale"),
    (lambda: A.RandomErasing(ratio=(0.0, 3.0)), "ratio"),
    (lambda: A.RandomErasing(p=2.0), "p must be in"),
    (lambda: A.RandomChoice([A.RandomGrayscale()], p=[1, 2]), "one non-negative weight"),
    (lambda: A.RandomChoice([A.GaussianBlur(3), A.GaussianBlur(5)]), "at most one"),
    (lambda: A.RandomChoice([A.RandomVerticalFlip()]), "flips"),
    (lambda: A.Compose([A.RandomResizedCrop(16), A.Pad(16, padding_mode="reflect")]), "reflect padding"),
    (lambda: A.Resize(8, interpolation="lanczos"), "interpolation"),
    (lambda: A.GaussianBlur((3, 5, 7)), "kernel_size"),
    (lambda: A.RandomApply([A.RandomHorizontalFlip()]), "flips"),
    (lambda: A.RandomRotation(-5), "non-negative"),
    (lambda: A.RandomRotation((1, 2, 3)), "sequence of 2"),
    (lambda: A.RandomRotation(10, expand=True), "expand"),
    (lambda: A.RandomRotation(10, interpolation="bicubic"), "nearest or bilinear"),
    (lambda: A.RandomRotation(10, fill=255), "0-1 scale"),
    (lambda: A.RandomRotation(10, center=(1, 2, 3)), "center"),
    (lambda: A.RandomAffine(10, translate=(0.1, 1.5)), "translate"),
    (lambda: A.RandomAffine(10, scale=(0.0, 1.0)), "scale"),
    (lambda: A.RandomAffine(10, shear=(1, 2, 3)), "shear"),
    (lambda: A.RandomAffine(float("nan")), "finite"),
    (lambda: A.RandomPerspective(1.5), "distortion_scale"),
    (lambda: A.RandomPerspective(p=-0.1), "p must be in"),
    (lambda: A.RandomApply([A.RandomRotation(5), A.RandomGrayscale()]), "exactly one"),
    (lambda: A.RandomPosterize(9), "bits"),
    (lambda: A.RandAugment(num_ops=-1), "num_ops"),
    (lambda: A.RandAugment(magnitude=31), "magnitude"),
    (lambda: A.RandAugment(magnitude=1.5), "magnitude"),
    (lambda: A.RandAugment(num_magnitude_bins=1, magnitude=0), "num_magnitude_bins"),
    (lambda: A.RandAugment(interpolation="bicubic"), "nearest or bilinear"),
    (lambda: A.RandAugment(fill=2.0), "0-1 scale"),
    (lambda: A.RandomApply([A.RandAugment()]), "policies"),
    (lambda: A.Compose([A.RandAugment(), A.RandomResizedCrop(64)]), "must come first"),
    (lambda: A.GaussianNoise(sigma=-0.1), "non-negative"),
    (lambda: A.GaussianNoise(mean=float("nan")), "finite"),
    (lambda: A.Compose([A.Normalize(MEAN, STD), A.GaussianNoise()]), "Normalize must come after"),
    (lambda: A.TrivialAugmentWide(num_magnitude_bins=1), "num_magnitude_bins"),
    (lambda: A.TrivialAugmentWide(interpolation="bicubic"), "nearest or bilinear"),
    (lambda: A.AutoAugment("reduced_imagenet"), "imagenet, cifar10 or svhn"),
    (lambda: A.AutoAugment(fill=-0.1), "0-1 scale"),
    (lambda: A.RandomChoice([A.AutoAugment()]), "policies"),
    (lambda: A.TimmRandAugment("augmix-m5"), "start with 'rand'"),
    (lambda: A.TimmRandAugment("rand-m9-tweights"), "transform sets"),
    (lambda: A.TimmRandAugment("rand-m9-w0"), "unknown config section"),
    (lambda: A.TimmRandAugment("rand-m9.5"), "bad value"),
    (lambda: A.TimmRandAugment(interpolation="nearest"), "random, bilinear or bicubic"),
    (lambda: A.TimmRandAugment(fill=124), "0-1 scale"),
    (lambda: A.TimmRandAugment(translate_pct=float("nan")), "finite"),
    (lambda: A.RandomChoice([A.TimmRandAugment()]), "policies"),
    (lambda: A.RandomPosterize(4.0), "bits"),
    (lambda: A.RandomAdjustSharpness(-1.0), "non-negative"),
    (lambda: A.RandomEqualize(p=1.5), "p must be in"),
    (lambda: A.RandomChoice([A.RandomAdjustSharpness(2.0), A.GaussianBlur(3)]), "at most one"),
    (lambda: A.RandomChoice([A.RandomRotation(5), A.GaussianBlur(3)]), "at most one"),
    (lambda: A.RandomApply([A.RandomHorizontalFlip(), A.ColorJitter(0.1)]), "flips"),
    (lambda: A.RandomChoice([A.RandomHorizontalFlip(), A.CenterCrop(8)]), "flips"),
])
def test_argument_errors(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_numpy_scalars_and_short_sequences_are_accepted():
    np = pytest.importorskip("numpy")
    assert A.Resize(np.int64(224)).size == (224, 224)
    assert A.ColorJitter(np.float32(0.4)).ranges[0] is not None
    assert A.GaussianBlur([3]).kernel_size == (3, 3)

    class Mode:                     # stands in for torchvision's InterpolationMode
        value = "bicubic"

    assert A.RandomResizedCrop(8, interpolation=Mode()).interpolation == "bicubic"
    assert A.Resize(8, interpolation="BICUBIC").interpolation == "bicubic"
    assert A.GaussianBlur(np.int64(5), sigma=np.float64(1.0)).kernel_size == (5, 5)


def test_custom_op_without_super_init_or_params():
    class Invert(A.ColorOp):
        slang = "float4 apply(float4 c) { return 1.0 - c; }"

        def __init__(self):            # forgets super().__init__()
            pass

    op = Invert()
    assert op.p == 1.0 and op.same_on_batch is False and dict(op.params) == {}
    A.Compose([A.CenterCrop(8), op])
    with pytest.raises(TypeError):
        op.params["x"] = 1.0           # the shared default is read-only


def test_custom_op_declaration_errors():
    class NoApply(A.ColorOp):
        slang = "float4 f(float4 c) { return c; }"

    class BadParams(A.ColorOp):
        params = {"gamma": "big"}
        slang = "float4 apply(float4 c, float g) { return c; }"

    class BadName(A.ColorOp):
        params = {"not a name": 1.0}
        slang = "float4 apply(float4 c, float g) { return c; }"

    class BadBorder(Box):
        border = "wrap"

    with pytest.raises(ValueError, match="must define"):
        A.Compose([NoApply()])
    with pytest.raises(TypeError, match="Uniform, Choice or a number"):
        A.Compose([BadParams()])
    with pytest.raises(ValueError, match="valid name"):
        A.Compose([BadName()])
    with pytest.raises(ValueError, match="border"):
        A.Compose([A.RandomApply([BadBorder()])])
    with pytest.raises(TypeError, match="slangvision transforms"):
        A.Compose([lambda img: img])
    with pytest.raises(TypeError, match="slangvision transforms"):
        A.RandomApply([lambda img: img])
