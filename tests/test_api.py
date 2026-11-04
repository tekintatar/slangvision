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
    ([A.Resize(256), A.CenterCrop(240), A.CenterCrop(224)], "one CenterCrop"),
    ([A.Resize(256)], "follow it with CenterCrop"),
    ([A.Normalize(MEAN, STD), A.ColorJitter(0.4)], "Normalize must be the last"),
    ([A.RandomResizedCrop(64), Box(), A.RandomHorizontalFlip()], "before your NeighbourhoodOps"),
    ([A.RandomResizedCrop(8), A.GaussianBlur(23)], "larger than its radius"),
    ([A.Resize((64, 6)), A.RandomApply([A.GaussianBlur((13, 3))])], "larger than its radius"),
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
    (lambda: A.Normalize((0.5, 0.5), (0.5, 0.5)), "three values"),
    (lambda: A.Normalize(MEAN, (0.2, 0.0, 0.2)), "non-zero"),
    (lambda: A.RandomApply([]), "at least one"),
    (lambda: A.RandomApply([A.GaussianBlur(3), A.ColorJitter(0.1)]), "exactly one"),
    (lambda: A.RandomApply([A.CenterCrop(8)]), "geometric"),
    (lambda: A.Compose([A.ColorJitter(0.4)], dtype="float64"), "dtype"),
    (lambda: A.Uniform(1.0, 0.0), "low <= high"),
    (lambda: A.Compose([A.RandomResizedCrop(8)] + [A.RandomHorizontalFlip()] * 2), "appears twice"),
    (lambda: A.Choice([]), "at least one"),
    (lambda: A.GaussianBlur(3, sigma=[1.0]), "sigma"),
    (lambda: A.GaussianBlur((3, 5, 7)), "kernel_size"),
    (lambda: A.RandomApply([A.RandomHorizontalFlip()]), "flips"),
])
def test_argument_errors(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_numpy_scalars_and_short_sequences_are_accepted():
    np = pytest.importorskip("numpy")
    assert A.Resize(np.int64(224)).size == (224, 224)
    assert A.ColorJitter(np.float32(0.4)).ranges[0] is not None
    assert A.GaussianBlur([3]).kernel_size == (3, 3)
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
