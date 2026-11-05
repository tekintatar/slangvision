// shaders/neighbourhood.slang — what ops that read neighbouring pixels see
// (GaussianBlur, sharpness and user NeighborhoodOps): the IImage accessor,
// border handling, the Gaussian weight and torchvision's sharpness.

// Neighbourhood ops read the previous pass's output through this accessor;
// load applies the op's border mode.
interface IImage
{
    float4 load(int x, int y);
    int width();
    int height();
}

// torch "reflect" padding (edge pixel not repeated), then clamped so radii
// larger than the image stay in bounds.
int reflect(int i, int n)
{
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return clamp(i, 0, n - 1);
}

// torchvision's adjust_sharpness_image with alpha = 1 - sharpness_factor:
// blend toward its 3x3 smoothing kernel (1/13 around, 5/13 centre); border
// pixels are only clamped, as torchvision clamps the whole output
float4 sharpen<I : IImage>(I img, int x, int y, float alpha)
{
    float4 c = img.load(x, y);
    if (x == 0 || y == 0 || x == img.width() - 1 || y == img.height() - 1)
        return clamp01(c);
    float4 ring = float4(0.0);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (dx != 0 || dy != 0)
                ring += img.load(x + dx, y + dy);
    float4 blur = ring * (1.0 / 13.0) + c * (5.0 / 13.0);
    return clamp01(c + alpha * (blur - c));
}

// torchvision's gaussian kernel weight (unnormalized) at offset d
float gauss(int d, float sigma) { float x = float(d) / sigma; return exp(-0.5 * x * x); }
