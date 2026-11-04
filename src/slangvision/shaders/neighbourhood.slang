// shaders/neighbourhood.slang — what ops that read neighbouring pixels see
// (GaussianBlur and user NeighbourhoodOps): the IImage accessor, border
// handling and the Gaussian weight.

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

// torchvision's gaussian kernel weight (unnormalized) at offset d
float gauss(int d, float sigma) { float x = float(d) / sigma; return exp(-0.5 * x * x); }
