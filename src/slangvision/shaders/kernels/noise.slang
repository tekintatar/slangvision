// shaders/kernels/noise.slang — BasicSR's Gaussian and Poisson noise
// (basicsr.data.degradations, *_pt). Not part of the op library:
// slangvision/sr.py loads it after a `typealias T` line and the ld/st
// converters.
//
// prm row b: strength (Gaussian: sigma on the 0-255 scale; Poisson: scale),
// gray weight g (the noise is colour * (1 - g) + gray * g), for Poisson the
// image's levels for colour and for gray (2**ceil(log2(distinct values)),
// from `levels`), and last the call's noise key. Noise is drawn from a
// counter-based stream per pixel: key, image, position.

// core.slang's counter-based generator (this module does not include core)
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint s)
{
    s = pcg(s);
    return float(s >> 8) * (1.0 / 16777216.0);
}

uint pixel_stream(uint key, int b, int x, int y)
{
    return pcg(key ^ pcg(uint(b) * 2654435769u ^ pcg(uint(x) * 40503u + uint(y) * 2246822519u)));
}

// torch.round: half to even
float rne(float x)
{
    float r = floor(x);
    float f = x - r;
    if (f > 0.5 || (f == 0.5 && fmod(r, 2.0) != 0.0))
        r += 1.0;
    return r;
}

// torch.round(v * 255) clamped to 0-255
int level255(float v) { return int(clamp(rne(v * 255.0), 0.0, 255.0)); }

// torchvision's rgb_to_grayscale, in its order of operations
float gray_tv(float r, float g, float b) { return (0.2989 * r + 0.587 * g) + 0.114 * b; }

float finish(float v, int clip, int rounds)
{
    if (rounds != 0) v = rne(v * 255.0) / 255.0;
    return clip != 0 ? clamp(v, 0.0, 1.0) : v;
}

float normal1(inout uint s)
{
    float u1 = max(rnd(s), 1e-7);
    float u2 = rnd(s);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

void gaussian_noise(int3 pix, Tensor<T, 4> x, RWTensor<T, 4> out, Tensor<float, 2> prm, int clip,
                    int rounds)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    float sigma = prm[uint2(0, b)] * (1.0 / 255.0);
    float g = prm[uint2(1, b)];
    uint s = pixel_stream(uint(prm[uint2(2, b)]), b, X, Y);
    float gn = g > 0.0 ? normal1(s) * sigma : 0.0;
    for (int c = 0; c < 3; ++c)
    {
        float n = g < 1.0 ? normal1(s) * sigma : 0.0;
        float v = ld(x[uint4(X, Y, c, b)]) + (n * (1.0 - g) + gn * g);
        out[uint4(X, Y, c, b)] = st(finish(v, clip, rounds));
    }
}

// which of the 256 levels each image uses: colour (all channels) and gray
void levels(int3 pix, Tensor<T, 4> x, RWTensor<int, 2> color, RWTensor<int, 2> gray)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    float r = ld(x[uint4(X, Y, 0, b)]);
    float g = ld(x[uint4(X, Y, 1, b)]);
    float bl = ld(x[uint4(X, Y, 2, b)]);
    // read first: most pixels find their level already marked, and skipping
    // the store avoids every thread writing the same few cells
    int l0 = level255(r); int l1 = level255(g); int l2 = level255(bl);
    int lg = level255(gray_tv(r, g, bl));
    if (color[uint2(l0, b)] == 0) color[uint2(l0, b)] = 1;
    if (color[uint2(l1, b)] == 0) color[uint2(l1, b)] = 1;
    if (color[uint2(l2, b)] == 0) color[uint2(l2, b)] = 1;
    if (gray[uint2(lg, b)] == 0) gray[uint2(lg, b)] = 1;
}

// numpy's random_loggam: log(Gamma(x)), in float
float loggam(float x)
{
    if (x == 1.0 || x == 2.0)
        return 0.0;
    int n = x < 7.0 ? int(7.0 - x) : 0;
    float x0 = x + float(n);
    float x2 = (1.0 / x0) * (1.0 / x0);
    float gl0 = -1.39243221690590;
    gl0 = gl0 * x2 + 1.796443723688307e-01;
    gl0 = gl0 * x2 - 2.955065359477124e-02;
    gl0 = gl0 * x2 + 6.410256410256410e-03;
    gl0 = gl0 * x2 - 1.917526917526918e-03;
    gl0 = gl0 * x2 + 8.417508417508418e-04;
    gl0 = gl0 * x2 - 5.952380952380952e-04;
    gl0 = gl0 * x2 + 7.936507936507937e-04;
    gl0 = gl0 * x2 - 2.777777777777778e-03;
    gl0 = gl0 * x2 + 8.333333333333333e-02;
    float gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * log(x0) - x0;
    for (int k = 1; k <= n; ++k)
    {
        gl -= log(x0 - 1.0);
        x0 -= 1.0;
    }
    return gl;
}

// A Poisson(lam) sample: multiplication of uniforms below 10, numpy's PTRS
// (Hormann's transformed rejection with squeeze) above.
float poisson(float lam, inout uint s)
{
    if (!(lam > 0.0))
        return 0.0;
    if (lam < 10.0)
    {
        float limit = exp(-lam);
        float prod = rnd(s);
        int k = 0;
        while (prod > limit)
        {
            prod *= rnd(s);
            ++k;
        }
        return float(k);
    }
    float slam = sqrt(lam);
    float loglam = log(lam);
    float bb = 0.931 + 2.53 * slam;
    float a = -0.059 + 0.02483 * bb;
    float invalpha = 1.1239 + 1.1328 / (bb - 3.4);
    float vr = 0.9277 - 3.6224 / (bb - 2.0);
    for (int it = 0; it < 1000; ++it)
    {
        float U = rnd(s) - 0.5;
        float V = rnd(s);
        float us = 0.5 - abs(U);
        float k = floor((2.0 * a / us + bb) * U + lam + 0.43);
        if (us >= 0.07 && V <= vr)
            return k;
        if (k < 0.0 || (us < 0.013 && V > us))
            continue;
        // float32 here: its rounding moves the acceptance edge by ~1e-4 of
        // the density, far below what the tests (exact pmf) can see
        if (log(V) + log(invalpha) - log(a / (us * us) + bb) <= -lam + k * loglam - loggam(k + 1.0))
            return k;
    }
    return floor(lam + 0.5);
}

void poisson_noise(int3 pix, Tensor<T, 4> x, RWTensor<T, 4> out, Tensor<float, 2> prm, int clip,
                   int rounds)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    float scale = prm[uint2(0, b)];
    float g = prm[uint2(1, b)];
    float vals = prm[uint2(2, b)];
    float gvals = prm[uint2(3, b)];
    uint s = pixel_stream(uint(prm[uint2(4, b)]), b, X, Y);
    float v0 = ld(x[uint4(X, Y, 0, b)]);
    float v1 = ld(x[uint4(X, Y, 1, b)]);
    float v2 = ld(x[uint4(X, Y, 2, b)]);
    float gn = 0.0;
    if (g > 0.0)
    {
        float q = float(level255(gray_tv(v0, v1, v2))) / 255.0;
        gn = poisson(q * gvals, s) / gvals - q;
    }
    for (int c = 0; c < 3; ++c)
    {
        float v = c == 0 ? v0 : (c == 1 ? v1 : v2);
        float n = 0.0;
        if (g < 1.0)
        {
            float q = float(level255(v)) / 255.0;
            n = poisson(q * vals, s) / vals - q;
        }
        out[uint4(X, Y, c, b)] = st(finish(v + (n * (1.0 - g) + gn * g) * scale, clip, rounds));
    }
}
