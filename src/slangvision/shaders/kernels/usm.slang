// shaders/kernels/usm.slang — BasicSR's USMSharp (Real-ESRGAN sharpens its
// high-resolution targets with it), with its 2D Gaussian split into a row
// and a column pass. Not part of the op library: slangvision/sr.py loads it
// on its own, after a `typealias T` line and the ld/st converters.
//
// out = soft * sharp + (1 - soft) * x, with blur = G * x (reflect padding),
// residual = x - blur, sharp = clamp(x + weight * residual, 0, 1), and soft =
// G * (|residual| * 255 > threshold) per channel. w holds the K = 2r + 1
// Gaussian taps.

// torch "reflect" padding (edge pixel not repeated); the caller keeps r < n
int reflect_index(int i, int n)
{
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i;
}

// pass 1: the image blurred along rows
void usm_rows(int3 pix, Tensor<T, 4> x, Tensor<float, 1> w, RWTensor<float, 4> tmp, int r,
              int W)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    for (int c = 0; c < 3; ++c)
    {
        float acc = 0.0;
        for (int j = 0; j < 2 * r + 1; ++j)
            acc += w[uint1(j)] * ld(x[uint4(reflect_index(X + j - r, W), Y, c, b)]);
        tmp[uint4(X, Y, c, b)] = acc;
    }
}

// pass 2: then along columns; keeps the residual x - blur
void usm_residual(int3 pix, Tensor<T, 4> x, Tensor<float, 4> tmp, Tensor<float, 1> w,
                  RWTensor<float, 4> res, int r, int H)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    for (int c = 0; c < 3; ++c)
    {
        float blur = 0.0;
        for (int i = 0; i < 2 * r + 1; ++i)
            blur += w[uint1(i)] * tmp[uint4(X, reflect_index(Y + i - r, H), c, b)];
        res[uint4(X, Y, c, b)] = ld(x[uint4(X, Y, c, b)]) - blur;
    }
}

// pass 3: the mask |residual| * 255 > threshold, blurred along rows
void usm_mask_rows(int3 pix, Tensor<float, 4> res, Tensor<float, 1> w, RWTensor<float, 4> tmp,
                   int r, float threshold, int W)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    for (int c = 0; c < 3; ++c)
    {
        float acc = 0.0;
        for (int j = 0; j < 2 * r + 1; ++j)
            acc += abs(res[uint4(reflect_index(X + j - r, W), Y, c, b)]) * 255.0 > threshold
                   ? w[uint1(j)] : 0.0;
        tmp[uint4(X, Y, c, b)] = acc;
    }
}

// pass 4: then along columns, and the blend
void usm_blend(int3 pix, Tensor<T, 4> x, Tensor<float, 4> tmp, Tensor<float, 4> res,
               Tensor<float, 1> w, RWTensor<T, 4> out, int r, float weight, int H)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    for (int c = 0; c < 3; ++c)
    {
        float soft = 0.0;
        for (int i = 0; i < 2 * r + 1; ++i)
            soft += w[uint1(i)] * tmp[uint4(X, reflect_index(Y + i - r, H), c, b)];
        float v = ld(x[uint4(X, Y, c, b)]);
        float sharp = clamp(v + weight * res[uint4(X, Y, c, b)], 0.0, 1.0);
        out[uint4(X, Y, c, b)] = st(soft * sharp + (1.0 - soft) * v);
    }
}
