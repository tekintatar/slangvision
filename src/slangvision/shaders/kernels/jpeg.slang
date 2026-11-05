// shaders/kernels/jpeg.slang — BasicSR's DiffJPEG (JPEG compression and
// decompression in float, as Real-ESRGAN degrades its inputs) on a batch
// padded to a multiple of 16. Not part of the op library: slangvision/sr.py
// loads it after a `typealias T` line and the ld/st converters.
//
// One float buffer per image holds the three planes, so every 8x8 block is
// at a multiple of 8 and one dispatch covers them all: rows [0, Hp) luma
// (Hp x Wp), rows [Hp, 3Hp/2) Cb in columns [0, Wp/2) and Cr in [Wp/2, Wp).
// p holds samples (0-255 scale), q the dequantized DCT coefficients: (u, v)
// of a block at (row0 + u, col0 + v), u the vertical frequency. Both DCTs
// run separably, one thread per block row, with
//   cs[x*8 + u] = cos((2x + 1) u pi / 16) and alpha[u*8 + v] = alpha(u)
//   alpha(v), alpha(0) = 1/sqrt(2), else 1 (BasicSR's float32 values).
// tab: the base Y then C tables (64 each, [u*8 + v]).

// torch.round: half to even
float rne(float x)
{
    float r = floor(x);
    float f = x - r;
    if (f > 0.5 || (f == 0.5 && fmod(r, 2.0) != 0.0))
        r += 1.0;
    return r;
}

// one input value (pad 0: black beyond the image, as BasicSR; 1: the edge
// pixels repeated, as libjpeg)
float sample(Tensor<T, 4> x, int X, int Y, int c, int b, int H, int W, int pad)
{
    if (pad == 0 && (X >= W || Y >= H))
        return 0.0;
    return ld(x[uint4(min(X, W - 1), min(Y, H - 1), c, b)]);
}

// pass 1, per chroma sample: RGB -> YCbCr (JFIF) on the 0-255 scale for its
// 2x2 pixels, Cb and Cr averaged over them
void jpeg_ycc(int3 pix, Tensor<T, 4> x, RWTensor<float, 3> p, int H, int W, int Hp, int Wp,
              int pad)
{
    int j = pix.x; int i = pix.y; int b = pix.z;
    float cb = 0.0;
    float cr = 0.0;
    for (int dy = 0; dy < 2; ++dy)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            int X = 2 * j + dx; int Y = 2 * i + dy;
            float r = sample(x, X, Y, 0, b, H, W, pad) * 255.0;
            float g = sample(x, X, Y, 1, b, H, W, pad) * 255.0;
            float bl = sample(x, X, Y, 2, b, H, W, pad) * 255.0;
            p[uint3(X, Y, b)] = 0.299 * r + 0.587 * g + 0.114 * bl;
            cb += (-0.168736 * r - 0.331264 * g + 0.5 * bl) + 128.0;
            cr += (0.5 * r - 0.418688 * g - 0.081312 * bl) + 128.0;
        }
    }
    p[uint3(j, Hp + i, b)] = cb / 4.0;
    p[uint3(Wp / 2 + j, Hp + i, b)] = cr / 4.0;
}

float4 row4(Tensor<float, 3> t, int col, int row, int b)
{
    return float4(t[uint3(col, row, b)], t[uint3(col + 1, row, b)], t[uint3(col + 2, row, b)],
                  t[uint3(col + 3, row, b)]);
}

float4 tab4(Tensor<float, 1> t, int k)
{
    return float4(t[uint1(k)], t[uint1(k + 1)], t[uint1(k + 2)], t[uint1(k + 3)]);
}

// cs[y*8 + v] for y = y0..y0+3: a column of the cosine table
float4 col4(Tensor<float, 1> cs, int y0, int v)
{
    return float4(cs[uint1(y0 * 8 + v)], cs[uint1(y0 * 8 + 8 + v)], cs[uint1(y0 * 8 + 16 + v)],
                  cs[uint1(y0 * 8 + 24 + v)]);
}

// the image's quality factor (table = base * factor) or, with integer
// tables, libjpeg's scale (table = clamp(floor((base * scale + 50) / 100), 1,
// 255)): from its quality in qual (BasicSR's quality_to_factor in float32,
// as on a quality tensor; libjpeg's jpeg_quality_scaling of the rounded
// quality), or the call's value when per_image is 0. Done here rather than
// in torch, where it takes about seven launches (+0.085 ms per call).
float table_scale(Tensor<float, 1> qual, int b, int per_image, float value, int int_tables)
{
    if (per_image == 0)
        return value;
    float q = qual[uint1(b)];
    if (int_tables == 0)
        return (q < 50.0 ? 5000.0 / q : 200.0 - q * 2.0) / 100.0;
    q = clamp(rne(q), 1.0, 100.0);
    return q < 50.0 ? floor(5000.0 / q) : 200.0 - q * 2.0;
}

// pass 2, per block row u: the forward DCT's row u of coefficients,
// quantized and dequantized; soft: BasicSR's differentiable rounding
// round(t) + (t - round(t))^3
void jpeg_quant(int3 pix, Tensor<float, 3> p, Tensor<float, 1> cs, Tensor<float, 1> alpha,
                Tensor<float, 1> tab, Tensor<float, 1> qual, RWTensor<float, 3> q, int Hp,
                int per_image, float value, int int_tables, int soft)
{
    int col0 = pix.x * 8; int row = pix.y; int b = pix.z;
    int u = row & 7; int row0 = row - u;
    int t0 = row < Hp ? 0 : 64;                    // luma or chroma table
    // the columns weighted by cos((2x + 1) u pi / 16), summed over the rows x
    float4 lo = float4(0.0);
    float4 hi = float4(0.0);
    for (int x = 0; x < 8; ++x)
    {
        float c = cs[uint1(x * 8 + u)];
        lo += c * (row4(p, col0, row0 + x, b) - 128.0);
        hi += c * (row4(p, col0 + 4, row0 + x, b) - 128.0);
    }
    float f = table_scale(qual, b, per_image, value, int_tables);
    for (int v = 0; v < 8; ++v)
    {
        float base = tab[uint1(t0 + u * 8 + v)];
        float qt = int_tables != 0 ? clamp(floor((base * f + 50.0) / 100.0), 1.0, 255.0) : base * f;
        // BasicSR's scale: alpha(u) alpha(v) / 4 (exact in float)
        float t = 0.25 * alpha[uint1(u * 8 + v)] * (dot(lo, col4(cs, 0, v)) + dot(hi, col4(cs, 4, v)))
                  / qt;
        float r = rne(t);
        if (soft != 0)
        {
            float d = t - r;
            r += d * d * d;
        }
        q[uint3(col0 + v, row, b)] = r * qt;
    }
}

// pass 3, per block row sx: the inverse DCT's row sx of samples (written
// over pass 1's samples)
void jpeg_idct(int3 pix, Tensor<float, 3> q, Tensor<float, 1> cs, Tensor<float, 1> alpha,
               RWTensor<float, 3> p)
{
    int col0 = pix.x * 8; int row = pix.y; int b = pix.z;
    int sx = row & 7; int row0 = row - sx;
    float4 lo = float4(0.0);
    float4 hi = float4(0.0);
    for (int u = 0; u < 8; ++u)
    {
        float c = cs[uint1(sx * 8 + u)];
        lo += c * (row4(q, col0, row0 + u, b) * tab4(alpha, u * 8));
        hi += c * (row4(q, col0 + 4, row0 + u, b) * tab4(alpha, u * 8 + 4));
    }
    for (int sy = 0; sy < 8; ++sy)
        p[uint3(col0 + sy, row, b)] = 0.25 * (dot(lo, tab4(cs, sy * 8)) + dot(hi, tab4(cs, sy * 8 + 4)))
                                      + 128.0;
}

// chroma at the output pixel (X, Y) from the plane at column c0: the sample
// repeated over 2x2 pixels, or libjpeg's "fancy" triangle filter (9/16,
// 3/16, 3/16, 1/16 of the nearest samples; edge samples repeated) within the
// image's ch x cw samples
float chroma_at(Tensor<float, 3> p, int X, int Y, int c0, int Hp, int b, int ch, int cw, int fancy)
{
    int i = Y >> 1; int j = X >> 1;
    float s = p[uint3(c0 + j, Hp + i, b)];
    if (fancy == 0)
        return s;
    int i2 = clamp((Y & 1) != 0 ? i + 1 : i - 1, 0, ch - 1);
    int j2 = clamp((X & 1) != 0 ? j + 1 : j - 1, 0, cw - 1);
    float near = 0.75 * s + 0.25 * p[uint3(c0 + j2, Hp + i, b)];
    float far = 0.75 * p[uint3(c0 + j, Hp + i2, b)] + 0.25 * p[uint3(c0 + j2, Hp + i2, b)];
    return 0.75 * near + 0.25 * far;
}

// pass 4, per output pixel: chroma upsampled, YCbCr -> RGB, clamped to 0-255
// and back to 0-1
void jpeg_out(int3 pix, Tensor<float, 3> p, RWTensor<T, 4> out, int Hp, int Wp, int ch, int cw,
              int fancy)
{
    int X = pix.x; int Y = pix.y; int b = pix.z;
    float y = p[uint3(X, Y, b)];
    float cb = chroma_at(p, X, Y, 0, Hp, b, ch, cw, fancy) - 128.0;
    float cr = chroma_at(p, X, Y, Wp / 2, Hp, b, ch, cw, fancy) - 128.0;
    out[uint4(X, Y, 0, b)] = st(clamp(y + 1.402 * cr, 0.0, 255.0) / 255.0);
    out[uint4(X, Y, 1, b)] = st(clamp(y - 0.344136 * cb - 0.714136 * cr, 0.0, 255.0) / 255.0);
    out[uint4(X, Y, 2, b)] = st(clamp(y + 1.772 * cb, 0.0, 255.0) / 255.0);
}
