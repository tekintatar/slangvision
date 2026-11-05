// shaders/pil.slang — the Pillow ops timm's RandAugment runs, on float
// values. Pillow works on uint8; here the integer parameters it derives stay
// (thresholds and bit masks applied to the value as uint8, the contrast
// mean's level, equalize's 256 bins) and the uint8 rounding of each result
// goes, as the torchvision ops in color.slang drop it.

// Pillow's convert("L") luma (ITU-R 601-2)
float gray_pil(float4 c) { return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z; }

// that luma as Pillow computes it for a uint8 pixel (fixed point, rounded)
int luma_pil(float4 c)
{
    return (q255(c.x) * 19595 + q255(c.y) * 38470 + q255(c.z) * 7471 + 32768) >> 16;
}

// ImageEnhance.Color: blend with the luma image by f
float4 color_pil(float4 c, float f)
{
    float g = gray_pil(c);
    return clamp01(float4(g + f * (c.x - g), g + f * (c.y - g), g + f * (c.z - g), 0.0));
}

// timm's posterize: 8 or more bits keep the image; otherwise ImageOps.posterize
// keeps the top `bits` bits of each uint8 value (none at 0 or fewer, as Pillow 12)
float4 posterize_pil(float4 c, int bits)
{
    if (bits >= 8)
        return c;
    int s = min(8 - bits, 8);
    return float4(float((q255(c.x) >> s) << s), float((q255(c.y) >> s) << s),
                  float((q255(c.z) >> s) << s), 0.0) * (1.0 / 255.0);
}

// ImageOps.solarize: invert uint8 values at or above `threshold` (0-256)
float4 solarize_pil(float4 c, int threshold)
{
    return float4(q255(c.x) >= threshold ? 1.0 - c.x : c.x, q255(c.y) >= threshold ? 1.0 - c.y : c.y,
                  q255(c.z) >= threshold ? 1.0 - c.z : c.z, 0.0);
}

// timm's solarize_add: add `add` (0-1 scale) to uint8 values below 128, capped at 1
float4 solarize_add(float4 c, float add)
{
    return float4(q255(c.x) < 128 ? min(c.x + add, 1.0) : c.x,
                  q255(c.y) < 128 ? min(c.y + add, 1.0) : c.y,
                  q255(c.z) < 128 ? min(c.z + add, 1.0) : c.z, 0.0);
}

// Pillow's affine matrix for timm's geometric op `op` with level t (degrees
// for Rotate, the factor for ShearX/Y, the fraction of the size for
// TranslateX/YRel), converted to pixel units (pixel i covers [i - 0.5,
// i + 0.5]): u = m0 x + m1 y + m2, v = m3 x + m4 y + m5. Pillow maps the
// output pixel centre (x + 0.5, y + 0.5) through the matrix.
void timm_affine(int op, double t, int W, int H, out double m[6])
{
    double a0 = 1.0l; double a1 = 0.0l; double a2 = 0.0l;
    double a3 = 0.0l; double a4 = 1.0l; double a5 = 0.0l;
    if (op == 3)            // Image.rotate(t) about (W / 2, H / 2)
    {
        double th = -(t * (3.141592653589793l / 180.0l));
        double cs = cos(th);
        double sn = sin(th);
        double cx = 0.5l * double(W);
        double cy = 0.5l * double(H);
        a0 = cs; a1 = sn; a2 = cs * -cx + sn * -cy + cx;
        a3 = -sn; a4 = cs; a5 = -sn * -cx + cs * -cy + cy;
    }
    if (op == 11) a1 = t;
    if (op == 12) a3 = t;
    if (op == 13) a2 = t * double(W);
    if (op == 14) a5 = t * double(H);
    m[0] = a0; m[1] = a1; m[2] = a2 + 0.5l * (a0 + a1) - 0.5l;
    m[3] = a3; m[4] = a4; m[5] = a5 + 0.5l * (a3 + a4) - 0.5l;
}

// Pillow's cubic (a = -1) through four samples at fraction d past v2
float4 cubic_pil(float4 v1, float4 v2, float4 v3, float4 v4, float d)
{
    float4 p2 = v3 - v1;
    float4 p3 = 2.0 * (v1 - v2) + v3 - v4;
    float4 p4 = v2 - v1 - v3 + v4;
    return v2 + d * (p2 + d * (p3 + d * p4));
}

// Pillow's affine resampling at (u, v): `fill` outside the image (a hard
// edge, no blend), neighbours clamped to the edge inside it; bilinear, or
// bicubic clamped to [0, 1]. img must clamp its loads.
float4 pil_sample<I : IImage>(I img, float u, float v, int bicubic, float4 fill)
{
    if (!(u >= -0.5 && u < float(img.width()) - 0.5 && v >= -0.5 && v < float(img.height()) - 0.5))
        return fill;
    float fx = floor(u);
    float fy = floor(v);
    int x0 = int(fx);
    int y0 = int(fy);
    float dx = u - fx;
    float dy = v - fy;
    if (bicubic == 0)
    {
        float4 top = lerp(img.load(x0, y0), img.load(x0 + 1, y0), dx);
        float4 bottom = lerp(img.load(x0, y0 + 1), img.load(x0 + 1, y0 + 1), dx);
        return lerp(top, bottom, dy);
    }
    float4 r0 = cubic_pil(img.load(x0 - 1, y0 - 1), img.load(x0, y0 - 1), img.load(x0 + 1, y0 - 1),
                          img.load(x0 + 2, y0 - 1), dx);
    float4 r1 = cubic_pil(img.load(x0 - 1, y0), img.load(x0, y0), img.load(x0 + 1, y0),
                          img.load(x0 + 2, y0), dx);
    float4 r2 = cubic_pil(img.load(x0 - 1, y0 + 1), img.load(x0, y0 + 1), img.load(x0 + 1, y0 + 1),
                          img.load(x0 + 2, y0 + 1), dx);
    float4 r3 = cubic_pil(img.load(x0 - 1, y0 + 2), img.load(x0, y0 + 2), img.load(x0 + 1, y0 + 2),
                          img.load(x0 + 2, y0 + 2), dx);
    return clamp01(cubic_pil(r0, r1, r2, r3, dy));
}
