// shaders/color.slang — torchvision's colour adjustments (ColorJitter, the
// policy ops) on float images, and the per-image order of ColorJitter's ops.

float4 brightness(float4 c, float f) { return clamp01(c * f); }

float4 contrast(float4 c, float f, float mean) { return clamp01(f * c + (1.0 - f) * mean); }

float4 saturation(float4 c, float f) { return clamp01(f * c + (1.0 - f) * gray(c)); }

float4 invert(float4 c) { return float4(1.0 - c.x, 1.0 - c.y, 1.0 - c.z, 0.0); }

// Channel order i of 012, 021, 102, 120, 201, 210 (output channel j reads
// input channel perm[j], as torchvision's permute_channels).
float4 permute_channels(float4 c, int i)
{
    switch (i)
    {
    case 0: return c;
    case 1: return float4(c.x, c.z, c.y, 0.0);
    case 2: return float4(c.y, c.x, c.z, 0.0);
    case 3: return float4(c.y, c.z, c.x, 0.0);
    case 4: return float4(c.z, c.x, c.y, 0.0);
    default: return float4(c.z, c.y, c.x, 0.0);
    }
}

float4 solarize(float4 c, float t)
{
    return float4(c.x >= t ? 1.0 - c.x : c.x, c.y >= t ? 1.0 - c.y : c.y,
                  c.z >= t ? 1.0 - c.z : c.z, 0.0);
}

// torchvision's float posterize: n = 2**bits levels, floor(value * n) / n
float4 posterize(float4 c, float n)
{
    float top = n - 1.0;
    float k = 1.0 / n;
    return float4(clamp(floor(c.x * n), 0.0, top) * k, clamp(floor(c.y * n), 0.0, top) * k,
                  clamp(floor(c.z * n), 0.0, top) * k, 0.0);
}

// torchvision adjust_hue: RGB -> HSV, shift the hue, HSV -> RGB
float4 hue(float4 c, float shift)
{
    float r = c.x, g = c.y, b = c.z;
    float maxc = max(r, max(g, b));
    float minc = min(r, min(g, b));
    bool eqc = maxc == minc;
    float cr = maxc - minc;
    float s = cr / (eqc ? 1.0 : maxc);
    float div = eqc ? 1.0 : cr;
    float rc = (maxc - r) / div;
    float gc = (maxc - g) / div;
    float bc = (maxc - b) / div;
    float h;
    if (maxc == r)      h = bc - gc;
    else if (maxc == g) h = 2.0 + rc - bc;
    else                h = 4.0 + gc - rc;
    h = fmod(h * (1.0 / 6.0) + 1.0, 1.0) + shift;
    h = h - floor(h);
    float v = maxc;
    float h6 = h * 6.0;
    float fi = floor(h6);
    float f = h6 - fi;
    int i = int(fi) % 6;
    float sxf = s * f;
    float q = clamp((1.0 - sxf) * v, 0.0, 1.0);
    float t = clamp((sxf + 1.0 - s) * v, 0.0, 1.0);
    float pp = clamp((1.0 - s) * v, 0.0, 1.0);
    switch (i)
    {
    case 0: return float4(v, t, pp, 0.0);
    case 1: return float4(q, v, pp, 0.0);
    case 2: return float4(pp, v, t, 0.0);
    case 3: return float4(pp, q, v, 0.0);
    case 4: return float4(t, pp, v, 0.0);
    default: return float4(v, pp, q, 0.0);
    }
}

// A random order of 0..3 packed in 2-bit fields (Fisher-Yates).
int random_order(inout uint s)
{
    int code = 0 | (1 << 2) | (2 << 4) | (3 << 6);
    for (int i = 3; i > 0; --i)
    {
        int j = min(int(rnd(s) * float(i + 1)), i);
        int a = (code >> (2 * i)) & 3;
        int c = (code >> (2 * j)) & 3;
        code &= ~((3 << (2 * i)) | (3 << (2 * j)));
        code |= (c << (2 * i)) | (a << (2 * j));
    }
    return code;
}
