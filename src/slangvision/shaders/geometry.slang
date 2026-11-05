// shaders/geometry.slang — torchvision's resampling and crop geometry:
// RandomResizedCrop and Resize (bilinear and bicubic filters), CenterCrop,
// RandomCrop and Pad (padding modes).

// PyTorch's resampling filters (aten UpSample kernels). Antialias filters:
// a triangle for bilinear and Keys' cubic with a = -0.5 for bicubic, both
// stretched by the downscale factor. Without antialias, bicubic is Keys'
// cubic with a = -0.75 on the four nearest taps.
float aa_filter(float x, int bicubic)
{
    x = abs(x);
    if (bicubic == 0)
        return max(0.0, 1.0 - x);
    if (x < 1.0)
        return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
    return 0.0;
}

float cubic_near(float x) { return (1.25 * x - 2.25) * x * x + 1.0; }               // |x| <= 1
float cubic_far(float x) { return ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0; }        // 1 < |x| < 2

// One output pixel of torchvision's resize of the box (left, top, cw, ch)
// to (OW, OH), align_corners=False: bilinear (bicubic == 0) or bicubic, with
// aa != 0 the antialias filters above. Taps stay inside the box. Returns
// 0-1 values; bicubic can overshoot, as torchvision's float output does.
float4 resample<S : IImageSource>(S src, int left, int top, int cw, int ch,
                                  int ox, int oy, int OW, int OH, int aa, int bicubic)
{
    if (cw == OW && ch == OH)       // no scaling: every filter reduces to a copy
        return src.load(left + ox, top + oy) * (1.0 / 255.0);
    float sx = float(cw) / float(OW);
    float sy = float(ch) / float(OH);
    if (aa == 0 && bicubic == 0)
    {
        float fx = max((float(ox) + 0.5) * sx - 0.5, 0.0);
        float fy = max((float(oy) + 0.5) * sy - 0.5, 0.0);
        int x0 = min(int(fx), cw - 1);
        int y0 = min(int(fy), ch - 1);
        int x1 = min(x0 + 1, cw - 1);
        int y1 = min(y0 + 1, ch - 1);
        float ax = fx - float(x0);
        float ay = fy - float(y0);
        float4 r0 = lerp(src.load(left + x0, top + y0), src.load(left + x1, top + y0), ax);
        float4 r1 = lerp(src.load(left + x0, top + y1), src.load(left + x1, top + y1), ax);
        return lerp(r0, r1, ay) * (1.0 / 255.0);
    }
    if (aa == 0)
    {
        // four taps per axis around floor(real), clamped to the box
        float rx = (float(ox) + 0.5) * sx - 0.5;
        float ry = (float(oy) + 0.5) * sy - 0.5;
        int ix = int(floor(rx));
        int iy = int(floor(ry));
        float tx = rx - float(ix);
        float ty = ry - float(iy);
        float wx0 = cubic_far(tx + 1.0), wx1 = cubic_near(tx), wx2 = cubic_near(1.0 - tx), wx3 = cubic_far(2.0 - tx);
        float wy0 = cubic_far(ty + 1.0), wy1 = cubic_near(ty), wy2 = cubic_near(1.0 - ty), wy3 = cubic_far(2.0 - ty);
        float4 acc = float4(0.0);
        for (int j = 0; j < 4; ++j)
        {
            int y = top + clamp(iy - 1 + j, 0, ch - 1);
            float wy = j == 0 ? wy0 : (j == 1 ? wy1 : (j == 2 ? wy2 : wy3));
            int xa = left + clamp(ix - 1, 0, cw - 1);
            int xb = left + clamp(ix, 0, cw - 1);
            int xc = left + clamp(ix + 1, 0, cw - 1);
            int xd = left + clamp(ix + 2, 0, cw - 1);
            acc += wy * (wx0 * src.load(xa, y) + wx1 * src.load(xb, y) +
                         wx2 * src.load(xc, y) + wx3 * src.load(xd, y));
        }
        return acc * (1.0 / 255.0);
    }
    float half = bicubic != 0 ? 2.0 : 1.0;          // filter support at scale 1
    float supx = half * (sx >= 1.0 ? sx : 1.0);
    float supy = half * (sy >= 1.0 ? sy : 1.0);
    float invx = sx >= 1.0 ? 1.0 / sx : 1.0;
    float invy = sy >= 1.0 ? 1.0 / sy : 1.0;
    float cx = (float(ox) + 0.5) * sx;
    float cy = (float(oy) + 0.5) * sy;
    int xmin = max(int(cx - supx + 0.5), 0);
    int xmax = min(int(cx + supx + 0.5), cw);
    int ymin = max(int(cy - supy + 0.5), 0);
    int ymax = min(int(cy + supy + 0.5), ch);
    float wxsum = 0.0;
    for (int x = xmin; x < xmax; ++x)
        wxsum += aa_filter((float(x) - cx + 0.5) * invx, bicubic);
    float4 acc = float4(0.0);
    float wysum = 0.0;
    for (int y = ymin; y < ymax; ++y)
    {
        float wy = aa_filter((float(y) - cy + 0.5) * invy, bicubic);
        wysum += wy;
        if (wy == 0.0) continue;
        float4 row = float4(0.0);
        for (int x = xmin; x < xmax; ++x)
            row += aa_filter((float(x) - cx + 0.5) * invx, bicubic) * src.load(left + x, top + y);
        acc += wy * row;
    }
    return acc * (1.0 / (255.0 * wxsum * wysum));
}

// Out-of-image indices for torchvision's padding modes ("reflect" is in
// neighbourhood.slang): edge repeats the border pixel, symmetric mirrors
// including it. Compose rejects mirror padding wider than the image, as
// torchvision does; the clamp only keeps indices in bounds.
int clamp_index(int i, int n) { return clamp(i, 0, n - 1); }

int symmetric(int i, int n)
{
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2 * n - 1 - i;
    return clamp(i, 0, n - 1);
}

// torchvision's centre-crop offset: int(round((size - crop) / 2.0)), where
// Python's round sends halves to the even neighbour. A crop larger than the
// image pads (crop - size) // 2 zeros before it: a negative offset.
int center_offset(int size, int crop)
{
    int d = size - crop;
    if (d < 0)
        return -((-d) / 2);
    int q = d / 2;
    if ((d & 1) != 0 && (q & 1) != 0)   // exactly x.5 with an odd floor
        q += 1;
    return q;
}

// torchvision Resize(int): the shorter side becomes `size`, the longer side
// int(size * long / short). Writes the new (w, h).
void resize_shorter(int W, int H, int size, out int nw, out int nh)
{
    if (W <= H) { nw = size; nh = (size * H) / W; }
    else        { nh = size; nw = (size * W) / H; }
}

// torchvision RandomResizedCrop box for an image of W x H: 10 tries, then a
// centre crop clamped to the ratio range.
void sample_rrc(inout uint s, int W, int H, float smin, float smax, float rmin, float rmax,
                out int left, out int top, out int w, out int h)
{
    float area = float(H * W);
    float lo = log(rmin);
    float hi = log(rmax);
    w = 0;
    h = 0;
    bool found = false;
    for (int t = 0; t < 10; ++t)
    {
        float ta = area * urand(s, smin, smax);
        float ar = exp(urand(s, lo, hi));
        int cw = int(round(sqrt(ta * ar)));
        int ch = int(round(sqrt(ta / ar)));
        if (!found && cw > 0 && cw <= W && ch > 0 && ch <= H)
        {
            w = cw;
            h = ch;
            found = true;
        }
    }
    float u0 = rnd(s);
    float u1 = rnd(s);
    if (found)
    {
        top = min(int(u0 * float(H - h + 1)), H - h);
        left = min(int(u1 * float(W - w + 1)), W - w);
        return;
    }
    float ratio = float(W) / float(H);
    if (ratio < rmin)      { w = W; h = int(round(float(W) / rmin)); }
    else if (ratio > rmax) { h = H; w = int(round(float(H) * rmax)); }
    else                   { w = W; h = H; }
    top = (H - h) / 2;
    left = (W - w) / 2;
}

// torchvision RandomErasing's box for a W x H image: 10 tries at an area in
// [smin, smax] of the image and a log-uniform aspect in [rmin, rmax]; the
// first that fits strictly inside is placed uniformly. Returns false (no
// erasing) when none fits. Always draws the same count, so a shared
// (same_on_batch) stream stays aligned across images.
bool sample_box(inout uint s, int W, int H, float smin, float smax, float rmin, float rmax,
                out int top, out int left, out int h, out int w)
{
    float area = float(H * W);
    float lo = log(rmin);
    float hi = log(rmax);
    h = 0;
    w = 0;
    bool found = false;
    for (int t = 0; t < 10; ++t)
    {
        float ea = area * urand(s, smin, smax);
        float ar = exp(urand(s, lo, hi));
        int eh = int(round(sqrt(ea * ar)));
        int ew = int(round(sqrt(ea / ar)));
        if (!found && eh < H && ew < W)
        {
            h = eh;
            w = ew;
            found = true;
        }
    }
    top = min(int(rnd(s) * float(H - h + 1)), H - h);
    left = min(int(rnd(s) * float(W - w + 1)), W - w);
    return found;
}
