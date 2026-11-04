// shaders/geometry.slang — torchvision's resampling and crop geometry:
// RandomResizedCrop, Resize, CenterCrop.

// One output pixel of torchvision's resize of the box (left, top, cw, ch)
// to (OW, OH): bilinear, align_corners=False; with aa != 0 the antialias
// triangle filter, whose support widens with the downscale factor. Taps stay
// inside the box. Returns [0, 1].
float4 resample<S : IImageSource>(S src, int left, int top, int cw, int ch,
                                  int ox, int oy, int OW, int OH, int aa)
{
    float sx = float(cw) / float(OW);
    float sy = float(ch) / float(OH);
    if (aa == 0)
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
    float supx = sx >= 1.0 ? sx : 1.0;
    float supy = sy >= 1.0 ? sy : 1.0;
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
        wxsum += max(0.0, 1.0 - abs((float(x) - cx + 0.5) * invx));
    float4 acc = float4(0.0);
    float wysum = 0.0;
    for (int y = ymin; y < ymax; ++y)
    {
        float wy = max(0.0, 1.0 - abs((float(y) - cy + 0.5) * invy));
        if (wy == 0.0) continue;
        float4 row = float4(0.0);
        for (int x = xmin; x < xmax; ++x)
            row += max(0.0, 1.0 - abs((float(x) - cx + 0.5) * invx)) * src.load(left + x, top + y);
        acc += wy * row;
        wysum += wy;
    }
    return acc * (1.0 / (255.0 * wxsum * wysum));
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
