// shaders/statistics.slang — ops that need a per-image statistic, computed
// by a pre-pass over the image just before the op (Compose's stats_<k>
// kernels): RandomAutocontrast (per-channel min and max) and RandomEqualize
// (per-channel histograms, turned into lookup tables by Compose).

// Histogram counting for one channel: runs of equal bins are added at once,
// which keeps flat regions from piling atomics onto one counter.
void hist_add(uint* h, int q, inout int prev, inout uint run)
{
    if (q == prev)
    {
        run += 1u;
        return;
    }
    if (run > 0u)
        InterlockedAdd(h[prev], run);
    prev = q;
    run = 1u;
}

void hist_flush(uint* h, int prev, uint run)
{
    if (run > 0u)
        InterlockedAdd(h[prev], run);
}

// torchvision's autocontrast_image: (c - min) / (max - min) per channel,
// clamped; Compose stores (0, 1) for a constant channel
float4 autocontrast(float4 c, float l0, float l1, float l2, float s0, float s1, float s2)
{
    return clamp01(float4((c.x - l0) / s0, (c.y - l1) / s1, (c.z - l2) / s2, 0.0));
}

// torchvision's equalize_image: the quantized value through this image's
// lookup tables (statistic slots k .. k + 767, already scaled to 0-1)
float4 equalize(float4 c, Tensor<float, 2> st, int b, int k)
{
    return float4(prm(st, b, k + q255(c.x)), prm(st, b, k + 256 + q255(c.y)),
                  prm(st, b, k + 512 + q255(c.z)), 0.0);
}
