// shaders/sources.slang — where the input images live: uint8 CHW or HWC
// images read in place through their device addresses.

// Where one uint8 RGB image lives. Kernels are generic over the source, and
// Slang specializes each one, so the indirection costs nothing.
interface IImageSource
{
    float4 load(int x, int y);   // 0-255 values
}

// one (3, H, W) image read in place through its device address
struct SourceCHW : IImageSource
{
    uint8_t* p;
    int w;
    int plane;
    float4 load(int x, int y)
    {
        int i = y * w + x;
        return float4(float(p[i]), float(p[i + plane]), float(p[i + 2 * plane]), 0.0);
    }
}

// one (H, W, 3) image, as DALI and nvImageCodec decode
struct SourceHWC : IImageSource
{
    uint8_t* p;
    int w;
    float4 load(int x, int y)
    {
        int i = (y * w + x) * 3;
        return float4(float(p[i]), float(p[i + 1]), float(p[i + 2]), 0.0);
    }
}

// The image table: ptrs holds device addresses, meta rows are (H, W).
SourceCHW source_chw(Tensor<int64_t, 1> ptrs, Tensor<int, 2> meta, int b)
{
    int h = meta[uint2(0, b)];
    int w = meta[uint2(1, b)];
    SourceCHW s = { (uint8_t*)(uint64_t(ptrs[uint1(b)])), w, h * w };
    return s;
}

SourceHWC source_hwc(Tensor<int64_t, 1> ptrs, Tensor<int, 2> meta, int b)
{
    SourceHWC s = { (uint8_t*)(uint64_t(ptrs[uint1(b)])), meta[uint2(1, b)] };
    return s;
}
