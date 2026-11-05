// shaders/core.slang — shared by every pass: parameter access, pixel helpers
// and the counter-based random generator.
//
// slangvision joins the files in this folder (core first, then the others in
// alphabetical order) into one library and appends each pipeline's generated
// entry points to it, so every pipeline compiles into one module; the file
// split is for reading only. Every op is a small function on one pixel:
// float4 with rgb in xyz, values in [0, 1], fp32 math. No float3 or float2
// anywhere (NVRTC spills them). Semantics follow torchvision.transforms.v2
// on float images, including the clamp after each colour op.
//
// Tensor vector subscripts are reversed (x = last dim): a (B, P) parameter
// table is read as p[uint2(k, b)].

import slangpy;

float prm(Tensor<float, 2> p, int b, int k) { return p[uint2(k, b)]; }

float gray(float4 c) { return 0.2989 * c.x + 0.587 * c.y + 0.114 * c.z; }

float4 clamp01(float4 c) { return clamp(c, 0.0, 1.0); }

// torchvision's float -> uint8 conversion (to_dtype_image, scale=True):
// value * 255.999 truncated. Clamped here; torchvision's cast of a value
// outside [0, 1] (bicubic overshoot) is undefined and in practice wraps.
int q255(float v) { return clamp(int(v * 255.999), 0, 255); }

// The bfloat16 bit pattern of f, rounded to nearest even like torch's
// .to(torch.bfloat16); NaN stays NaN.
int16_t bf16_bits(float f)
{
    uint u = asuint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return int16_t(0x7fc0);
    u += 0x7fffu + ((u >> 16) & 1u);
    return int16_t(u >> 16);
}

// ---------------------------------------------------------------- random
// Counter-based: every value is a function of (call seed, image, draw).
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

float urand(inout uint s, float lo, float hi) { return lo + (hi - lo) * rnd(s); }

int irand(inout uint s, int n) { return min(int(rnd(s) * float(n)), n - 1); }

// a standard normal value (Box-Muller)
float nrand(inout uint s)
{
    float u1 = max(rnd(s), 1e-7);
    float u2 = rnd(s);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

// Four independent standard normal values for one pixel of image b, keyed by
// a per-call random key (Box-Muller on counter-based uniforms).
float4 normal4(uint key, int b, int x, int y)
{
    uint s = pcg(key ^ pcg(uint(b) * 2654435769u ^ pcg(uint(x) * 40503u + uint(y) * 2246822519u)));
    float u1 = max(rnd(s), 1e-7), u2 = rnd(s), u3 = max(rnd(s), 1e-7), u4 = rnd(s);
    float r1 = sqrt(-2.0 * log(u1)), r2 = sqrt(-2.0 * log(u3));
    return float4(r1 * cos(6.2831853 * u2), r1 * sin(6.2831853 * u2),
                  r2 * cos(6.2831853 * u4), r2 * sin(6.2831853 * u4));
}
