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
