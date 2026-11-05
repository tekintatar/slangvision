// shaders/kernels/mix.slang — MixUp and CutMix on an augmented batch.
// Not part of the op library: slangvision/mixing.py loads it on its own,
// after a `typealias T` line and the ld/st converters for the batch's dtype.
//
// Per image b, row b of prm: partner j, own weight a, partner weight m, and a
// CutMix box (x1, y1, x2, y2). Inside the box the pixel is the partner's;
// outside it is a * own + m * partner (a = 1, m = 0 for CutMix).

void mix(int3 pix, Tensor<T, 4> x, RWTensor<T, 4> y, Tensor<float, 2> prm)
{
    int X = pix.x;
    int Y = pix.y;
    int b = pix.z;
    int j = int(prm[uint2(0, b)]);
    float a = prm[uint2(1, b)];
    float m = prm[uint2(2, b)];
    bool box = X >= int(prm[uint2(3, b)]) && Y >= int(prm[uint2(4, b)]) &&
               X < int(prm[uint2(5, b)]) && Y < int(prm[uint2(6, b)]);
    for (int c = 0; c < 3; ++c)
    {
        float v = ld(x[uint4(X, Y, c, b)]);
        float w = ld(x[uint4(X, Y, c, j)]);
        y[uint4(X, Y, c, b)] = st(box ? w : a * v + m * w);
    }
}
