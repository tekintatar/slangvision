// shaders/warp.slang — torchvision's RandomRotation, RandomAffine and
// RandomPerspective: per-image coefficients, computed by the sampler in
// double precision as torchvision's Python and float64 least squares do,
// and the grid sample of the previous pass.
//
// Every warp maps an output pixel (x, y) to a source position (u, v) in
// pixel units, pixel i covering [i - 0.5, i + 0.5] (align_corners=False):
//     u = (a x + b y + c) / (g x + h y + 1),  v = (d x + e y + f) / (g x + h y + 1)
// stored as m[0..7] = a, b, c, d, e, f, g, h.

static const double DEG = 3.141592653589793l / 180.0l;     // math.radians

// torchvision's _get_inverse_affine_matrix (inverted) followed by
// _affine_grid, for a W x H image. Angles in degrees; (cx, cy) is the
// rotation centre's offset from the image centre.
void affine_coeffs(int W, int H, double angle, double tx, double ty, double scale,
                   double shx, double shy, double cx, double cy, out double m[8])
{
    double rot = angle * DEG;
    double sx = shx * DEG;
    double sy = shy * DEG;
    double cos_sy = cos(sy);
    double tan_sx = tan(sx);
    double a = cos(rot - sy) / cos_sy;
    double b = -(a * tan_sx + sin(rot));
    double c = sin(rot - sy) / cos_sy;
    double d = cos(rot) - c * tan_sx;
    double m0 = d / scale;
    double m1 = -b / scale;
    double m3 = -c / scale;
    double m4 = a / scale;
    double m2 = cx - m0 * (cx + tx) - m1 * (cy + ty);
    double m5 = cy - m3 * (cx + tx) - m4 * (cy + ty);
    // the grid's centred coordinates: u = m0 X + m1 Y + m2 + hw with X = x - hw
    double hw = 0.5l * double(W - 1);
    double hh = 0.5l * double(H - 1);
    m[0] = m0; m[1] = m1; m[2] = m2 + hw - m0 * hw - m1 * hh;
    m[3] = m3; m[4] = m4; m[5] = m5 + hh - m3 * hw - m4 * hh;
    m[6] = 0.0l; m[7] = 0.0l;
}

// torchvision's _get_perspective_coeffs for a W x H image whose corners
// (top-left, top-right, bottom-right, bottom-left) move to (ex, ey): the
// exact solution of its 8x8 system (Gaussian elimination with partial
// pivoting), rounded to float as torchvision does, then folded with
// _perspective_grid's half-pixel offsets into the form above.
void perspective_coeffs(int W, int H, int4 ex, int4 ey, out double m[8])
{
    double A[8][9];
    for (int i = 0; i < 4; ++i)
    {
        double qx = double(ex[i]);
        double qy = double(ey[i]);
        double px = (i == 1 || i == 2) ? double(W - 1) : 0.0l;
        double py = i >= 2 ? double(H - 1) : 0.0l;
        A[2 * i][0] = qx;   A[2 * i][1] = qy;   A[2 * i][2] = 1.0l;
        A[2 * i][3] = 0.0l; A[2 * i][4] = 0.0l; A[2 * i][5] = 0.0l;
        A[2 * i][6] = -px * qx; A[2 * i][7] = -px * qy; A[2 * i][8] = px;
        A[2 * i + 1][0] = 0.0l; A[2 * i + 1][1] = 0.0l; A[2 * i + 1][2] = 0.0l;
        A[2 * i + 1][3] = qx;   A[2 * i + 1][4] = qy;   A[2 * i + 1][5] = 1.0l;
        A[2 * i + 1][6] = -py * qx; A[2 * i + 1][7] = -py * qy; A[2 * i + 1][8] = py;
    }
    for (int col = 0; col < 8; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < 8; ++r)
            if (abs(A[r][col]) > abs(A[piv][col]))
                piv = r;
        for (int j = 0; j < 9; ++j)
        {
            double t = A[col][j]; A[col][j] = A[piv][j]; A[piv][j] = t;
        }
        for (int r = col + 1; r < 8; ++r)
        {
            double f = A[r][col] / A[col][col];
            for (int j = col; j < 9; ++j)
                A[r][j] -= f * A[col][j];
        }
    }
    double s[8];
    for (int i = 7; i >= 0; --i)
    {
        double acc = A[i][8];
        for (int j = i + 1; j < 8; ++j)
            acc -= A[i][j] * s[j];
        s[i] = acc / A[i][i];
    }
    for (int i = 0; i < 8; ++i)
        s[i] = double(float(s[i]));
    // torchvision samples at (x + 0.5, y + 0.5) and subtracts 0.5 after
    double K = 1.0l + 0.5l * (s[6] + s[7]);
    m[0] = (s[0] - 0.5l * s[6]) / K;
    m[1] = (s[1] - 0.5l * s[7]) / K;
    m[2] = (s[2] + 0.5l * (s[0] + s[1]) - 0.5l * K) / K;
    m[3] = (s[3] - 0.5l * s[6]) / K;
    m[4] = (s[4] - 0.5l * s[7]) / K;
    m[5] = (s[5] + 0.5l * (s[3] + s[4]) - 0.5l * K) / K;
    m[6] = s[6] / K;
    m[7] = s[7] / K;
}

// round half to even, as torch's nearest sampling (nearbyint) does
float round_even(float x)
{
    float r = floor(x);
    float f = x - r;
    if (f > 0.5 || (f == 0.5 && fmod(r, 2.0) != 0.0))
        r += 1.0;
    return r;
}

// torch's grid_sample at (u, v) with zero padding, then torchvision's fill:
// a mask channel of ones is sampled alongside, and the result is
// (img - fill) * mask + fill for bilinear, fill where mask < 0.5 for nearest.
float4 warp_sample<I : IImage>(I img, float u, float v, int nearest, float4 fill)
{
    int w = img.width();
    int h = img.height();
    if (!(abs(u) < 1e7 && abs(v) < 1e7))       // NaN, or far outside
        return fill;
    if (nearest != 0)
    {
        int ix = int(round_even(u));
        int iy = int(round_even(v));
        return (ix < 0 || iy < 0 || ix >= w || iy >= h) ? fill : img.load(ix, iy);
    }
    float fx = floor(u);
    float fy = floor(v);
    int x0 = int(fx);
    int y0 = int(fy);
    float ax = u - fx;
    float ay = v - fy;
    bool inx0 = x0 >= 0 && x0 < w;
    bool inx1 = x0 + 1 >= 0 && x0 + 1 < w;
    bool iny0 = y0 >= 0 && y0 < h;
    bool iny1 = y0 + 1 >= 0 && y0 + 1 < h;
    float4 acc = float4(0.0);
    float mask = 0.0;
    if (inx0 && iny0) { float t = (1.0 - ax) * (1.0 - ay); acc += t * img.load(x0, y0); mask += t; }
    if (inx1 && iny0) { float t = ax * (1.0 - ay); acc += t * img.load(x0 + 1, y0); mask += t; }
    if (inx0 && iny1) { float t = (1.0 - ax) * ay; acc += t * img.load(x0, y0 + 1); mask += t; }
    if (inx1 && iny1) { float t = ax * ay; acc += t * img.load(x0 + 1, y0 + 1); mask += t; }
    return (acc - fill) * mask + fill;
}
