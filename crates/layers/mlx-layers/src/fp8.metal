#include <metal_stdlib>
using namespace metal;

// The row-major scalar encoding and the MXFP8 128x4 Comfy scale-atom layout
// are shared with gen_core::comfy_quant. Only one tile is expanded at a time.
inline float sc_decode_fp8(uchar bits, bool e5) {
    uint m = e5 ? 2u : 3u;
    uint exponent = (bits & 127u) >> m;
    uint fraction = bits & ((1u << m) - 1u);
    int offset = e5 ? 15 : 7;
    if ((e5 && exponent == 31u && fraction != 0u) || (!e5 && (bits & 127u) == 127u)) return NAN;
    float magnitude = e5 && exponent == 31u ? INFINITY :
        ldexp(float(exponent == 0u ? fraction : (1u << m) + fraction),
            int(max(1u, exponent)) - offset - int(m));
    return (bits & 128u) ? -magnitude : magnitude;
}

inline float sc_round_fp8(float value, bool e5) {
    if (isnan(value)) return value;
    float maximum = e5 ? 57344.0f : 448.0f;
    float a = min(abs(value), maximum);
    if (a == 0.0f) return value;
    int mantissa = e5 ? 2 : 3;
    int exponent;
    frexp(a, exponent);
    int shift = max(exponent - 1, e5 ? -14 : -6) - mantissa;
    return copysign(min(ldexp(rint(ldexp(a, -shift)), shift), maximum), value);
}
