#include <metal_stdlib>
using namespace metal;

// SW-6141 (SW-F287 M3): shared helpers of the INT8 QK attention POC kernels. Original Metal source
// written for this repository (see the module docs of `int8_attention_metal.rs`); the 8x8
// simdgroup-matrix fragment layout below is the Apple GPU layout MLX's steel kernels use, as in the
// in-tree Veda and INT8 GEMM kernels.

template <typename T> struct sw6141_elem;
template <typename T> struct sw6141_elem<device T*> { using type = T; };

template <bool C, typename A, typename B> struct sw6141_select { using type = A; };
template <typename A, typename B> struct sw6141_select<false, A, B> { using type = B; };

// Row and first column of this lane's two elements of an 8x8 simdgroup matrix.
inline ushort2 sw6141_frag(uint lane) {
    const uint qid = lane / 4u;
    return ushort2((qid & 4u) + ((lane / 2u) % 4u), (qid & 2u) * 2u + (lane % 2u) * 2u);
}

// The signed Walsh-Hadamard rotation's sign diagonal (the CUDA twin's `SAGE_SIGN_WORDS`, SW-6131):
// channel `c` is negated when bit `c & 31` of word `c >> 5` is clear.
constant uint SW6141_SIGN[4] = {0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};

// One row of 128 channels held four per lane (lane l owns channels 4l..4l+3) of one SIMD group:
// y = H_128 · diag(sign) · y with H_128 the orthonormal Sylvester Walsh-Hadamard matrix. Butterfly
// strides 1 and 2 run inside the lane, 4..64 across lanes (stride 4m = lane bit m): the element
// whose index has the stride bit clear keeps the sum, the other the difference partner - self.
// Every lane of the SIMD group must call it (it shuffles).
inline float4 sw6141_rotate(float4 y, uint lane) {
    const uint c0 = lane * 4u;
#pragma clang loop unroll(full)
    for (uint e = 0u; e < 4u; ++e) {
        const uint c = c0 + e;
        if (((SW6141_SIGN[c >> 5] >> (c & 31u)) & 1u) == 0u) y[e] = -y[e];
    }
    const float a0 = y.x + y.y;
    const float a1 = y.x - y.y;
    const float a2 = y.z + y.w;
    const float a3 = y.z - y.w;
    y = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
#pragma clang loop unroll(full)
    for (uint m = 1u; m < 32u; m <<= 1) {
        const float4 p = simd_shuffle_xor(y, ushort(m));
        y = (lane & m) != 0u ? p - y : y + p;
    }
    return y * 0.08838834764831845f;  // 1 / sqrt(128)
}
