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
