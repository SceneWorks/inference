
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// SW-6141 neural-accelerator (NAX) helpers: MLX's `BaseNAXFrag` fragment layout and one
// `mpp::tensor_ops::matmul2d` product per SIMD group, as the in-tree packed-KV (SC-20676) and Veda
// NAX attention kernels use them, generalised over the destination element type so the same
// product runs INT8 x INT8 -> INT32 for Q·Kᵀ.

// (column, row) of this lane's first element of a 16x16 fragment; the lane holds rows `fm` and
// `fm + 8`, columns `fn..fn+3`.
inline __attribute__((always_inline)) short2 sw6141_nax_coord(uint lane) {
    const short qid = short(lane >> 2);
    const short fm = (qid & 4) | short((lane >> 1) & 3u);
    const short fn = ((qid & 2) | short(lane & 1u)) * 4;
    return short2(fn, fm);
}

// C[16x32] += A[16x16] . B[16x32], B given as two 16x16 fragments (rows of B^T when `TB`).
template <bool TB, typename AT, typename BT, typename CT>
inline __attribute__((always_inline)) void sw6141_nax_mma(thread vec<CT, 8>& c0, thread vec<CT, 8>& c1,
                            thread const vec<AT, 8>& a, thread const vec<BT, 8>& b0,
                            thread const vec<BT, 8>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), CT>();
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = a[i];
        ct_b[i] = b0[i];
        ct_b[8 + i] = b1[i];
        ct_c[i] = c0[i];
        ct_c[8 + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[8 + i];
    }
}

// Rows fm, fm+8 x columns fn..fn+3 of the 16x16 fragment at (`row0`, `col0`) of a threadgroup tile.
template <typename T, int LD>
inline __attribute__((always_inline)) vec<T, 8> sw6141_nax_frag(const threadgroup T* tile, short2 coord, uint row0, uint col0) {
    const threadgroup T* base = tile + (row0 + uint(coord.y)) * uint(LD) + col0 + uint(coord.x);
    const vec<T, 4> lo = *(const threadgroup vec<T, 4>*)(base);
    const vec<T, 4> hi = *(const threadgroup vec<T, 4>*)(base + 8 * LD);
    return vec<T, 8>(lo, hi);
}
