// Shared device primitives of the INT8 attention kernels (SW-6131 sage_attention.cu, SW-6137
// sol_attention.cu). The nvrtc seam compiles single sources with no include paths, so this file is
// prepended to each kernel's own text by `concat!(include_str!(..), include_str!(..))` in Rust: one
// definition of the typedefs, 16-bit loads/packs, warp reductions, cp.async, ldmatrix, the INT8 /
// U8 / 16-bit mma.sync wrappers and the U8 pack, no copies. Extracted verbatim from SW-6131's
// sage_attention.cu (whose attribution header covers it), with `sage_` renamed `ia_`.

typedef unsigned int u32;
typedef unsigned short u16;
typedef signed char i8;
typedef long long i64;

#define IA_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float ia_load(const u16* p, bool bf) {
    u16 h = *p;
    if (bf) return __uint_as_float(((u32)h) << 16);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// Pack (lo, hi) into a bf16x2 / f16x2 register, round to nearest even.
template <bool BF>
__device__ __forceinline__ u32 ia_pack2(float lo, float hi) {
    u32 r;
    if (BF) asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    else asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// 2^x on the SFU (ex2.approx, ~2 ulp; exactly 0 for -inf), as FlashAttention-2's fast-math build.
__device__ __forceinline__ float ia_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ia_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float ia_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ u32 ia_smem(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ia_cp16(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(ia_smem(dst)), "l"(src),
                 "r"(bytes));
}

// .ca: L1-cached, for sources many CTAs re-read (sol_attn routing's pooled arrays). Added by
// SW-6137 (comfy-kitchen sol_layout.cuh `cp_async16_ca`).
__device__ __forceinline__ void ia_cp16_ca(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(ia_smem(dst)), "l"(src));
}

__device__ __forceinline__ void ia_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void ia_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ia_ldm4(u32* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(ia_smem(p)));
}

__device__ __forceinline__ void ia_ldm4t(u32* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(ia_smem(p)));
}

__device__ __forceinline__ void ia_mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool BF>
__device__ __forceinline__ void ia_mma_16(float* c, const u32* a, u32 b0, u32 b1) {
    if (BF)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ia_mma_u8s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Round to nearest even and saturate to [0, 255] (comfy-kitchen `pack_u8x4` on sm_89+); byte j
// of the result is the j-th argument.
__device__ __forceinline__ u32 ia_pack_u8x4(float a, float b, float c, float d) {
    const float v[4] = {a, b, c, d};
    u32 out = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int q;
        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(q) : "f"(v[j]));
        q = max(0, min(255, q));
        out |= ((u32)q) << (8 * j);
    }
    return out;
}

__device__ __forceinline__ float ia_u8_sum(u32 p) {
    return (float)__dp4a(p, 0x01010101u, 0u);
}
