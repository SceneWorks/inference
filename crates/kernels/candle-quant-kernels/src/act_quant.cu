// SW-6126 (SW-F287 S1): fused per-tensor dynamic activation quantizer — amax -> scale -> cast in
// two device passes, emitting E4M3FN bytes or int8 codes directly, with the scale left on the
// device. Written from the in-tree reference semantics (candle-quant-kernels `cublaslt.rs`
// `quantize_activation_fp8` / `quantize_activation_int8` / `CublasLt::scaled_fp8_activation`);
// the two-pass amax/cast structure follows comfy-kitchen's `per_tensor_quantize.cu` design, but no
// comfy-kitchen source is copied.
//
// BYTE IDENTITY CONTRACT (each step is the reference's own operation, in the reference's order):
//   amax    max |x| in the reference's reduction dtype. The candle op chain reduces in x's dtype:
//           f32 with fmaxf (NaN ignored), f16/bf16 with __hmax_nan (any NaN wins). The Fp8Linear
//           route (`scaled_fp8_activation`) reduces the F32 cast, so NaN is ignored for every
//           dtype there. `nan_propagates` selects which.
//   scale   fmaxf(amax / qmax, FLT_MIN) in f32 — Rust's `(a / q).max(f32::MIN_POSITIVE)`
//           (f32::max ignores NaN like fmaxf).
//   scaled  candle chain: `x.to_dtype(F32) / scale as f64` is `affine(1 / scale, 0)`: the
//           multiplier is (float)(1.0 / (double)scale) and candle's affine_f32 computes
//           `x * mul + add`, which nvcc contracts to one fma(x, mul, 0) — reproduced explicitly.
//           Fp8Linear route: IEEE x / scale.
//   fp8     the `float8` crate's `F8E4M3::from_f32` (a port of cuda_fp8.hpp's host conversion):
//           round-to-nearest-even, saturate-finite (|v| > 464 -> 0x7E), NaN -> 0x7F, signed
//           zero kept, sign bit OR-ed last. Ported bit for bit below on binary32 (the binary64
//           thresholds it uses are exact binary32 values), so no sm_89 cvt instruction is needed.
//   int8    roundf (half away from zero), then fmaxf(., -127) and fminf(., 127) — candle's
//           round().clamp(-127, 127); emitted either as those F32 codes or as the two's-
//           complement i8 byte the cuBLASLt stage narrows them to.
//
// Single file, no includes (nvrtc seam rule). No per-thread local arrays.

#define AQ_THREADS 256

// A 16-bit input element as f32, exactly as candle's cast does it: bf16 is the high half of the
// binary32 word; f16 widens with cvt.f32.f16 (`__half2float`).
__device__ __forceinline__ float aq_h2f(unsigned short h, int bf16) {
    if (bf16) return __uint_as_float(((unsigned)h) << 16);
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}

// The float8 crate's E4M3 conversion (`convert_to_fp8`, SatFinite), on binary32 bits.
__device__ __forceinline__ unsigned char aq_e4m3(float v) {
    const unsigned x = __float_as_uint(v);
    const unsigned char sign = (unsigned char)((x >> 31) << 7);
    const unsigned absx = x & 0x7FFFFFFFu;
    const unsigned half_ulp = 1u << 19;  // binary64 1 << 48, rescaled to the 23-bit mantissa
    unsigned char res;
    if (absx <= 0x3A800000u) {            // <= 2^-10: zero or underflow
        res = 0;
    } else if (absx > 0x7F800000u) {      // NaN
        res = 0x7F;
    } else if (absx > 0x43E80000u) {      // > 464 (incl. inf): saturate to 448
        res = 0x7E;
    } else {
        const unsigned char exp = (unsigned char)(((x >> 23) & 0xFFu) - 127u + 7u);
        const unsigned char mantissa = (unsigned char)((x >> 20) & 0x7u);
        if (absx >= 0x3C800000u) {        // >= 2^-6: normal
            res = (unsigned char)((exp << 3) | mantissa);
            const unsigned round = x & ((half_ulp << 1) - 1u);
            if (round > half_ulp || (round == half_ulp && (mantissa & 1u))) res += 1;
        } else {                          // E4M3 subnormal
            const unsigned shift = (unsigned char)(1u - (unsigned)exp);
            const unsigned char m = (unsigned char)(mantissa | (1u << 3));
            res = (unsigned char)(m >> shift);
            const unsigned round = (x | (1u << 23)) & ((half_ulp << (shift + 1u)) - 1u);
            const unsigned half = half_ulp << shift;
            if (round > half || (round == half && (res & 1u))) res += 1;
        }
    }
    return (unsigned char)(res | sign);
}

// ---- pass 1: amax ---------------------------------------------------------------------------
// |x| of a non-negative float orders like its bits, and every NaN's bits exceed +inf's, so one
// atomicMax over the bits is the max — NaN-propagating for free; the NaN-ignoring variant skips
// NaN lanes (fmaxf). `amax` must be zero-initialised (+0.0, the max of an all-NaN f32 input under
// fmaxf maps to the same scale as candle's -inf seed: both clamp to FLT_MIN).
__device__ __forceinline__ unsigned aq_abs_bits(float v, int nan_propagates) {
    const unsigned b = __float_as_uint(v) & 0x7FFFFFFFu;
    return (!nan_propagates && b > 0x7F800000u) ? 0u : b;
}

__device__ __forceinline__ void aq_block_max(unsigned peak, unsigned* amax) {
    __shared__ unsigned warp_peak[AQ_THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) peak = max(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_peak[warp] = peak;
    __syncthreads();
    if (warp == 0) {
        peak = lane < AQ_THREADS / 32 ? warp_peak[lane] : 0u;
        for (int o = 16; o > 0; o >>= 1) peak = max(peak, __shfl_xor_sync(0xffffffffu, peak, o));
        if (lane == 0 && peak) atomicMax(amax, peak);
    }
}

extern "C" __global__ void act_quant_amax_f32(const float* x, unsigned long long n,
                                              unsigned* amax, int nan_propagates) {
    unsigned peak = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * AQ_THREADS;
    const unsigned long long groups = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (unsigned long long g = (unsigned long long)blockIdx.x * AQ_THREADS + threadIdx.x;
         g < groups; g += stride) {
        const float4 v = x4[g];
        peak = max(peak, aq_abs_bits(v.x, nan_propagates));
        peak = max(peak, aq_abs_bits(v.y, nan_propagates));
        peak = max(peak, aq_abs_bits(v.z, nan_propagates));
        peak = max(peak, aq_abs_bits(v.w, nan_propagates));
    }
    for (unsigned long long i = groups * 4 + (unsigned long long)blockIdx.x * AQ_THREADS + threadIdx.x;
         i < n; i += stride) {
        peak = max(peak, aq_abs_bits(x[i], nan_propagates));
    }
    aq_block_max(peak, amax);
}

// f16 (bf16 = 0) and bf16 (bf16 = 1) inputs, four 16-bit values per 8-byte load.
extern "C" __global__ void act_quant_amax_16(const unsigned short* x, unsigned long long n,
                                             unsigned* amax, int nan_propagates, int bf16) {
    unsigned peak = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * AQ_THREADS;
    const unsigned long long groups = n / 4;
    const uint2* x4 = reinterpret_cast<const uint2*>(x);
    for (unsigned long long g = (unsigned long long)blockIdx.x * AQ_THREADS + threadIdx.x;
         g < groups; g += stride) {
        const uint2 w = x4[g];
        peak = max(peak, aq_abs_bits(aq_h2f((unsigned short)(w.x & 0xFFFFu), bf16), nan_propagates));
        peak = max(peak, aq_abs_bits(aq_h2f((unsigned short)(w.x >> 16), bf16), nan_propagates));
        peak = max(peak, aq_abs_bits(aq_h2f((unsigned short)(w.y & 0xFFFFu), bf16), nan_propagates));
        peak = max(peak, aq_abs_bits(aq_h2f((unsigned short)(w.y >> 16), bf16), nan_propagates));
    }
    for (unsigned long long i = groups * 4 + (unsigned long long)blockIdx.x * AQ_THREADS + threadIdx.x;
         i < n; i += stride) {
        peak = max(peak, aq_abs_bits(aq_h2f(x[i], bf16), nan_propagates));
    }
    aq_block_max(peak, amax);
}

// ---- pass 2: scale + cast -------------------------------------------------------------------
// Every block derives the scale from the finished amax (pass 1 is ordered before it on the
// stream); block 0 publishes it to `scale_out[0]`, the device-resident scale the GEMM reads.
struct AqScale { float scale, mul; };

__device__ __forceinline__ AqScale aq_scale(const unsigned* amax, float qmax) {
    __shared__ float s_scale, s_mul;
    if (threadIdx.x == 0) {
        const float a = __uint_as_float(*amax);
        const float scale = fmaxf(__fdiv_rn(a, qmax), 1.17549435082228750797e-38f);
        s_scale = scale;
        s_mul = __double2float_rn(__ddiv_rn(1.0, (double)scale));
    }
    __syncthreads();
    return AqScale{s_scale, s_mul};
}

__device__ __forceinline__ float aq_scaled(float v, AqScale s, int divide) {
    return divide ? __fdiv_rn(v, s.scale) : __fmaf_rn(v, s.mul, 0.0f);
}

__device__ __forceinline__ float aq_int8(float v, AqScale s) {
    return fminf(fmaxf(roundf(__fmaf_rn(v, s.mul, 0.0f)), -127.0f), 127.0f);
}

__device__ __forceinline__ unsigned aq_pack4(unsigned char a, unsigned char b, unsigned char c,
                                             unsigned char d) {
    return (unsigned)a | ((unsigned)b << 8) | ((unsigned)c << 16) | ((unsigned)d << 24);
}

__device__ __forceinline__ unsigned char aq_i8_byte(float code) {
    return (unsigned char)(signed char)(int)code;
}

// Shared body: `load(i)` reads element i as f32; groups of four are written with one store.
#define AQ_CAST_BODY(LOAD4, LOAD1)                                                              \
    const unsigned long long stride = (unsigned long long)gridDim.x * AQ_THREADS;              \
    const unsigned long long groups = n / 4;                                                    \
    for (unsigned long long g = (unsigned long long)blockIdx.x * AQ_THREADS + threadIdx.x;     \
         g < groups; g += stride) {                                                             \
        float v0, v1, v2, v3;                                                                   \
        LOAD4;                                                                                  \
        AQ_EMIT4;                                                                               \
    }                                                                                           \
    for (unsigned long long i = groups * 4 + (unsigned long long)blockIdx.x * AQ_THREADS +     \
                                threadIdx.x;                                                    \
         i < n; i += stride) {                                                                  \
        float v;                                                                                \
        LOAD1;                                                                                  \
        AQ_EMIT1;                                                                               \
    }

#define AQ_LOAD4_F32                                                                            \
    {                                                                                           \
        const float4 w = reinterpret_cast<const float4*>(x)[g];                                 \
        v0 = w.x; v1 = w.y; v2 = w.z; v3 = w.w;                                                 \
    }
#define AQ_LOAD1_F32 v = x[i]
#define AQ_LOAD4_16                                                                             \
    {                                                                                           \
        const uint2 w = reinterpret_cast<const uint2*>(x)[g];                                   \
        v0 = aq_h2f((unsigned short)(w.x & 0xFFFFu), bf16);                                     \
        v1 = aq_h2f((unsigned short)(w.x >> 16), bf16);                                         \
        v2 = aq_h2f((unsigned short)(w.y & 0xFFFFu), bf16);                                     \
        v3 = aq_h2f((unsigned short)(w.y >> 16), bf16);                                         \
    }
#define AQ_LOAD1_16 v = aq_h2f(x[i], bf16)

// E4M3 codes. `divide` selects the Fp8Linear route's x / scale over the chain's fma(x, 1/scale).
#define AQ_EMIT4                                                                                \
    reinterpret_cast<unsigned*>(codes)[g] =                                                    \
        aq_pack4(aq_e4m3(aq_scaled(v0, s, divide)), aq_e4m3(aq_scaled(v1, s, divide)),         \
                 aq_e4m3(aq_scaled(v2, s, divide)), aq_e4m3(aq_scaled(v3, s, divide)))
#define AQ_EMIT1 codes[i] = aq_e4m3(aq_scaled(v, s, divide))

extern "C" __global__ void act_quant_fp8_f32(const float* x, unsigned long long n,
                                             const unsigned* amax, float* scale_out,
                                             unsigned char* codes, int divide) {
    const AqScale s = aq_scale(amax, 448.0f);
    if (blockIdx.x == 0 && threadIdx.x == 0) scale_out[0] = s.scale;
    AQ_CAST_BODY(AQ_LOAD4_F32, AQ_LOAD1_F32)
}

extern "C" __global__ void act_quant_fp8_16(const unsigned short* x, unsigned long long n,
                                            const unsigned* amax, float* scale_out,
                                            unsigned char* codes, int divide, int bf16) {
    const AqScale s = aq_scale(amax, 448.0f);
    if (blockIdx.x == 0 && threadIdx.x == 0) scale_out[0] = s.scale;
    AQ_CAST_BODY(AQ_LOAD4_16, AQ_LOAD1_16)
}

#undef AQ_EMIT4
#undef AQ_EMIT1

// int8 codes: `f32_codes` writes the rounded codes as F32 (`QuantizedActivation::q`'s contract),
// otherwise the i8 bytes cuBLASLt reads.
#define AQ_EMIT4                                                                                \
    {                                                                                           \
        const float c0 = aq_int8(v0, s), c1 = aq_int8(v1, s), c2 = aq_int8(v2, s),              \
                    c3 = aq_int8(v3, s);                                                        \
        if (f32_codes) {                                                                        \
            reinterpret_cast<float4*>(codes)[g] = make_float4(c0, c1, c2, c3);                  \
        } else {                                                                                \
            reinterpret_cast<unsigned*>(codes)[g] =                                            \
                aq_pack4(aq_i8_byte(c0), aq_i8_byte(c1), aq_i8_byte(c2), aq_i8_byte(c3));      \
        }                                                                                       \
    }
#define AQ_EMIT1                                                                                \
    {                                                                                           \
        const float c = aq_int8(v, s);                                                          \
        if (f32_codes) {                                                                        \
            reinterpret_cast<float*>(codes)[i] = c;                                             \
        } else {                                                                                \
            reinterpret_cast<unsigned char*>(codes)[i] = aq_i8_byte(c);                         \
        }                                                                                       \
    }

extern "C" __global__ void act_quant_int8_f32(const float* x, unsigned long long n,
                                              const unsigned* amax, float* scale_out,
                                              void* codes, int f32_codes) {
    const AqScale s = aq_scale(amax, 127.0f);
    if (blockIdx.x == 0 && threadIdx.x == 0) scale_out[0] = s.scale;
    AQ_CAST_BODY(AQ_LOAD4_F32, AQ_LOAD1_F32)
}

extern "C" __global__ void act_quant_int8_16(const unsigned short* x, unsigned long long n,
                                             const unsigned* amax, float* scale_out, void* codes,
                                             int f32_codes, int bf16) {
    const AqScale s = aq_scale(amax, 127.0f);
    if (blockIdx.x == 0 && threadIdx.x == 0) scale_out[0] = s.scale;
    AQ_CAST_BODY(AQ_LOAD4_16, AQ_LOAD1_16)
}
