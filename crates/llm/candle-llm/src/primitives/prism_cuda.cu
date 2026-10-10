// Native compact Prism/Bonsai CUDA kernels. The packed source is decoded inside each dot product or
// selected embedding row; no dense checkpoint-sized buffer is ever allocated.

// NVRTC is invoked with no SDK or host compiler include paths. Keep the runtime source
// self-contained; these CUDA ABI types match the Rust u8/u16/u32/i64 launch buffers on
// both Windows (LLP64) and Unix (LP64). In particular, `long` is not a portable i64.
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
static_assert(sizeof(uint8_t) == 1, "packed byte ABI");
static_assert(sizeof(uint16_t) == 2, "packed half bits ABI");
static_assert(sizeof(uint32_t) == 4, "packed word ABI");
static_assert(sizeof(int64_t) == 8, "embedding index ABI");

__device__ __forceinline__ float prism_half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t prism_stored_row(
    uint32_t row, uint32_t prefix, uint32_t groups, uint32_t repetitions, uint32_t unit) {
    if (groups == 0 || row < prefix) return row;
    uint32_t logical = row - prefix;
    uint32_t group = logical / (repetitions * unit);
    uint32_t rem = logical % (repetitions * unit);
    uint32_t repetition = rem / unit;
    uint32_t lane = rem % unit;
    return prefix + (repetition * groups + group) * unit + lane;
}

__device__ __forceinline__ float prism_pq2_value(
    const uint8_t* packed, uint32_t row, uint32_t width, uint32_t col) {
    uint32_t blocks = width / 128;
    uint32_t lane = col & 127u;
    const uint8_t* block = packed + (row * blocks + col / 128) * 34;
    float scale = prism_half_to_float((uint16_t)block[0] | ((uint16_t)block[1] << 8));
    uint8_t code = (block[2 + lane / 4] >> (2 * (lane & 3u))) & 3u;
    return ((float)code - 1.0f) * scale;
}

// 3^trit for trit in 0..=4, as a select chain. Do not turn this back into an indexed local array:
// `trit` is data-dependent, so nvrtc places such an array in per-thread local memory. Since
// sc-24137 the kernels compile for the device's own architecture (`compute_120` on Blackwell), and
// there the array spilled 24 B of local memory into every PTQ dot product. That made
// `prism_ptq_matmul_f32` 3.2x slower and Bonsai GGUF prefill 2.5x slower (sc-24164). The integer
// result, and so every decoded value, is unchanged.
__device__ __forceinline__ uint32_t prism_pow3(uint32_t trit) {
    return trit == 0u ? 1u : trit == 1u ? 3u : trit == 2u ? 9u : trit == 3u ? 27u : 81u;
}

__device__ __forceinline__ float prism_ptq_value(
    const uint8_t* packed, uint32_t row, uint32_t width, uint32_t col) {
    uint32_t blocks = width / 128;
    uint32_t lane = col & 127u;
    const uint8_t* block = packed + (row * blocks + col / 128) * 28;
    float scale = prism_half_to_float((uint16_t)block[26] | ((uint16_t)block[27] << 8));
    uint32_t byte_at, trit;
    if (lane < 80) {
        byte_at = lane % 16;
        trit = lane / 16;
    } else if (lane < 120) {
        lane -= 80;
        byte_at = 16 + lane % 8;
        trit = lane / 8;
    } else {
        lane -= 120;
        byte_at = 24 + lane % 2;
        trit = lane / 2;
    }
    uint32_t code = ((((uint32_t)block[byte_at] * prism_pow3(trit)) & 255u) * 3u) >> 8;
    return ((float)code - 1.0f) * scale;
}

extern "C" __global__ void prism_mlx_affine2_matmul_f32(
    const float* x, const uint32_t* words, const float* scales, float* output,
    uint32_t tokens, uint32_t rows, uint32_t width,
    uint32_t prefix, uint32_t groups, uint32_t repetitions, uint32_t unit) {
    uint32_t job = blockIdx.x;
    if (job >= tokens * rows) return;
    uint32_t token = job / rows;
    uint32_t logical_row = job % rows;
    uint32_t row = prism_stored_row(logical_row, prefix, groups, repetitions, unit);
    uint32_t words_per_row = width / 16;
    uint32_t groups_per_row = width / 128;
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
        uint32_t word = words[row * words_per_row + col / 16];
        uint32_t code = (word >> (2 * (col & 15u))) & 3u;
        float value = ((float)code - 1.0f) * scales[row * groups_per_row + col / 128];
        sum += x[token * width + col] * value;
    }
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[job] = scratch[0];
}

#define PRISM_GGUF_MATMUL(NAME, VALUE_FN) \
extern "C" __global__ void NAME( \
    const float* x, const uint8_t* packed, float* output, \
    uint32_t tokens, uint32_t rows, uint32_t width, \
    uint32_t prefix, uint32_t groups, uint32_t repetitions, uint32_t unit) { \
    uint32_t job = blockIdx.x; \
    if (job >= tokens * rows) return; \
    uint32_t token = job / rows; \
    uint32_t logical_row = job % rows; \
    uint32_t row = prism_stored_row(logical_row, prefix, groups, repetitions, unit); \
    float sum = 0.0f; \
    for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) \
        sum += x[token * width + col] * VALUE_FN(packed, row, width, col); \
    extern __shared__ float scratch[]; \
    scratch[threadIdx.x] = sum; \
    __syncthreads(); \
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) { \
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride]; \
        __syncthreads(); \
    } \
    if (threadIdx.x == 0) output[job] = scratch[0]; \
}

PRISM_GGUF_MATMUL(prism_pq2_matmul_f32, prism_pq2_value)
PRISM_GGUF_MATMUL(prism_ptq_matmul_f32, prism_ptq_value)

extern "C" __global__ void prism_mlx_affine2_embedding_f32(
    const int64_t* ids, const uint32_t* words, const float* scales, float* output,
    uint32_t count, uint32_t rows, uint32_t width) {
    uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count * width) return;
    uint32_t item = at / width, col = at % width;
    int64_t row64 = ids[item];
    if (row64 < 0 || row64 >= rows) { output[at] = __int_as_float(0x7fffffff); return; }
    uint32_t row = (uint32_t)row64;
    uint32_t word = words[row * (width / 16) + col / 16];
    uint32_t code = (word >> (2 * (col & 15u))) & 3u;
    output[at] = ((float)code - 1.0f) * scales[row * (width / 128) + col / 128];
}

#define PRISM_GGUF_EMBED(NAME, VALUE_FN) \
extern "C" __global__ void NAME( \
    const int64_t* ids, const uint8_t* packed, float* output, \
    uint32_t count, uint32_t rows, uint32_t width) { \
    uint32_t at = blockIdx.x * blockDim.x + threadIdx.x; \
    if (at >= count * width) return; \
    uint32_t item = at / width, col = at % width; \
    int64_t row64 = ids[item]; \
    if (row64 < 0 || row64 >= rows) { output[at] = __int_as_float(0x7fffffff); return; } \
    output[at] = VALUE_FN(packed, (uint32_t)row64, width, col); \
}

PRISM_GGUF_EMBED(prism_pq2_embedding_f32, prism_pq2_value)
PRISM_GGUF_EMBED(prism_ptq_embedding_f32, prism_ptq_value)

// ---------------------------------------------------------------------------------------------
// Fused block-Hadamard rotation (sc-24440, assigned from sc-24444's review). One launch replaces
// the device-tensor chain `transform_forward_tensor` / `transform_inverse` runs:
//
//   forward: [GDN gather] -> x * sign -> FWHT (log2(B) butterfly stages) -> * bf(1/sqrt(B)) + 0
//   inverse: FWHT -> * bf(1/sqrt(B)) + 0 -> * sign
//
// BIT-IDENTICAL to that chain: every value is rounded to the activation dtype exactly where the
// chain stores a tensor of it — after the sign product, after EVERY butterfly add / sub, after the
// affine's product and again after its `+ 0` — and each such op is computed in f32 then rounded
// once, which equals the dtype's own correctly-rounded op (bf16 / f16: a product or sum of two
// such values is exact in f32 or rounds innocuously; 24 >= 2p + 2 for p = 8, 11). `mul` is the
// affine factor already rounded to the dtype on the host (candle's `T::from_f64`). The forward's
// trailing `.to_dtype(F32)` is fused as an f32 store.
//
// One block per (activation row, Hadamard block); the block's values live in shared memory
// (`block` f32), butterflies split over the threads, a barrier between stages.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float prism_bf16_round(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float((u | 0x00400000u) & 0xffff0000u);
    uint32_t lsb = (u >> 16) & 1u;
    return __uint_as_float(((u + 0x7fffu + lsb) >> 16) << 16);
}

__device__ __forceinline__ float prism_f16_round(float f) {
    uint16_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

// dtype codes: 0 = f32, 1 = bf16, 2 = f16.
__device__ __forceinline__ float prism_round(float v, int dtype) {
    return dtype == 1 ? prism_bf16_round(v) : dtype == 2 ? prism_f16_round(v) : v;
}

__device__ __forceinline__ float prism_load(const void* x, size_t i, int dtype) {
    if (dtype == 1) return __uint_as_float(((uint32_t)((const uint16_t*)x)[i]) << 16);
    if (dtype == 2) {
        float f;
        asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(((const uint16_t*)x)[i]));
        return f;
    }
    return ((const float*)x)[i];
}

__device__ __forceinline__ void prism_store(void* y, size_t i, float v, int dtype) {
    if (dtype == 1) {
        ((uint16_t*)y)[i] = (uint16_t)(__float_as_uint(prism_bf16_round(v)) >> 16);
    } else if (dtype == 2) {
        uint16_t h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
        ((uint16_t*)y)[i] = h;
    } else {
        ((float*)y)[i] = v;
    }
}

// x: [rows, width] of `dtype`; signs: [width] f32 (+-1); gather: [width] u32 source column of each
// column (forward only; null = identity); y: [rows, width] of `out_dtype` (0 = f32, else
// `dtype`). `inverse` selects the inverse order. `mul` is the dtype-rounded 1/sqrt(block).
extern "C" __global__ void prism_rotate(
    const void* __restrict__ x, const float* __restrict__ signs,
    const uint32_t* __restrict__ gather, void* __restrict__ y,
    uint32_t width, uint32_t block, float mul, int dtype, int out_dtype, int inverse) {
    extern __shared__ float v[];
    const uint32_t row = blockIdx.y;
    const uint32_t base = blockIdx.x * block;
    const size_t row_at = (size_t)row * width;
    for (uint32_t c = threadIdx.x; c < block; c += blockDim.x) {
        const uint32_t col = base + c;
        const uint32_t src = (gather && !inverse) ? gather[col] : col;
        float value = prism_load(x, row_at + src, dtype);
        if (!inverse) value = prism_round(value * signs[col], dtype);
        v[c] = value;
    }
    __syncthreads();
    const uint32_t half = block / 2;
    for (uint32_t step = 1; step < block; step <<= 1) {
        for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
            const uint32_t lo = (i / step) * 2 * step + (i % step);
            const uint32_t hi = lo + step;
            const float a = v[lo];
            const float b = v[hi];
            v[lo] = prism_round(a + b, dtype);
            v[hi] = prism_round(a - b, dtype);
        }
        __syncthreads();
    }
    for (uint32_t c = threadIdx.x; c < block; c += blockDim.x) {
        const uint32_t col = base + c;
        // The affine: `x * mul` rounded, then `+ 0` rounded (turns -0 into +0, like the chain).
        float value = prism_round(prism_round(v[c] * mul, dtype) + 0.0f, dtype);
        if (inverse) value = prism_round(value * signs[col], dtype);
        prism_store(y, row_at + col, value, out_dtype == 0 ? 0 : dtype);
    }
}
