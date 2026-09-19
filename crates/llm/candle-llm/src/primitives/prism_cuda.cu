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

__device__ __forceinline__ float prism_ptq_value(
    const uint8_t* packed, uint32_t row, uint32_t width, uint32_t col) {
    const uint32_t pow3[5] = {1u, 3u, 9u, 27u, 81u};
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
    uint32_t code = ((((uint32_t)block[byte_at] * pow3[trit]) & 255u) * 3u) >> 8;
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
