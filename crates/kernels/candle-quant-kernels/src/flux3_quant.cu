// FLUX3 publisher-format activation kernels (sc-26241, epic 26233).
//
// Exactly the frozen BFL inference recipes, NOT the repo's generic NVFP4/FP8 recipes:
//
//  * FP8R  — W8A8 rowwise: each activation row is scaled by max(|row|)/448 (clamped at 1e-12) and
//            rounded to E4M3; the FP8 GEMM accumulates in f32 and the epilogue applies the per-row
//            activation scale and per-output weight scale BEFORE the single BF16 rounding
//            (`torch._scaled_mm(..., scale_a=[M,1], scale_b=[1,N], out_dtype=bf16)`).
//  * NVFP4 — W4A4 with a STATIC calibrated `input_scale`: per-16 block, scale code
//            e4m3(min(amax / (6 * max(input_scale, 1e-8)), 448)), elements e2m1(v / (decode(code) *
//            max(input_scale, 1e-8))), zero codes when that divisor is 0. The GEMM alpha is the RAW
//            `input_scale * weight_scale_2` (the clamp applies only to the quantizer reciprocal).
//
// The CPU mirrors in `flux3_quant.rs` are bit-for-bit transcriptions of every device helper here and
// are tested against the gen-core oracles (`gen_core::flux3::quantization`). nvrtc, no headers.

#define F3_LN2      0.6931471805599453f
#define F3_INV_LN2  1.4426950408889634f

extern "C" {

// Same E2M1 nearest-with-even-ties encoder as nvfp4_quant.cu (sign in bit 3, saturate at 6).
__device__ __forceinline__ unsigned int f3_e2m1_code(float v) {
    if (isnan(v)) return 0u;
    unsigned int sign = signbit(v) ? 0x8u : 0x0u;
    float mag = fminf(fabsf(v), 6.0f);
    unsigned int idx;
    if      (mag <= 0.25f) idx = 0;
    else if (mag <  0.75f) idx = 1;
    else if (mag <= 1.25f) idx = 2;
    else if (mag <  1.75f) idx = 3;
    else if (mag <= 2.5f)  idx = 4;
    else if (mag <  3.5f)  idx = 5;
    else if (mag <= 5.0f)  idx = 6;
    else                   idx = 7;
    return sign | idx;
}

// Same E4M3 grid rounding (ties-to-even, saturate 448) as nvfp4_quant.cu.
__device__ __forceinline__ float f3_e4m3_round(float v) {
    v = fminf(fmaxf(v, 0.0f), 448.0f);
    float vc = fminf(fmaxf(v, 0.015625f), 448.0f);
    float e = floorf(logf(vc) * F3_INV_LN2);
    float ulp = expf((e - 3.0f) * F3_LN2);
    float q = (float)__float2int_rn(v / ulp) * ulp;
    return fminf(fmaxf(q, 0.0f), 448.0f);
}

__device__ __forceinline__ unsigned int f3_e4m3_byte_from_decoded(float q) {
    if (q < (1.0f / 512.0f) * 0.5f) return 0u;
    if (q < 0.015625f) {
        int b = (int)roundf(q * 512.0f);
        return (unsigned int)(b < 0 ? 0 : (b > 254 ? 254 : b));
    }
    float e = floorf(logf(q) * F3_INV_LN2);
    float pow_e = expf(e * F3_LN2);
    int mant = (int)roundf(q / pow_e * 8.0f);
    int byte = (int)e * 8 + 48 + mant;
    if (byte < 0) byte = 0;
    if (byte > 254) byte = 254;
    return (unsigned int)byte;
}

// f32 -> bf16 bits, round-to-nearest-even, NaN kept quiet.
__device__ __forceinline__ unsigned short f3_bf16_bits(float x) {
    unsigned int b = __float_as_uint(x);
    if (isnan(x)) return (unsigned short)((b >> 16) | 0x40u);
    unsigned int round = 0x7fffu + ((b >> 16) & 1u);
    return (unsigned short)((b + round) >> 16);
}

// ---- FP8R activation: one CTA (256 threads) per row. Writes the DECODED on-grid E4M3 value as f32
// (exactly representable, so a later f32 -> F8E4M3 cast is lossless) and the row scale.
__global__ void flux3_fp8_rowwise_quantize_f32(
    const float* __restrict__ x,   // [m, k]
    float* __restrict__ q,         // [m, k] on-grid E4M3 values (signed)
    float* __restrict__ scales,    // [m]
    int m, int k)
{
    __shared__ float red[256];
    int r = blockIdx.x;
    if (r >= m) return;
    const float* row = x + (long)r * k;
    float a = 0.0f;
    for (int c = threadIdx.x; c < k; c += blockDim.x) a = fmaxf(a, fabsf(row[c]));
    red[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float scale = fmaxf(red[0] / 448.0f, 1e-12f);
    if (threadIdx.x == 0) scales[r] = scale;
    float* out = q + (long)r * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        float v = row[c] / scale;
        float mag = f3_e4m3_round(fabsf(v));
        out[c] = signbit(v) ? -mag : mag;
    }
}

// ---- FP8R epilogue: bf16(acc[m,n] * sa[m] * sw[n]) — one rounding of the f32 accumulate.
__global__ void flux3_fp8_rowwise_epilogue(
    const float* __restrict__ acc,   // [m, n] f32 GEMM accumulate (unit operand scales)
    const float* __restrict__ sa,    // [m] activation row scales
    const float* __restrict__ sw,    // [n] weight row scales
    unsigned short* __restrict__ out,// [m, n] bf16 bits
    int m, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)m * n;
    if (idx >= total) return;
    int r = (int)(idx / n);
    int c = (int)(idx % n);
    out[idx] = f3_bf16_bits(acc[idx] * sa[r] * sw[c]);
}

// ---- NVFP4 static activation pack: one thread per (row, 16-block). `packed` and `scales` must be
// pre-zeroed so K padding and the 128x4 atom padding stay 0. Scale bytes are scattered into the
// cuBLASLt / FlashInfer row-major-atom layout (K atoms fastest).
__global__ void flux3_nvfp4_static_pack_f32(
    const float* __restrict__ x,          // [m, k]
    unsigned char* __restrict__ packed,   // [m, cols_padded / 2]
    unsigned char* __restrict__ scales,   // [sf_rows * sf_cols]
    int m, int k, int n_blocks, int cols_padded, int sf_cols,
    float qscale)                          // max(input_scale, 1e-8)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)m * n_blocks;
    if (idx >= total) return;
    int r = (int)(idx / n_blocks);
    int blk = (int)(idx % n_blocks);
    int c0 = blk * 16;
    const float* row = x + (long)r * k;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int c = c0 + j;
        if (c < k) amax = fmaxf(amax, fabsf(row[c]));
    }
    float sf = fminf(amax / (6.0f * qscale), 448.0f);
    float decoded = f3_e4m3_round(sf);
    unsigned int code = f3_e4m3_byte_from_decoded(decoded);
    float divisor = decoded * qscale;
    unsigned char* out = packed + (long)r * (cols_padded / 2) + blk * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c = c0 + 2 * j;
        unsigned int lo = 0u, hi = 0u;
        if (divisor != 0.0f) {
            if (c < k) lo = f3_e2m1_code(row[c] / divisor);
            if (c + 1 < k) hi = f3_e2m1_code(row[c + 1] / divisor);
        }
        out[j] = (unsigned char)(lo | (hi << 4));
    }
    int m_atom = r / 128, mr = r % 128;
    int k_atom = blk / 4, kc = blk % 4;
    long atom = (long)m_atom * (sf_cols / 4) + k_atom;
    scales[atom * 512 + (mr % 32) * 16 + (mr / 32) * 4 + kc] = (unsigned char)code;
}

} // extern "C"
