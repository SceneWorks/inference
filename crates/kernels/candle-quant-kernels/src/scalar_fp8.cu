// E4M3FN saturating round-to-nearest-even, available on sm_89 and later.
extern "C" __global__ void scalar_fp8_quantize(const float* x, unsigned char* codes,
                                                float scale, unsigned int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(x[i] / scale));
    codes[i] = (unsigned char)pair;
}

extern "C" __global__ void scalar_e5m2_quantize(const float* x, unsigned char* codes,
                                                float scale, unsigned int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned short pair;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(x[i] / scale));
    codes[i] = (unsigned char)pair;
}

// The FP8 tensor-core instruction supports E5M2 x E5M2. CUDA 12 cuBLASLt's
// published scalar GEMM combinations require at least one E4M3 input.
extern "C" __global__ void scalar_e5m2_gemm(const unsigned char* x, const unsigned char* w,
                                             void* out, unsigned m, unsigned n,
                                             unsigned k, float sx, float sw, unsigned output_f32) {
    const unsigned lane = threadIdx.x % 32, group = lane / 4, thread = lane % 4;
    const unsigned mr = blockIdx.y * 16 + group;
    const unsigned nc = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * 8;
    if (nc >= n) return;
    float acc[4] = {};
    for (unsigned j = 0; j < k; j += 16) {
        const unsigned col = j + thread * 4;
        const unsigned a0 = mr < m ? *reinterpret_cast<const unsigned*>(x + (unsigned long long)mr*k + col) : 0;
        const unsigned a1 = mr+8 < m ? *reinterpret_cast<const unsigned*>(x + (unsigned long long)(mr+8)*k + col) : 0;
        const unsigned b = *reinterpret_cast<const unsigned*>(w + (unsigned long long)(nc+group)*k + col);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32 "
                     "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                     : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                     : "r"(a0), "r"(a1), "r"(b));
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned row = mr + (i/2)*8;
        if (row < m) {
            const unsigned long long index = (unsigned long long)row*n + nc + thread*2 + i%2;
            const float value = acc[i] * sx * sw;
            if (output_f32) {
                static_cast<float*>(out)[index] = value;
            } else {
                unsigned short bits;
                asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(bits) : "f"(value));
                static_cast<unsigned short*>(out)[index] = bits;
            }
        }
    }
}

// Every finite FP8 value has an exact binary16 representation, including FP8 subnormals.
__device__ __forceinline__ unsigned fp8_pair_to_f16(unsigned short codes, unsigned e5m2) {
    unsigned value;
    if (e5m2) {
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(value) : "h"(codes));
    } else {
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(value) : "h"(codes));
    }
    return value;
}

__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

template <bool BF16>
__device__ __forceinline__ unsigned short round_to_16(float value) {
    unsigned short bits;
    if (BF16) {
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    } else {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    }
    return bits;
}

template <bool BF16>
__device__ __forceinline__ unsigned weight_pair_16(const unsigned char* codes, float scale,
                                                   unsigned e5m2) {
    const unsigned pair = fp8_pair_to_f16(*reinterpret_cast<const unsigned short*>(codes), e5m2);
    if (!BF16 && scale == 1.0f) return pair;
    // Match the dense reference: apply the scale in F32, then round the weight operand.
    const unsigned lo = round_to_16<BF16>(f16_bits_to_f32((unsigned short)pair) * scale);
    const unsigned hi = round_to_16<BF16>(f16_bits_to_f32((unsigned short)(pair >> 16)) * scale);
    return (hi << 16) | lo;
}

template <bool BF16>
__device__ __forceinline__ void weight_only_mma_16(float acc[4], unsigned a0, unsigned a1,
                                                   unsigned a2, unsigned a3, unsigned b0,
                                                   unsigned b1) {
    if (BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

// One warp writes a 16-by-8 output tile. Each four-lane group loads the two
// input rows and one output-channel weight row belonging to its MMA fragments.
template <bool BF16>
__device__ __forceinline__ void weight_only_fp8_16(const unsigned short* x, const unsigned char* w,
    unsigned short* out, unsigned m, unsigned n, unsigned k, float scale, unsigned e5m2) {
    const unsigned lane = threadIdx.x % 32, group = lane / 4, thread = lane % 4;
    const unsigned mr = blockIdx.y * 16 + group;
    const unsigned nc = blockIdx.x * 32 + (threadIdx.x / 32) * 8;
    if (nc >= n) return; // Whole warp. The packed layout aligns N and K to 16.
    const unsigned char* weight = w + (unsigned long long)(nc + group) * k;
    float acc[4] = {};
    for (unsigned tile = 0; tile < k; tile += 16) {
        const unsigned col = tile + thread * 2;
        unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (mr < m) {
            const unsigned short* row = x + (unsigned long long)mr * k + col;
            a0 = *reinterpret_cast<const unsigned*>(row);
            a2 = *reinterpret_cast<const unsigned*>(row + 8);
        }
        if (mr + 8 < m) {
            const unsigned short* row = x + (unsigned long long)(mr + 8) * k + col;
            a1 = *reinterpret_cast<const unsigned*>(row);
            a3 = *reinterpret_cast<const unsigned*>(row + 8);
        }
        const unsigned b0 = weight_pair_16<BF16>(weight + col, scale, e5m2);
        const unsigned b1 = weight_pair_16<BF16>(weight + col + 8, scale, e5m2);
        weight_only_mma_16<BF16>(acc, a0, a1, a2, a3, b0, b1);
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned row = mr + (i / 2) * 8;
        if (row < m) {
            const unsigned long long offset = (unsigned long long)row * n + nc + thread * 2 + i % 2;
            out[offset] = round_to_16<BF16>(acc[i]);
        }
    }
}

extern "C" __global__ void scalar_fp8_weight_only_f16(const unsigned short* x, const unsigned char* w,
    unsigned short* out, unsigned m, unsigned n, unsigned k, float scale, unsigned e5m2) {
    weight_only_fp8_16<false>(x, w, out, m, n, k, scale, e5m2);
}

extern "C" __global__ void scalar_fp8_weight_only_bf16(const unsigned short* x, const unsigned char* w,
    unsigned short* out, unsigned m, unsigned n, unsigned k, float scale, unsigned e5m2) {
    weight_only_fp8_16<true>(x, w, out, m, n, k, scale, e5m2);
}

// F32 activations require full precision, including bits that TF32 would discard.
// Decode the 32-channel weight tile once into shared memory, and reuse it for
// the 16 activation rows in this block.
extern "C" __global__ void scalar_fp8_weight_only_f32(const float* x, const unsigned char* w,
    float* out, unsigned m, unsigned n, unsigned k, float scale, unsigned e5m2) {
    __shared__ float activation[16][17], weight[16][32];
    const unsigned tid = threadIdx.x;
    const unsigned m0 = blockIdx.y * 16, n0 = blockIdx.x * 32;
    const unsigned row = tid / 8, col = (tid % 8) * 4;
    float acc[4] = {};
    for (unsigned tile = 0; tile < k; tile += 16) {
        for (unsigned i = tid; i < 16 * 16; i += 128) {
            const unsigned r = i / 16, c = i % 16;
            activation[r][c] = m0 + r < m ? x[(unsigned long long)(m0 + r) * k + tile + c] : 0.0f;
        }
        for (unsigned i = tid; i < 32 * 8; i += 128) {
            const unsigned r = i / 8, c = (i % 8) * 2;
            unsigned short codes = 0;
            if (n0 + r < n) {
                const unsigned long long offset = (unsigned long long)(n0 + r) * k + tile + c;
                codes = *reinterpret_cast<const unsigned short*>(w + offset);
            }
            const unsigned pair = fp8_pair_to_f16(codes, e5m2);
            weight[c][r] = f16_bits_to_f32((unsigned short)pair) * scale;
            weight[c + 1][r] = f16_bits_to_f32((unsigned short)(pair >> 16)) * scale;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < 16; ++i) {
            const float a = activation[row][i];
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) acc[j] = fmaf(a, weight[i][col + j], acc[j]);
        }
        __syncthreads();
    }
    if (m0 + row < m && n0 + col < n) {
        const unsigned long long offset = (unsigned long long)(m0 + row) * n + n0 + col;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) out[offset + j] = acc[j];
    }
}

// One warp computes the activation scale and codes for a block of 32 elements.
// Pad both dimensions for the block-scaled GEMM. Unused scale atom entries are zero.
extern "C" __global__ void mxfp8_activation(const float* x, unsigned char* codes,
    unsigned char* scales, unsigned m, unsigned k, unsigned k_padded, unsigned m_padded) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned block = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned blocks = k_padded / 32;
    const unsigned row = block / blocks, col_block = block % blocks;
    if (row >= m_padded) return;
    const unsigned col = col_block * 32 + lane;
    const float v = row < m && col < k ? x[(unsigned long long)row*k + col] : 0;
    float amax = fabsf(v);
    for (unsigned offset = 16; offset; offset /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    const float need = fmaxf(amax / 448.f, 0x1p-127f);
    int power;
    const float mantissa = frexpf(need, &power);
    const int exp = max(0, min(254, power - (mantissa == 0.5f) + 127));
    const float scaled = ldexpf(v, 127 - exp);
    unsigned short bits;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;" : "=h"(bits) : "f"(0.f),"f"(scaled));
    codes[(unsigned long long)row*k_padded + col] = (unsigned char)bits;
    if (lane == 0) {
        const unsigned atom = (row / 128) * ((blocks + 3) / 4) + col_block / 4;
        scales[atom*512 + (row%32)*16 + ((row%128)/32)*4 + col_block%4] = amax == 0.f ? 0 : (unsigned char)exp;
    }
}
