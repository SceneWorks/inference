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
