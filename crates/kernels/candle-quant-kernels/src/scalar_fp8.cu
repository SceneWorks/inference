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
                                             unsigned short* out, unsigned m, unsigned n,
                                             unsigned k, float sx, float sw) {
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
            unsigned short bits;
            asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(bits) : "f"(acc[i] * sx * sw));
            out[(unsigned long long)row*n + nc + thread*2 + i%2] = bits;
        }
    }
}
