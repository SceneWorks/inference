// E4M3FN saturating round-to-nearest-even, available on sm_89 and later.
extern "C" __global__ void scalar_fp8_quantize(const float* x, unsigned char* codes,
                                                float scale, unsigned int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(x[i] / scale));
    codes[i] = (unsigned char)pair;
}
