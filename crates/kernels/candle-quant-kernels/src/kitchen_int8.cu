// sc-26683: Kitchen f8b7e97a1418d2e529bd5405fbf86312be198ae6 rowwise semantics.
// The input has already been ConvRot-rotated and rounded to the source dtype.
__device__ float source_round(float x, int dtype) {
    if (dtype == 1) {
        unsigned int u = __float_as_uint(x);
        if ((u & 0x7fffffffU) > 0x7f800000U) return x;
        u += 0x7fffU + ((u >> 16) & 1U);
        return __uint_as_float(u & 0xffff0000U);
    }
    if (dtype == 2) {
        unsigned short h;
        float y;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
        asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
        return y;
    }
    return x;
}
extern "C" __global__ void kitchen_int8_quantize(
    const float* x, unsigned char* q, float* scales, int k, int dtype) {
    __shared__ float peaks[256];
    int tid = threadIdx.x;
    unsigned long long offset = (unsigned long long)blockIdx.x * k;
    float peak = 0.0f;
    for (int j=tid;j<k;j+=256) peak=fmaxf(peak,fabsf(x[offset+j]));
    peaks[tid]=peak;
    __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if(tid<stride) peaks[tid]=fmaxf(peaks[tid],peaks[tid+stride]);
        __syncthreads();
    }
    float limit=dtype==1 ? 3.38953139e38f : dtype==2 ? 65504.0f : 3.402823466e38f;
    float scale=fmaxf(fminf(peaks[0],limit)*(1.0f/127.0f),1.0e-30f);
    if(tid==0) scales[blockIdx.x]=scale;
    float divisor=source_round(scale,dtype);
    for (int j=tid;j<k;j+=256) {
        float z=source_round(x[offset+j]/divisor,dtype);
        float rounded=fminf(127.0f,fmaxf(-128.0f,nearbyintf(z)));
        q[offset+j]=(unsigned char)(signed char)rounded;
    }
}
