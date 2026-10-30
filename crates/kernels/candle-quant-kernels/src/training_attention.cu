// sc-26856: streaming SDPA candidate. No score/probability matrix, atomics or
// cross-stream work. A warp owns one row; reductions/accumulators are f32.
// This deliberately simple SIMT candidate is NOT a tensor-core FlashAttention
// implementation. Adoption requires measured forward+backward go/no-go.
// Load the public input dtype directly; arithmetic and saved output/LSE stay f32.
// F16 conversion is exact and BF16 widening only shifts the representation.
__device__ __forceinline__ float input_f32(const void* ptr, int index, int dtype) {
    if (dtype == 0) return ((const float*)ptr)[index];
    unsigned short bits = ((const unsigned short*)ptr)[index];
    if (dtype == 1) return __uint_as_float(((unsigned int)bits) << 16);
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}
__device__ float warp_sum(float x) {
    for (int n = 16; n; n >>= 1) x += __shfl_down_sync(0xffffffff, x, n);
    return __shfl_sync(0xffffffff, x, 0);
}
extern "C" __global__ void attention_forward(
    const void* q, const void* k, const void* v, float* out,
    int rows, int hq, int hk, int sq, int sk, int d, int dtype, float scale) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int batch = row / (hq * sq), head = (row / sq) % hq;
    int kb = (batch * hk + head / (hq / hk)) * sk * d;
    float acc[8] = {}, m = -__int_as_float(0x7f800000), z = 0;
    for (int j = 0; j < sk; ++j) {
        float score = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) score += input_f32(q, row*d+lane+32*t, dtype) * input_f32(k, kb+j*d+lane+32*t, dtype);
        score = warp_sum(score) * scale;
        float next = fmaxf(m, score), a = expf(m-next), p = expf(score-next);
        #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) acc[t] = a*acc[t] + p*input_f32(v, kb+j*d+lane+32*t, dtype);
        z = a*z + p; m = next;
    }
    #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) out[row*(d+1)+lane+32*t] = acc[t]/z;
    if (lane == 0) out[row*(d+1)+d] = m + logf(z);
}
extern "C" __global__ void attention_dq(
    const void* q, const void* k, const void* v, const float* out,
    const float* grad, float* dst,
    int rows, int hq, int hk, int sq, int sk, int d, int dtype, float scale) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int batch = row/(hq*sq), head = (row/sq)%hq;
    int kb = (batch*hk + head/(hq/hk))*sk*d;
    float delta = 0, acc[8] = {};
    #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) delta += out[row*(d+1)+lane+32*t]*grad[row*(d+1)+lane+32*t];
    delta = warp_sum(delta);
    float lse = out[row*(d+1)+d];
    for (int j = 0; j < sk; ++j) {
        float score = 0, dp = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) {
            score += input_f32(q, row*d+lane+32*t, dtype)*input_f32(k, kb+j*d+lane+32*t, dtype);
            dp += grad[row*(d+1)+lane+32*t]*input_f32(v, kb+j*d+lane+32*t, dtype);
        }
        float p = expf(warp_sum(score)*scale-lse);
        float ds = p*(warp_sum(dp)-delta)*scale;
        #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) acc[t] += ds*input_f32(k, kb+j*d+lane+32*t, dtype);
    }
    #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) dst[row*d+lane+32*t] = acc[t];
}
extern "C" __global__ void attention_dkv(
    const void* q, const void* k, const void* v, const float* out,
    const float* grad, float* dst,
    int rows, int hq, int hk, int sq, int sk, int d, int dtype, float scale) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int batch = row/(hk*sk), head = (row/sk)%hk;
    int groups = hq/hk, qrows = rows/sk/hk*hq*sq;
    float dk[8] = {}, dv[8] = {};
    for (int g = 0; g < groups; ++g) {
        int qb = (batch*hq+head*groups+g)*sq;
        for (int i = 0; i < sq; ++i) {
            int qr = qb+i;
            float score = 0, dp = 0, delta = 0;
            #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) {
                score += input_f32(q, qr*d+lane+32*t, dtype)*input_f32(k, row*d+lane+32*t, dtype);
                dp += grad[qr*(d+1)+lane+32*t]*input_f32(v, row*d+lane+32*t, dtype);
                delta += grad[qr*(d+1)+lane+32*t]*out[qr*(d+1)+lane+32*t];
            }
            float p = expf(warp_sum(score)*scale-out[qr*(d+1)+d]);
            float ds = p*(warp_sum(dp)-warp_sum(delta))*scale;
            #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) {
                dk[t] += ds*input_f32(q, qr*d+lane+32*t, dtype);
                dv[t] += p*grad[qr*(d+1)+lane+32*t];
            }
        }
    }
    #pragma unroll
        for (int t = 0; t < 8; ++t) if (lane+32*t < d) {
        dst[qrows*d+row*d+lane+32*t] = dk[t];
        dst[(qrows+rows)*d+row*d+lane+32*t] = dv[t];
    }
}
