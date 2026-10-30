// sc-26856: streaming SDPA candidate. No score/probability matrix, atomics or
// cross-stream work. A warp owns one row; reductions/accumulators are f32.
// This deliberately simple SIMT candidate is NOT a tensor-core FlashAttention
// implementation. Adoption requires measured forward+backward go/no-go.
__device__ float warp_sum(float x) {
    for (int n = 16; n; n >>= 1) x += __shfl_down_sync(0xffffffff, x, n);
    return __shfl_sync(0xffffffff, x, 0);
}
extern "C" __global__ void attention_forward(
    const float* q, const float* k, const float* v, float* out,
    int rows, int hq, int hk, int sq, int sk, int d, float scale) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int batch = row / (hq * sq), head = (row / sq) % hq;
    int kb = (batch * hk + head / (hq / hk)) * sk * d;
    float acc[8] = {}, m = -__int_as_float(0x7f800000), z = 0;
    for (int j = 0; j < sk; ++j) {
        float score = 0;
        for (int t = lane; t < d; t += 32) score += q[row*d+t] * k[kb+j*d+t];
        score = warp_sum(score) * scale;
        float next = fmaxf(m, score), a = expf(m-next), p = expf(score-next);
        for (int t = lane; t < d; t += 32) acc[t/32] = a*acc[t/32] + p*v[kb+j*d+t];
        z = a*z + p; m = next;
    }
    for (int t = lane; t < d; t += 32) out[row*(d+1)+t] = acc[t/32]/z;
    if (lane == 0) out[row*(d+1)+d] = m + logf(z);
}
extern "C" __global__ void attention_dq(
    const float* q, const float* k, const float* v, const float* out,
    const float* grad, float* dst,
    int rows, int hq, int hk, int sq, int sk, int d, float scale) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int batch = row/(hq*sq), head = (row/sq)%hq;
    int kb = (batch*hk + head/(hq/hk))*sk*d;
    float delta = 0, acc[8] = {};
    for (int t = lane; t < d; t += 32) delta += out[row*(d+1)+t]*grad[row*(d+1)+t];
    delta = warp_sum(delta);
    float lse = out[row*(d+1)+d];
    for (int j = 0; j < sk; ++j) {
        float score = 0, dp = 0;
        for (int t = lane; t < d; t += 32) {
            score += q[row*d+t]*k[kb+j*d+t];
            dp += grad[row*(d+1)+t]*v[kb+j*d+t];
        }
        float p = expf(warp_sum(score)*scale-lse);
        float ds = p*(warp_sum(dp)-delta)*scale;
        for (int t = lane; t < d; t += 32) acc[t/32] += ds*k[kb+j*d+t];
    }
    for (int t = lane; t < d; t += 32) dst[row*d+t] = acc[t/32];
}
extern "C" __global__ void attention_dkv(
    const float* q, const float* k, const float* v, const float* out,
    const float* grad, float* dst,
    int rows, int hq, int hk, int sq, int sk, int d, float scale) {
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
            for (int t = lane; t < d; t += 32) {
                score += q[qr*d+t]*k[row*d+t];
                dp += grad[qr*(d+1)+t]*v[row*d+t];
                delta += grad[qr*(d+1)+t]*out[qr*(d+1)+t];
            }
            float p = expf(warp_sum(score)*scale-out[qr*(d+1)+d]);
            float ds = p*(warp_sum(dp)-warp_sum(delta))*scale;
            for (int t = lane; t < d; t += 32) {
                dk[t/32] += ds*q[qr*d+t];
                dv[t/32] += p*grad[qr*(d+1)+t];
            }
        }
    }
    for (int t = lane; t < d; t += 32) {
        dst[qrows*d+row*d+t] = dk[t/32];
        dst[(qrows+rows)*d+row*d+t] = dv[t/32];
    }
}
