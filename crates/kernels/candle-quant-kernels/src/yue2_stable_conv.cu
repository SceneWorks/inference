// YuE2 VAE BF16 convolution leaves. One output owns its complete reduction, so neither the
// reduction order nor its CUDA launch geometry depends on the length of the surrounding song.
// Inputs, folded weights, and outputs remain BF16; only the documented kernel accumulator is F32.
// The explicit fma prevents nvrtc from choosing a shape-dependent GEMM reduction.
typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

__device__ __forceinline__ bf16_t f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (bf16_t)((bits >> 16) | 0x0040u);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    return (bf16_t)((bits + 0x7fffu + lsb) >> 16);
}

extern "C" __global__ void yue2_conv1d_bf16(
    const bf16_t* input, const bf16_t* weight, bf16_t* output,
    int batch, int in_channels, int out_channels, int in_length, int out_length,
    int kernel, int stride, int padding, int dilation) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)batch * out_channels * out_length;
    if (index >= count) return;
    int t = (int)(index % out_length);
    int co = (int)((index / out_length) % out_channels);
    int b = (int)(index / ((unsigned long long)out_length * out_channels));
    float sum = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int k = 0; k < kernel; ++k) {
            long long xi = (long long)t * stride + (long long)k * dilation - padding;
            if (xi >= 0 && xi < in_length) {
                unsigned long long x_index = ((unsigned long long)b * in_channels + ci) * in_length + xi;
                unsigned long long w_index = ((unsigned long long)co * in_channels + ci) * kernel + k;
                sum = __fmaf_rn(bf16_to_f32(input[x_index]), bf16_to_f32(weight[w_index]), sum);
            }
        }
    }
    output[index] = f32_to_bf16(sum);
}

extern "C" __global__ void yue2_conv_transpose1d_bf16(
    const bf16_t* input, const bf16_t* weight, bf16_t* output,
    int batch, int in_channels, int out_channels, int in_length, int out_length,
    int kernel, int stride) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)batch * out_channels * out_length;
    if (index >= count) return;
    int t = (int)(index % out_length);
    int co = (int)((index / out_length) % out_channels);
    int b = (int)(index / ((unsigned long long)out_length * out_channels));
    float sum = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        // Only taps congruent to this output position can contribute. Their ascending order is
        // fixed for every full/tiled length, while avoiding stride-1 zero-insertion work.
        for (long long k = t % stride; k < kernel; k += stride) {
            long long source = (long long)t - k;
            if (source >= 0) {
                long long xi = source / stride;
                if (xi < in_length) {
                    unsigned long long x_index = ((unsigned long long)b * in_channels + ci) * in_length + xi;
                    unsigned long long w_index = ((unsigned long long)ci * out_channels + co) * kernel + k;
                    sum = __fmaf_rn(bf16_to_f32(input[x_index]), bf16_to_f32(weight[w_index]), sum);
                }
            }
        }
    }
    output[index] = f32_to_bf16(sum);
}
