    // SW-6141 quantizer: one SIMD group per token row of D = 128 channels (four per lane), QB rows
    // per threadgroup sharing one INT8 scale. Row t of head bh is
    //   y = H_128 · diag(sign) · (x[t] - shift[bh])   (when ROTATE; otherwise x[t] - shift[bh])
    // with H_128 the orthonormal Sylvester Walsh-Hadamard matrix, then
    //   scale = max(amax(block) / 127, 1e-12),  code = clamp(rint(y / scale), -127, 127).
    // The per-block scale is written to every row of the block (the attention kernels read one
    // scale per row). Inputs: x [B*H, S, 128] (bf16/f16), shift [B*H, 128] f32 (K's channel mean,
    // or zeros), params {S}.
    // Outputs: codes [B*H, S, 128] int8, scales [B*H, S] f32. Template: QB (1..32), ROTATE.
    constexpr uint D = 128u;
    threadgroup float block_peak[32];
    using T = metal::remove_cv_t<typename sw6141_elem<decltype(x)>::type>;
    const uint S = uint(params[0]);
    const uint bh = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x * uint(QB) + sg;
    const bool live = t < S;
    const uint c0 = lane * 4u;
    float4 y = float4(0.0f);
    if (live) {
        const device T* src = x + (bh * S + t) * D + c0;
        const device float* mu = shift + bh * D + c0;
        y = float4(float(src[0]) - mu[0], float(src[1]) - mu[1], float(src[2]) - mu[2],
                   float(src[3]) - mu[3]);
    }
    if (ROTATE) {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 4u; ++e) {
            const uint c = c0 + e;
            if (((SW6141_SIGN[c >> 5] >> (c & 31u)) & 1u) == 0u) y[e] = -y[e];
        }
        // Butterfly strides 1 and 2 inside the lane, then 4..64 across lanes (stride 4m = lane
        // bit m): the element whose index has the stride bit clear keeps the sum, the other the
        // difference partner - self.
        const float a0 = y.x + y.y;
        const float a1 = y.x - y.y;
        const float a2 = y.z + y.w;
        const float a3 = y.z - y.w;
        y = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
#pragma clang loop unroll(full)
        for (uint m = 1u; m < 32u; m <<= 1) {
            const float4 p = simd_shuffle_xor(y, ushort(m));
            y = (lane & m) != 0u ? p - y : y + p;
        }
        y *= 0.08838834764831845f;  // 1 / sqrt(128)
    }
    float peak = max(max(abs(y.x), abs(y.y)), max(abs(y.z), abs(y.w)));
    peak = simd_max(peak);
    if (lane == 0u) block_peak[sg] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = block_peak[0];
    for (uint i = 1u; i < uint(QB); ++i) amax = max(amax, block_peak[i]);
    const float scale = max(amax * (1.0f / 127.0f), 1.0e-12f);
    if (live) {
        const float4 r = clamp(rint(y / scale), float4(-127.0f), float4(127.0f));
        device int8_t* dst = codes + (bh * S + t) * D + c0;
        dst[0] = int8_t(r.x);
        dst[1] = int8_t(r.y);
        dst[2] = int8_t(r.z);
        dst[3] = int8_t(r.w);
        if (lane == 0u) scales[bh * S + t] = scale;
    }
