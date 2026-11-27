    // SW-6141 quantizer: rows of D = 128 channels (four per lane of a SIMD group), QB rows per
    // threadgroup sharing one INT8 scale, min(QB, 8) SIMD groups per threadgroup each owning rows
    // sg, sg + SG, ... of the block. Row t of head bh is
    //   y = H_128 · diag(sign) · (x[t] - shift[bh])   (when ROTATE; otherwise x[t] - shift[bh])
    // (`sw6141_rotate`), then
    //   scale = max(amax(block) / 127, 1e-12),  code = clamp(rint(y / scale), -127, 127).
    // The per-block scale is written to every row of the block (the attention kernels read one
    // scale per row). Inputs: x [B*H, S, 128] (bf16/f16), shift [B*H, 128] f32 (K's channel mean,
    // or zeros), params {S}.
    // Outputs: codes [B*H, S, 128] int8, scales [B*H, S] f32. Template: QB (1..32), ROTATE.
    constexpr uint D = 128u;
    constexpr uint SG = uint(QB) < 8u ? uint(QB) : 8u;
    constexpr uint PER = (uint(QB) + SG - 1u) / SG;
    threadgroup float block_peak[8];
    using T = metal::remove_cv_t<typename sw6141_elem<decltype(x)>::type>;
    const uint S = uint(params[0]);
    const uint bh = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint first = threadgroup_position_in_grid.x * uint(QB);
    const uint c0 = lane * 4u;
    float4 y[PER];
    float peak = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < PER; ++i) {
        const uint local = sg + i * SG;
        const uint t = first + local;
        float4 r = float4(0.0f);
        if (local < uint(QB) && t < S) {
            const device T* src = x + (bh * S + t) * D + c0;
            const device float* mu = shift + bh * D + c0;
            r = float4(float(src[0]) - mu[0], float(src[1]) - mu[1], float(src[2]) - mu[2],
                       float(src[3]) - mu[3]);
        }
        // Uniform across the SIMD group (it depends on sg and i only); a dead row rotates zeros.
        if (ROTATE) r = sw6141_rotate(r, lane);
        y[i] = r;
        peak = max(peak, max(max(abs(r.x), abs(r.y)), max(abs(r.z), abs(r.w))));
    }
    peak = simd_max(peak);
    if (lane == 0u) block_peak[sg] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = block_peak[0];
    for (uint i = 1u; i < SG; ++i) amax = max(amax, block_peak[i]);
    const float scale = max(amax * (1.0f / 127.0f), 1.0e-12f);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < PER; ++i) {
        const uint local = sg + i * SG;
        const uint t = first + local;
        if (local < uint(QB) && t < S) {
            const float4 r = clamp(rint(y[i] / scale), float4(-127.0f), float4(127.0f));
            device int8_t* dst = codes + (bh * S + t) * D + c0;
            dst[0] = int8_t(r.x);
            dst[1] = int8_t(r.y);
            dst[2] = int8_t(r.z);
            dst[3] = int8_t(r.w);
            if (lane == 0u) scales[bh * S + t] = scale;
        }
    }
