    // SW-6141 INT8 QK attention on the SIMD-group matrix units (every Apple GPU): one threadgroup
    // of four SIMD groups owns 32 query rows (8 per SIMD group) of one (batch, head) and streams
    // K/V in 16-token blocks through FP32 threadgroup tiles. Q·Kᵀ multiplies the INT8 codes as
    // exact FP32 integers (|sum| <= 128 * 127^2 < 2^24, so every partial sum is exact in any order),
    // then dequantizes with the two row scales (the Q scale carrying 1/sqrt(D) · log2(e)); the
    // online softmax is FP32 in the log2 domain and P·V accumulates in FP32.
    // Inputs: q [B*H, SQ, 128] int8, k [B*H, SK, 128] int8, v [B*H, SK, 128] (bf16/f16),
    // sq [B*H, SQ] f32, sk [B*H, SK] f32, params {SQ, SK}. Output: out [B*H, SQ, 128] (v's dtype).
    constexpr int D = 128;
    constexpr int BK = 16;
    constexpr int DT = D / 8;
    constexpr int KT8 = BK / 8;
    constexpr uint LD = D + 4;
    constexpr uint QUADS = BK * D / 4;
    using T = metal::remove_cv_t<typename sw6141_elem<decltype(out)>::type>;
    threadgroup float4 k_store[BK * LD / 4];
    threadgroup float4 v_store[BK * LD / 4];
    threadgroup float* k_tile = (threadgroup float*)k_store;
    threadgroup float* v_tile = (threadgroup float*)v_store;

    const uint SQ = uint(params[0]);
    const uint SK = uint(params[1]);
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint bh = threadgroup_position_in_grid.y;
    const ushort2 frag = sw6141_frag(lane);
    const uint row = threadgroup_position_in_grid.x * 32u + sg * 8u + uint(frag.x);
    const bool live = row < SQ;
    const uint qrow = live ? row : SQ - 1u;
    const float rscale = sq[bh * SQ + qrow] * (rsqrt(float(D)) * M_LOG2E_F);

    simdgroup_float8x8 qm[DT];
    simdgroup_float8x8 om[DT];
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
        const uint at = (bh * SQ + qrow) * uint(D) + uint(d * 8) + uint(frag.y);
        qm[d].thread_elements()[0] = float(q[at]);
        qm[d].thread_elements()[1] = float(q[at + 1u]);
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    for (uint t0 = 0u; t0 < SK; t0 += uint(BK)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < QUADS; idx += 128u) {
            const uint tt = idx / uint(D / 4);
            const uint c0 = (idx % uint(D / 4)) * 4u;
            float4 kq = float4(0.0f);
            float4 vq = float4(0.0f);
            if (t0 + tt < SK) {
                const uint at = (bh * SK + t0 + tt) * uint(D) + c0;
                kq = float4(float(k[at]), float(k[at + 1u]), float(k[at + 2u]), float(k[at + 3u]));
                vq = float4(float(v[at]), float(v[at + 1u]), float(v[at + 2u]), float(v[at + 3u]));
            }
            *(threadgroup float4*)(k_tile + tt * LD + c0) = kq;
            *(threadgroup float4*)(v_tile + tt * LD + c0) = vq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sm[KT8];
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) sm[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) {
                simdgroup_float8x8 km;
                simdgroup_load(km, k_tile + j * 8 * LD + d * 8, LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
            }
        }
        float block_max = -INFINITY;
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const uint t = t0 + uint(j * 8) + uint(frag.y) + uint(e);
                const float s = t < SK
                    ? sm[j].thread_elements()[e] * rscale * sk[bh * SK + t]
                    : -INFINITY;
                sm[j].thread_elements()[e] = s;
                block_max = max(block_max, s);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float next_max = max(run_max, block_max);
        const float factor = run_max == -INFINITY ? 0.0f : fast::exp2(run_max - next_max);
        float block_sum = 0.0f;
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const float s = sm[j].thread_elements()[e];
                const float p = s == -INFINITY ? 0.0f : fast::exp2(s - next_max);
                sm[j].thread_elements()[e] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * factor + block_sum;
        run_max = next_max;
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
            om[d].thread_elements()[0] *= factor;
            om[d].thread_elements()[1] *= factor;
        }
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; ++d) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, v_tile + j * 8 * LD + d * 8, LD);
                simdgroup_multiply_accumulate(om[d], sm[j], vm, om[d]);
            }
        }
    }
    if (live) {
        const uint base = (bh * SQ + row) * uint(D) + uint(frag.y);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const float o = run_sum > 0.0f ? om[d].thread_elements()[e] / run_sum : 0.0f;
                out[base + uint(d * 8 + e)] = static_cast<T>(o);
            }
        }
    }
