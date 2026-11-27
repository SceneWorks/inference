    // SW-6141 attention on the neural accelerators: one threadgroup of eight SIMD groups owns 128
    // query rows (16 per SIMD group) of one (batch, head) and streams K/V in 32-token blocks
    // through threadgroup tiles. QUANT = 1: q/k are the INT8 codes and Q·Kᵀ is an INT8 x INT8 ->
    // INT32 `matmul2d`, dequantized with the two row scales (the Q scale carrying
    // 1/sqrt(D) · log2(e)). QUANT = 0 is the 16-bit control: q/k are the bf16/f16 operands
    // themselves with FP32 accumulation, i.e. the same kernel without quantization. The online
    // softmax is FP32 in the log2 domain and P·V runs on the accelerator with FP32 accumulation.
    // Inputs: q [B*H, SQ, 128], k [B*H, SK, 128], v [B*H, SK, 128], sq [B*H, SQ] f32,
    // sk [B*H, SK] f32 (one-element placeholders when QUANT = 0), params {SQ, SK}.
    // Output: out [B*H, SQ, 128] (v's dtype).
    constexpr int D = 128;
    constexpr int BK = 32;
    constexpr int LD = D + 8;
    constexpr int DT = D / 16;
    constexpr int KF = BK / 16;
    constexpr uint TSTEP = 256u / uint(D / 4);
    using QT = metal::remove_cv_t<typename sw6141_elem<decltype(q)>::type>;
    using VT = metal::remove_cv_t<typename sw6141_elem<decltype(v)>::type>;
    using AccT = typename sw6141_select<(QUANT != 0), int, float>::type;
    threadgroup float4 k_store[BK * LD * sizeof(QT) / 16];
    threadgroup float4 v_store[BK * LD * sizeof(VT) / 16];
    threadgroup QT* k_tile = (threadgroup QT*)k_store;
    threadgroup VT* v_tile = (threadgroup VT*)v_store;

    const uint SQ = uint(params[0]);
    const uint SK = uint(params[1]);
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint bh = threadgroup_position_in_grid.y;
    const short2 coord = sw6141_nax_coord(lane);
    const uint c_quad = (tid % uint(D / 4)) * 4u;
    const uint t_row = tid / uint(D / 4);
    const float scale2 = rsqrt(float(D)) * M_LOG2E_F;
    uint row[2];
    bool live[2];
    float rscale[2];
    const device QT* q_row[2];
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        row[i] = threadgroup_position_in_grid.x * 128u + sg * 16u + uint(coord.y) + 8u * uint(i);
        live[i] = row[i] < SQ;
        const uint r = live[i] ? row[i] : SQ - 1u;
        q_row[i] = q + (bh * SQ + r) * uint(D) + uint(coord.x);
        rscale[i] = QUANT != 0 ? sq[bh * SQ + r] * scale2 : scale2;
    }
    vec<float, 8> o[DT];
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) o[d] = vec<float, 8>(0.0f);
    float run_max[2] = {-INFINITY, -INFINITY};
    float run_sum[2] = {0.0f, 0.0f};

    for (uint t0 = 0u; t0 < SK; t0 += uint(BK)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint r = 0u; r < uint(BK) / TSTEP; ++r) {
            const uint tt = t_row + r * TSTEP;
            vec<QT, 4> kq = vec<QT, 4>(0);
            vec<VT, 4> vq = vec<VT, 4>(0);
            if (t0 + tt < SK) {
                const uint at = (bh * SK + t0 + tt) * uint(D) + c_quad;
                kq = *(const device vec<QT, 4>*)(k + at);
                vq = *(const device vec<VT, 4>*)(v + at);
            }
            *(threadgroup vec<QT, 4>*)(k_tile + tt * uint(LD) + c_quad) = kq;
            *(threadgroup vec<VT, 4>*)(v_tile + tt * uint(LD) + c_quad) = vq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        vec<AccT, 8> acc[KF];
#pragma clang loop unroll(full)
        for (int f = 0; f < KF; ++f) acc[f] = vec<AccT, 8>(0);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
            // Query fragments are re-read per block (cache-resident), as MLX's NAX attention does.
            const vec<QT, 8> qd = vec<QT, 8>(*(const device vec<QT, 4>*)(q_row[0] + d * 16),
                                             *(const device vec<QT, 4>*)(q_row[1] + d * 16));
            const vec<QT, 8> k0 = sw6141_nax_frag<QT, LD>(k_tile, coord, 0u, uint(d * 16));
            const vec<QT, 8> k1 = sw6141_nax_frag<QT, LD>(k_tile, coord, 16u, uint(d * 16));
            sw6141_nax_mma<true>(acc[0], acc[1], qd, k0, k1);
        }
        vec<float, 8> s[KF];
        float block_max[2] = {-INFINITY, -INFINITY};
#pragma clang loop unroll(full)
        for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
            for (short a = 0; a < 2; ++a) {
#pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    const uint t = t0 + uint(f * 16 + coord.x + j);
                    float x = -INFINITY;
                    if (t < SK) {
                        const float cs = QUANT != 0 ? sk[bh * SK + t] : 1.0f;
                        x = float(acc[f][4 * a + j]) * rscale[a] * cs;
                    }
                    s[f][4 * a + j] = x;
                    block_max[a] = max(block_max[a], x);
                }
            }
        }
        float factor[2];
#pragma clang loop unroll(full)
        for (short a = 0; a < 2; ++a) {
            block_max[a] = max(block_max[a], simd_shuffle_xor(block_max[a], 1));
            block_max[a] = max(block_max[a], simd_shuffle_xor(block_max[a], 8));
            const float next_max = max(run_max[a], block_max[a]);
            factor[a] = run_max[a] == -INFINITY ? 0.0f : fast::exp2(run_max[a] - next_max);
            float block_sum = 0.0f;
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    const float x = s[f][4 * a + j];
                    const float p = x == -INFINITY ? 0.0f : fast::exp2(x - next_max);
                    s[f][4 * a + j] = p;
                    block_sum += p;
                }
            }
            block_sum += simd_shuffle_xor(block_sum, 1);
            block_sum += simd_shuffle_xor(block_sum, 8);
            run_sum[a] = run_sum[a] * factor[a] + block_sum;
            run_max[a] = next_max;
        }
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                o[d][j] *= factor[0];
                o[d][4 + j] *= factor[1];
            }
        }
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; d += 2) {
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; ++f) {
                const vec<VT, 8> v0 = sw6141_nax_frag<VT, LD>(v_tile, coord, uint(f * 16), uint(d * 16));
                const vec<VT, 8> v1 = sw6141_nax_frag<VT, LD>(v_tile, coord, uint(f * 16), uint(d * 16 + 16));
                sw6141_nax_mma<false>(o[d], o[d + 1], s[f], v0, v1);
            }
        }
    }
#pragma clang loop unroll(full)
    for (short a = 0; a < 2; ++a) {
        if (!live[a]) continue;
        const uint base = (bh * SQ + row[a]) * uint(D);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                const float x = run_sum[a] > 0.0f ? o[d][4 * a + j] / run_sum[a] : 0.0f;
                out[base + uint(d * 16 + coord.x + j)] = static_cast<VT>(x);
            }
        }
    }
