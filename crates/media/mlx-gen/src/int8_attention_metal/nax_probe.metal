    // SW-6141 fragment-layout probe: one SIMD group runs the same per-lane element values through
    // the INT8 x INT8 -> INT32 product and the proven 16-bit x 16-bit -> FP32 product. Equal
    // outputs for every lane and element mean the INT8 cooperative tensors use the 16-bit
    // `BaseNAXFrag` layout the attention kernel assumes. Inputs: pa int8 [256], pb int8 [512],
    // ha / hb the same values as bf16. Outputs: ci int32 [512], cf f32 [512] (lane-major, c0 then c1).
    using T16 = metal::remove_cv_t<typename sw6141_elem<decltype(ha)>::type>;
    const uint lane = thread_index_in_simdgroup;
    vec<int8_t, 8> a;
    vec<int8_t, 8> b0;
    vec<int8_t, 8> b1;
    vec<T16, 8> fa;
    vec<T16, 8> fb0;
    vec<T16, 8> fb1;
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        a[i] = pa[lane * 8u + uint(i)];
        b0[i] = pb[lane * 16u + uint(i)];
        b1[i] = pb[lane * 16u + 8u + uint(i)];
        fa[i] = ha[lane * 8u + uint(i)];
        fb0[i] = hb[lane * 16u + uint(i)];
        fb1[i] = hb[lane * 16u + 8u + uint(i)];
    }
    vec<float, 8> f0 = vec<float, 8>(0.0f);
    vec<float, 8> f1 = vec<float, 8>(0.0f);
    sw6141_nax_mma<true>(f0, f1, fa, fb0, fb1);
    vec<int, 8> c0 = vec<int, 8>(0);
    vec<int, 8> c1 = vec<int, 8>(0);
    sw6141_nax_mma<true>(c0, c1, a, b0, b1);
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        cf[lane * 16u + uint(i)] = f0[i];
        cf[lane * 16u + 8u + uint(i)] = f1[i];
        ci[lane * 16u + uint(i)] = c0[i];
        ci[lane * 16u + 8u + uint(i)] = c1[i];
    }
