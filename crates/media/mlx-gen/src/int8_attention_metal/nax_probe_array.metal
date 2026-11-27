    // SW-6141 diagnostic probe: the INT8 x INT8 -> INT32 `matmul2d` of `nax_probe.metal` over
    // plain thread arrays (no 8-wide INT8 vectors). Inputs: pa int8 [256], pb int8 [512].
    // Output: ci int32 [512] (lane-major, c0 then c1).
    const uint lane = thread_index_in_simdgroup;
    int8_t a[8];
    int8_t b0[8];
    int8_t b1[8];
    int c0[8];
    int c1[8];
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        a[i] = pa[lane * 8u + uint(i)];
        b0[i] = pb[lane * 16u + uint(i)];
        b1[i] = pb[lane * 16u + 8u + uint(i)];
        c0[i] = 0;
        c1[i] = 0;
    }
    sw6141_nax_mma_arr<true, int8_t, int8_t, int>(c0, c1, a, b0, b1);
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        ci[lane * 16u + uint(i)] = c0[i];
        ci[lane * 16u + 8u + uint(i)] = c1[i];
    }
