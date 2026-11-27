
// SW-6141 diagnostic: the same INT8 `matmul2d` product as `sw6141_nax_mma`, over plain thread
// arrays instead of 8-wide vectors. Only the array probe kernel includes it, so it tells "INT8
// cooperative tensors are unsupported" apart from "8-wide INT8 vectors are unsupported".
template <bool TB, typename AT, typename BT, typename CT>
inline __attribute__((always_inline)) void sw6141_nax_mma_arr(thread CT* c0, thread CT* c1,
                            thread const AT* a, thread const BT* b0, thread const BT* b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), CT>();
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = a[i];
        ct_b[i] = b0[i];
        ct_b[8 + i] = b1[i];
        ct_c[i] = c0[i];
        ct_c[8 + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[8 + i];
    }
}
