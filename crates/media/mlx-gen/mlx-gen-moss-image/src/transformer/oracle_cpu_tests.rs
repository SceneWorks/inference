#[test]
fn rejects_nonfinite_truncated_and_wrong_shape_oracles_sc26325() {
    use super::checked_delta;
    assert_eq!(checked_delta(&[1., 2.], &[2], &[1., 2.], &[2]), 0.);
    for (a, shape, b, expected_shape) in [
        (vec![f32::NAN], vec![1], vec![0.], vec![1]),
        (vec![0.], vec![1], vec![f32::NAN], vec![1]),
        (vec![f32::INFINITY], vec![1], vec![0.], vec![1]),
        (vec![1.], vec![1], vec![1., 2.], vec![2]),
        (vec![1., 2.], vec![2], vec![1.], vec![2]),
        (vec![1., 2.], vec![1, 2], vec![1., 2.], vec![2, 1]),
    ] {
        assert!(
            std::panic::catch_unwind(|| checked_delta(&a, &shape, &b, &expected_shape)).is_err()
        );
    }
}
