use super::*;
#[test]
fn fraction_uses_selected_total_not_free_or_safety_and_intersects_bytes() {
    assert_eq!(TrainingConfig::default().memory_limit_fraction, None);
    let resolve = effective_memory_budget_with_fraction;
    assert_eq!(resolve(800, Some(1000), None, Some(0.5)).unwrap(), 500);
    assert_eq!(resolve(600, Some(1000), None, Some(0.5)).unwrap(), 500);
    assert_eq!(resolve(300, Some(1000), None, Some(0.5)).unwrap(), 300);
    assert_eq!(resolve(800, Some(2000), None, Some(0.5)).unwrap(), 800);
    assert_eq!(resolve(800, Some(1000), Some(173), Some(0.5)).unwrap(), 173);
    assert_eq!(resolve(800, Some(1000), Some(900), Some(0.5)).unwrap(), 500);
    assert_eq!(resolve(800, None, None, None).unwrap(), 800);
    assert_eq!(resolve(800, None, Some(173), None).unwrap(), 173);
    assert_eq!(resolve(800, Some(1000), None, Some(1.0)).unwrap(), 800);
    assert!(resolve(800, None, None, Some(0.5)).is_err());
    assert!(resolve(800, Some(0), None, Some(0.5)).is_err());
}
#[test]
fn fraction_invalid_values_and_sub_byte_caps_refuse_without_fallback() {
    for fraction in [
        0.0,
        -0.0,
        -0.5,
        1.01,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(matches!(
            effective_memory_budget_with_fraction(800, Some(1000), None, Some(fraction)),
            Err(crate::Error::Unsupported(_))
        ));
    }
    assert!(effective_memory_budget_with_fraction(800, Some(1000), Some(0), None).is_err());
    assert!(effective_memory_budget_with_fraction(0, Some(1000), None, None).is_err());
    assert!(
        effective_memory_budget_with_fraction(800, Some(1000), None, Some(f32::from_bits(1)))
            .is_err()
    );
}
#[test]
fn fraction_cap_rounds_down_exactly_even_at_u64_max() {
    let resolve = effective_memory_budget_with_fraction;
    assert_eq!(resolve(1000, Some(1001), None, Some(0.5)).unwrap(), 500);
    assert_eq!(
        resolve(u64::MAX, Some(u64::MAX), None, Some(1.0)).unwrap(),
        u64::MAX
    );
    assert_eq!(
        resolve(u64::MAX, Some(u64::MAX), None, Some(0.5)).unwrap(),
        u64::MAX / 2
    );
    assert_eq!(
        resolve(
            u64::MAX,
            Some(u64::MAX),
            None,
            Some(f32::from_bits(0x3f7fffff))
        )
        .unwrap(),
        u64::MAX - (1_u64 << 40)
    );
}
