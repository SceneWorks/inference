//! Teacher-forced greedy-step and exact-tie helpers for LLM-style parity tests.

/// Check one teacher-forced greedy step without turning an exact or narrow logit tie into a
/// backend-specific hard token id. Returns whether this was a wide, exact-argmax check.
/// The caller supplies its existing logit margin and minimum-wide-check policy; this helper
/// introduces no new tolerance and has no tensor/backend dependency.
pub fn assert_teacher_forced_logit_step(
    label: &str,
    expected_id: usize,
    reference: &[f32],
    actual: &[f32],
    margin: f32,
) -> bool {
    assert_eq!(reference.len(), actual.len(), "{label}: vocabulary");
    assert!(reference.len() >= 2 && expected_id < reference.len());
    assert!(
        reference.iter().chain(actual).all(|v| v.is_finite()),
        "{label}: finite scores"
    );
    let argmax = |values: &[f32]| {
        values.iter().enumerate().fold(
            0,
            |best, (i, value)| {
                if *value > values[best] {
                    i
                } else {
                    best
                }
            },
        )
    };
    let reference_max = reference[argmax(reference)];
    assert_eq!(
        reference[expected_id], reference_max,
        "{label}: reference greedy choice"
    );
    let second = reference
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != expected_id)
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let actual_id = argmax(actual);
    if reference_max - second >= margin {
        assert_eq!(actual_id, expected_id, "{label}: wide-margin argmax");
        true
    } else {
        assert!(
            reference_max - reference[actual_id] <= margin,
            "{label}: actual candidate is outside the reference narrow margin"
        );
        assert!(
            actual[actual_id] - actual[expected_id] <= margin,
            "{label}: reference candidate is outside the native narrow margin"
        );
        false
    }
}

/// Validate a public greedy token against the complete exact-tie set, not a CPU argmax tie-break.
/// Autoregressive comparison ends here; a teacher-forced gate must verify the remaining suffix.
pub fn assert_exact_greedy_tie(reference: &[f32], actual_id: usize) {
    assert!(actual_id < reference.len());
    assert!(reference.iter().all(|value| value.is_finite()));
    let top = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(reference.iter().filter(|&&value| value == top).count() >= 2);
    assert_eq!(
        reference[actual_id], top,
        "public greedy token must belong to the exact tie"
    );
}
