//! Combine reset-separated high-water marks without dropping the first training step.
pub fn aggregate_training_peaks(first_overlap: (u64, u64), remaining: (u64, u64)) -> (u64, u64) {
    (
        first_overlap.0.max(remaining.0),
        first_overlap.1.max(remaining.1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_keeps_the_peak_before_reset() {
        assert_eq!(
            aggregate_training_peaks((900, 1000), (100, 200)),
            (900, 1000)
        );
    }

    #[test]
    fn every_counter_keeps_its_own_high_water_not_a_sum() {
        assert_eq!(
            aggregate_training_peaks((900, 500), (100, 1000)),
            (900, 1000)
        );
        assert_eq!(
            aggregate_training_peaks((100, 1000), (900, 500)),
            (900, 1000)
        );
    }
}
