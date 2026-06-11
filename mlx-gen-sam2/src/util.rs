//! Small host-side helpers shared across the SAM2 segmenter / video predictor.

/// Index of the maximum value, NaN-safe via [`f32::total_cmp`] — a NaN in the IoU-head output can't
/// panic the way `partial_cmp(..).unwrap()` would. Matches [`Iterator::max_by`] semantics (ties
/// resolve to the **last** maximum, so for finite inputs this is identical to the previous
/// `max_by(partial_cmp)`), and an empty slice returns `0`. Shared by the three SAM2 best-IoU
/// multimask selections (F-169).
pub(crate) fn argmax_f32(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_f32_max_ties_empty() {
        assert_eq!(argmax_f32(&[0.1, 0.9, 0.3]), 1);
        // Ties resolve to the last maximum — same as the replaced `max_by(partial_cmp)`.
        assert_eq!(argmax_f32(&[0.5, 0.5, 0.5]), 2);
        // Empty → 0, preserving the call sites' `unwrap_or(0)`.
        assert_eq!(argmax_f32(&[]), 0);
    }

    #[test]
    fn argmax_f32_is_nan_safe() {
        // The point of F-169: a NaN must NOT panic (vs `partial_cmp(..).unwrap()`). `total_cmp`
        // gives a total order, so this returns some valid in-bounds index instead.
        let with_nan = [0.2_f32, f32::NAN, 0.8];
        let idx = argmax_f32(&with_nan);
        assert!(idx < with_nan.len());
    }
}
