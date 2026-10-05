//! Original 1000-step style donor protocol, frozen before the edit diagnostic.

pub const ORIGINAL_STYLE_PROMPT: &str = "zxq style, a lighthouse on a rocky coast at dusk";
pub const DONOR_NAME: &str = "mlx_t2i_1000_steps";
pub const DONOR_SHA256: &str = "165ef06aa4374084f5efec26756e1f01c1e41a8c0cec6f5057892d41e33ee8e0";
pub const DONOR_BYTES: u64 = 167_849_666;
pub const TRAINING_SOURCE: &str = "6cca130b55939e1223261a82b9eeaea876a9ba6b";
pub const TRAINING_RUN: u64 = 37127726908;
pub const PALETTE: [[u8; 3]; 3] = [[0, 150, 150], [240, 120, 20], [250, 220, 60]];

pub fn uses_original_style_request(adapter: &str, mode: &str) -> bool {
    adapter == DONOR_NAME && mode == "t2i"
}

/// Identical continuous nearest-training-color metric used by the original run.
pub fn palette_distance(pixels: &[u8]) -> f64 {
    let px = pixels.chunks_exact(3);
    let n = px.len().max(1) as f64;
    px.map(|px| {
        PALETTE
            .iter()
            .map(|c| {
                px.iter()
                    .zip(c)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    })
    .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_completed_donor_identity_is_retained() {
        assert_eq!(DONOR_NAME, "mlx_t2i_1000_steps");
        assert_eq!(
            DONOR_SHA256,
            "165ef06aa4374084f5efec26756e1f01c1e41a8c0cec6f5057892d41e33ee8e0"
        );
        assert_eq!(DONOR_BYTES, 167_849_666);
        assert_eq!(TRAINING_SOURCE, "6cca130b55939e1223261a82b9eeaea876a9ba6b");
        assert_eq!(TRAINING_RUN, 37127726908);
    }

    #[test]
    fn only_retained_style_t2i_uses_the_original_request() {
        for mode in ["t2i", "two_reference_edit"] {
            for donor in [
                DONOR_NAME,
                "cuda_lora",
                "cuda_lokr",
                "mlx_corrected_edit_lokr",
            ] {
                assert_eq!(
                    uses_original_style_request(donor, mode),
                    donor == "mlx_t2i_1000_steps" && mode == "t2i"
                );
            }
        }
        assert_eq!(
            ORIGINAL_STYLE_PROMPT,
            "zxq style, a lighthouse on a rocky coast at dusk"
        );
        assert!(!ORIGINAL_STYLE_PROMPT.contains("zxq edit"));
        assert!(!ORIGINAL_STYLE_PROMPT.contains("zxq invert"));
    }

    #[test]
    fn targets_remain_the_three_original_training_colors() {
        assert_eq!(PALETTE, [[0, 150, 150], [240, 120, 20], [250, 220, 60]]);
        for color in PALETTE {
            assert_eq!(palette_distance(&color), 0.0);
        }
        // An edit-palette color is not a style target.
        assert!(palette_distance(&[85, 170, 255]) > 1.0);
    }

    #[test]
    fn direction_discriminates_learning_from_movement_noop_and_reversal() {
        let base = [0, 140, 140];
        let toward = [0, 150, 150];
        let away = [0, 130, 130];
        let gain = |adapted: &[u8]| palette_distance(&base) - palette_distance(adapted);
        assert!((palette_distance(&base) - 200.0_f64.sqrt()).abs() < 1e-12);
        assert!(gain(&toward) >= 1.0);
        assert_eq!(gain(&base), 0.0);
        assert!(gain(&away) < 0.0);
        // Both equally large movements have opposite learned-direction signs.
        assert_eq!(
            base.iter()
                .zip(toward)
                .map(|(&a, b)| (i16::from(a) - i16::from(b)).abs())
                .sum::<i16>(),
            20
        );
        assert_eq!(
            base.iter()
                .zip(away)
                .map(|(&a, b)| (i16::from(a) - i16::from(b)).abs())
                .sum::<i16>(),
            20
        );
    }
}
