//! Canonical diffusion-noise schedule parameters for the Candle SDXL family.
//!
//! These values are consumed by the SDXL and Kolors inference and training paths. Keeping them in
//! the Candle commons prevents the sampler, curated path, and DDPM trainer from silently using
//! different `scaled_linear` grids.

/// SDXL's diffusers `scaled_linear` beta schedule (re-exported from `gen_core::sampling`, `PIPE-9`).
pub use gen_core::sampling::{
    KOLORS_BETA_END, KOLORS_BETA_START, KOLORS_TRAIN_STEPS, SDXL_BETA_END, SDXL_BETA_START,
    SDXL_TRAIN_STEPS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdxl_and_kolors_schedule_constants_match_the_pinned_configs() {
        assert_eq!(SDXL_TRAIN_STEPS, 1_000);
        assert_eq!(SDXL_BETA_START, 0.00085);
        assert_eq!(SDXL_BETA_END, 0.012);
        assert_eq!(KOLORS_TRAIN_STEPS, 1_100);
        assert_eq!(KOLORS_BETA_START, 0.00085);
        assert_eq!(KOLORS_BETA_END, 0.014);
    }
}
