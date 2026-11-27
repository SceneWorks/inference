//! # gen-core-family
//!
//! Small tensor-free per-family host policy split out of gen-core in sc-27316, each module used by
//! one model family's MLX and Candle providers: schedulers (Ideogram, Krea multiphase/schedule),
//! resolution and geometry (Lens, SeedVR2 video, SDXL LDM), keypoint and face-alignment math
//! (InstantID, face/FX), the Qwen-Image 2.1 config/reference/scheduler, and the SD3 text-encoder
//! artifact resolution and seeded-output mapping. Grouped so a change to one of them rebuilds
//! only these families, never gen-core.

pub mod bernini_plan;
pub mod face_align;
pub mod fx_program;
pub mod ideogram_scheduler;
pub mod instantid_kps;
pub mod krea_multiphase;
pub mod krea_schedule;
pub mod lens_resolution;
pub mod mage_plan;
pub mod mochi;
pub mod qwen_image_2_1;
pub mod scail2;
pub mod sd3_encoder_artifacts;
pub mod sd3_request;
pub mod sdxl_ldm;
pub mod seedvr2_video;
pub mod svd;

pub use sd3_encoder_artifacts::{
    resolve_sd3_text_encoder_artifacts, Sd3TextEncoderArtifactError, Sd3TextEncoderArtifacts,
};
pub use sd3_request::map_sd3_seeded_outputs;

/// Eager Candle nearest-even RGB8 round, including its full f32 scaled input. At
/// `where_cond` the scaled input, floor, parity, both f32 branches and result coexist
/// with the u8 tie mask. The earlier chained parity expression has the same peak.
/// `retained_bytes` is distinct, still-live caller storage on this same device.
pub fn candle_round_rgb8_live_bytes(
    rgb_elements: u64,
    retained_bytes: u64,
) -> gen_core::Result<u64> {
    let per_element = (6 * std::mem::size_of::<f32>() + std::mem::size_of::<u8>()) as u64;
    rgb_elements
        .checked_mul(per_element)
        .and_then(|n| n.checked_add(retained_bytes))
        .ok_or_else(|| {
            gen_core::Error::Unsupported("Candle RGB8 rounding footprint overflows".into())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candle_round_rgb8_prices_eager_branches_and_external_storage_once() {
        assert_eq!(candle_round_rgb8_live_bytes(30, 120).unwrap(), 870);
        assert!(candle_round_rgb8_live_bytes(u64::MAX, 0).is_err());
        assert!(candle_round_rgb8_live_bytes(1, u64::MAX - 24).is_err());
    }
}
