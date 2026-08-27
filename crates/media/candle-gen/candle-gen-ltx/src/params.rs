//! Explicit per-checkpoint-generation pipeline parameters (sc-18759) — the candle sibling of
//! `mlx-gen-ltx::params`. Kept in lockstep with that crate's copy; see its module docs for the full
//! upstream-fallthrough rationale.
//!
//! Upstream `ltx_pipelines.utils.constants._PARAMS_SINCE_VERSION` (reference `Lightricks/LTX-2`
//! `d1511477`) resolves `PipelineParams` by walking a "newest generation at or below" table with
//! rows for 2.4 and 2.3 only. A checkpoint whose `model_version` is newer than every row —
//! including a 2.5 checkpoint — silently inherits the nearest older row (`LTX_2_4_PARAMS`: 30
//! steps, STG block 28, CFG 3.0 video / 7.0 audio, rescale 0.7, `default_image_crf: 18`).
//!
//! We do not replicate the fallthrough: every `model_version` this codebase actually loads gets
//! its own explicit row (2.3 and 2.5), and [`resolve_generation_params`] is an **exact**
//! `(major, minor)` lookup — an unrecognized version is a loud [`Error`], never a silent
//! hand-me-down from a neighboring row.
//!
//! Distilled sampling (this crate's fixed [`crate::config::STAGE1_SIGMAS`] schedule) is unchanged
//! across every row here and is **not** gated by `model_version` — see the
//! `distilled_sigma_schedule_is_shared_across_generations` test below. The guidance fields on
//! [`GuiderParams`] mirror upstream's non-distilled ("dev") pipeline knobs; this crate does not yet
//! run a CFG/STG-guided denoise loop, so those fields are pinned data today, not yet consumed by a
//! sampler. `default_image_crf` IS consumed today — see [`crate::image_crf`].

use candle_gen::candle_core::{Error, Result};

/// Multi-modal guidance knobs for one modality (video or audio) — mirrors
/// `mlx_gen_ltx::params::GuiderParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuiderParams {
    pub cfg_scale: f32,
    /// Selective token guidance strength.  The dev sampler executes this only on the
    /// STG-perturbed branch; distilled checkpoints keep the value as checkpoint metadata.
    pub stg_scale: f32,
    pub stg_blocks: &'static [u32],
    pub rescale_scale: f32,
    pub modality_scale: f32,
}

/// Resolved generation parameters for one checkpoint generation (upstream `PipelineParams`,
/// narrowed to the fields this story cares about).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LtxGenerationParams {
    pub num_inference_steps: u32,
    pub video_guider: GuiderParams,
    pub audio_guider: GuiderParams,
    /// H.264 CRF an I2V conditioning image is re-compressed at before VAE encode, matching the
    /// compression the checkpoint was trained against.
    pub default_image_crf: u8,
}

const LTX_2_3_VIDEO_GUIDER: GuiderParams = GuiderParams {
    cfg_scale: 3.0,
    stg_scale: 1.0,
    stg_blocks: &[28],
    rescale_scale: 0.7,
    modality_scale: 3.0,
};

const LTX_2_3_AUDIO_GUIDER: GuiderParams = GuiderParams {
    cfg_scale: 7.0,
    stg_scale: 1.0,
    stg_blocks: &[28],
    rescale_scale: 0.7,
    modality_scale: 3.0,
};

/// LTX-2.3's explicit row (upstream `LTX_2_3_PARAMS`): 30 steps, STG block 28 on both modalities,
/// CFG 3.0 video / 7.0 audio, rescale 0.7, modality scale 3.0, `default_image_crf: 33`.
pub const LTX_2_3_PARAMS: LtxGenerationParams = LtxGenerationParams {
    num_inference_steps: 30,
    video_guider: LTX_2_3_VIDEO_GUIDER,
    audio_guider: LTX_2_3_AUDIO_GUIDER,
    default_image_crf: 33,
};

/// LTX-2.5's explicit row (sc-18759): unchanged step count and STG block from 2.3;
/// `default_image_crf` moves to 18. See `mlx_gen_ltx::params::LTX_2_5_PARAMS` doc comment for the
/// full rationale (this is its candle-side twin, kept numerically identical).
pub const LTX_2_5_PARAMS: LtxGenerationParams = LtxGenerationParams {
    num_inference_steps: 30,
    video_guider: LTX_2_3_VIDEO_GUIDER,
    audio_guider: LTX_2_3_AUDIO_GUIDER,
    default_image_crf: 18,
};

/// Resolve the generation params for a checkpoint's declared `model_version` string. Exact
/// `(major, minor)` match only — never a "newest at or below" fallthrough. See the mlx-gen-ltx
/// twin for the full contract.
pub fn resolve_generation_params(model_version: &str) -> Result<&'static LtxGenerationParams> {
    match parse_major_minor(model_version) {
        Some((2, 3)) => Ok(&LTX_2_3_PARAMS),
        Some((2, 5)) => Ok(&LTX_2_5_PARAMS),
        _ => Err(Error::Msg(format!(
            "ltx: no explicit generation params row for model_version {model_version:?} — add \
             one rather than falling through to a neighboring generation's defaults (sc-18759)"
        ))),
    }
}

fn parse_major_minor(v: &str) -> Option<(u32, u32)> {
    let mut it = v.split(['.', '-']);
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-18759 acceptance: pins the resolved params for `model_version: 2.5.0` — fails on drift.
    #[test]
    fn ltx_2_5_params_pinned() {
        let p = resolve_generation_params("2.5.0").unwrap();
        assert_eq!(p.num_inference_steps, 30);
        assert_eq!(p.video_guider.stg_blocks, &[28]);
        assert_eq!(p.audio_guider.stg_blocks, &[28]);
        assert_eq!(p.video_guider.cfg_scale, 3.0);
        assert_eq!(p.audio_guider.cfg_scale, 7.0);
        assert_eq!(p.video_guider.stg_scale, 1.0);
        assert_eq!(p.audio_guider.stg_scale, 1.0);
        assert_eq!(p.video_guider.rescale_scale, 0.7);
        assert_eq!(p.video_guider.modality_scale, 3.0);
        assert_eq!(p.default_image_crf, 18); // NOT 33 -- the silent-fallthrough bug fixed here
    }

    /// sc-18759 acceptance: "2.3 renders keep CRF 33 and 30 steps / STG 28."
    #[test]
    fn ltx_2_3_params_unchanged() {
        let p = resolve_generation_params("2.3.0").unwrap();
        assert_eq!(p.num_inference_steps, 30);
        assert_eq!(p.video_guider.stg_blocks, &[28]);
        assert_eq!(p.default_image_crf, 33);
    }

    /// No fallthrough: an unrecognized generation errors instead of inheriting a neighboring row.
    #[test]
    fn unrecognized_version_errors_not_falls_through() {
        assert!(resolve_generation_params("2.6.0").is_err());
        assert!(resolve_generation_params("2.4.0").is_err());
        assert!(resolve_generation_params("").is_err());
    }

    /// sc-18759 acceptance: confirms the distilled schedule is not version-gated — 2.5 uses the
    /// exact same fixed [`crate::config::STAGE1_SIGMAS`] this crate already runs for 2.3 (this
    /// crate's stage-1-only distilled T2V/I2V path — see `crate::config` doc comments).
    #[test]
    fn distilled_sigma_schedule_is_shared_across_generations() {
        use crate::config::{NATIVE_STEPS, STAGE1_SIGMAS};
        assert_eq!(STAGE1_SIGMAS.len(), 9);
        assert_eq!(NATIVE_STEPS, 8, "stage-1: 8 distilled denoise steps");
    }
}
