//! The typed body-loss config of
//! [`TrainingConfig::body_losses`](super::TrainingConfig::body_losses) (epic 2123, sc-24832). The
//! model hyper-parameters, footprints and host-side geometry of the three body losses live in
//! `gen_core_train::body` (sc-27316).

use std::path::PathBuf;

use super::AuxLossSchedule;

/// Body ratios (upstream `NUM_BODY_RATIOS`).
pub const NUM_BODY_RATIOS: usize = 8;
/// Extra head ratios with [`BodyLossesConfig::include_head`] (upstream `NUM_HEAD_RATIOS`).
pub const NUM_HEAD_RATIOS: usize = 2;
/// [`TrainingConfig::body_losses`](super::TrainingConfig::body_losses) — the three body losses'
/// schedules, their knobs and the frozen checkpoints they run. Every loss is off by default.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyLossesConfig {
    /// ViTPose bone-length-ratio loss. Off by default.
    pub proportion: AuxLossSchedule,
    /// Add the two head ratios (nose→shoulders height, ear-to-ear width) to the eight body ratios
    /// (upstream `body_proportion_include_head`, default `false`).
    pub include_head: bool,
    /// HybrIK SMPL-beta loss. Off by default.
    pub shape: AuxLossSchedule,
    /// The shape loss only counts while the live betas' cosine to the reference exceeds this
    /// (upstream `body_shape_loss_min_cos`, default `0.2`), in [`Self::SHAPE_MIN_COS_RANGE`].
    pub shape_min_cos: f32,
    /// Sapiens surface-normal loss. Off by default.
    pub normal: AuxLossSchedule,
    /// Average the normal loss over the item's subject mask only (upstream
    /// `perceptual_restrict_to_body`). Needs every item's
    /// [`subject_mask_path`](super::TrainingItem::subject_mask_path); the trainer hands each
    /// reference its item's mask (`gen_core_train::subject_mask::PerceptualSubjectMasks`).
    pub normal_restrict_to_subject: bool,
    /// The ViTPose+ base checkpoint (`usyd-community/vitpose-plus-base`, `model.safetensors`).
    /// Required whenever **any** body loss is on: it is the proportion encoder and the
    /// reference-time person detector of all three.
    pub pose_model_dir: Option<PathBuf>,
    /// The HybrIK ResNet-34 checkpoint (`model.safetensors`, HybrIK key layout). Required with the
    /// shape loss on.
    pub shape_model_dir: Option<PathBuf>,
    /// The Sapiens-0.3B normal checkpoint (`model.safetensors`, Sapiens key layout). Required with
    /// the normal loss on.
    pub normal_model_dir: Option<PathBuf>,
}

impl BodyLossesConfig {
    /// Upstream `body_shape_loss_min_cos` default.
    pub const DEFAULT_SHAPE_MIN_COS: f32 = 0.2;
    /// Inclusive bounds of [`shape_min_cos`](Self::shape_min_cos) (a cosine).
    pub const SHAPE_MIN_COS_RANGE: (f32, f32) = (-1.0, 1.0);

    /// Whether any of the three losses is on.
    pub fn any_enabled(&self) -> bool {
        self.proportion.is_enabled() || self.shape.is_enabled() || self.normal.is_enabled()
    }

    /// Ratios the proportion loss compares (8, or 10 with the head ratios).
    pub fn ratio_count(&self) -> usize {
        NUM_BODY_RATIOS
            + if self.include_head {
                NUM_HEAD_RATIOS
            } else {
                0
            }
    }

    /// Reject malformed schedules or knobs (`label` prefixes the message).
    pub fn validate(&self, label: &str) -> Result<(), String> {
        for (name, s) in [
            ("body proportion loss", &self.proportion),
            ("body shape loss", &self.shape),
            ("normal loss", &self.normal),
        ] {
            s.validate(name).map_err(|m| format!("{label}: {m}"))?;
        }
        let (lo, hi) = Self::SHAPE_MIN_COS_RANGE;
        let c = self.shape_min_cos;
        if !c.is_finite() || c < lo || c > hi {
            return Err(format!(
                "{label}: body_losses.shape_min_cos must be in [{lo}, {hi}], got {c}"
            ));
        }
        Ok(())
    }
}

impl Default for BodyLossesConfig {
    fn default() -> Self {
        Self {
            proportion: super::PIXEL_PERCEPTUAL_SCHEDULE,
            include_head: false,
            shape: super::BODY_SURFACE_SCHEDULE,
            shape_min_cos: Self::DEFAULT_SHAPE_MIN_COS,
            normal: super::BODY_SURFACE_SCHEDULE,
            normal_restrict_to_subject: false,
            pose_model_dir: None,
            shape_model_dir: None,
            normal_model_dir: None,
        }
    }
}
