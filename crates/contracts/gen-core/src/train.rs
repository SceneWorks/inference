//! The `Trainer` contract — LoRA/LoKr fine-tuning of a registered model (epic 3039), the training
//! analog of [`Generator`](crate::generator::Generator). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! SceneWorks owns the training *product* surface (datasets, plan normalization, validation, the
//! queue) in Rust; this is the *execution* surface that replaces the Python kernel for the
//! MLX-native families. The worker maps its normalized `TrainingPlan` onto a [`TrainingRequest`]
//! and calls [`Trainer::train`] — exactly as it maps `ImageRequest` → `GenerationRequest` and calls
//! `Generator::generate`. mlx-gen owns these shapes; it does not depend on the SceneWorks contract.
//!
//! The spike (sc-3042) proved the per-family training mechanism (functional autograd over an
//! external LoRA factor map, re-injected as `Adapter::Lora`, stepped with `keyed_value_and_grad` +
//! AdamW). This module is the family-agnostic glue around it: the config/progress/request shapes,
//! the [`schedule`] LR helpers, dataset bucketing, and checkpointing. Each family crate implements
//! [`Trainer`] (Z-Image in sc-3044) and publishes a [`crate::registry::TrainerRegistration`].

// The pure LR-schedule policy lives here (gen-core); the MLX training kernels
// (checkpoint/dataset/lora/optim, incl. `TrainOptimizer`) stay in mlx-gen's `train` module.
pub mod aspect_ratio;
pub use aspect_ratio::AspectRatioBucket;
pub mod aux_schedule;
pub mod body;
pub mod face_loss;
pub mod resume;
pub mod schedule;
pub mod subject_mask;

use std::path::PathBuf;

pub use aux_schedule::{
    combine_step_terms, perceptual_footprint_bytes, plan_in_band, plan_step, AltKey,
    AuxAlternation, AuxModelFootprint, StepPlan,
};
pub use body::BodyLossesConfig;
pub use schedule::LrSchedule;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::generator::Modality;
use crate::media::Image;
use crate::runtime::CancelFlag;

mod config;
pub use config::*;

mod buckets;
pub use buckets::*;

mod request;
pub use request::*;

mod techniques;
pub use techniques::*;

mod validate;
pub use validate::*;

#[cfg(test)]
mod tests;
