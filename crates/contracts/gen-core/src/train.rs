//! The `Trainer` contract — LoRA/LoKr fine-tuning of a registered model (epic 3039), the training
//! analog of [`Generator`](crate::generator::Generator). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! SceneWorks owns the training *product* surface (datasets, plan normalization, validation, the
//! queue) in Rust; this is the *execution* surface that replaces the Python kernel for the
//! MLX-native families. The worker maps its normalized `TrainingPlan` onto a [`TrainingRequest`]
//! and calls [`Trainer::train`] — exactly as it maps `ImageRequest` → `GenerationRequest` and calls
//! `Generator::generate`. mlx-gen owns these shapes; it does not depend on the SceneWorks contract.
//!
//! This module is the contract half: the config/progress/request shapes the registry's
//! [`TrainerRegistration`](crate::registry::TrainerRegistration) and every [`Trainer`] exchange,
//! the [`schedule`] LR policy and dataset bucketing. The family-agnostic trainer helpers (request
//! validation, aspect-ratio assignment, subject-mask decode, auxiliary-loss scheduling, resume
//! identity, body/face loss geometry) live in the sibling `gen-core-train` crate (sc-27316).

// The pure LR-schedule policy lives here (gen-core); the MLX training kernels
// (checkpoint/dataset/lora/optim, incl. `TrainOptimizer`) stay in mlx-gen's `train` module.
pub mod aspect_ratio;
pub use aspect_ratio::AspectRatioBucket;
pub mod body;
pub mod schedule;

use std::path::PathBuf;

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
