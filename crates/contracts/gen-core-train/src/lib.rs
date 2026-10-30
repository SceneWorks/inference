//! Family-agnostic trainer helpers for the backend-neutral [`Trainer`] contract (split out of
//! gen-core in sc-27316): request validation against a [`TrainerDescriptor`], rectangular
//! aspect-ratio assignment and cropping, subject-mask decode, auxiliary-loss scheduling, resume
//! identity, and the body/face loss geometry. The contract itself — [`Trainer`],
//! [`TrainingRequest`], [`TrainingConfig`] and the shapes they carry — stays in
//! [`gen_core::train`] and is re-exported here so a trainer reaches the whole surface from one
//! path. Tensor-free, like gen-core.

pub use gen_core::train::*;

pub mod aspect_ratio;
pub mod aux_schedule;
pub mod body;
pub mod face_loss;
pub mod resume;
pub mod subject_mask;

pub use aux_schedule::{
    combine_step_terms, perceptual_footprint_bytes, plan_in_band, plan_step, AltKey,
    AuxAlternation, AuxModelFootprint, StepPlan,
};
use serde_json::{Map as JsonMap, Value as JsonValue};

mod validate;
pub use validate::*;

#[cfg(test)]
mod tests;
