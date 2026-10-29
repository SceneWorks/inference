//! The serialized training geometry row of
//! [`TrainingConfig::aspect_ratio_buckets`](super::TrainingConfig::aspect_ratio_buckets).
//! Assignment, cropping and the homogeneous microbatch schedule live in
//! `gen_core_train::aspect_ratio` (sc-27316).
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AspectRatioBucket {
    pub width: u32,
    pub height: u32,
    pub repeats: u32,
}
