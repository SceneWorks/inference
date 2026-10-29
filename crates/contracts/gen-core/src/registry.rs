//! Explicit model + transform discovery: provider crates publish registration constants, family
//! crates add them to a [`ProviderRegistryBuilder`], and platform catalogs select the families they
//! ship. This is the Rust equivalent of an ordinary DI composition root with resolve-by-id.

use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
use crate::audio_transform::{AudioTransform, AudioTransformDescriptor, AudioTransformKind};
use crate::caption::{Captioner, CaptionerDescriptor};
use crate::checkpoint_codec::{CheckpointCodecRegistration, CheckpointCodecRegistry};
use crate::generator::{ConditioningKind, Generator, Modality, ModelDescriptor, StepSupport};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::memory_strategy::{
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport,
};
use crate::runtime::{LoadSpec, Quant, WeightsSource};
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor};
use crate::transcribe::{Transcriber, TranscriberDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor};
use crate::weightsmeta::safetensors_path_bytes;
use crate::{Error, Result};

use std::path::Path;

mod registration;
pub use registration::*;

mod checkpoint_adapters;
pub use checkpoint_adapters::*;

mod memory_surface;
pub use memory_surface::*;

mod builder;
pub use builder::*;

mod providers;
pub use providers::*;

mod conformance;
pub use conformance::*;

#[cfg(test)]
mod tests;
