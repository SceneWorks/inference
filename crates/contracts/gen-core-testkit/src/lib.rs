//! # gen-core-testkit
//!
//! A **contract conformance suite** for gen-core providers — the image/video/pure-audio
//! [`gen_core::Generator`] (this module), [`gen_core::Trainer`](crate::trainer),
//! [`gen_core::Captioner`](crate::captioner), and the audio-contract family (sc-12853):
//! [`gen_core::Generator`] under [`Modality::Audio`]
//! ([`crate::audio_generator`]), [`gen_core::VoiceEmbedder`](crate::voice_embedder),
//! [`gen_core::AudioTransform`](crate::audio_transform),
//! [`gen_core::Transcriber`](crate::transcriber), and
//! [`gen_core::AudioEmbedder`](crate::audio_embedder). Given any boxed provider — an MLX family from
//! `mlx-gen`, a candle-gen provider, or a `crates/audio` candle provider — it exercises the
//! behavioral guarantees the contract *promises but cannot express in the type system*: typed
//! cancellation, progress monotonicity, seed determinism, and capability honesty. Both backends run
//! it in CI, so a provider that silently ignores `CancelFlag` or reports no progress (the sc-4380
//! class of bug) becomes a CI failure instead of a field report (epic 3720, sc-4481/sc-4895).
//!
//! The testkit has **zero tensor dependencies** — it depends only on `gen-core` and drives the
//! provider purely through the public contract, so it builds and runs on the Linux gen-core lane
//! against an in-crate stub exactly as it does on the macOS lane against a real MLX family.
//!
//! ## Usage
//!
//! ```ignore
//! // macOS lane, real family — generator, trainer, captioner:
//! let registry = mlx_gen_z_image::provider_registry().unwrap();
//! gen_core_testkit::conformance(
//!     || registry.load("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::Profile::cheap(),
//! );
//! gen_core_testkit::trainer_conformance(
//!     || registry.load_trainer("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::TrainerProfile::cheap(items, out_dir),
//! );
//! let registry = mlx_gen_joycaption::provider_registry().unwrap();
//! gen_core_testkit::captioner_conformance(
//!     || registry.load_captioner("joy_caption", &spec).unwrap(),
//!     &gen_core_testkit::CaptionerProfile::cheap(),
//! );
//! ```
//!
//! The individual `check_*` functions are public so a provider's own tests can target one guarantee
//! at a time; the `*_conformance` entry points run them all and panic with the aggregated failures.

pub mod audio_embedder;
pub mod audio_generator;
pub mod audio_transform;
pub mod captioner;
pub mod h3_memory;
pub mod host;
pub mod memory_strategy;
pub mod moss_training_fixture;
pub mod moss_training_probe;
pub mod trainer;
pub mod transcriber;
pub mod voice_embedder;
pub mod wan_i2v;

pub use audio_embedder::{
    audio_embedder_conformance, check_audio_embed_joint, check_audio_embedder_registry,
    AudioEmbedderProfile,
};
pub use audio_generator::{
    audio_conformance, check_audio_cancellation, check_audio_multi_speaker, check_audio_output,
    check_audio_precancellation, check_audio_progress, check_audio_progress_contract,
    check_audio_seed_determinism, check_audio_streaming, check_audio_validate_honesty,
    check_multi_turn, check_video_to_audio, AudioProfile,
};
pub use audio_transform::{
    audio_transform_conformance, check_audio_transform_cardinality,
    check_audio_transform_coherence, check_audio_transform_registry,
    check_audio_transform_validate, AudioTransformProfile,
};
pub use captioner::{
    captioner_conformance, check_captioner_cancellation, check_captioner_progress,
    check_captioner_registry, check_captioner_validate, CaptionerProfile,
};
pub use memory_strategy::{
    assert_memory_contract_asset_facts_conform, assert_memory_contract_facts_conform,
    check_memory_contract_asset_facts, check_memory_contract_facts,
    check_memory_contract_surface_registry, check_memory_contract_surface_registry_facts,
    check_memory_strategy_contract, check_memory_strategy_registry,
    memory_contract_surface_registry_conformance,
    memory_contract_surface_registry_facts_conformance, memory_strategy_conformance,
    memory_strategy_registry_conformance, MaterializedRootLookup,
    MemoryContractSurfaceFactsCoverage,
};
pub use trainer::{
    check_trainer_bucketed_progress, check_trainer_cancellation, check_trainer_progress,
    check_trainer_registry, check_trainer_technique_refusal, check_trainer_train_floors,
    check_trainer_validate, trainer_conformance, TrainerProfile,
};
pub use transcriber::{
    check_transcriber_cancellation, check_transcriber_output, check_transcriber_progress,
    check_transcriber_registry, check_transcriber_validate, transcriber_conformance,
    TranscriberProfile,
};
pub use voice_embedder::{
    check_voice_embed, check_voice_embed_rejects_short, check_voice_embedder_registry,
    voice_embedder_conformance, VoiceEmbedderProfile,
};

use gen_core::{
    Capabilities, Conditioning, EncoderContract, Error, GenerationOutput, GenerationRequest,
    Generator, Image, Modality, OutputChannels, Progress, RgbaImage, StepSupport,
    VisionEncoderContract,
};

mod fixtures;
pub use fixtures::*;

mod generator_checks;
pub use generator_checks::*;

mod registry_checks;
pub use registry_checks::*;

mod logits;
pub use logits::*;

#[cfg(test)]
mod tests;
