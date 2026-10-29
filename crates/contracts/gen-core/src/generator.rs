//! The `Generator` contract — prompt-conditioned synthesis of image, video, **or** audio
//! (or a mix), including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio), and pure audio synthesis (TTS / music). Modality is a
//! [`ModelDescriptor`] property plus a [`GenerationOutput`] variant — *not* a per-modality
//! trait split (which breaks on multi-modal models).

use crate::approximation::{ApproximationPlan, ApproximationRequest, ApproximationSurface};
use crate::execution_domains::{
    CfgBatching, ExecutionSurface, FfnChunk, GraphEvalCadence, PrefixKvCacheMode,
    PrefixKvCacheReport,
};
use crate::hdr::HdrColorSpace;
use crate::media::{AudioChunk, AudioTrack, Image, RgbaImage};
use crate::runtime::{
    CancelFlag, HdrFrameSink, PreviewSink, Progress, PromptEnhancementSink, Quant,
};
use crate::voice_embed::VoiceEmbedding;
use crate::{
    default_memory_strategy_safety_check, Error, MemoryPeakBreakdown, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStructuralResidentEvidence, Result,
};

mod traits;
pub use traits::*;

mod output;
pub use output::*;

mod request;
pub use request::*;

mod params;
pub use params::*;

mod views;
pub use views::*;

mod conditioning;
pub use conditioning::*;

mod descriptor;
pub use descriptor::*;

mod capabilities;
pub use capabilities::*;

#[cfg(test)]
mod tests;
