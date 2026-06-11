//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: active** — multiple merged, parity-validated provider crates spanning image,
//! video, identity, and understanding models, consumed in-process as a Rust library.
//!
//! Families: FLUX.1 / FLUX.2, Chroma, Qwen-Image (+ Edit), SDXL, Kolors, Z-Image,
//! SenseNova-U1 (image); Wan2.2, LTX-2.3, SVD (video); PuLID-FLUX, InstantID (identity);
//! JoyCaption, SAM2 (understanding). Adapters: LoRA, LoKr (with stacking), ControlNet,
//! IP-Adapter. Plus native MLX LoRA/LoKr training and group-wise Q4/Q8 quantization.
//!
//! Architecture: a *disciplined hybrid* of the frozen Python mflux fork — see
//! [`ARCHITECTURE.md`](https://github.com/michaeltrefry/mlx-gen/blob/main/ARCHITECTURE.md).

pub mod adapters;
pub mod array;
pub mod caption;
pub mod error;
pub mod generator;
pub mod image;
pub mod media;
pub mod nn;
pub mod quant;
pub mod registry;
pub mod runtime;
pub mod sampler;
pub mod scheduler;
pub mod tiling;
pub mod tokenizer;
pub mod train;
pub mod transform;
pub mod weights;

pub use caption::{
    CaptionCapabilities, CaptionFinishReason, CaptionOptions, CaptionOutput, CaptionRequest,
    CaptionSampling, Captioner, CaptionerDescriptor,
};
pub use error::{Error, Result};
pub use generator::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlClipRef, ControlKind,
    GenerationOutput, GenerationRequest, Generator, KeyframeRef, Modality, ModelDescriptor,
    ReplacementMode, VideoClipRef,
};
pub use media::{AudioTrack, Image};
pub use registry::{
    load, load_captioner, load_transform, CaptionerRegistration, ModelRegistration,
    TransformRegistration,
};
pub use registry::{load_trainer, TrainerRegistration};
pub use runtime::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, Precision, Progress, Quant,
    WeightsSource,
};
pub use sampler::{
    AlphaSchedule, DiffusionSampler, FlowMatchSampler, LcmSampler, LightningSampler, TcdSampler,
};
pub use scheduler::FlowMatchEuler;
pub use tiling::{TilingConfig, VaeTiling};
pub use train::{
    LrSchedule, NetworkType, TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig,
    TrainingItem, TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};
