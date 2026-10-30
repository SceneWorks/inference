//! # gen-core-h3
//!
//! The tensor-free MiniMax-H3 planners and policy that only the H3 providers need, split out of
//! gen-core in sc-27316: request/component native allocation pricing, adapter and conditioning
//! memory pricing, the Veda sparse-attention predictor, selection and workspace pricing, reference
//! normalization, audio config, keyframes, the denoise schedule, chunking and pruned adaLN. The data
//! shapes the contract carries (`MemoryPhaseFacts::native_quant`, the Veda request/report, the
//! workspace bound and the reference resize mode) stay in gen-core and are re-exported from the
//! matching modules here; the methods of `NativeAllocationFacts` are [`h3_memory::NativeAllocationFactsExt`].

pub mod h3_adapter_memory;
pub mod h3_audio_config;
pub mod h3_chunking;
pub mod h3_conditioning_memory;
pub mod h3_keyframe;
pub mod h3_memory;
pub mod h3_reference;
pub mod h3_schedule;
pub mod pruned_adaln;
pub mod veda;

pub use h3_reference::{resolve_reference_image_size, truncate_reference_audio};
