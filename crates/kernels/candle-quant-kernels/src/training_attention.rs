//! Streaming training attention's kernel descriptor, shared with the all-kernels checks.
//!
//! The media engine owns selection, dispatch and autograd; this crate owns the CUDA source
//! so the shared NVRTC inventory validates every entry point without an engine dependency.

use crate::nvrtc::KernelSource;

/// Shared compile-once source for the media engine's streaming attention candidate.
pub const SOURCE: KernelSource = KernelSource {
    name: "candle_gen_training_attention_v1",
    src: include_str!("training_attention.cu"),
    cc_floor: (7, 0),
};
