//! Lower-level Candle linear, quantization, and residual adapter primitives shared below media and
//! LLM engines (`SW-6955` / `SW-F335`).
//!
//! This crate depends only on tensor-free contracts (`sceneworks-gen-core`, `core-llm`), Candle core
//! crates (`candle-core`, `candle-nn`), and `candle-quant-kernels`, with zero upward dependency on
//! `candle-gen*`, `candle-llm*`, `mlx-*`, `runtime-*`, or `candle-audio*`.

pub use ::gen_core;

#[macro_use]
#[path = "../../../llm/test-support/model_proof.rs"]
mod model_proof;

pub mod act_probe;
pub mod adapt;
pub mod convrot;
pub mod convrot_int8;
#[cfg(feature = "cuda")]
pub mod eight_bit_linear;
pub mod fused;
pub mod llm_projection;
pub mod nvfp4_kitchen;
pub mod nvfp4_path;
pub mod prism;
pub mod quant;
pub mod repack;
pub mod switch;
pub mod w4a8_int8;

pub use adapt::{AdaptLinear, Adapter, Base, Linear, LokrFactors, ScaleContext};
pub use candle_quant_kernels::{cublaslt, nvfp4, nvfp4_linear, nvfp4_outlier};
pub use convrot::{convrot_fht_f32, convrot_rotate};
#[cfg(feature = "cuda")]
pub use cublaslt::{CublasLt, W4A8DecodeTile};
#[cfg(feature = "cuda")]
pub use eight_bit_linear::{Fp8Linear, Int8Linear};
pub use llm_projection::{
    from_ggml_block_tensor, to_ggml_block_tensor, KvProjection, Projection, ProjectionCensus,
    ProjectionFormat, ProjectionKind, ProjectionTally, Q8RowEmbedding, QuantSpec, QuantizedLinear,
    WeightCensus,
};
pub use prism::{GdnRowMap, PrismPackedWeight, PrismRegistry};
pub use quant::{
    DenseLinear, MatmulStrategy, PackedConfig, QEmbedding, QLinear, QuantForward, QuantWeight,
};

use candle_core::{Device, Tensor};
use std::sync::{Mutex, MutexGuard};
use thiserror::Error;

/// Acquire `mutex`, recovering the inner value if a previous holder panicked.
pub fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Minimal tensor-lookup contract over a loaded Candle weight table so [`convrot_int8::ConvRotLinear`],
/// [`w4a8_int8::W4A8Linear`], [`quant::linear_from_weights`], and [`quant::linear_from_weights_gs`]
/// construct directly in `candle-layers` without depending on `candle-gen`'s loader/weight types.
pub trait CandleWeightSource {
    /// Whether `key` is present in the loaded weight table.
    fn contains(&self, key: &str) -> bool;

    /// Load or fetch a required tensor by exact key.
    fn require(&self, key: &str) -> Result<Tensor>;

    /// Optional final execution device for host-staged packed weights.
    fn host_packed_target(&self) -> Option<&Device>;

    /// Shared cuBLASLt INT8 context for `device`.
    fn int8_context(&self, device: &Device) -> Result<candle_quant_kernels::Int8Context>;

    /// Cached `[group, group]` orthogonal Walsh-Hadamard rotation matrix on `device`.
    fn convrot_rotation(&self, group: usize, device: &Device) -> Result<Tensor>;
}

/// The candle-backed crate error. gen-core cannot name candle types, so device/tensor failures
/// arrive boxed in [`gen_core::Error::Backend`] via the [`From`] bridge below. This mirrors
/// mlx-gen's `From<mlx_gen::Error> for gen_core::Error` seam — legal under the orphan rule because
/// the source type ([`CandleError`]) is local to this crate.
#[derive(Debug, Error)]
pub enum CandleError {
    /// A candle op (matmul, conv, device alloc, …) failed.
    #[error("candle op failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// A contextual message (config/validation/shape errors).
    #[error("{0}")]
    Msg(String),

    /// A typed **capability refusal**: the request asks for something this route deliberately does
    /// not do (an unadvertised conditioning kind, a strength on a condition image, more references
    /// than a joint layout can express, a quant tier the backend cannot produce). Kept a typed
    /// variant — NOT a `Msg` — so the [`From`] bridge lifts it to the contract-load-bearing
    /// [`gen_core::Error::Unsupported`]: the worker and the gen-core-testkit validate-honesty check
    /// key off that variant to tell "refused by design" from "broke" (sc-24114). Mirrors mlx-gen's
    /// `Error::Unsupported`.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Cooperative cancellation tripped mid-generation (the request's `CancelFlag`). Kept a typed
    /// variant — NOT a `Msg` — so a provider's rich-`Result` body can `return Err(CandleError::Canceled)`
    /// between denoise steps and the [`From`] bridge lifts it to the contract-load-bearing
    /// [`gen_core::Error::Canceled`] (the worker + gen-core-testkit conformance suite key off the typed
    /// variant, sc-4481). Mirrors mlx-gen's `Error::Canceled`.
    #[error("cancelled")]
    Canceled,

    /// Explicit process-local backend poison. Ordinary Candle failures remain backend errors.
    #[error("process poisoned: {0}")]
    ProcessPoisoned(String),

    /// A measured pre-render refusal. Keep every decision-surface field typed across the Candle seam
    /// so worker/API telemetry can distinguish it from an opaque backend failure.
    #[error(
        "geometry refused: {reason}; requested {requested_width}x{requested_height}; verified alternative: {alternative:?}"
    )]
    GeometryRefused {
        reason: String,
        requested_width: u32,
        requested_height: u32,
        alternative: Option<(u32, u32)>,
    },
}

impl From<CandleError> for gen_core::Error {
    fn from(e: CandleError) -> Self {
        match e {
            // candle's Error is `Send + Sync + 'static`, so it boxes straight into Backend.
            CandleError::Candle(c) => gen_core::Error::backend(c),
            CandleError::Msg(s) => gen_core::Error::Msg(s),
            // Preserve the typed refusal across the bridge (do NOT stringify to Msg).
            CandleError::Unsupported(s) => gen_core::Error::Unsupported(s),
            // Preserve the typed cancellation signal across the bridge (do NOT stringify to Msg).
            CandleError::Canceled => gen_core::Error::Canceled,
            CandleError::ProcessPoisoned(s) => gen_core::Error::ProcessPoisoned(s),
            CandleError::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            } => gen_core::Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            },
        }
    }
}

/// Reverse bridge: lift a backend-neutral [`gen_core::Error`] back into [`CandleError`]. The unified
/// curated-sampler driver (`sampler::run_curated_sampler`) runs over the gen-core `Sampler` trait
/// (which returns `gen_core::Result`), so a provider's rich-`Result` denoise body needs this to `?` the
/// driver's output. The load-bearing arm is `Canceled -> Canceled`: a cooperative cancel tripped inside
/// the driver's `denoise` callback surfaces as `gen_core::Error::Canceled` and MUST stay the typed
/// [`CandleError::Canceled`] (not a stringified `Msg`) so the worker + conformance suite key off it.
/// Mirrors mlx-gen's `From<gen_core::Error> for mlx_gen::Error`.
impl From<gen_core::Error> for CandleError {
    fn from(e: gen_core::Error) -> Self {
        match e {
            gen_core::Error::Canceled => CandleError::Canceled,
            gen_core::Error::ProcessPoisoned(s) => CandleError::ProcessPoisoned(s),
            gen_core::Error::MissingTensor(s) => CandleError::Msg(format!("missing tensor: {s}")),
            gen_core::Error::Unsupported(s) => CandleError::Unsupported(s),
            gen_core::Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            } => CandleError::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            },
            gen_core::Error::Io(io) => CandleError::Msg(io.to_string()),
            gen_core::Error::Backend(b) => CandleError::Msg(b.to_string()),
            gen_core::Error::Msg(s) => CandleError::Msg(s),
        }
    }
}

impl From<String> for CandleError {
    fn from(s: String) -> Self {
        CandleError::Msg(s)
    }
}

impl From<&str> for CandleError {
    fn from(s: &str) -> Self {
        CandleError::Msg(s.to_string())
    }
}

/// Crate-wide result over [`CandleError`] (the rich candle-side `Result`; provider `Generator`
/// bodies bridge the tail into `gen_core::Result` via `?` + the [`From`] above).
pub type Result<T> = std::result::Result<T, CandleError>;
