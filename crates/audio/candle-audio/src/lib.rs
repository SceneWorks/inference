//! # candle-audio
//!
//! Shared **candle** commons for the SceneWorks audio-generation provider family
//! (epic sc-12833, scaffolded by sc-12835). Audio generation is Candle-native on
//! **every** platform (`docs/architecture/audio-backend-strategy.md`): one Candle
//! implementation of each audio model serves `runtime-cpu`, `runtime-cuda`, and
//! `runtime-macos`, where it rides the runtime catalog's dedicated audio section
//! alongside the mlx media graph.
//!
//! Audio providers implement the **existing** generator contract — [`gen_core::Generator`]
//! with [`gen_core::Modality::Audio`] descriptors emitting
//! [`gen_core::GenerationOutput::Audio`] — registered through
//! `candle-audio-catalog`, the audio composition root. No new trait, no linker
//! discovery. This crate is the single audited home for the machinery every candle
//! audio provider needs (the sibling of `candle-gen` for the media families):
//!
//! - [`dsp`] — Hann windowing, forward STFT, and the inverse-STFT overlap-add
//!   reconstruction an iSTFT-Net-style vocoder head needs (Kokoro / StyleTTS2,
//!   sc-12836).
//! - [`mel`] — HTK mel filterbank construction and application for mel-spectrogram
//!   front-ends (reference-audio conditioning, model preprocessing parity).
//! - [`wav`] — 16-bit PCM WAV encoding of a [`gen_core::AudioTrack`], the audio
//!   sibling of the media families' image/video encode step.
//! - [`harness`] — the audio validation & quality harness (sc-12854): per-run
//!   latency/warmup/peak-memory/duration/clipping/LUFS/true-peak measurement (loudness
//!   meters reused from `gen_core::audio_dsp`, never reimplemented), the PCM
//!   repeatability hash, and the [`harness::MetricEnvelope`] the per-model regression
//!   fixtures assert against.
//!
//! Scope discipline (sc-12835): these modules are exactly what the first shipped
//! provider (Kokoro, sc-12836) needs — no speculative surface. Grow this crate
//! per-model, the way `candle-gen` grew with its provider families.

// Re-export the backend-neutral contract so downstream audio provider crates resolve
// `gen_core::…` through `candle_audio::gen_core` (single gen-core resolution — the same
// pattern candle-gen / mlx-gen use for the media families).
pub use gen_core;
// Re-export the generator registration macro. The audio lane is **generators-only**
// (audio-backend-strategy.md; enforced by `runtime-catalog::validate_audio`), so only the
// generator macro is re-exported — a provider needing another kind belongs in a media family.
pub use gen_core::register_generators;
// Re-export the candle backend so provider crates share this crate's exact candle build.
pub use candle_core;

use thiserror::Error;

pub mod dsp;
pub mod harness;
pub mod mel;
pub mod wav;

// Test-support helpers shared across the candle audio provider crates. Feature-gated so it never
// compiles into a production build — provider crates enable `candle-audio/testkit` under
// `[dev-dependencies]`, mirroring `candle-gen`'s testkit seam. CI compiles it explicitly with
// `--features candle-audio/testkit` (ci.yml) so the module never sits behind an unexercised cfg
// (the sc-11990 cfg-hole guard). The HF-cache snapshot scanners this module once held were removed
// under epic 13657 — inference never self-fetches or derives an HF-cache location; tests take a
// passed-in snapshot dir via env instead.
#[cfg(feature = "testkit")]
pub mod testkit;

/// The candle-backed audio-crate error. gen-core cannot name candle types, so device/tensor
/// failures arrive boxed in [`gen_core::Error::Backend`] via the [`From`] bridge below —
/// the same seam `candle-gen`'s `CandleError` provides for the media families (legal under
/// the orphan rule because the source type is local to this crate).
#[derive(Debug, Error)]
pub enum AudioError {
    /// A candle op (matmul, conv, device alloc, …) failed.
    #[error("candle op failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// A contextual message (config/validation/shape errors).
    #[error("{0}")]
    Msg(String),

    /// Cooperative cancellation tripped mid-synthesis (the request's `CancelFlag`). A typed
    /// variant — NOT a `Msg` — so a provider's rich-`Result` body can bail between synthesis
    /// steps and the [`From`] bridge lifts it to the contract-load-bearing
    /// [`gen_core::Error::Canceled`] (the worker + gen-core-testkit conformance suite key off
    /// the typed variant, sc-4481). Mirrors `candle-gen`'s `CandleError::Canceled`.
    #[error("cancelled")]
    Canceled,
}

impl From<AudioError> for gen_core::Error {
    fn from(e: AudioError) -> Self {
        match e {
            // candle's Error is `Send + Sync + 'static`, so it boxes straight into Backend.
            AudioError::Candle(c) => gen_core::Error::backend(c),
            AudioError::Msg(s) => gen_core::Error::Msg(s),
            // Preserve the typed cancellation signal across the bridge (do NOT stringify to Msg).
            AudioError::Canceled => gen_core::Error::Canceled,
        }
    }
}

/// Reverse bridge: lift a backend-neutral [`gen_core::Error`] back into [`AudioError`] so a
/// provider's rich-`Result` body can `?` gen-core helper results. The load-bearing arm is
/// `Canceled -> Canceled` — cancellation must stay typed across both bridges (sc-4481).
impl From<gen_core::Error> for AudioError {
    fn from(e: gen_core::Error) -> Self {
        match e {
            gen_core::Error::Canceled => AudioError::Canceled,
            other => AudioError::Msg(other.to_string()),
        }
    }
}

impl From<String> for AudioError {
    fn from(s: String) -> Self {
        AudioError::Msg(s)
    }
}

impl From<&str> for AudioError {
    fn from(s: &str) -> Self {
        AudioError::Msg(s.to_string())
    }
}

/// Crate-wide result over [`AudioError`] (the rich candle-side `Result`; provider `Generator`
/// bodies bridge the tail into `gen_core::Result` via `?` + the [`From`] above).
pub type Result<T> = std::result::Result<T, AudioError>;

/// The process-default compute device for the audio lane, selected at compile time by feature:
/// CUDA (`cuda`) → Metal (`metal`) → CPU (default) — the same seam as `candle-gen`'s
/// `default_device`. The audio lane ships CPU-first everywhere (the walking-skeleton models
/// synthesize in real time on CPU — audio-backend-strategy.md); GPU device selection is a
/// per-model implementation option behind the platform bundle's feature choice.
pub fn default_device() -> Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    let dev = candle_core::Device::new_cuda(0)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = candle_core::Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let dev = candle_core::Device::Cpu;
    Ok(dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_constructs() {
        // CPU on the default build; Metal/CUDA when those features are on. Proves candle is
        // linked and a Device is constructible on whatever backend the build selected.
        let dev = default_device().expect("default device constructs");
        let t = candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, &dev).expect("alloc");
        assert_eq!(t.dims(), &[2, 2]);
    }

    #[test]
    fn audio_error_bridges_to_backend() {
        // A candle error must box into gen_core::Error::Backend (the parity-critical seam).
        let bad =
            candle_core::Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap()
                .matmul(
                    &candle_core::Tensor::zeros(
                        (4, 5),
                        candle_core::DType::F32,
                        &candle_core::Device::Cpu,
                    )
                    .unwrap(),
                );
        let audio_err = AudioError::from(bad.unwrap_err());
        let neutral: gen_core::Error = audio_err.into();
        assert!(matches!(neutral, gen_core::Error::Backend(_)));
    }

    #[test]
    fn cancellation_stays_typed_across_both_bridges() {
        let neutral: gen_core::Error = AudioError::Canceled.into();
        assert!(matches!(neutral, gen_core::Error::Canceled));
        let back: AudioError = gen_core::Error::Canceled.into();
        assert!(matches!(back, AudioError::Canceled));
    }
}
