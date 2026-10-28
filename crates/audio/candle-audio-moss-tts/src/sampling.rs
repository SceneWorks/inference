//! Deterministic per-channel token sampling for the MOSS-TTSD delay-pattern AR loop (sc-13360).
//!
//! MOSS-TTSD samples each channel token — it is **not** a greedy model (`generation_config` ships
//! `do_sample=True` with `temperature=0.9`, `top_k=50`, `top_p=0.95`, `repetition_penalty=1.0`).
//! Greedy (argmax) decoding of a discrete-codebook TTS collapses into a repeating loop that decodes
//! to silence, so the reference (and this port) samples. This module reproduces the reference
//! per-channel `LogitsProcessorList` (Temperature → TopK → TopP warpers + an optional
//! RepetitionPenalty) faithfully, but with a **seeded** PRNG so the gen-core reproducibility law
//! holds: the same `seed` yields byte-identical frames run to run.
//!
//! Order (matching HF's `RepetitionPenaltyLogitsProcessor` → `TemperatureLogitsWarper` →
//! `TopKLogitsWarper` → `TopPLogitsWarper` → softmax → multinomial): repetition penalty on the raw
//! logits → divide by temperature → top-k mask → top-p (nucleus) mask → softmax → multinomial draw.
//! Parity with PyTorch's audio is not required (a different RNG) — a valid, deterministic
//! distribution is.

// The pipeline itself (PRNG, penalty → temperature → top-k → top-p → multinomial) is shared with
// the sibling MOSS crate through the audio commons (sc-27272); this module keeps the model's own
// reference defaults.
pub use candle_audio::sampling::{sample, Rng, SamplingParams};

/// MOSS-TTSD-v0.5 `generation_config` decoding defaults.
pub const TEMPERATURE: f32 = 0.9;
pub const TOP_P: f32 = 0.95;
pub const TOP_K: usize = 50;
pub const REPETITION_PENALTY: f32 = 1.0;
pub const REPETITION_WINDOW: usize = 50;

/// The MOSS-TTSD-v0.5 `generation_config` defaults as one [`SamplingParams`] value —
/// what the AR loop samples with.
pub const DEFAULT_PARAMS: SamplingParams = SamplingParams {
    temperature: TEMPERATURE,
    top_p: TOP_P,
    top_k: TOP_K,
    repetition_penalty: REPETITION_PENALTY,
    repetition_window: REPETITION_WINDOW,
};
