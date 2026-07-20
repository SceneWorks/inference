//! # candle-audio-moss-tts
//!
//! **MOSS-TTSD** multi-speaker *dialogue* text-to-speech provider for the SceneWorks Candle audio
//! lane (sc-13360, epic sc-12833) — OpenMOSS's Apache-2.0 dialogue-TTS model ported natively onto the
//! workspace's pinned candle revision. One candle implementation targets `runtime-cpu`,
//! `runtime-cuda`, and `runtime-macos` through the audio composition root.
//!
//! ## The architecture (delay-pattern autoregressive RVQ)
//!
//! MOSS-TTSD is a discrete multi-codebook dialogue-TTS model. Unlike the sibling
//! `candle-audio-moss-tts-realtime` (a CSM-style backbone + local/depth transformer), MOSS-TTSD is a
//! **delay-pattern** model (MusicGen/Parler-style):
//!
//! - [`backbone`] — a standard **Qwen3** causal LM (MOSS-TTSD-v0.5: 2048 hidden, 28 layers, GQA
//!   16/8, head-dim 128). Its input at every position is a **`channels`-wide** (8) token: a
//!   text/speech id (channel 0, whose vocab also carries speech codebook 0) plus the remaining audio
//!   codebooks, each embedded and **summed**. `tie_word_embeddings` makes each channel's prediction
//!   head its own embedding matrix.
//! - [`decode`] — the **delay-pattern** AR loop (`MossTTSDGenerationMixin._sample`): one backbone
//!   step yields all 8 channel logits at once; channel `j` is time-shifted by `j` positions (the
//!   delay pattern), with a start-of-stream teacher-forced ramp and an end-of-stream delay-tail
//!   drain. Sampled per-channel ([`sampling`]) from a **seeded** PRNG (the reproducibility law), then
//!   un-shifted into clean 8-codebook frames.
//! - [`prepare`] — the audio-lane snapshot probe + validated passthrough preparer.
//!
//! ## Port status — HONEST PARTIAL (AR brain landed; XY_Tokenizer codec split off)
//!
//! This slice lands and verifies the **AR brain**: on the real **MOSS-TTSD-v0.5** weights (the
//! smallest single-shard dialogue checkpoint; the 8B v1.0 `moss_tts_delay` is the quality ceiling)
//! the loop emits real, in-range, deterministic delay-pattern RVQ token frames — proven by the
//! real-weights conformance test. The RVQ codec, OpenMOSS's **XY_Tokenizer**
//! (`OpenMOSS-Team/XY_Tokenizer_TTSD_V0`, a 2.1 GB raw-pickle codec whose architecture lives only in
//! the OpenMOSS reference code — *not* candle's Mimi/SNAC/DAC), is a large separate port and is **not
//! yet landed**, exactly the sc-13334 → sc-13392 split precedent. So [`model::MossTtsdGenerator`]'s
//! `generate` returns a typed error at the codec boundary rather than fabricate audio, and this
//! generator is **NOT registered** into `candle-audio-catalog`.
//! [`model::MossTtsdGenerator::rvq_frames`] exposes the AR token stream the codec will consume.
//!
//! Weights resolve through the audio lane's pinned-SHA hub path (F-029): the AR checkpoint at
//! [`model::HUB_REPO`]@[`model::HUB_REVISION`]; the codec source is pinned at
//! [`model::CODEC_HUB_REPO`]@[`model::CODEC_HUB_REVISION`] for the follow-up.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod backbone;
pub mod blocks;
pub mod config;
pub mod decode;
pub mod model;
pub mod prepare;
pub mod sampling;

pub use model::{
    descriptor, load, load_generator, resolve_pinned_snapshot, CODEC_HUB_REPO, CODEC_HUB_REVISION,
    HUB_REPO, HUB_REVISION, LANGUAGES, MAX_DURATION_SECS, MAX_SPEAKERS, MODEL_ID, SAMPLE_RATE,
    WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY,
};

/// This crate's model-weight-license entries — one row keyed by [`MODEL_ID`]. Ready for catalog
/// aggregation when the codec lands and the provider ships (it is not registered yet).
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];
