//! # candle-audio-moss-tts-realtime
//!
//! **MOSS-TTS-Realtime-1.7B** streaming-TTS provider for the SceneWorks Candle audio lane
//! (sc-13334, epic sc-12833) — OpenMOSS's context-aware, real-time streaming text-to-speech model
//! (Apache-2.0 weights + code) ported natively onto the workspace's pinned candle revision. One
//! candle implementation targets `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the
//! audio composition root, per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The architecture (CSM-style autoregressive RVQ)
//!
//! MOSS-TTS-Realtime is a discrete multi-codebook TTS model. It pairs a **Qwen3-1.7B** causal LM
//! backbone with a small **local/depth transformer** over 16 residual-vector-quantization (RVQ)
//! speech codebooks — the Sesame-CSM / Moshi pattern:
//!
//! - [`backbone`] — the Qwen3-1.7B backbone (2048 hidden, 28 layers, GQA 16/8, head-dim 128). Its
//!   input at every position is a **multi-channel** token: one text-channel id plus 16 audio
//!   codebook ids, each embedded and **summed** (`config.json.language_config`).
//! - [`local`] — the 4-layer local/depth transformer (`config.json.local_config`, `rvq = 16`).
//!   Run once per audio frame, seeded by the backbone's last hidden state, it autoregressively
//!   emits the frame's 16 RVQ codebook tokens through 16 per-codebook LM heads. Sampling here is
//!   deterministic greedy (the gen-core reproducibility law).
//! - [`decode`] — the assembled AR loop (the reference `prefill` + `step`): backbone → RVQ frame →
//!   feed the frame back as the next position's audio channels → repeat until the audio-EOS or a
//!   frame budget. Each iteration emits one RVQ frame incrementally; cancellation is consulted every
//!   frame.
//! - [`chunk`] — the streaming PCM-block → `gen_core::AudioChunk` mechanism (a pure, offline-tested
//!   helper) the streaming path will emit incrementally once the codec lands.
//!
//! ## Port status (honest partial — sc-13334)
//!
//! sc-13334 ports the **AR brain** (backbone, local transformer, AR loop, the provider contract,
//! the pinned-SHA hub path, and the preparer probe) and verifies, on real weights, that it emits
//! real in-range 16-codebook RVQ speech-token frames (the `conformance` test).
//!
//! Turning those RVQ frames into a 24 kHz waveform requires the **MOSS-Audio-Tokenizer** codec
//! (`OpenMOSS-Team/MOSS-Audio-Tokenizer`) — a **separate ~7 GB** model (a novel RLFQ streaming codec
//! with 32 quantizers and ~44 causal-transformer layers, NOT the same as candle-transformers' Mimi)
//! that is **not yet ported**. Consequently [`model::MossTtsRealtimeGenerator`]'s `generate` runs the AR
//! loop to produce real frames and then returns a typed error at the codec boundary rather than
//! fabricate audio, and this generator is **not yet registered** into `candle-audio-catalog`'s
//! shipping surface (registering an audio generator that cannot render audio would fail the gen-core
//! audio conformance suite and mis-advertise the lane). That registration, the ordered-id surface
//! extension, and the three bundle smokes are deliberately deferred to the codec follow-up — the same
//! discipline `candle-audio-chatterbox` applies while its S3Gen vocoder stack is unported. This
//! crate is present as a workspace member so its AR stack builds, is unit-tested, and is exercised
//! end-to-end on real weights by the conformance test.
//!
//! Weights resolve through the audio lane's pinned-SHA hub path (F-029):
//! [`model::HUB_REPO`] at [`model::HUB_REVISION`].

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod backbone;
pub mod blocks;
pub mod chunk;
pub mod config;
pub mod decode;
pub mod local;
pub mod model;
pub mod prepare;

pub use model::{
    descriptor, load, load_generator, resolve_pinned_snapshot, CODEC_HUB_REPO, CODEC_HUB_REVISION,
    HUB_REPO, HUB_REVISION, LANGUAGES, MAX_DURATION_SECS, MODEL_ID, REGISTRATION, SAMPLE_RATE,
    WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY,
};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332) — one row keyed by
/// [`MODEL_ID`]. The audio catalog concatenates every provider's slice into the model-licenses
/// manifest SceneWorks lists on its end-product licenses page. (Wired in with the deferred catalog
/// registration; declared here now so the license is never lost.)
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];
