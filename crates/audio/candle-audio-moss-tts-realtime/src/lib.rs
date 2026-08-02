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
//!   emits the frame's 16 RVQ codebook tokens through 16 per-codebook LM heads, **sampled** with the
//!   reference pipeline ([`sampling`]: temperature / top-k / top-p + a per-codebook cross-frame
//!   repetition penalty) from a **seeded** PRNG — greedy argmax collapses this model into a repeating
//!   loop that decodes to silence, so the reference (and this port) sample; seeding keeps it
//!   deterministic (the gen-core reproducibility law).
//! - [`decode`] — the assembled AR loop (the reference `prefill` + `step`): backbone → RVQ frame →
//!   feed the frame back as the next position's audio channels → repeat until the audio-EOS or a
//!   frame budget. Each iteration emits one RVQ frame incrementally; cancellation is consulted every
//!   frame.
//! - [`codec`] — the **MOSS-Audio-Tokenizer** decoder (sc-13392): RVQ codes → 24 kHz waveform via the
//!   RLFQ quantizer decode + the causal RoPE-transformer / `PatchedPretransform` upsampling stack.
//! - [`chunk`] — the streaming PCM-block → `gen_core::AudioChunk` mechanism (a pure, offline-tested
//!   helper); the streaming path emits blocks incrementally as the codec decodes them.
//!
//! ## Port status (complete — sc-13334 + sc-13392)
//!
//! sc-13334 ported the **AR brain** (backbone, local transformer, AR loop, the provider contract,
//! the pinned-SHA hub path, and the preparer probe), verified on real weights to emit real in-range
//! 16-codebook RVQ speech-token frames. sc-13392 ports the **MOSS-Audio-Tokenizer codec**
//! (`OpenMOSS-Team/MOSS-Audio-Tokenizer`, a ~7.1 GB RLFQ streaming codec — 32 quantizers, `rvq_dim`
//! 512, causal RoPE transformers with `PatchedPretransform` channel→time upsampling; distinct from
//! candle-transformers' Mimi) natively onto the pinned candle revision — the **decode path** (the TTS
//! direction). The AR's 16 codebooks drive the codec's first 16 quantizers (the codec's documented
//! variable-bitrate decode). [`model::MossTtsRealtimeGenerator`] now renders real 24 kHz audio through
//! `generate` / `generate_streaming`, and the generator is **registered** into `candle-audio-catalog`.
//!
//! The **AR** stack (the model's own `spec.weights`) resolves through the audio lane's pinned-SHA
//! hub path (F-029) at [`model::HUB_REPO`]@[`model::HUB_REVISION`]. The **MOSS-Audio-Tokenizer
//! codec** is a passed-in component (epic 13657, sc-13662): the caller stages its snapshot directory
//! under [`model::CODEC_COMPONENT_ID`] in [`gen_core::LoadSpec::components`] and it is validated at
//! load, never self-fetched; the pin the consumer must provision is
//! [`model::CODEC_HUB_REPO`]@[`model::CODEC_HUB_REVISION`].

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod backbone;
pub mod blocks;
pub mod chunk;
pub mod codec;
pub mod config;
pub mod conversation;
pub mod decode;
pub mod local;
pub mod model;
pub mod prepare;
pub mod sampling;

pub use model::{
    descriptor, load, load_generator, provider_registry, register_providers, CODEC_COMPONENT_ID,
    CODEC_HUB_REPO, CODEC_HUB_REVISION, COMPONENT_KEY, COMPONENT_LICENSE, HUB_REPO, HUB_REVISION,
    LANGUAGES, MAX_DURATION_SECS, MODEL_ID, REGISTRATION, SAMPLE_RATE,
};

/// This crate's schema-3 component licence rows — one per loaded artifact (sc-16663). The audio
/// catalog concatenates every provider crate's slice into the licence table it ships.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[model::COMPONENT_LICENSE];

/// The provider→component mapping this crate contributes. A provider's effective terms are
/// **derived** from these components by [`gen_core::provider_terms`] — never hand-authored — so
/// they cannot drift from the component rows they summarize.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: model::MODEL_ID,
    components: &[model::COMPONENT_KEY],
}];
