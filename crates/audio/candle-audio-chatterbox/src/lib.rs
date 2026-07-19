//! # candle-audio-chatterbox
//!
//! **Chatterbox clone-TTS** generator for the SceneWorks Candle audio lane (sc-13222, epic
//! sc-12833) — Resemble AI's zero-shot voice-cloning speech synthesizer (MIT; commercial use OK)
//! ported natively onto the workspace's pinned candle revision. One candle implementation serves
//! `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio composition root, per
//! `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The pipeline
//!
//! Chatterbox clones a voice from a reference clip and speaks arbitrary text in it via two stages:
//!
//! 1. **T3** ([`t3`]) — a Llama-520M speech-token LM: it embeds the text plus a speaker/voice
//!    conditioning prefix (the `chatterbox_ve` 256-d vector projected in, an optional
//!    Perceiver-resampled prompt, and an emotion-advisor scalar) and autoregressively decodes S3
//!    speech tokens with classifier-free guidance. **Ported here on real `t3_cfg.safetensors`
//!    weights.**
//! 2. **S3Gen** ([`s3gen`]) — a CosyVoice-derived stack (s3tokenizer FSQ + CAMPPlus x-vector +
//!    flow-matching token→mel decoder + HiFTNet NSF/iSTFT vocoder) that renders those speech tokens
//!    into a 24 kHz waveform in the reference voice.
//!
//! ## Port status (honest partial — sc-13222, sc-13235)
//!
//! sc-13222 ported the **T3 LM** (the clone's text→speech-token brain), the text front-end
//! ([`text`] — `punc_norm` + the `EnTokenizer` BPE), the full provider contract, and the
//! conditioning mapping ([`model`]). sc-13235 ports the **s3tokenizer** ([`s3tokenizer`]) — the
//! first of S3Gen's four networks (a Whisper-v2 FSMN mel encoder + FSQ quantizer → 25 Hz speech
//! tokens); it now fills T3's reference-conditioning prompt (empty before). sc-13236 ports the
//! **CAMPPlus speaker encoder** ([`campplus`]) — the D-TDNN x-vector network (an 80-bin Kaldi-fbank
//! → 192-d x-vector) S3Gen's flow conditions on, plus its L2-norm + `spk_embed_affine_layer`
//! (192→80) derivation. The remaining **two** S3Gen networks (the CosyVoice flow-matching decoder
//! and the HiFTNet vocoder — see [`s3gen`]) are **not yet ported**; the generator's `generate()`
//! runs T3 to produce real speech tokens and then returns a typed error at the S3Gen boundary
//! rather than fabricate audio.
//! Consequently the generator is **not yet registered into `candle-audio-catalog`'s shipping
//! surface** (registering a generator that cannot render audio would be a false advertisement, and
//! would fail the gen-core generator conformance suite): that registration, the ordered-id surface
//! extension, and the three bundle smokes are deliberately deferred to the remaining S3Gen slices.
//! This crate is present as a workspace member so its T3 + s3tokenizer stages build, are
//! unit-tested, and are exercised end-to-end on real weights by the conformance test.
//!
//! Weights resolve through the audio lane's pinned-SHA hub path (F-029): `ResembleAI/chatterbox`
//! at the same immutable commit the [`candle_audio_chatterbox_ve`] sibling pins.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod campplus;
pub mod config;
pub mod model;
pub mod prepare;
pub mod s3gen;
pub mod s3tokenizer;
pub mod t3;
pub mod text;

pub use campplus::Campplus;
pub use config::{S3GenConfig, S3TokenizerConfig, T3Config};
pub use model::{
    descriptor, load, load_generator, resolve_pinned_snapshot, ChatterboxGenerator, HUB_REPO,
    HUB_REVISION, MODEL_ID, REGISTRATION, T3_WEIGHTS_FILE, TOKENIZER_FILE,
};
pub use s3tokenizer::S3Tokenizer;

/// Add the Chatterbox generator to an explicit audio registry builder.
///
/// NOTE: `candle-audio-catalog` does **not** call this yet — see the crate-level "Port status" note.
/// It exists so this crate's own [`provider_registry`] can validate the descriptor, and so the
/// catalog wiring is a one-line add once the S3Gen stack lands.
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build this crate's own explicit provider catalog (descriptor introspection / conformance).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_resolves_through_an_explicit_registry() {
        let registry = provider_registry().unwrap();
        let ids: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["chatterbox_tts"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
