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
//! (192→80) derivation. sc-13237 ports the **flow-matching token→mel decoder** ([`flow`]) — the
//! third S3Gen network: token `Embedding(6561→512)` + an [`flow_encoder::UpsampleConformerEncoder`]
//! (output 512, 8 heads, 6+4 blocks, 25 Hz→50 Hz) + `encoder_proj(512→80)` feeding a
//! `CausalConditionalCFM` (Euler flow-matching, 10 steps, cosine schedule, CFG 0.7) over a
//! `ConditionalDecoder` U-Net estimator (in 320, 12 DiT mid-blocks), plus the 24 kHz prompt-mel
//! front-end ([`mel24`]). sc-13238 ports the **HiFTNet vocoder** ([`hift`]) — the fourth and last
//! S3Gen network: a `ConvRNNF0Predictor`, an NSF harmonic-plus-noise source (`nb_harmonics = 8`), a
//! weight-normed `ConvTranspose1d` upsample trunk (`[8, 5, 3]`) with MRF resblocks and per-stage
//! source injection, and an iSTFT head (`n_fft = 16`, `hop = 4`) → a 24 kHz waveform (480
//! samples/mel-frame). With it, **all four** S3Gen networks are now ported.
//!
//! sc-13239 lands the end-to-end **token→waveform integration** ([`s3gen::S3Gen`]: tokenize → flow
//! → vocode → PerTh watermark) so `generate()` renders a real 24 kHz cloned-voice WAV, and the
//! catalog **registration** — `chatterbox_tts` is added to `candle-audio-catalog`'s ordered
//! generator surface and the three bundle smokes. The sc-12838 clone-WAV DoD (a cloned WAV whose
//! `chatterbox_ve` embedding is closer to the reference than to a different-voice control) is gated
//! on real weights by the crate's conformance test.
//!
//! Weights resolve through the audio lane's pinned-SHA hub path (F-029): `ResembleAI/chatterbox`
//! at the same immutable commit the [`candle_audio_chatterbox_ve`] sibling pins.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod campplus;
pub mod config;
pub mod flow;
pub mod flow_encoder;
pub mod hift;
pub mod mel24;
pub mod model;
pub mod perth;
pub mod prepare;
pub mod s3gen;
pub mod s3tokenizer;
pub mod t3;
pub mod text;

pub use campplus::Campplus;
pub use config::{S3GenConfig, S3TokenizerConfig, T3Config};
pub use flow::Flow;
pub use hift::HiftGenerator;
pub use mel24::Mel24Extractor;
pub use model::{
    descriptor, load, load_generator, resolve_pinned_snapshot, ChatterboxGenerator, HUB_REPO,
    HUB_REVISION, MODEL_ID, REGISTRATION, T3_WEIGHTS_FILE, TOKENIZER_FILE, WEIGHT_LICENSE,
    WEIGHT_LICENSE_ENTRY,
};
pub use perth::{
    resolve_perth_weights, snr_db, PerthWatermarker, PERTH_HUB_REPO, PERTH_HUB_REVISION, PERTH_SR,
    PERTH_WEIGHTS_FILE,
};
pub use s3gen::S3Gen;
pub use s3tokenizer::S3Tokenizer;

/// Every model-weight license this crate ships (sc-13332) — the `chatterbox_tts` generator's pinned
/// `ResembleAI/chatterbox` (MIT) checkpoint. `candle-audio-catalog` aggregates this into the
/// release-tooling weight-license manifest.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];

/// Add the Chatterbox clone-TTS generator (`chatterbox_tts`) to an explicit audio registry builder.
/// `candle-audio-catalog` calls this in stable catalog order (sc-13239); this crate's own
/// [`provider_registry`] also uses it for descriptor introspection / conformance.
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
