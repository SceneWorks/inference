//! S3Gen — the speech-token→waveform stack (sc-13222). **This is the port boundary of the current
//! slice.**
//!
//! Chatterbox's S3Gen is a CosyVoice-derived stack of **four large neural networks** plus custom
//! DSP, none of which candle-transformers provides (verified against the pinned candle revision).
//! Its weights (`s3gen.safetensors`, 2489 tensors) decompose as:
//!
//! | reference submodule          | tensors | what it is                                            |
//! |------------------------------|---------|-------------------------------------------------------|
//! | `tokenizer` (s3tokenizer)    | 103     | a Whisper-v2 mel encoder + **FSQ** quantizer → 25 Hz codes |
//! | `speaker_encoder` (CAMPPlus) | 937     | an 80-fbank → 192-d x-vector speaker network          |
//! | `flow` (CausalMaskedDiffWithXvec + CausalConditionalCFM) | 1121 | an UpsampleConformerEncoder + a ConditionalDecoder U-Net/DiT CFM estimator (flow-matching token→mel) |
//! | `mel2wav` (HiFTGenerator)    | 328     | an NSF harmonic-source + F0-predictor + iSTFT vocoder → 24 kHz |
//!
//! Reproducing intelligible, voice-similar speech requires **all four** to be numerically exact
//! simultaneously (any single error yields noise), plus a non-power-of-two STFT mel front-end
//! (`n_fft = 1920`) and arbitrary-rate resampling — neither available in the shared audio commons.
//! FSQ in particular is absent from the entire candle ecosystem and must be ported from scratch.
//!
//! The T3 LM (the clone's text→speech-token brain — see [`crate::t3`]) and the full provider
//! contract/conditioning surface were ported in sc-13222. The **s3tokenizer** — the first of the
//! four S3Gen networks (the Whisper-v2 FSMN mel encoder + FSQ quantizer) — is ported natively
//! (sc-13235; see [`crate::s3tokenizer`]): it derives the 25 Hz reference speech tokens T3's
//! conditioning prompt and S3Gen's `prompt_token` need. The **CAMPPlus speaker encoder** — the
//! second network (an 80-bin Kaldi-fbank → 192-d D-TDNN x-vector) — is ported (sc-13236;
//! see [`crate::campplus`]): it derives the S3Gen flow's speaker conditioning. The **flow-matching
//! token→mel decoder** — the third network — is ported (sc-13237; see [`crate::flow`]). The
//! **HiFTNet vocoder** — the fourth and last network (an NSF harmonic-source + F0-predictor + iSTFT
//! mel→waveform) — is now ported too (sc-13238; see [`crate::hift`]). So **all four** S3Gen networks
//! are ported, each exercised end-to-end on real weights by the conformance test.
//!
//! What remains is not a *network* but the end-to-end **token→waveform integration** and the catalog
//! **registration** (sc-13239): assembling s3tokenizer → flow → [`crate::hift`] behind
//! [`decode`], then registering the generator into `candle-audio-catalog`. Until that lands
//! [`decode`] returns a typed, precise error rather than emit fake audio (the honest-partial law).

use candle_audio::{AudioError, Result};

/// The relative filename of the S3Gen checkpoint inside a Chatterbox snapshot.
pub const S3GEN_WEIGHTS_FILE: &str = "s3gen.safetensors";

/// Number of S3Gen **networks** still to port — now **zero**: the s3tokenizer (sc-13235;
/// [`crate::s3tokenizer`]), the CAMPPlus x-vector (sc-13236; [`crate::campplus`]), the flow-matching
/// token→mel decoder (sc-13237; [`crate::flow`]), and the HiFTNet vocoder (sc-13238; [`crate::hift`])
/// are all ported. What remains is the end-to-end token→waveform *integration* + catalog
/// registration (sc-13239), not a model port.
pub const S3GEN_REMAINING_NETWORKS: usize = 0;

/// The S3Gen token→waveform decode. The end-to-end assembly (s3tokenizer → flow → HiFTNet vocoder)
/// and the provider registration are sc-13239; until they land this returns a typed error naming the
/// gap rather than fabricated audio (the honest-partial law — a fake waveform would pass a naive
/// "non-silent" check while the clone gate must fail honestly). Every S3Gen *network* — including
/// the mel→waveform HiFTNet vocoder ([`crate::hift`]) — is ported and runs on real weights.
pub fn decode(_speech_tokens: &[u32]) -> Result<Vec<f32>> {
    Err(AudioError::Msg(
        "chatterbox: all four S3Gen networks (s3tokenizer, CAMPPlus x-vector, flow-matching \
         token\u{2192}mel decoder, and HiFTNet mel\u{2192}waveform vocoder) ARE ported and run on \
         real weights, but the end-to-end S3Gen token\u{2192}waveform integration and provider \
         registration are not yet wired (sc-13239); this stops at the S3Gen boundary rather than \
         emit fake audio. See the crate docs."
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_is_an_honest_typed_boundary_not_fake_audio() {
        let err = decode(&[1, 2, 3]).unwrap_err();
        match err {
            AudioError::Msg(m) => {
                assert!(m.contains("S3Gen"));
                // All four networks are ported; the honest boundary is now the end-to-end
                // integration (sc-13239), not a missing model.
                assert!(m.contains("integration"));
                assert!(m.contains("sc-13239"));
                // It must NOT silently return samples.
            }
            other => panic!("expected a typed Msg boundary, got {other:?}"),
        }
    }

    #[test]
    fn all_four_networks_are_ported() {
        assert_eq!(S3GEN_REMAINING_NETWORKS, 0);
    }
}
