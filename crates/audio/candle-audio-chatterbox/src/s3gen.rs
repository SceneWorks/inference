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
//! This slice ports the **T3 LM** (the clone's text→speech-token brain — see [`crate::t3`]) and the
//! full provider contract/conditioning surface. The S3Gen stack is **not yet ported**: rather than
//! emit fake audio, [`decode`] returns a typed, precise error naming exactly what remains. The
//! honest partial is tracked by the follow-up stories referenced in the crate docs; the T3 stage is
//! exercised end-to-end on real weights by the conformance test.

use candle_audio::{AudioError, Result};

/// The relative filename of the S3Gen checkpoint inside a Chatterbox snapshot.
pub const S3GEN_WEIGHTS_FILE: &str = "s3gen.safetensors";

/// Number of neural networks in the S3Gen stack still to port (s3tokenizer, CAMPPlus, flow,
/// HiFTNet) — surfaced in the boundary error so the gap is never silent.
pub const S3GEN_REMAINING_NETWORKS: usize = 4;

/// The S3Gen token→waveform decode. Not yet implemented in this slice: returns a typed error
/// describing precisely which components remain, never fabricated audio (the honest-partial law —
/// a fake waveform would pass a naive "non-silent" check while the clone gate must fail honestly).
pub fn decode(_speech_tokens: &[u32]) -> Result<Vec<f32>> {
    Err(AudioError::Msg(format!(
        "chatterbox: the S3Gen token\u{2192}waveform stack is not yet ported ({} networks: \
         s3tokenizer FSQ, CAMPPlus x-vector, CosyVoice flow-matching decoder, HiFTNet vocoder). \
         The T3 speech-token LM IS ported and runs on real weights; this slice stops at the S3Gen \
         boundary rather than emit fake audio. See the crate docs and sc-13222 follow-ups.",
        S3GEN_REMAINING_NETWORKS
    )))
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
                assert!(m.contains("not yet ported"));
                // It must NOT silently return samples.
            }
            other => panic!("expected a typed Msg boundary, got {other:?}"),
        }
    }
}
