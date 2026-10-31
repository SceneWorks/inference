//! S3Gen — the speech-token→waveform stack, assembled end-to-end (sc-13239).
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
//! Each of the four networks is ported natively and gated on real weights on its own
//! (s3tokenizer — sc-13235, [`crate::s3tokenizer`]; CAMPPlus — sc-13236, [`crate::campplus`]; the
//! flow-matching token→mel decoder — sc-13237, [`crate::flow`]; the HiFTNet vocoder — sc-13238,
//! [`crate::hift`]). This module is the sc-13239 **integration**: it assembles them behind
//! [`S3Gen`] so a real 24 kHz cloned-voice waveform is rendered from T3's speech tokens plus the
//! reference clip's S3Gen conditioning (prompt tokens + prompt mel + the CAMPPlus x-vector), the
//! stage [`crate::model`]'s `generate()` calls after the T3 LM.

use std::path::Path;

use candle_audio::gen_core::{AudioTrack, Progress};
use candle_audio::{AudioError, Result};

use crate::campplus::Campplus;
use crate::config::{DEC_COND_LEN, S3GEN_SR};
use crate::flow::Flow;
use crate::hift::HiftGenerator;
use crate::s3tokenizer::S3Tokenizer;

/// The relative filename of the S3Gen checkpoint inside a Chatterbox snapshot.
pub const S3GEN_WEIGHTS_FILE: &str = "s3gen.safetensors";

/// Number of S3Gen **networks** still to port — **zero**: the s3tokenizer (sc-13235), the CAMPPlus
/// x-vector (sc-13236), the flow-matching token→mel decoder (sc-13237), and the HiFTNet vocoder
/// (sc-13238) are all ported, and sc-13239 assembles them here into an end-to-end token→waveform
/// pipeline ([`S3Gen`]).
pub const S3GEN_REMAINING_NETWORKS: usize = 0;

/// Cap a reference clip at the S3Gen decoder-conditioning length ([`DEC_COND_LEN`], 10 s at
/// [`S3GEN_SR`]) expressed in the clip's own sample rate — the reference truncates the S3Gen
/// reference to 10 s before deriving its mel / tokens / x-vector.
fn cap_reference(samples: &[f32], sample_rate: u32) -> &[f32] {
    // DEC_COND_LEN is defined at 24 kHz; scale to the clip's rate so 10 s is 10 s at any input rate.
    let max = (DEC_COND_LEN as u64 * sample_rate as u64 / S3GEN_SR as u64) as usize;
    &samples[..samples.len().min(max)]
}

/// The assembled S3Gen token→waveform stack: the s3tokenizer (reference prompt tokens), the CAMPPlus
/// speaker encoder (the 80-d flow x-vector), the flow-matching token→mel decoder, and the HiFTNet
/// vocoder. Built once from a Chatterbox snapshot and reused across requests.
pub struct S3Gen {
    tokenizer: S3Tokenizer,
    campplus: Campplus,
    flow: Flow,
    hift: HiftGenerator,
}

// The production stage driver accepts plain operations so its event, error and cancellation
// protocol can be verified without executing any of the learned networks.
fn render_stages<C, M, W>(
    speech_tokens: &[u32],
    on_progress: &mut dyn FnMut(Progress),
    should_cancel: &dyn Fn() -> bool,
    reference_conditioning: impl FnOnce() -> Result<C>,
    flow: impl FnOnce(&C) -> Result<M>,
    vocode: impl FnOnce(&M) -> Result<W>,
) -> Result<Option<W>> {
    if speech_tokens.is_empty() {
        return Err(AudioError::Msg(
            "s3gen: T3 produced no speech tokens to render".into(),
        ));
    }
    if should_cancel() {
        return Ok(None);
    }
    on_progress(Progress::Step {
        current: 1,
        total: 3,
    });
    let conditioning = reference_conditioning()?;
    if should_cancel() {
        return Ok(None);
    }
    on_progress(Progress::Step {
        current: 2,
        total: 3,
    });
    let mel = flow(&conditioning)?;
    if should_cancel() {
        return Ok(None);
    }
    on_progress(Progress::Step {
        current: 3,
        total: 3,
    });
    on_progress(Progress::Decoding);
    Ok(Some(vocode(&mel)?))
}

impl S3Gen {
    /// Load all four S3Gen networks from a Chatterbox snapshot directory (each reads its own
    /// prefix of `s3gen.safetensors`).
    pub fn from_snapshot(dir: &Path) -> Result<Self> {
        let tokenizer = S3Tokenizer::from_snapshot(dir)?;
        let campplus = Campplus::from_snapshot(dir)?;
        let flow = Flow::from_snapshot(dir)?;
        let hift = HiftGenerator::from_snapshot(dir)?;
        Ok(Self {
            tokenizer,
            campplus,
            flow,
            hift,
        })
    }

    /// Derive S3Gen's reference conditioning from a clip: the 25 Hz prompt speech tokens, the 24 kHz
    /// 80-bin prompt mel, and the 80-d flow speaker embedding (L2-normalized CAMPPlus x-vector →
    /// `spk_embed_affine_layer`). VoiceEmbedding-only conditioning cannot supply any of these (see
    /// [`crate::model`]), so a full clone requires a reference clip.
    fn reference_conditioning(
        &self,
        reference: &AudioTrack,
    ) -> Result<(Vec<u32>, candle_audio::candle_core::Tensor, Vec<f32>)> {
        if reference.channels != 1 {
            return Err(AudioError::Msg(format!(
                "s3gen: expected provider-prepared mono reference audio, got {} channels",
                reference.channels
            )));
        }
        let sr = reference.sample_rate;
        let clip = cap_reference(&reference.samples, sr);
        if clip.is_empty() {
            return Err(AudioError::Msg(
                "s3gen: the reference clip is empty — cannot derive the S3Gen conditioning".into(),
            ));
        }
        let prompt_tokens: Vec<u32> = self
            .tokenizer
            .encode(clip, sr)?
            .into_iter()
            .map(|c| c as u32)
            .collect();
        let prompt_mel = self
            .flow
            .mel_extractor()
            .mel(clip, sr, self.flow.device())?;
        let spk_embed = self.campplus.spk_embed_flow(clip, sr)?;
        Ok((prompt_tokens, prompt_mel, spk_embed))
    }

    /// Render T3's `speech_tokens` into a **24 kHz** cloned-voice waveform in the reference voice:
    /// derive the reference conditioning, run the flow-matching token→mel decoder, then vocode the
    /// mel with HiFTNet. `seed` seeds the flow noise and the NSF source (the reproducibility law:
    /// same tokens + reference + seed ⇒ byte-identical samples). Progress is reported over the three
    /// stages, with one `Decoding` event immediately before HiFTNet vocoding;
    /// `should_cancel` is polled between them and returns `Ok(None)` when tripped (mirroring
    /// the T3 stage's cooperative cancellation).
    pub fn render(
        &self,
        speech_tokens: &[u32],
        reference: &AudioTrack,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
        should_cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        render_stages(
            speech_tokens,
            on_progress,
            should_cancel,
            || self.reference_conditioning(reference),
            |(prompt_tokens, prompt_mel, spk_embed)| {
                self.flow
                    .inference(speech_tokens, prompt_tokens, prompt_mel, spk_embed, seed)
            },
            |mel| {
                let wav = self.hift.decode(mel, seed)?;
                Ok(wav.flatten_all()?.to_vec1::<f32>()?)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_driver_announces_one_decode_at_the_actual_vocoder_call() {
        use std::cell::RefCell;
        #[derive(Debug, PartialEq)]
        enum Mark {
            Event(Progress),
            Body(u8),
        }
        let marks = RefCell::new(Vec::new());
        let out = render_stages(
            &[1],
            &mut |p| marks.borrow_mut().push(Mark::Event(p)),
            &|| false,
            || {
                marks.borrow_mut().push(Mark::Body(1));
                Ok(7u8)
            },
            |conditioning| {
                assert_eq!(*conditioning, 7);
                marks.borrow_mut().push(Mark::Body(2));
                Ok(9u8)
            },
            |mel| {
                assert_eq!(*mel, 9);
                marks.borrow_mut().push(Mark::Body(3));
                Ok(vec![0.25f32])
            },
        )
        .unwrap();
        assert_eq!(out, Some(vec![0.25]));
        assert_eq!(
            *marks.borrow(),
            [
                Mark::Event(Progress::Step {
                    current: 1,
                    total: 3
                }),
                Mark::Body(1),
                Mark::Event(Progress::Step {
                    current: 2,
                    total: 3
                }),
                Mark::Body(2),
                Mark::Event(Progress::Step {
                    current: 3,
                    total: 3
                }),
                Mark::Event(Progress::Decoding),
                Mark::Body(3),
            ]
        );
    }

    #[test]
    fn stage_driver_keeps_empty_input_cancellation_and_errors_before_unreached_work() {
        use std::cell::Cell;
        for cancel_at in 0..3 {
            let checks = Cell::new(0);
            let calls = Cell::new(0);
            let mut events = Vec::new();
            let out = render_stages(
                &[1],
                &mut |p| events.push(p),
                &|| {
                    let checkpoint = checks.get();
                    checks.set(checkpoint + 1);
                    checkpoint == cancel_at
                },
                || {
                    calls.set(calls.get() + 1);
                    Ok(1u8)
                },
                |_| {
                    calls.set(calls.get() + 1);
                    Ok(2u8)
                },
                |_| {
                    calls.set(calls.get() + 1);
                    Ok(3u8)
                },
            )
            .unwrap();
            assert_eq!(out, None);
            assert_eq!(calls.get(), cancel_at);
            assert!(!events.contains(&Progress::Decoding));
        }
        let empty: Result<Option<u8>> = render_stages(
            &[],
            &mut |_| panic!("empty input emitted progress"),
            &|| panic!("empty input entered a cancellation checkpoint"),
            || -> Result<u8> { panic!("empty input entered reference conditioning") },
            |_| -> Result<u8> { panic!("empty input entered flow") },
            |_| -> Result<u8> { panic!("empty input entered vocoding") },
        );
        assert!(matches!(empty, Err(AudioError::Msg(_))));
        for fail_at in 1..=3 {
            let calls = Cell::new(0);
            let mut body = || {
                let stage = calls.get() + 1;
                calls.set(stage);
                if stage == fail_at {
                    Err(AudioError::Msg("stage failure".into()))
                } else {
                    Ok(stage)
                }
            };
            let body = std::cell::RefCell::new(&mut body);
            let mut events = Vec::new();
            let out = render_stages(
                &[1],
                &mut |p| events.push(p),
                &|| false,
                || (body.borrow_mut())(),
                |_| (body.borrow_mut())(),
                |_| (body.borrow_mut())(),
            );
            assert!(matches!(out, Err(AudioError::Msg(ref message)) if message == "stage failure"));
            assert_eq!(calls.get(), fail_at);
            assert_eq!(
                events.iter().filter(|p| **p == Progress::Decoding).count(),
                usize::from(fail_at == 3)
            );
        }
    }

    #[test]
    fn all_four_networks_are_ported() {
        assert_eq!(S3GEN_REMAINING_NETWORKS, 0);
    }

    #[test]
    fn cap_reference_bounds_to_ten_seconds_at_any_rate() {
        // 24 kHz: 10 s cap = 240_000 samples.
        let long = vec![0.0f32; 24_000 * 15];
        assert_eq!(cap_reference(&long, 24_000).len(), 240_000);
        // 16 kHz: the same 10 s window = 160_000 samples.
        let long16 = vec![0.0f32; 16_000 * 15];
        assert_eq!(cap_reference(&long16, 16_000).len(), 160_000);
        // A short clip is returned whole.
        let short = vec![0.0f32; 4_000];
        assert_eq!(cap_reference(&short, 24_000).len(), 4_000);
    }
}
